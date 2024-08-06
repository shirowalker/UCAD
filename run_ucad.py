import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch
import tqdm
import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import label
from bisect import bisect
import time
from metric_utils import find_optimal_threshold
import cv2
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

import argparse

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
@click.option("--memory_size", type=int, default=196, show_default=True)
@click.option("--epochs_num", type=int, default=25, show_default=True)
@click.option("--key_size", type=int, default=196, show_default=True)
@click.option("--basic_size", type=int, default=1960, show_default=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
    memory_size,
    epochs_num,
    key_size,
    basic_size,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )
    run_save_path_nolimit = patchcore.utils.create_storage_folder(
        results_path+'_nolimit', log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []
    result_collect_nolimit = []
    
    key_feature_list = [0]*15
    memory_feature_list = [0]*15
    prompt_list = [0]*15
    print(prompt_list)
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name
        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](
                device,
            )
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                # current_task_id task_num
                PatchCore.set_dataloadercount(dataloader_count)
                key_feature = PatchCore.fit_with_limit_size(dataloaders["training"], key_size)
                key_feature_list[dataloader_count] = key_feature


            aggregator = {"scores": [], "segmentations": []}
            basic_aggregator = {"scores": [], "segmentations": []}
            start_time = time.time()
            pr_auroc = 0
            basic_pr_auroc = 0
            args = np.load('./args_dict.npy',allow_pickle=True).item()
            args.lr = 0.0005
            args.decay_epochs = 15#30
            args.warmup_epochs = 3#5
            args.cooldown_epochs = 5#10
            args.patience_epochs = 5#10
            optimizer = create_optimizer(args, PatchCore.prompt_model)
            epochs = epochs_num
            PatchCore.prompt_model.train()
            PatchCore.prompt_model.train_contrastive = True
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None
            best_auroc,best_full_pixel_auroc,best_img_ap,best_pixel_ap,best_pixel_pro,best_time_cost = 0,0,0,0,0,0
            best_basic_auroc,best_basic_full_pixel_auroc,best_basic_img_ap,best_basic_pixel_ap,best_basic_pixel_pro,best_basic_time_cost = 0,0,0,0,0,0
            for epoch in range(epochs):
                for i, PatchCore in enumerate(PatchCore_list):
                    torch.cuda.empty_cache()
                    # '''
                    PatchCore.prompt_model.train()
                    loss_list = []
                    with tqdm.tqdm(dataloaders["training"], desc="training...", leave=False) as data_iterator:
                        for image in data_iterator:
                            # if(image["image"].shape[0]<2):
                            #     continue
                            if isinstance(image, dict):
                                image_paths = image["image_path"]
                                image = image["image"].cuda()
                            # res = self._embed_train_false(image, provide_patch_shapes=True)
                            res = PatchCore._embed_train_sam(image, provide_patch_shapes=True, image_path=image_paths)
                            loss = res['loss']
                            loss_list.append(loss.item())
                            optimizer.zero_grad()
                            if(loss!=0):
                                loss.backward()
                            torch.nn.utils.clip_grad_norm_(PatchCore.prompt_model.parameters(), args.clip_grad)
                            optimizer.step()
                        print("epoch:{} loss:{}".format(epoch,np.mean(loss_list)))    
                    if lr_scheduler:
                        lr_scheduler.step(i)

                    PatchCore.prompt_model.eval()
                    nolimimit_memory_feature = PatchCore.fit_with_limit_size_prompt(dataloaders["training"], basic_size)
                    PatchCore.anomaly_scorer.fit(detection_features=[nolimimit_memory_feature])
                    basic_scores, basic_segmentations, basic_labels_gt, basic_masks_gt = PatchCore.predict_prompt(
                        dataloaders["testing"]
                    )
                    basic_aggregator["scores"].append(basic_scores)
                    basic_aggregator["segmentations"].append(basic_segmentations)
                    basic_end_time = time.time()

                    memory_feature = PatchCore.fit_with_limit_size_prompt(dataloaders["training"], memory_size)
                    PatchCore.anomaly_scorer.fit(detection_features=[memory_feature])
                    scores, segmentations, labels_gt, masks_gt = PatchCore.predict_prompt(
                        dataloaders["testing"]
                    )
                    aggregator["scores"].append(scores)
                    aggregator["segmentations"].append(segmentations)
                    end_time = time.time()
                
                scores = np.array(aggregator["scores"])
                min_scores = scores.min(axis=-1).reshape(-1, 1)
                max_scores = scores.max(axis=-1).reshape(-1, 1)
                scores = (scores - min_scores) / (max_scores - min_scores)
                scores = np.mean(scores, axis=0)
                segmentations = np.array(aggregator["segmentations"])
                min_scores = (
                    segmentations.reshape(len(segmentations), -1)
                    .min(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                max_scores = (
                    segmentations.reshape(len(segmentations), -1)
                    .max(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                segmentations = (segmentations - min_scores) / (max_scores - min_scores)
                segmentations = np.mean(segmentations, axis=0)
                
                
                time_cost = (end_time - basic_end_time)/len(dataloaders["testing"])
                anomaly_labels = [
                    x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                ap_seg = np.asarray(segmentations)
                ap_seg = ap_seg.flatten()
                # LOGGER.info("Computing evaluation metrics.")
                auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                    scores, anomaly_labels
                )["auroc"]
                ap_mask = np.asarray(masks_gt)
                ap_mask = ap_mask.flatten().astype(np.int32)
                pixel_ap = average_precision_score(ap_mask,ap_seg)

                # metric without limit
                basic_scores = np.array(basic_aggregator["scores"])
                basic_min_scores = basic_scores.min(axis=-1).reshape(-1, 1)
                basic_max_scores = basic_scores.max(axis=-1).reshape(-1, 1)
                basic_scores = (basic_scores - basic_min_scores) / (basic_max_scores - basic_min_scores)
                basic_scores = np.mean(basic_scores, axis=0)
                basic_segmentations = np.array(basic_aggregator["segmentations"])
                basic_min_scores = (
                    basic_segmentations.reshape(len(basic_segmentations), -1)
                    .min(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                basic_max_scores = (
                    basic_segmentations.reshape(len(basic_segmentations), -1)
                    .max(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                basic_segmentations = (basic_segmentations - basic_min_scores) / (basic_max_scores - basic_min_scores)
                basic_segmentations = np.mean(basic_segmentations, axis=0)
                
                basic_time_cost = (basic_end_time - start_time)/len(dataloaders["testing"])
                basic_anomaly_labels = [
                    x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                basic_ap_seg = np.asarray(basic_segmentations)
                basic_ap_seg = basic_ap_seg.flatten()
                # LOGGER.info("Computing evaluation metrics.")
                basic_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                    basic_scores, basic_anomaly_labels
                )["auroc"]
                basic_ap_mask = np.asarray(basic_masks_gt)
                basic_ap_mask = basic_ap_mask.flatten().astype(np.int32)
                basic_pixel_ap = average_precision_score(basic_ap_mask,basic_ap_seg)

                if(auroc>pr_auroc):
                    memory_feature_list[dataloader_count] = memory_feature
                    prompt_list[dataloader_count] = PatchCore.prompt_model.get_cur_prompt()
                    if(pr_auroc!=0):
                        result_collect.pop()
                    pr_auroc = auroc
                    img_ap = average_precision_score(anomaly_labels,scores)
                    # Compute PRO score & PW Auroc for all images
                    segmentations = ap_seg.reshape(-1,224,224)
                    # (Optional) Plot example images.
                    if save_segmentation_images:
                        image_paths = [
                            x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                        ]
                        mask_paths = [
                            x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                        ]

                        def image_transform(image):
                            in_std = np.array(
                                dataloaders["testing"].dataset.transform_std
                            ).reshape(-1, 1, 1)
                            in_mean = np.array(
                                dataloaders["testing"].dataset.transform_mean
                            ).reshape(-1, 1, 1)
                            image = dataloaders["testing"].dataset.transform_img(image)
                            return np.clip(
                                (image.numpy() * in_std + in_mean) * 255, 0, 255
                            ).astype(np.uint8)

                        def mask_transform(mask):
                            return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                        image_save_path = os.path.join(
                            run_save_path, "segmentation_images", dataset_name
                        )
                        os.makedirs(image_save_path, exist_ok=True)
                        patchcore.utils.plot_segmentation_images(
                            image_save_path,
                            image_paths,
                            segmentations,
                            scores,
                            mask_paths,
                            image_transform=image_transform,
                            mask_transform=mask_transform,
                        )
                    pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                        segmentations, masks_gt
                    )
                    full_pixel_auroc = pixel_scores["auroc"]
                    # Compute PRO score & PW Auroc only images with anomalies
                    sel_idxs = []
                    for i in range(len(masks_gt)):
                        if np.sum(masks_gt[i]) > 0:
                            sel_idxs.append(i)
                    pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                        [segmentations[i] for i in sel_idxs],
                        [masks_gt[i] for i in sel_idxs],
                    )
                    anomaly_pixel_auroc = pixel_scores["auroc"]
                    for i,mask in enumerate(masks_gt):
                        masks_gt[i] = np.array(mask[0])
                    for i,seg in enumerate(segmentations):
                        segmentations[i] = np.array(seg)
                    pixel_pro, pro_curve  = calculate_au_pro(np.array(masks_gt),np.array(segmentations))
                    result_collect.append(
                        {
                            "dataset_name": dataset_name,
                            "instance_auroc": auroc,
                            "full_pixel_auroc": full_pixel_auroc,
                            "anomaly_pixel_auroc": anomaly_pixel_auroc,
                            "image_ap": img_ap,
                            "pixel_ap": pixel_ap,
                            "pixel_pro": pixel_pro,
                            "time_cost:": time_cost
                        }
                    )

                    print('current task:{}/train task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, pixel_pro:{}, time_cost:{}'.format
                            (dataloader_count+1,dataloader_count+1,auroc,full_pixel_auroc,img_ap,pixel_ap,pixel_pro,time_cost))
                    best_auroc,best_full_pixel_auroc,best_img_ap,best_pixel_ap,best_pixel_pro,best_time_cost = auroc,full_pixel_auroc,img_ap,pixel_ap,pixel_pro,time_cost
                    if(auroc==1):
                        break
                
                # calc aupro and save metric without memory limit
                if(basic_auroc>basic_pr_auroc):
                    if(basic_pr_auroc!=0):
                        result_collect_nolimit.pop()
                    basic_pr_auroc = basic_auroc
                    basic_img_ap = average_precision_score(basic_anomaly_labels,basic_scores)
                    # Compute PRO score & PW Auroc for all images
                    basic_segmentations = basic_ap_seg.reshape(-1,224,224)
                    # (Optional) Plot example images.
                    
                    basic_pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                        basic_segmentations, basic_masks_gt
                    )
                    basic_full_pixel_auroc = basic_pixel_scores["auroc"]
                    # Compute PRO score & PW Auroc only images with anomalies
                    basic_sel_idxs = []
                    for i in range(len(basic_masks_gt)):
                        if np.sum(basic_masks_gt[i]) > 0:
                            basic_sel_idxs.append(i)
                    basic_pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                        [basic_segmentations[i] for i in basic_sel_idxs],
                        [basic_masks_gt[i] for i in basic_sel_idxs],
                    )
                    basic_anomaly_pixel_auroc = basic_pixel_scores["auroc"]
                    for i,mask in enumerate(basic_masks_gt):
                        basic_masks_gt[i] = np.array(mask[0])
                    for i,seg in enumerate(basic_segmentations):
                        segmentations[i] = np.array(seg)
                    basic_pixel_pro, basic_pro_curve  = calculate_au_pro(np.array(basic_masks_gt),np.array(basic_segmentations))
                    result_collect_nolimit.append(
                        {
                            "dataset_name": dataset_name,
                            "instance_auroc": basic_auroc,
                            "full_pixel_auroc": basic_full_pixel_auroc,
                            "anomaly_pixel_auroc": basic_anomaly_pixel_auroc,
                            "image_ap": basic_img_ap,
                            "pixel_ap": basic_pixel_ap,
                            "pixel_pro": basic_pixel_pro,
                            "time_cost:": basic_time_cost
                        }
                    )
                    
                #     print('Nolimlit current task:{}/train task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, pixel_pro:{}, time_cost:{}'.format
                #             (dataloader_count+1,dataloader_count+1,basic_auroc,basic_full_pixel_auroc,basic_img_ap,basic_pixel_ap,basic_pixel_pro,basic_time_cost))
                    best_basic_auroc,best_basic_full_pixel_auroc,best_basic_img_ap,best_basic_pixel_ap,best_basic_pixel_pro,best_basic_time_cost = basic_auroc,basic_full_pixel_auroc,basic_img_ap,basic_pixel_ap,basic_pixel_pro,basic_time_cost
            print('Limited current task:{}/train task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, pixel_pro:{}, time_cost:{}'.format
                    (dataloader_count+1,dataloader_count+1,best_auroc,best_full_pixel_auroc,best_img_ap,best_pixel_ap,best_pixel_pro,best_time_cost))
            print('Nolimlited current task:{}/train task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, pixel_pro:{}, time_cost:{}'.format
                    (dataloader_count+1,dataloader_count+1,best_basic_auroc,best_basic_full_pixel_auroc,best_basic_img_ap,best_basic_pixel_ap,best_basic_pixel_pro,best_basic_time_cost))

            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")
    # Inference
    '''
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name
        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](
                device,
            )
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                # current_task_id task_num
                # memory_feature = PatchCore.fit_with_limit_size(dataloaders["training"], memory_size)
                # query_feature = PatchCore.get_mem_limit_size(dataloaders["training"], key_size)
                cur_query_list = []
                for key_count in range(len(list_of_dataloaders)):
                    PatchCore.anomaly_scorer.fit(detection_features=[key_feature_list[key_count]])
                    query_scores, query_seg, labels_gt_query, masks_gt_query = PatchCore.predict(
                        dataloaders["testing"]
                    )
                    cur_query_list.append(np.sum(query_scores))
                print(cur_query_list)
                print('get query dataloader')
                print(np.argmin(cur_query_list))
                query_data_id = np.argmin(cur_query_list)
                PatchCore.set_dataloadercount(query_data_id)
                PatchCore.prompt_model.set_cur_prompt(prompt_list[query_data_id])
                PatchCore.prompt_model.eval()
                PatchCore.anomaly_scorer.fit(detection_features=[memory_feature_list[query_data_id]])
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict_prompt(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
                
            scores = np.array(aggregator["scores"])
            # print(scores.shape)
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)
            
            end_time = time.time()
            time_cost = (end_time - start_time)/len(dataloaders["testing"])
            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            ap_seg = np.asarray(segmentations)
            ap_seg = ap_seg.flatten()
            
            LOGGER.info("Computing evaluation metrics.")
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]
            ap_mask = np.asarray(masks_gt)
            ap_mask = ap_mask.flatten().astype(np.int32)
            pixel_ap = average_precision_score(ap_mask,ap_seg)
            print('current task:{}/test task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, pixel_pro:{}, time_cost:{}'.format
                        (dataloader_count+1,len(list_of_dataloaders),auroc,full_pixel_auroc,img_ap,pixel_ap,pixel_pro,time_cost))
            
    '''
    # Store all results and mean scores to a csv-file.
    # limited result
    print('Average result with limited')
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

    print('Average result without limited memory')
    basic_result_metric_names = list(result_collect_nolimit[-1].keys())[1:]
    basic_result_dataset_names = [results["dataset_name"] for results in result_collect_nolimit]
    basic_result_scores = [list(results.values())[1:] for results in result_collect_nolimit]
    patchcore.utils.compute_and_store_final_results(
        run_save_path_nolimit,
        basic_result_scores,
        column_names=basic_result_metric_names,
        row_names=basic_result_dataset_names,
    )


@main.command("ucad")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=8, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=224, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)

class GroundTruthComponent:
    """
    Stores sorted anomaly scores of a single ground truth component.
    Used to efficiently compute the region overlap for many increasing thresholds.
    """

    def __init__(self, anomaly_scores):
        """
        Initialize the module.

        Args:
            anomaly_scores: List of all anomaly scores within the ground truth
                            component as numpy array.
        """
        # Keep a sorted list of all anomaly scores within the component.
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()

        # Pointer to the anomaly score where the current threshold divides the component into OK / NOK pixels.
        self.index = 0

        # The last evaluated threshold.
        self.last_threshold = None

    def compute_overlap(self, threshold):
        """
        Compute the region overlap for a specific threshold.
        Thresholds must be passed in increasing order.

        Args:
            threshold: Threshold to compute the region overlap.

        Returns:
            Region overlap for the specified threshold.
        """
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold

        # Increase the index until it points to an anomaly score that is just above the specified threshold.
        while (self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold):
            self.index += 1

        # Compute the fraction of component pixels that are correctly segmented as anomalous.
        return 1.0 - self.index / len(self.anomaly_scores)


def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by x- and corresponding y-values.
    In contrast to, e.g., 'numpy.trapz()', this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a warning.

    Args:
        x:     Samples from the domain of the function to integrate need to be sorted in ascending order. May contain
               the same value multiple times. In that case, the order of the corresponding y values will affect the
               integration with the trapezoidal rule.
        y:     Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be determined by interpolating between its
               neighbors. Must not lie outside of the range of x.

    Returns:
        Area under the curve.
    """

    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print(
            """WARNING: Not all x and y values passed to trapezoid are finite. Will continue with only the finite values.""")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between the last x[ins-1] and x_max. Since we do not
            # know the exact value of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    """
    Extract anomaly scores for each ground truth connected component as well as anomaly scores for each potential false
    positive pixel from anomaly maps.

    Args:
        anomaly_maps:      List of anomaly maps (2D numpy arrays) that contain a real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that contain binary-valued ground truth labels
                           for each pixel. 0 indicates that a pixel is anomaly-free. 1 indicates that a pixel contains
                           an anomaly.

    Returns:
        ground_truth_components: A list of all ground truth connected components that appear in the dataset. For each
                                 component, a sorted list of its anomaly scores is stored.

        anomaly_scores_ok_pixels: A sorted list of anomaly scores of all anomaly-free pixels of the dataset. This list
                                  can be used to quickly select thresholds that fix a certain false positive rate.
    """
    # Make sure an anomaly map is present for each ground truth map.
    assert len(anomaly_maps) == len(ground_truth_maps)

    # Initialize ground truth components and scores of potential fp pixels.
    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    # Collect anomaly scores within each ground truth region and for all potential fp pixels.
    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)

        # Store all potential fp scores.
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index:ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
        ok_index += num_ok_pixels

        # Fetch anomaly scores within each GT component.
        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(GroundTruthComponent(component_scores))

    # Sort all potential false positive scores.
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()

    return ground_truth_components, anomaly_scores_ok_pixels


def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    """
    Compute the PRO curve at equidistant interpolation points for a set of anomaly maps with corresponding ground
    truth maps. The number of interpolation points can be set manually.

    Args:
        anomaly_maps:      List of anomaly maps (2D numpy arrays) that contain a real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that contain binary-valued ground truth labels
                           for each pixel. 0 indicates that a pixel is anomaly-free. 1 indicates that a pixel contains
                           an anomaly.

        num_thresholds:    Number of thresholds to compute the PRO curve.
    Returns:
        fprs: List of false positive rates.
        pros: List of correspoding PRO values.
    """
    # Fetch sorted anomaly scores.
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)

    # Select equidistant thresholds.
    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)

    fprs = [1.0]
    pros = [1.0]
    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]

        # Compute the false positive rate for this threshold.
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)

        # Compute the PRO value for this threshold.
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)

        fprs.append(fpr)
        pros.append(pro)

    # Return (FPR/PRO) pairs in increasing FPR order.
    fprs = fprs[::-1]
    pros = pros[::-1]

    return fprs, pros


def calculate_au_pro(gts, predictions, integration_limit=0.3, num_thresholds=100):
    """
    Compute the area under the PRO curve for a set of ground truth images and corresponding anomaly images.
    Args:
        gts:         List of tensors that contain the ground truth images for a single dataset object.
        predictions: List of tensors containing anomaly images for each ground truth image.
        integration_limit:    Integration limit to use when computing the area under the PRO curve.
        num_thresholds:       Number of thresholds to use to sample the area under the PRO curve.

    Returns:
        au_pro:    Area under the PRO curve computed up to the given integration limit.
        pro_curve: PRO curve values for localization (fpr,pro).
    """
    # Compute the PRO curve.
    pro_curve = compute_pro(anomaly_maps=predictions, ground_truth_maps=gts, num_thresholds=num_thresholds)

    # Compute the area under the PRO curve.
    au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit

    # Return the evaluation metrics.
    return au_pro, pro_curve


if __name__ == "__main__":
    # parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    # config = parser.parse_known_args()[-1][0]
    # subparser = parser.add_subparsers(dest='subparser_name')

    # from patchcore.configs.mvtecad_dualprompt import get_args_parser
    # config_parser = subparser.add_parser('mvtecad_dualprompt', help='MVTec AD')
    # get_args_parser(config_parser)
    # args = parser.parse_args()
    # print(args)

    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
