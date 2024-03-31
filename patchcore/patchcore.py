"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

import patchcore.models
from timm.models import create_model
import argparse
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from sklearn.cluster import KMeans


LOGGER = logging.getLogger(__name__)



class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler
        self.dataloader_count = 0

        

        self.model = create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=15,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )
        self.prompt_model = create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=15,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
            prompt_length=1,
            embedding_key="cls",
            prompt_init="uniform",
            prompt_pool=True,
            prompt_key=True,
            pool_size=1,
            top_k=1,
            batchwise_prompt=True,
            prompt_key_init="uniform",
            head_type='token',
            use_prompt_mask=True,
            use_g_prompt=False,
            g_prompt_length=0,
            g_prompt_layer_idx=[],
            use_prefix_tune_for_g_prompt=True,
            use_e_prompt=True,
            e_prompt_layer_idx=[0,1,2,3,4,5,6,7,8,9,10,11],
            use_prefix_tune_for_e_prompt=True,
            same_key_value=False,
            prototype_size=5, # failure version
        )
        self.model.to('cuda')
        self.prompt_model.to('cuda')
        for p in self.model.parameters():
            p.requires_grad = False
        
        ## freeze args.freeze[blocks, patch_embed, cls_token] parameters
        ## print parameters
        for n, p in self.prompt_model.named_parameters():
            if n.startswith(tuple(['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'])):
                p.requires_grad = False
                # print(n)
    
    def set_dataloadercount(self, dataloader_count):
        self.dataloader_count = dataloader_count

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            # features = self.forward_modules["feature_aggregator"](images)
            features = self.model(images)['seg_feat']
            for i in range(len(features)):
                features[i] = features[i].reshape(-1,14,14,768).permute(0,3,1,2)
    
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]


        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def embed_prompt(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed_prompt(input_image))
            return features
        return self._embed_prompt(data)

    def _embed_prompt(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            # features = self.forward_modules["feature_aggregator"](images)
            features = self.prompt_model(images)['seg_feat']
            for i in range(len(features)):
                features[i] = features[i].reshape(-1,14,14,768).permute(0,3,1,2)
    
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def _embed_train(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        # _ = self.model.eval()
        res = self.prompt_model(images,task_id=self.dataloader_count, cls_features=None, train=True) # TODO: Train=True==neg
            # print(features.shape)
        return res

    def _embed_train_false(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        # _ = self.model.eval()
        res = self.prompt_model(images,task_id=self.dataloader_count, cls_features=None, train=False) # TODO: Train=True==neg
            # print(features.shape)
        return res

    def _embed_train_sam(self, images, detach=True, provide_patch_shapes=False, image_path=None):
        """Returns feature embeddings for images."""
        res = self.prompt_model(images,task_id=self.dataloader_count, cls_features=None, train=True, image_path=image_path)
        return res

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
    
    def fit_with_return_feature(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
        return features
    
    def get_all_features(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        return features
    
    def fit_with_limit_size(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        # self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features
    
    def get_mem_limit_size(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._get_mem_limit_size(training_data, limit_size)
        
    def _get_mem_limit_size(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        return features
    
    def fit_with_limit_size_prompt(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_prompt(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_prompt(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed_prompt(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def get_normal_prototypes(self, data, args):
        # switch to evaluation mode
        with torch.no_grad():
            cls_memory = list()
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()

                        output = self.model(image)
                        cls_features = output['pre_logits']
                        cls_memory.append(cls_features.cpu())
        cls_prototypes = torch.cat([cls_f for cls_f in cls_memory],dim=0).numpy()
        kmeans = KMeans(n_clusters=args.prototype_size,random_state=0)
        labels = kmeans.fit_predict(cls_prototypes)
        representatives = torch.zeros(args.prototype_size,768)
        for i in range(args.prototype_size):
            cluster_tensors = cls_prototypes[labels==i]
            representative = np.mean(cluster_tensors,axis=0)
            representatives[i] = torch.from_numpy(representative)

        return representatives
    
    def get_normal_prototypes_instance(self, data, args):
        # switch to evaluation mode

        with torch.no_grad():
            cls_memory = list()
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()

                        output = self.model(image)
                        cls_features = output['pre_logits']
                        cls_memory.append(cls_features.cpu())
        cls_prototypes = torch.cat([cls_f for cls_f in cls_memory],dim=0).numpy()
        kmeans = KMeans(n_clusters=args.prototype_size,n_init=10,max_iter=300).fit(cls_prototypes)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        representatives = torch.zeros(args.prototype_size,768)
        for i in range(args.prototype_size):
            representatives[i] = torch.from_numpy(centers[i])

        return representatives
    
    def get_normal_prototypes_seg(self, data, args):
        with torch.no_grad():
            seg_feat_memory = list()
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()

                        output = self.model(image)
                        seg_feat = output['seg_feat']
                        seg_feat_memory.append(seg_feat[0].cpu())
        seg_prototypes = torch.cat([seg_feat.reshape(-1,196*4*768) for seg_feat in seg_feat_memory],dim=0).numpy()
        kmeans = KMeans(n_clusters=args.prototype_size,n_init=10,max_iter=300).fit(seg_prototypes)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        representatives = torch.zeros(args.prototype_size,196*4,768)
        for i in range(args.prototype_size):
            representatives[i] = torch.from_numpy(centers[i]).reshape(196*4,768)

        return representatives
    
    def get_normal_prototypes_seg_mean(self, data, args):
        with torch.no_grad():
            seg_feat_memory = list()
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()

                        output = self.model(image)
                        seg_feat = output['seg_feat'][0]
                        seg_feat_memory.append(seg_feat.cpu())
        seg_prototypes = torch.cat([seg_feat.reshape(-1,196*4*768) for seg_feat in seg_feat_memory],dim=0).numpy()
        kmeans = KMeans(n_clusters=args.prototype_size,random_state=0)
        labels = kmeans.fit_predict(seg_prototypes)
        representatives = torch.zeros(args.prototype_size,196*4,768)
        for i in range(args.prototype_size):
            cluster_tensors = seg_prototypes[labels==i]
            representative = np.mean(cluster_tensors,axis=0)
            representatives[i] = torch.from_numpy(representative).reshape(196*4,768)

        return representatives

    def train(self, data, dataloader_count, memory_feature):
        args = np.load('../args_dict.npy',allow_pickle=True).item()
        args.prototype_size = 5
        args.lr = 0.0005
        args.decay_epochs = 3#30
        args.warmup_epochs = 1#5
        args.cooldown_epochs = 1#10
        args.patience_epochs = 1#10
        optimizer = create_optimizer(args, self.prompt_model)
        self.dataloader_count = dataloader_count

        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        elif args.sched == 'constant':
            lr_scheduler = None
        # self.prompt_model.set_prompt_seg(dataloader_count,torch.from_numpy(memory_feature).cuda())
        prompt_cls_feature = self.get_normal_prototypes(data, args=args)
        self.prompt_model.set_prompt_cls(dataloader_count,prompt_cls_feature)
        prompt_seg_feature = self.get_normal_prototypes_seg(data, args=args)
        # prompt_seg_feature = self.get_normal_prototypes_seg_mean(data, args=args)
        self.prompt_model.set_prompt_seg(dataloader_count,prompt_seg_feature)
        # self.anomaly_scorer.fit(detection_features=[prompt_seg_feature.clone().detach().reshape(-1,768).cpu().numpy()])


        epochs = 10
        self.prompt_model.train()
        for i in range(epochs):
            loss_list = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()
                    res = self._embed_train(image, provide_patch_shapes=True)
                    loss = res['loss']
                    loss_list.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), args.clip_grad)
                    optimizer.step()
                print("epoch:{} loss:{}".format(i,np.mean(loss_list)))    
            if lr_scheduler:
                lr_scheduler.step(i)
            # prompt_seg_feature = self.get_normal_prototypes_seg(data, args=args)
            # self.prompt_model.set_prompt_seg(dataloader_count,prompt_seg_feature)
        
        # prompt_seg_feature = prompt_seg_feature.reshape(-1,768)
        # print(prompt_seg_feature.shape)

        return prompt_seg_feature
    
    
    def train_contrastive(self, data, dataloader_count, memory_feature=None):
        args = np.load('../args_dict.npy',allow_pickle=True).item()
        args.prototype_size = 5
        args.lr = 0.0005
        # args.decay_epochs = 3#30
        # args.warmup_epochs = 1#5
        # args.cooldown_epochs = 1#10
        # args.patience_epochs = 1#10
        args.decay_epochs = 10#30
        args.warmup_epochs = 2#5
        args.cooldown_epochs = 3#10
        args.patience_epochs = 3#10
        optimizer = create_optimizer(args, self.prompt_model)
        self.dataloader_count = dataloader_count

        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        elif args.sched == 'constant':
            lr_scheduler = None
        epochs = 20
        self.prompt_model.train()
        self.prompt_model.train_contrastive = True
        for i in range(epochs):
            loss_list = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if(image["image"].shape[0]<2):
                        continue
                    if isinstance(image, dict):
                        image = image["image"].cuda()
                    res = self._embed_train_false(image, provide_patch_shapes=True)
                    loss = res['loss']
                    loss_list.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), args.clip_grad)
                    optimizer.step()
                print("epoch:{} loss:{}".format(i,np.mean(loss_list)))    
            if lr_scheduler:
                lr_scheduler.step(i)
        
        # prompt_seg_feature = prompt_seg_feature.reshape(-1,768)
        # print(prompt_seg_feature.shape)

        # return prompt_seg_feature

    #aug data1,data2 contrastive
    # def train_con(self, data, dataloader_count, memory_feature):
        
    def train_sam(self, data, dataloader_count, memory_feature=None):
        args = np.load('../args_dict.npy',allow_pickle=True).item()
        args.lr = 0.0005
        args.decay_epochs = 3#30
        args.warmup_epochs = 1#5
        args.cooldown_epochs = 1#10
        args.patience_epochs = 1#10
        # args.decay_epochs = 10#30
        # args.warmup_epochs = 2#5
        # args.cooldown_epochs = 3#10
        # args.patience_epochs = 3#10
        optimizer = create_optimizer(args, self.prompt_model)
        self.dataloader_count = dataloader_count

        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        elif args.sched == 'constant':
            lr_scheduler = None
        epochs = 10
        self.prompt_model.train()
        self.prompt_model.train_contrastive = True
        for i in range(epochs):
            loss_list = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    # if(image["image"].shape[0]<2):
                    #     continue
                    if isinstance(image, dict):
                        image_paths = image["image_path"]
                        image = image["image"].cuda()
                    # res = self._embed_train_false(image, provide_patch_shapes=True)
                    res = self._embed_train_sam(image, provide_patch_shapes=True, image_path=image_paths)
                    loss = res['loss']
                    loss_list.append(loss.item())
                    optimizer.zero_grad()
                    if(loss!=0):
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), args.clip_grad)
                    optimizer.step()
                print("epoch:{} loss:{}".format(i,np.mean(loss_list)))    
            if lr_scheduler:
                lr_scheduler.step(i)

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)#TODO: baseline
            # features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)
            # features = np.repeat(features,2,axis=1)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]
    
    def predict_prompt(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_prompt(data)
        return self._predict_prompt(data)

    def _predict_dataloader_prompt(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict_prompt(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict_prompt(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed_prompt(images, provide_patch_shapes=True)#TODO: baseline
            # features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            # print(patch_shapes) [32,32]
            features = np.asarray(features)
            # print(features.shape)
            # features = np.repeat(features,2,axis=1)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]
    
    def _predict_past_tasks(self, features, data):
        pass
            
    def _fit_past_tasks(self, features, data):
        pass
        

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
