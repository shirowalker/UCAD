import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse

def show_anns(anns,path):
    if len(anns) == 0:
        print('error '+path)
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    # img[:,:,3] = 0
    cur = 1
    img_color = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
    for ann in sorted_anns:
        m = ann['segmentation']
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        # img[m] = color_mask
        # color = np.random.random(3)
        img_color[m] = cur
        cur += 1
        # cv2.imwrite('/apdcephfs/private_jiaqiiliu/PromptAD/prompt_seg_ad/test_results/bottle_224_origin_color'+str(cur)+'.png',img_color*255)
        # cur+=1
    cv2.imwrite(path,img_color)


# image = cv2.resize(image,[224,224])

# print(image.shape)
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import glob

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_type', default='vit_b')
    parser.add_argument('--sam_checkpoint', default='/hhd3/ljq/checkpoints/sam_vit_b_01ec64.pth') # checkpoint path for sam
    parser.add_argument('--data_path', default='/hhd3/m3lab/data/mvtec2d-sam-b') # a special copy for sam segmentation result of dataset
    opt = parser.parse_args()
    sam_checkpoint = opt.sam_checkpoint
    model_type = opt.sam_type
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    data_path = opt.data_path
    train_data = glob.glob(data_path+'/*/train/*/*.png')
    test_data = glob.glob(data_path+'/*/test/*/*.png')
    print(len(train_data))
    print(len(test_data))
    for data in train_data:
        image = cv2.imread(data)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if(image.shape[1]<=224):
            pass
            # print('pass')
        else:
            # print('do')
            image = cv2.resize(image,[224,224])
            mask_generator = SamAutomaticMaskGenerator(sam)
            masks = mask_generator.generate(image)
            show_anns(masks,data)

    for data in test_data:
        image = cv2.imread(data)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if(image.shape[1]<=224):
            pass
            # print('pass')
        else:
            # print('do')
            image = cv2.resize(image,[224,224])
            mask_generator = SamAutomaticMaskGenerator(sam)
            masks = mask_generator.generate(image)
            show_anns(masks,data)