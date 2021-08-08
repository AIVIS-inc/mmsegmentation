#************For System*********************
import argparse
from numpy import dtype, mod
import torch
import os
import shutil
from tqdm import tqdm
import math
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

#************For segmentation*********************
import os.path as osp
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.image import tensor2imgs
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision import models

#**************** Environment*********************
# HomePC: /media/mingfan/DATASSD/PAIP2021/mmSegUptoData

#@ouput Path ********************************
#dir_mask_Seg = "/media/ming-aivis/SSD500_linux/PAIP2021/Eval_Framework/Validate_vit_uper_84"
dir_mask_Seg_binary = "/media/mingfan/DATASSD/PAIP2021/Eval_Framework/Test_Phase/Test_result_Overlap256_Prob95_small_512_vis"
os.makedirs(dir_mask_Seg_binary, exist_ok=True)
dir_mask_Seg_contour = "/media/mingfan/DATASSD/PAIP2021/Eval_Framework/Test_Phase/Test_result_Overlap256_Prob95_small_512_contour"
os.makedirs(dir_mask_Seg_contour, exist_ok=True)

dir_mask_Seg_polygon = "/media/mingfan/DATASSD/PAIP2021/Eval_Framework/Test_Phase/Test_result_Overlap256_Prob95_small_512_polygon"
os.makedirs(dir_mask_Seg_polygon, exist_ok=True)
#@config file *******************************
config ='/media/mingfan/DATASSD/PAIP2021/Eval_Framework/configs/PAIP_SWIN.py'
#@weight Path *******************************
weightPath = '/media/mingfan/DATASSD/PAIP2021/Eval_Framework/weight/SWIN_Upnet_ALL/PAIP_SWIN/latest.pth'
#weightPath = '/media/mingfan/DATASSD/PAIP2021/Eval_Framework/weight/PAIP_SWIN/latest.pth'



def contour(bw_img_swin):
    bw_contour_swin = np.zeros((512,512), np.uint8)
    bw_contour_swin_border = np.zeros((512,512), np.uint8)
    contours, hierarchy = cv2.findContours(bw_img_swin.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bw_contour_swin = cv2.polylines(bw_contour_swin, contours, isClosed=False, color=1, thickness=16) 
    bw_contour_swin_border[10:512-10, 10:512-10] = bw_contour_swin[10:512-10, 10:512-10]
    #cv2.imshow("Binary Image",bw_contour_swin_border*255)
    #cv2.imshow("Binary Image",bw_contour_swin_border*255)
    #cv2.waitKey(0)
    return bw_contour_swin_border

    cv2.waitKey(0)
def main():
    cfg = mmcv.Config.fromfile(config)
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, weightPath, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))
            out_file = osp.join(dir_mask_Seg_binary, img_meta['ori_filename'])
            model.module.show_result(
                img_show,
                result,
                palette=dataset.PALETTE,
                show=True,
                out_file=out_file,
                opacity=0.3)
        #outputs, img_name = single_gpu_test(model, data_loader, show, dir_mask_Seg, None, 0.3)
        img_meta_folder_name = img_meta['ori_filename'].split('/')
        out_name_binary = os.path.join(dir_mask_Seg_contour, img_meta_folder_name[0])
        os.makedirs(out_name_binary, exist_ok=True)
        out_contour_name = os.path.join(out_name_binary, img_meta_folder_name[1])

        out_name_binary_polygon = os.path.join(dir_mask_Seg_polygon, img_meta_folder_name[0])
        os.makedirs(out_name_binary_polygon, exist_ok=True)
        out_polygon_name = os.path.join(out_name_binary_polygon, img_meta_folder_name[1])

        batch_size = len(result)
        bw_contour_swin_border = contour(result[0])
        for _ in range(batch_size):
            prog_bar.update()
            cv2.imwrite(out_polygon_name, result[0]*255)
            cv2.imwrite(out_contour_name, bw_contour_swin_border*255)
            #For safe saving
            cv2.waitKey(1)
        
if __name__ == "__main__":
    main()


