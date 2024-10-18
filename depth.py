import os
import glob
from torchvision.transforms import Compose

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from tqdm import tqdm
import numpy as np
import matplotlib


import torch
import cv2
import torch.nn.functional as F


dates = ['2011_09_29', '2011_09_26', '2011_10_03', '2011_09_30', '2011_09_28']

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
])


depth_v2_load_from = '/home/qiwei/program/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitb.pth'

def get_images_from_directory(root_dir, extensions=['.jpg', '.png', '.jpeg', '.bmp', '.gif'], grd_image_width=1024, grd_image_height=256, sat_width=101):
    
    depth_anything_v2 = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth': 80})
    depth_anything_v2.load_state_dict(torch.load(depth_v2_load_from, map_location='cpu'))
    depth_anything_v2 = depth_anything_v2.to('cuda').eval()
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    # depth_anything_v2 = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 80})
    # depth_anything_v2.load_state_dict(torch.load(depth_v2_load_from, map_location='cpu'))
    # depth_anything_v2 = depth_anything_v2.to('cuda').eval()

    v, u = torch.meshgrid(torch.arange(0, grd_image_height, dtype=torch.float32),
                            torch.arange(0, grd_image_width, dtype=torch.float32))
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0).to('cuda')
    grd_image_files = []
    with torch.no_grad():
        for idx in range(20):
            image_path='/home/qiwei/program/deep-burst-sr/indoor/viz_rggb/{}.3_lr.png'.format(idx)
            image_sr_path = '/home/qiwei/program/deep-burst-sr/indoor/viz_rggb/{}.2_sr.png'.format(idx)
            
            image_cv2 = cv2.imread(image_path)
            depth = depth_anything_v2.infer_image(image_cv2)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            cv2.imwrite('/home/qiwei/program/deep-burst-sr/indoor/depth_rggb/{}.lr.png'.format(idx), depth)

            image_cv2 = cv2.imread(image_sr_path)
            depth = depth_anything_v2.infer_image(image_cv2)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            cv2.imwrite('/home/qiwei/program/deep-burst-sr/indoor/depth_rggb/{}.sr.png'.format(idx), depth)
# Example usage
root_directory = '/home/qiwei/program/deep-burst-sr/indoor/viz_rggb/'  # Replace with your directory path
images = get_images_from_directory(root_directory)

for img in images:
    print(img)