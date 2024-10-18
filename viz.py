import cv2
import torch
import numpy as np
from dataset import SyntheticBurstVal
from processing import SimplePostProcess
import os

mode = 'rggb_1'

def rggb_to_rgb(rggb_img):
    # Split the RGGB channels
    R = rggb_img[0, :, :]
    G1 = rggb_img[1, :, :]
    G2 = rggb_img[2, :, :]
    B = rggb_img[3, :, :]
    
    # Combine the two green channels into one by averaging
    G = (G1 + G2) / 2
    
    # Stack the R, G, and B channels to form an RGB image
    rgb_img = torch.stack((R, G, B), dim=0)
    
    return rgb_img

def grbg_to_rgb(grbg_img):
    # Split the GRBG channels
    G1 = grbg_img[0, :, :]  # Green 1
    R = grbg_img[1, :, :]   # Red
    B = grbg_img[2, :, :]   # Blue
    G2 = grbg_img[3, :, :]  # Green 2
    
    # Combine the two green channels into one by averaging
    G = (G1 + G2) / 2
    
    # Stack the R, G, and B channels to form an RGB image
    rgb_img = torch.stack((R, G, B), dim=0)
    
    return rgb_img

def gbrg_to_rgb(gbrg_img):
    # Split the GBRG channels
    G1 = gbrg_img[0, :, :]  # Green 1
    B = gbrg_img[1, :, :]   # Blue
    R = gbrg_img[2, :, :]   # Red
    G2 = gbrg_img[3, :, :]  # Green 2
    
    # Combine the two green channels into one by averaging
    G = (G1 + G2) / 2
    
    # Stack the R, G, and B channels to form an RGB image
    rgb_img = torch.stack((R, G, B), dim=0)
    
    return rgb_img

def bggr_to_rgb(bggr_img):
    # Split the BGGR channels
    B = bggr_img[0, :, :]   # Blue
    G1 = bggr_img[1, :, :]  # Green 1
    G2 = bggr_img[2, :, :]  # Green 2
    R = bggr_img[3, :, :]   # Red
    
    # Combine the two green channels into one by averaging
    G = (G1 + G2) / 2
    
    # Stack the R, G, and B channels to form an RGB image
    rgb_img = torch.stack((R, G, B), dim=0)
    
    return rgb_img

results_dir = '/home/qiwei/program/deep-burst-sr/indoor'
results_name = f'DBSR_syn_{mode}'
viz_dir = f'viz_{mode}'

dataset = SyntheticBurstVal(f'/home/qiwei/program/deep-burst-sr/indoor/synthetic_{mode}')
process_fn = SimplePostProcess(return_np=True)
os.makedirs('{}/{}'.format(results_dir, viz_dir), exist_ok=True)

for idx in range(20):
    burst, gt, meta_info = dataset[idx]
    burst_name = meta_info['burst_name']

    pred_all = []
    titles_all = []

    pred_path = '{}/{}/{}.png'.format(results_dir, results_name, burst_name)
    pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    pred = torch.from_numpy(pred.astype(np.float32) / 2 ** 14).permute(2, 0, 1)
    pred_all.append(pred)

    gt = process_fn.process(gt, meta_info)
    pred_all = [process_fn.process(p, meta_info) for p in pred_all]
    if mode == 'rggb' or mode == 'rggb_1':
        burst = process_fn.process(rggb_to_rgb(burst[0]), meta_info)
    elif mode == 'grbg':
        # burst = process_fn.process(grbg_to_rgb(burst[0]), meta_info)
        burst = process_fn.process(rggb_to_rgb(burst[0]), meta_info)
    elif mode == 'gbrg':
        burst = process_fn.process(gbrg_to_rgb(burst[0]), meta_info)
    elif mode == 'bggr':
        burst = process_fn.process(bggr_to_rgb(burst[0]), meta_info)

    cv2.imwrite('{}/{}/{}.1_gt.png'.format(results_dir, viz_dir, idx), gt)
    cv2.imwrite('{}/{}/{}.2_sr.png'.format(results_dir, viz_dir, idx), pred_all[0])
    cv2.imwrite('{}/{}/{}.3_lr.png'.format(results_dir, viz_dir, idx), burst)

