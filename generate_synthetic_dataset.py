import dataset as datasets
from data import processing, sampler
import data.transforms as tfm
import os
import numpy as np
import cv2
import pickle as pkl

mode = 'rggb_1'

def main():
    image_folder_path = '/home/qiwei/program/deep-burst-sr/indoor/original/'
    out_dir = f'/home/qiwei/program/deep-burst-sr/indoor/synthetic_{mode}/'

    crop_sz = (1024 + 24*2, 1024 + 24*2)
    burst_sz = 4
    downsample_factor = 1

    burst_transformation_params = {'max_translation': 24.0,
                                   'max_rotation': 1.0,
                                   'max_shear': 0.0,
                                   'max_scale': 0.0,
                                   'border_crop': 24}
    image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True,
                               'add_noise': True, 'noise_type': 'unprocessing'}

    image_dataset = datasets.ImageFolder(root=image_folder_path)
    transform_list = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True))
    # transform_list = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
    data_processing = processing.SyntheticBurstProcessing(crop_sz, burst_sz, downsample_factor,
                                                          burst_transformation_params=burst_transformation_params,
                                                          transform=transform_list,
                                                          image_processing_params=image_processing_params)
    dataset = sampler.IndexedImage(image_dataset, processing=data_processing)

    for i, d in enumerate(dataset):
        burst = d['burst']
        gt = d['frame_gt']
        meta_info = d['meta_info']
        meta_info['frame_num'] = i

        os.makedirs('{}/bursts/{:04d}'.format(out_dir, i), exist_ok=True)
        os.makedirs('{}/gt/{:04d}'.format(out_dir, i), exist_ok=True)
        burst_np = (burst.permute(0, 2, 3, 1).clamp(0.0, 1.0) * 2**14).numpy().astype(np.uint16)

        for bi, b in enumerate(burst_np):
            cv2.imwrite('{}/bursts/{:04d}/im_raw_{:02d}.png'.format(out_dir, i, bi), b)

        gt_np = (gt.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
        cv2.imwrite('{}/gt/{:04d}/im_rgb.png'.format(out_dir, i), gt_np)

        with open('{}/gt/{:04d}/meta_info.pkl'.format(out_dir, i), "wb") as file_:
            pkl.dump(meta_info, file_, -1)


if __name__ == '__main__':
    main()

