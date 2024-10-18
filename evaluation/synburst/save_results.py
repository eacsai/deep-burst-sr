# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

import torch

import argparse
import importlib
import numpy as np
import cv2
import tqdm
from admin.environment import env_settings
from dataset.synthetic_burst_val_set import SyntheticBurstVal


mode = 'rggb_1'

def save_results(setting_name):
    """ Saves network outputs on the SyntheticBurst validation set. setting_name denotes the name of the experiment
        setting to be used. """

    expr_module = importlib.import_module('evaluation.synburst.experiments.{}'.format(setting_name))
    expr_func = getattr(expr_module, 'main')
    network_list = expr_func()

    base_results_dir = '/home/qiwei/program/deep-burst-sr/indoor'
    dataset = SyntheticBurstVal(f'/home/qiwei/program/deep-burst-sr/indoor/synthetic_{mode}')

    for n in network_list:
        net = n.load_net()
        device = 'cuda'
        net.to(device).train(False)

        out_dir = '{}/{}_{}'.format(base_results_dir, n.get_unique_name(), mode)
        os.makedirs(out_dir, exist_ok=True)

        for idx in tqdm.tqdm(range(len(dataset))):
            burst, _, meta_info = dataset[idx]

            # if mode == 'grbg':
            #     G1 = burst[:, 0, :, :]
            #     R = burst[:, 1, :, :]
            #     B = burst[:, 2, :, :]
            #     G2 = burst[:, 3, :, :]
            #     burst = torch.stack([R, G1, G2, B], dim=1)
            # elif mode == 'rggb':
            #     R = burst[:, 0, :, :]
            #     G1 = burst[:, 1, :, :]
            #     G2 = burst[:, 2, :, :]
            #     B = burst[:, 3, :, :]
            #     burst = torch.stack([R, G1, G2, B], dim=1)
            # elif mode == 'bggr':
            #     B = burst[:, 0, :, :]
            #     G1 = burst[:, 1, :, :]
            #     G2 = burst[:, 2, :, :]
            #     R = burst[:, 3, :, :]
            #     burst = torch.stack([R, G1, G2, B], dim=1)
            # elif mode == 'gbrg':
            #     G1 = burst[:, 0, :, :]
            #     B = burst[:, 1, :, :]
            #     R = burst[:, 2, :, :]
            #     G2 = burst[:, 3, :, :]
            #     burst = torch.stack([R, G1, G2, B], dim=1)

            burst_name = meta_info['burst_name']

            burst = burst.to(device).unsqueeze(0)

            if n.burst_sz is not None:
                burst = burst[:, :n.burst_sz]

            with torch.no_grad():
                net_pred, _ = net(burst)

                # Normalize to 0  2^14 range and convert to numpy array
                net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(
                    np.uint16)

                # Save predictions as png
                cv2.imwrite('{}/{}.png'.format(out_dir, burst_name), net_pred_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saves network outputs on the SyntheticBurst validation set. '
                                                 'setting_name denotes the name of the experiment setting to be used.')
    parser.add_argument('setting', type=str, help='Name of experiment setting')

    args = parser.parse_args()

    save_results(args.setting)