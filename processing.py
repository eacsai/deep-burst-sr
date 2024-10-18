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
import torch 
import cv2 as cv
import numpy as np

def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1)


def torch_to_numpy(a: torch.Tensor):
    return a.permute(1, 2, 0).numpy()


def torch_to_npimage(a: torch.Tensor, unnormalize=True, input_bgr=False):
    a_np = torch_to_numpy(a.clamp(0.0, 1.0))

    if unnormalize:
        a_np = a_np * 255
    a_np = a_np.astype(np.uint8)

    if input_bgr:
        return a_np

    return cv.cvtColor(a_np, cv.COLOR_RGB2BGR)


def npimage_to_torch(a, normalize=True, input_bgr=True):
    if input_bgr:
        a = cv.cvtColor(a, cv.COLOR_BGR2RGB)
    a_t = numpy_to_torch(a)

    if normalize:
        a_t = a_t / 255.0

    return a_t


def rggb_to_rgb(im):
    im_out = im[..., [0, 1, 3], :, :]
    return im_out


def convert_dict(base_dict, batch_sz):
    out_dict = []
    for b_elem in range(batch_sz):
        b_info = {}
        for k, v in base_dict.items():
            if isinstance(v, (list, torch.Tensor)):
                b_info[k] = v[b_elem]
        out_dict.append(b_info)

    return out_dict

def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    assert image.dim() == 3 and image.shape[0] == 3

    shape = image.shape
    image = image.view(3, -1)
    ccm = ccm.to(image.device).type_as(image)

    image = torch.mm(ccm, image)

    return image.view(shape)

def apply_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    assert image.dim() == 3 and image.shape[0] in [3, 4]

    if image.shape[0] == 3:
        gains = torch.tensor([red_gain, 1.0, blue_gain]) * rgb_gain
    else:
        gains = torch.tensor([red_gain, 1.0, 1.0, blue_gain]) * rgb_gain
    gains = gains.view(-1, 1, 1)
    gains = gains.to(image.device).type_as(image)

    return (image * gains).clamp(0.0, 1.0)

def apply_smoothstep(image):
    """Apply global tone mapping curve."""
    image_out = 3 * image**2 - 2 * image**3
    return image_out

def gamma_compression(image):
    """Converts from linear to gammaspace."""
    # Clamps to prevent numerical instability of gradients near zero.
    return image.clamp(1e-8) ** (1.0 / 2.2)

class SimplePostProcess:
    def __init__(self, gains=True, ccm=True, gamma=True, smoothstep=True, return_np=False):
        self.gains = gains
        self.ccm = ccm
        self.gamma = gamma
        self.smoothstep = smoothstep
        self.return_np = return_np

    def process(self, image, meta_info):
        return process_linear_image_rgb(image, meta_info, self.gains, self.ccm, self.gamma,
                                        self.smoothstep, self.return_np)


def process_linear_image_rgb(image, meta_info, gains=True, ccm=True, gamma=True, smoothstep=True, return_np=False):
    if gains:
        image = apply_gains(image, meta_info['rgb_gain'], meta_info['red_gain'], meta_info['blue_gain'])

    if ccm:
        image = apply_ccm(image, meta_info['cam2rgb'])

    image = image.clamp(0.0, 1.0)
    if meta_info['gamma'] and gamma:
        image = gamma_compression(image)

    if meta_info['smoothstep'] and smoothstep:
        image = apply_smoothstep(image)

    image = image.clamp(0.0, 1.0)

    if return_np:
        image = torch_to_npimage(image)
    return image


class Identity:
    def __init__(self, return_np=False, clamp=True):
        self.return_np = return_np
        self.clamp = clamp

    def process(self, image, meta_info):
        if self.clamp:
            image = image.clamp(0.0, 1.0)

        if self.return_np:
            image = torch_to_npimage(image)
        return image
