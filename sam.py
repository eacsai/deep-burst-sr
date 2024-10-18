import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "/data/dataset/homework/SyntheticBurstVal/sam_vit_b_01ec64.pth"
model_type = "vit_b"
results_dir = '/home/qiwei/program/deep-burst-sr/indoor/SAM_Results'
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=20,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=4,
    min_mask_region_area=40,  # Requires open-cv to run post-processing
)

# image='/home/wushx/data/MySR/120240306_161348.jpg'
for idx in range(20):
    image_path='/home/qiwei/program/deep-burst-sr/indoor/viz_rggb/{}.3_lr.png'.format(idx)
    image_sr_path = '/home/qiwei/program/deep-burst-sr/indoor/viz_rggb/{}.2_sr.png'.format(idx)
        
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig('/home/qiwei/program/deep-burst-sr/indoor/sam_rggb/{}.lr.png'.format(idx), bbox_inches='tight', pad_inches=0)
    plt.close()

    image_sr = cv2.imread(image_sr_path)
    image_sr = cv2.cvtColor(image_sr, cv2.COLOR_BGR2RGB)
    # # 指定新的尺寸 (宽度, 高度)
    # new_size = (128, 128)
    # # 调整尺寸
    # image_sr_resized = cv2.resize(image_sr, new_size)
    masks_sr = mask_generator.generate(image_sr)

    plt.figure(figsize=(20,20))
    plt.imshow(image_sr)
    show_anns(masks_sr)
    plt.axis('off')
    plt.savefig('/home/qiwei/program/deep-burst-sr/indoor/sam_rggb/{}.sr.png'.format(idx), bbox_inches='tight', pad_inches=0)
    plt.close()


