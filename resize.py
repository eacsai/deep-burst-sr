import cv2
import os

def crop_center_square(img):
    """Crop the center square of an image."""
    height, width = img.shape[:2]
    new_size = min(height, width)
    start_x = (width // 2) - (new_size // 2)
    start_y = (height // 2) - (new_size // 2)
    return img[start_y:start_y+new_size, start_x:start_x+new_size]

def resize_images(source_dir, target_dir, size=(1072, 1072)):
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        try:
            img = cv2.imread(file_path)
            # Check if the image is not square
            if img.shape[0] != img.shape[1]:
                img = crop_center_square(img)
            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(target_dir,'1'+filename.replace('.jpg', '.png')), resized_img)
            print(f"Processed {filename}.")
        except Exception as e:
            print(f"Skipping {filename}, error: {e}")

# Specify your target directory here
source_dir = '/home/qiwei/program/deep-burst-sr/indoor/camera'
target_dir = '/home/qiwei/program/deep-burst-sr/indoor/original'
# Resize the images
resize_images(source_dir, target_dir)
