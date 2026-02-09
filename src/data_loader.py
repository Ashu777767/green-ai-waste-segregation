import cv2
import numpy as np
import os

IMG_SIZE = 128

def load_data(image_dir, mask_dir):
    images = []
    masks = []

    image_files = sorted(os.listdir(image_dir))

    for img_file in image_files:
        if not img_file.endswith(".jpg"):
            continue

        img_path = os.path.join(image_dir, img_file)

        # convert img0.jpg -> img0.png
        mask_file = img_file.replace(".jpg", ".png")
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            continue

        # Load image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        # Load mask
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)
