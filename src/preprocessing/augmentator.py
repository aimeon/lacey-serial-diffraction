import cv2
import os
import numpy as np
from tqdm import tqdm
import time

def random_rotation(image, mask):
    angle = np.random.randint(4) * 90  # Randomly choose 0, 90, 180, or 270 degrees (multiples of 90)
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h))
    return rotated_image, rotated_mask


def random_flip(image, mask, p = 0.6):
    if np.random.random() < p:
        image = cv2.flip(image, 1)  # Horizontal flip
        mask = cv2.flip(mask, 1)
    if np.random.random() < p:
        image = cv2.flip(image, 0)  # Vertical flip
        mask = cv2.flip(mask, 0)
    return image, mask


def random_brightness(image, mask, brightness_range=30):
    brightness = np.random.uniform(-brightness_range-10, brightness_range)
    out = np.clip(image.astype(int) + brightness, 0, 255).astype(np.uint8)
    return out, mask


def random_contrast(image, mask, contrast_range=(0.05, 1.1)):
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    out = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    return out, mask


def _pair_by_stem(images_dir, masks_dir, image_exts=(".png", ".tif", ".tiff")):
    imgs = [f for f in os.listdir(images_dir) if f.lower().endswith(image_exts)]
    masks = [f for f in os.listdir(masks_dir) if f.lower().endswith(image_exts)]
    img_map  = {os.path.splitext(f)[0]: f for f in imgs}
    mask_map = {os.path.splitext(f)[0]: f for f in masks}
    keys = sorted(set(img_map) & set(mask_map))
    return [(os.path.join(images_dir, img_map[k]), os.path.join(masks_dir, mask_map[k])) for k in keys]


def _binarize_mask(mask):
    # keep 0/255
    return np.where(mask > 0, 255, 0).astype(np.uint8)


# ---- main ----
def data_augmentation(
    images_directory,
    masks_directory,
    output_images_directory,
    output_masks_directory,
    num_augmented_images=5,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    if not os.path.exists(output_images_directory):
        os.makedirs(output_images_directory)
    if not os.path.exists(output_masks_directory):
        os.makedirs(output_masks_directory)

    valid_img_exts = (".png", ".tif", ".tiff")
    images_in_folder = len([f for f in os.listdir(images_directory) if f.lower().endswith(valid_img_exts)])
    masks_in_folder  = len([f for f in os.listdir(masks_directory)  if f.lower().endswith(valid_img_exts)])

    start = time.time()
    pairs_processed = 0
    total_augmented_saved = 0

    image_files = os.listdir(images_directory)
    for image_file in tqdm(image_files, desc="Augmenting pairs"):
        image_path = os.path.join(images_directory, image_file)
        mask_path  = os.path.join(masks_directory,  image_file)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(mask_path,  cv2.IMREAD_GRAYSCALE)


        base_filename, ext = os.path.splitext(image_file)
        for i in range(num_augmented_images):
            augmented_image, augmented_mask = random_rotation(image, mask)
            augmented_image, augmented_mask = random_flip(augmented_image, augmented_mask)
            augmented_image, augmented_mask = random_brightness(augmented_image, augmented_mask)
            augmented_image, augmented_mask = random_contrast(augmented_image, augmented_mask)

            augmented_filename = f"{base_filename}_aug_{i}{ext}"
            augmented_image_path = os.path.join(output_images_directory, augmented_filename)
            augmented_mask_path  = os.path.join(output_masks_directory,  augmented_filename)

            cv2.imwrite(augmented_image_path, augmented_image)
            cv2.imwrite(augmented_mask_path,  augmented_mask)
            total_augmented_saved += 1

        pairs_processed += 1

    elapsed = time.time() - start

    # --- Sanity check ---
    print(
        f"Created {total_augmented_saved} augmented images "
        f"from {pairs_processed} pairs in {round(elapsed, 2)} seconds."
    )

    return {
        "images_in_folder": images_in_folder,
        "masks_in_folder": masks_in_folder,
        "pairs_augmented": pairs_processed,
        "total_augmented_saved": total_augmented_saved,
        "time_spent_s": round(elapsed, 2),
    }


if __name__ == "__main__":
    input_images_directory = r"/home/anvy4548/projects/Crystal_recognition/test images/png"  # Change this to your input images directory
    input_masks_directory = r"/home/anvy4548/projects/Crystal_recognition/test images/masks"  # Change this to your input masks directory
    output_images_directory = r"/home/anvy4548/projects/Crystal_recognition/test images/aug/png"  # Change this to your output images directory
    output_masks_directory = r"/home/anvy4548/projects/Crystal_recognition/test images/aug/masks"  # Change this to your output masks directory
    num_augmented_images_per_input = 2

    data_augmentation(input_images_directory, input_masks_directory, output_images_directory, output_masks_directory, num_augmented_images_per_input)
