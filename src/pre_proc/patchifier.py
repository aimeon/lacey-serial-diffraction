import os
import cv2
import numpy as np
from patchify import patchify
import tifffile as tiff
from tqdm import tqdm

VALID_EXTS = (".png", ".tif", ".tiff")


def get_pairs(img_dir, mask_dir):
    """Return list of (img_path, mask_path, stem) for matching image/mask files."""
    def collect(d):
        return {
            os.path.splitext(f)[0]: os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith(VALID_EXTS)
        }

    img_map = collect(img_dir)
    mask_map = collect(mask_dir)
    stems = sorted(set(img_map) & set(mask_map))
    return [(img_map[s], mask_map[s], s) for s in stems]


def patchify_dataset(
    img_dir,
    mask_dir,
    out_img_dir,
    out_mask_dir,
    patch_size=256,
    step=256,
    max_side=512,
    use_max_side=True,
):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    pairs = get_pairs(img_dir, mask_dir)
    total_patches = 0

    for img_path, mask_path, stem in tqdm(pairs, desc="Patchifying"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"[WARN] Could not read image or mask for '{stem}': img={img_path}, mask={mask_path}")
            continue

        # binarize mask
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)

        # match shapes
        if img.shape != mask.shape:
            print(
                f"[WARN] Shape mismatch for '{stem}': "
                f"img={img.shape}, mask={mask.shape}. Skipping."
            )
            continue

        h, w = mask.shape

        # choose resize target
        if use_max_side:
            if h > max_side or w > max_side:
                target_h, target_w = max_side, max_side
            else:
                target_h, target_w = h, w
            # shrinking larger images to match smaller images resolution didn't appear to be effective for model training
        else:
            target_h = (h // patch_size) * patch_size
            target_w = (w // patch_size) * patch_size
            # avoid shrinking away everything
            if target_h < patch_size or target_w < patch_size:
                target_h, target_w = h, w

        # resize if needed
        if (target_h, target_w) != (h, w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # patchify
        img_patches = patchify(img, (patch_size, patch_size), step)
        mask_patches = patchify(mask, (patch_size, patch_size), step)

        # save patches
        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                img_patch = img_patches[i, j, :, :]
                mask_patch = mask_patches[i, j, :, :]

                tiff.imwrite(os.path.join(out_img_dir, f"{stem}_{i}{j}.tif"), img_patch)
                tiff.imwrite(os.path.join(out_mask_dir, f"{stem}_{i}{j}.tif"), mask_patch)
                total_patches += 1

    print(f"Processed {total_patches} patches from {len(pairs)} pairs.")


if __name__ == "__main__":
    img_dir = r"/home/anvy4548/projects/Crystal_recognition/test images/png"
    mask_dir = r"/home/anvy4548/projects/Crystal_recognition/test images/masks"
    out_img = r"/home/anvy4548/projects/Crystal_recognition/test images/patches_for_training/images"
    out_mask = r"/home/anvy4548/projects/Crystal_recognition/test images/patches_for_training/masks"

    # use_max_side=True  → clamp to max_side if too big
    # use_max_side=False → shrink to largest multiple of patch_size
    patchify_dataset(
        img_dir,
        mask_dir,
        out_img,
        out_mask,
        patch_size=256,
        step=256,
        max_side=512,
        use_max_side=False,
    )
