from sys import prefix

import matplotlib
matplotlib.use("Agg")

from scipy import ndimage
from skimage import measure, morphology, segmentation
from scipy.cluster.vq import kmeans2
from crystal_finder_from_instamatic import calibration
from crystal_finder_from_instamatic import  CrystalPosition, autoscale, whiten, is_edge
from find_grid_unet import prediction
import cv2
import numpy as np
from tensorflow.keras import models
import tensorflow as tf
from PIL import Image
from skimage.morphology import reconstruction
from skimage import filters, measure, morphology, segmentation
model = models.load_model('standard_aug_20_50epochs.keras', compile=False)


import numpy as np
from skimage import morphology


import os

HEADER = "image,x_px,y_px,isolated,n_clusters,area_um2,area_px\n"

def write_per_image_csv(crystals, image_name, out_path):
    with open(out_path, "w") as f:
        f.write("x_px,y_px,isolated,n_clusters,area_um2,area_px\n")
        for c in crystals:
            f.write(
                f"{c.x:.2f},{c.y:.2f},{int(c.isolated)},"
                f"{c.n_clusters},{c.area_micrometer:.6f},{c.area_pixel}\n"
            )

def append_to_master_csv(crystals, image_name, master_path):
    # Write header once
    if not os.path.exists(master_path) or os.path.getsize(master_path) == 0:
        with open(master_path, "w") as f:
            f.write(HEADER)

    with open(master_path, "a") as f:
        for c in crystals:
            f.write(
                f"{image_name},{c.x:.2f},{c.y:.2f},{int(c.isolated)},"
                f"{c.n_clusters},{c.area_micrometer:.6f},{c.area_pixel}\n"
            )

def _load_ref_mask(path, target_shape):
    """Load ref mask as bool, resize to target_shape (nearest-neighbor)."""
    ref = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        return None
    ref = cv2.resize(ref, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return ref > 127


def _as_bool_ndarray(x):
    # handle tf tensors etc
    if hasattr(x, "numpy"):
        x = x.numpy()
    x = np.asarray(x)
    # squeeze weird singleton dims if they exist (e.g. HxWx1)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return x.astype(np.bool_, copy=False)

def iou_and_recall(pred_mask, ref_mask):
    pred = _as_bool_ndarray(pred_mask)
    ref  = _as_bool_ndarray(ref_mask)

    # safety: make sure same shape (should be, but just in case)
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")

    inter = np.count_nonzero(pred & ref)
    union = np.count_nonzero(pred | ref)
    ref_sum = np.count_nonzero(ref)

    iou = inter / union if union else (1.0 if ref_sum == 0 and np.count_nonzero(pred) == 0 else 0.0)
    recall = inter / ref_sum if ref_sum else 1.0
    return iou, recall




def contrast_guided_grow(seed, img, radius=5, q_low=10, q_high=90):
    """
    Grow seed by 'radius' pixels, but only into pixels whose intensity
    lies within a robust intensity range of the seed.
    img is assumed normalized to [0, 1].
    """
    seed = seed.astype(bool)

    if not seed.any():
        return seed

    lo, hi = np.percentile(img[seed], [q_low, q_high])

    grown = morphology.binary_dilation(seed, morphology.disk(radius))
    grown = grown & (img >= lo) & (img <= hi)

    return grown



def segment_crystals(img, r=101, offset=25, footprint=5, remove_carbon_lacing=False):
    """
    r: int
        Block size for local thresholding (unused when fixed threshold is applied)
    footprint: int
        Radius for disk used in morphological operations
    offset: int
        Constant subtracted from threshold (scaled internally)
    """

    # Normalize image to [0, 1]
    offset = offset / 255.0
    img = img * (1.0 / img.max())


    # --- Initial thresholding ---
    arr = img > filters.threshold_local(img, r, method='mean', offset=offset)
    arr = np.invert(arr)

    # --- Remove carbon grid EARLY using U-Net ---
    mask = prediction(model, img, 256).astype(bool)
    arr = arr & np.logical_not(mask)

    # --- Morphological cleanup ---

    arr = morphology.remove_small_objects(arr, min_size=4 * 4, connectivity=0)
    arr = morphology.binary_closing(arr, morphology.disk(footprint))
    arr = morphology.binary_erosion(arr, morphology.disk(footprint))
    #arr = morphology.remove_small_objects(arr, min_size=4 * 4, connectivity=0)


    #arr = morphology.binary_dilation(arr, morphology.disk(footprint + 2))

    if remove_carbon_lacing:
        arr = morphology.remove_small_objects(arr, min_size=8 * 8, connectivity=0)
        arr = morphology.remove_small_holes(arr, area_threshold=32 * 32, connectivity=0)

    arr = morphology.binary_dilation(arr, morphology.disk(footprint))
    # --- Random walker preparation ---
    bkg = np.invert(
        morphology.binary_dilation(arr, morphology.disk(footprint * 1)) | arr
    )

    markers = arr * 2 + bkg

    segmented = segmentation.random_walker(
        img, markers, beta=50, spacing=(5, 5), mode='bf'
    )
    segmented = segmented.astype(int) - 1

    return arr, segmented, mask

import matplotlib.pyplot as plt


def save_step(image, title, out_dir, cmap="gray"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{title}.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay(base_img, overlay_mask, out_path, overlay_cmap="jet", alpha=0.3):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(base_img, cmap="gray")
    plt.imshow(np.ma.masked_where(overlay_mask == 0, overlay_mask), cmap=overlay_cmap, alpha=alpha)
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

def visualize_steps(img, out_dir="debug_steps", r=101, offset=25, footprint=5, remove_carbon_lacing=False):
    offset = offset / 255.0
    img = img.astype(np.float32)
    img = img * (1.0 / img.max())

    save_step(img, "01_normalized", out_dir)

    # threshold
    arr = img > filters.threshold_local(img, r, method="mean", offset=offset)
    arr = np.invert(arr)
    save_step(arr, "02_thresholded", out_dir)

    # predicted carbon mask
    mask = prediction(model, img, 256).astype(bool)
    save_step(mask, "03_carbon_mask", out_dir)

    # remove carbon early, same as real pipeline
    arr = arr & np.logical_not(mask)
    save_step(arr, "04_after_carbon_removal", out_dir)

    # original image + carbon mask overlay
    save_overlay(img, mask, out_dir / "04_original_with_carbon_mask.png")

    # cleanup
    arr = morphology.remove_small_objects(arr, min_size=4 * 4, connectivity=0)
    save_step(arr, "05_small_objects_removed", out_dir)

    arr = morphology.binary_closing(arr, morphology.disk(footprint))
    save_step(arr, "06_closing", out_dir)

    arr = morphology.binary_erosion(arr, morphology.disk(footprint))
    save_step(arr, "07_erosion", out_dir)

    if remove_carbon_lacing:
        arr = morphology.remove_small_objects(arr, min_size=8 * 8, connectivity=0)
        arr = morphology.remove_small_holes(arr, area_threshold=32 * 32, connectivity=0)
        save_step(arr, "08_remove_lacing", out_dir)

    arr = morphology.binary_dilation(arr, morphology.disk(footprint))
    save_step(arr, "09_dilation", out_dir)

    # background + markers
    bkg = np.invert(morphology.binary_dilation(arr, morphology.disk(footprint)) | arr)
    save_step(bkg, "10_background", out_dir)

    markers = arr * 2 + bkg
    save_step(markers, "11_markers", out_dir, cmap="gray")

    segmented = segmentation.random_walker(img, markers, beta=50, spacing=(5, 5), mode="bf")
    segmented = segmented.astype(int) - 1
    save_step(segmented, "12_random_walker", out_dir, cmap="gray")

    # --- final crystal mask ---
    crystal_mask = segmented > 0
    labels, _ = ndimage.label(crystal_mask)
    props = measure.regionprops(labels, img)

    crystals = []
    for prop in props:
        if is_edge(prop):
            continue
        y, x = prop.centroid
        crystals.append((x, y))  # x,y for plotting

    # --- final overlay (like your example) ---
    h, w = img.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)

    ax.imshow(img, cmap="gray")

    # yellow contours
    ax.contour(crystal_mask, [0.5], linewidths=1.2, colors='yellow')

    # red centroids
    if crystals:
        xs, ys = zip(*crystals)
        ax.scatter(xs, ys, color='red', s=40)

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.savefig(out_dir / "13_final_result.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return arr, segmented, mask


def find_crystals_new(img, magnification, spread, plot=False, img_return=False, return_mask=False, **kwargs):
    """Function for finding crystals in a low contrast images. Used adaptive
    thresholds to find local features. Edges are detected, and rejected, on the
    basis of a histogram. Kmeans clustering is used to spread points over the
    segmented area.

    img: 2d np.ndarray
        Input image to locate crystals on
    magnification: float
        value indicating the magnification used, needed in order to determine the size of the crystals
    spread: float
        Value in micrometer to roughly indicate the desired spread of centroids over individual regions
    plot: bool
        Whether to plot the results or not
    **kwargs:
    keywords to pass to segment_crystals
    """
    img, scale = autoscale(img, maxdim=512)  # scale down for faster
    # print(img.shape)
    # segment the image, and find objects
    arr, seg, mask = segment_crystals(img, **kwargs)
    # arr, seg = visualize_steps(img, remove_carbon_lacing=True, **kwargs)

    labels, numlabels = ndimage.label(seg)
    props = measure.regionprops(labels, img)

    # calculate the pixel dimensions in micrometer
    px = py = calibration['mag1']['pixelsize'][magnification] / 1000  # nm -> um


    # if magnification in magnification_factor:
    #     px = py = magnification_factor[magnification]

    iters = 20

    crystals = []
    for prop in props:
        area = prop.area * px * py
        bbox = np.array(prop.bbox)

        # origin of the prop
        origin = bbox[0:2]

        # edge detection
        if is_edge(prop):
            continue

        # number of centroids for kmeans clustering
        nclust = int(area // spread) + 1

        if nclust > 1:
            # use skmeans clustering to segment large blobs
            coordinates = np.argwhere(prop.image)

            # kmeans needs normalized data (w), store std to calculate coordinates after
            w, std = whiten(coordinates)

            # nclust must be an integer for some reason
            cluster_centroids, closest_centroids = kmeans2(w, nclust, iter=iters, minit='points')

            # convert to image coordinates
            xy = (cluster_centroids * std + origin[0:2]) / scale
            crystals.extend([CrystalPosition(x, y, False, nclust, area, prop.area) for x, y in xy])
        else:
            x, y = prop.centroid
            crystals.append(CrystalPosition(x / scale, y / scale, True, nclust, area, prop.area))

    if plot:
        h, w = img.shape[:2]
        dpi = 100
        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)


        plt.imshow(img, cmap="gray")
        plt.imshow(np.ma.masked_where(mask == 0, mask), cmap='jet', alpha=0.3)
        plt.contour(seg, [0.5], linewidths=1.2, colors='yellow')

        if len(crystals) > 0:
            x, y = np.array([(crystal.x * scale, crystal.y * scale) for crystal in crystals]).T
            plt.scatter(y, x, color='red')

        ax.set_axis_off()

        # remove borders without changing the rendering style much
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_position([0, 0, 1, 1])

        fig.canvas.draw()

        result = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        result = result.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        result = result[:, :, :3]  # Keep only RGB

        plt.close(fig)

        if img_return and return_mask:
            return crystals, result, seg, mask
        if img_return:
            return crystals, result

    if return_mask:
        return crystals, seg, mask

    return crystals


from pathlib import Path
import os
import time

import cv2
import numpy as np
from PIL import Image


def main(image_path, magnification, spread=1, plot=False, img_return=False, return_mask=False, **kwargs):
    test_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    test_img = np.array(Image.fromarray(test_img).resize((512, 512)))

    start_time = time.time()
    result = find_crystals_new(
        test_img,
        magnification,
        spread=spread,
        plot=plot,
        img_return=img_return,
        return_mask=return_mask,
        **kwargs,
    )
    print(f"Processed in {time.time() - start_time:.2f} s")
    return result


def ensure_metrics_file(path):
    if not path.exists() or path.stat().st_size == 0:
        path.write_text("image\tIoU\trecall\n")


def find_reference_mask(ref_dir, base_name):
    for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        candidate = ref_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def get_unique_result_path(processed_dir, base_name):
    stem = f"{base_name}_inst_unet_after_random_walk"
    i = 1
    while True:
        candidate = processed_dir / f"{stem}_{i}.png"
        if not candidate.exists():
            return candidate
        i += 1

if __name__ == "__main__":




    script_start = time.time()

    prefix = "_test_only_3"
    new_folder = f"unet20_50_noscaling_crystal_pos{prefix}"

    root = Path("/home/anvy4548/projects/crystal-recognition")
    test_images_dir = root / "training_test_all_images"
    ref_masks_dir = root / "training_test_all_masks"

    # ---- one main output folder ----
    output_root = root / "results_final" / new_folder

    image_path = r"/home/anvy4548/projects/montage_100im_orig/_png/mont_0042.png"
    test_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    test_img = np.array(Image.fromarray(test_img).resize((512, 512)))

    visualize_steps(test_img, out_dir=output_root / "debug_steps")

    #
    # processed_images_dir = output_root / "processed_images"
    # masks_dir = output_root / "grid_masks"
    # crystal_masks_dir = output_root / "crystal_masks"
    # coords_dir = output_root / "coords"
    # metrics_dir = output_root / "metrics"
    #
    # master_metrics_path = metrics_dir / "grid_metrics.tsv"
    #
    # # create all folders
    # for d in (
    #     processed_images_dir,
    #     masks_dir,
    #     crystal_masks_dir,
    #     coords_dir,
    #     metrics_dir,
    # ):
    #     d.mkdir(parents=True, exist_ok=True)
    #
    # ensure_metrics_file(master_metrics_path)
    #
    # valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    #
    # for image_path in test_images_dir.iterdir():
    #     if not image_path.is_file() or image_path.suffix.lower() not in valid_exts:
    #         continue
    #
    #     image_start = time.time()
    #     print(image_path)
    #
    #     crystals, result_image, crystal_mask, carbon_mask = main(
    #         image_path,
    #         25000,
    #         plot=True,
    #         img_return=True,
    #         return_mask=True,
    #     )
    #
    #     base = image_path.stem
    #
    #     # processed image
    #     result_image_path = processed_images_dir / f"{base}.png"
    #     Image.fromarray(np.uint8(result_image)).save(result_image_path)
    #
    #     # grid mask
    #     grid_mask_path = masks_dir / f"{base}.png"
    #     Image.fromarray((carbon_mask.astype(np.uint8) * 255)).save(grid_mask_path)
    #
    #     # crystal mask
    #     crystal_mask_path = crystal_masks_dir / f"{base}.png"
    #     Image.fromarray((crystal_mask.astype(np.uint8) * 255)).save(crystal_mask_path)
    #
    #     # metrics
    #     ref_path = find_reference_mask(ref_masks_dir, base)
    #     if ref_path is not None:
    #         ref_mask = _load_ref_mask(ref_path, target_shape=carbon_mask.shape)
    #         iou, recall = iou_and_recall(carbon_mask, ref_mask)
    #         with master_metrics_path.open("a") as f:
    #             f.write(f"{image_path.name}\t{iou:.6f}\t{recall:.6f}\n")
    #
    #     print(f"--- {time.time() - image_start:.2f} s ---")
    #
    # print(f"Total runtime: {time.time() - script_start:.2f} s")