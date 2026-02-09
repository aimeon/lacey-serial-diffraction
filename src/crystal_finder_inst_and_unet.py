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
# from keras.utils import normalize
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
    threshold_value = 0.4
    arr = img > filters.threshold_local(img, r, method='mean', offset=offset)
    arr = np.invert(arr)

    # --- Remove carbon grid EARLY using U-Net ---
    mask = prediction(model, img, 256).astype(bool)
    arr = arr & np.logical_not(mask)

    # --- Morphological cleanup ---

    arr = morphology.remove_small_objects(arr, min_size=4 * 4, connectivity=0)
    arr = morphology.binary_closing(arr, morphology.disk(footprint))
    arr = morphology.binary_erosion(arr, morphology.disk(footprint))
    #arr = contrast_guided_grow(arr, img, radius=footprint+2, q_low=40, q_high=60)
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


def visualize_steps(img, r=101, offset=25, footprint=5, remove_carbon_lacing=False):
    offset = offset / 255.0
    print("Original Image:")
    plt.imshow(img, cmap='gray')
    plt.show()

    img = img * (1.0 / img.max())
    print("Normalized Image:")
    plt.imshow(img, cmap='gray')
    plt.show()

    arr = img > filters.threshold_local(img, r, method='mean', offset=offset)
    #arr = img > 0.4
    # This one would be great if the grid recognition was perfect
    # arr = img > filters.threshold_otsu(img)
    arr = np.invert(arr)
    print("Thresholded Image:")
    plt.imshow(arr, cmap='gray')
    plt.show()



    arr = morphology.remove_small_objects(arr, min_size=4 * 4, connectivity=0)
    print("Thresholded Image:")
    plt.imshow(arr, cmap='gray')
    plt.show()
    arr = morphology.binary_closing(arr, morphology.disk(footprint))
    print("Thresholded Image:")
    plt.imshow(arr, cmap='gray')
    plt.show()
    arr = morphology.binary_erosion(arr, morphology.disk(footprint))
    print("Thresholded Image:")
    plt.imshow(arr, cmap='gray')
    plt.show()

    if remove_carbon_lacing:
        arr = morphology.remove_small_objects(arr, min_size=8 * 8, connectivity=0)
        arr = morphology.remove_small_holes(arr, area_threshold=32 * 32, connectivity=0)
        print("Thresholded Image:")
        plt.imshow(arr, cmap='gray')
        plt.show()
    arr = morphology.binary_dilation(arr, morphology.disk(footprint))
    print("Thresholded Image:")
    plt.imshow(arr, cmap='gray')
    plt.show()

    bkg = np.invert(morphology.binary_dilation(arr, morphology.disk(footprint * 2)) | arr)
    markers = arr * 2 + bkg

    print("Markers:")
    plt.imshow(markers, cmap='jet')
    plt.show()

    segmented = segmentation.random_walker(img, markers, beta=50, spacing=(5, 5), mode='bf')
    segmented = segmented.astype(int) - 1
    print("Markers:")
    plt.imshow(segmented, cmap='jet')
    plt.show()
    mask = prediction(model, img, 256)
    print("Markers:")
    plt.imshow(mask, cmap='jet')
    plt.show()
    # mask = morphology.binary_dilation(mask, morphology.disk(footprint))
    mask = morphology.binary_opening(mask, morphology.disk(5))
    mask = morphology.binary_closing(mask, morphology.disk(5))
    print("Markers:")
    plt.imshow(mask, cmap='jet')
    plt.show()
    mask = np.logical_not(mask)
    # mask = mask.astype(bool)
    segmented = segmented & mask
    print("Markers:")
    plt.imshow(segmented, cmap='jet')
    plt.show()

    return arr, segmented


def find_crystals_new(img, magnification, spread=20.0, plot=False, img_return=False, return_mask=False, **kwargs):
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
        fig, ax = plt.subplots()
        plt.imshow(img, cmap="gray")
        plt.imshow(np.ma.masked_where(mask == 0, mask), cmap='jet', alpha=0.3)
        plt.contour(seg, [0.5], linewidths=1.2, colors='yellow')
        if len(crystals) > 0:
            x, y = np.array([(crystal.x * scale, crystal.y * scale) for crystal in crystals]).T
            plt.scatter(y, x, color='red')
        ax.set_axis_off()
        fig.tight_layout(pad=0)
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


def main(image_path, magnification, spread=2.0, plot=False, img_return=False, return_mask=False, **kwargs):
    test_img = cv2.imread(image_path, 0)
    test_img = Image.fromarray(test_img)
    test_img = np.array(test_img.resize((512, 512)))

    start_time = time.time()

    result = find_crystals_new(
        test_img,
        magnification,
        spread=spread,
        plot=plot,
        img_return=img_return,
        return_mask=return_mask,
        **kwargs
    )

    print(f"Processed in {time.time() - start_time:.2f} s")
    return result



if __name__ == "__main__":
    import os
    import time

    start_time = time.time()

    prefix = "_test_only"

    new_folder = f"unet20_50_noscaling{prefix}"

    test_images_dir = r"/home/anvy4548/projects/crystal-recognition/test_all_images"
    ref_masks_dir = r"/home/anvy4548/projects/crystal-recognition/training_test_all_masks"
    processed_images_dir = f"/home/anvy4548/projects/crystal-recognition/test_images/{new_folder}/processed_images"
    os.makedirs(processed_images_dir, exist_ok=True)
    masks_dir = f"/home/anvy4548/projects/crystal-recognition/test_images/{new_folder}/grid_masks"
    os.makedirs(masks_dir, exist_ok=True)

    coords_dir = f"/home/anvy4548/projects/crystal-recognition/test_images/{new_folder}/coords"
    os.makedirs(coords_dir, exist_ok=True)



    #master_csv = os.path.join(coords_dir, "centroids_master.csv")


    for img in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, img)

        if not img.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue  # Skip non-image files

        if not os.path.isfile(image_path):
            continue  # Skip directories
            
        print(image_path)
        crystals, result_image, crystal_mask, carbon_mask = main(
            image_path, 25000,
            plot=True, img_return=True, return_mask=True
        )
        base = os.path.splitext(img)[0]

        #per_image_csv = os.path.join(coords_dir, f"{base}_centroids.csv")
        #write_per_image_csv(crystals, img, per_image_csv)

        #append_to_master_csv(crystals, img, master_csv)

        result_image_path = f"{img[:-4]}_inst"
        i = 1
        while os.path.exists(os.path.join(processed_images_dir, f"{result_image_path}_unet_after_random_walk_{i}.png")):
            i += 1
            # print(i)
        result_image_path = os.path.join(processed_images_dir, f"{result_image_path}_unet_after_random_walk_{i}.png")
        print(result_image_path)

        processed_image = Image.fromarray(np.uint8(result_image))
        processed_image.save(result_image_path)

        mask_path = os.path.join(masks_dir, f'{img[:-4]}.png')
        mask_to_save = (carbon_mask.astype(np.uint8) * 255)
        final_mask = Image.fromarray(mask_to_save)
        final_mask.save(mask_path)

        # crystal mask
        crystal_masks_dir = f"/home/anvy4548/projects/crystal-recognition/test_images/{new_folder}/crystal_masks"
        os.makedirs(crystal_masks_dir, exist_ok=True)

        crystal_mask_path = os.path.join(crystal_masks_dir, f"{img[:-4]}_crystals.png")
        Image.fromarray((crystal_mask.astype(np.uint8) * 255)).save(crystal_mask_path)

        # ---- compare to reference grid masks if available ----
        base = os.path.splitext(img)[0]

        # try common ref mask extensions
        ref_candidates = [
            os.path.join(ref_masks_dir, base + ".png"),
            os.path.join(ref_masks_dir, base + ".tif"),
            os.path.join(ref_masks_dir, base + ".tiff"),
            os.path.join(ref_masks_dir, base + ".jpg"),
            os.path.join(ref_masks_dir, base + ".jpeg"),
        ]
        ref_path = next((p for p in ref_candidates if os.path.exists(p)), None)

        metrics_dir = f"/home/anvy4548/projects/crystal-recognition/test_images/{new_folder}/grid_metrics"
        os.makedirs(metrics_dir, exist_ok=True)

        master_metrics_path = os.path.join(metrics_dir, "grid_metrics.tsv")

        # write header once
        if not os.path.exists(master_metrics_path) or os.path.getsize(master_metrics_path) == 0:
            with open(master_metrics_path, "w") as f:
                f.write("image\tIoU\trecall\n")

        # append one row per image
        with open(master_metrics_path, "a") as f:
            if ref_path is None:
                continue  # skip images without reference mask

            ref_mask = _load_ref_mask(ref_path, target_shape=carbon_mask.shape)
            iou, recall = iou_and_recall(carbon_mask, ref_mask)
            f.write(f"{img}\t{iou:.6f}\t{recall:.6f}\n")

        # print(crystals)
        print("--- %s seconds ---" % (time.time() - start_time))
        # break
#
