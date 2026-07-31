import os
import time
import sys
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import filters, measure, morphology, segmentation
from scipy._lib._util import _asarray_validated
from scipy.cluster.vq import kmeans2
from matplotlib import pyplot as plt
from collections import namedtuple
from pathlib import Path

calibration = {
    "mag1": {
        "pixelsize": {
            25000: 1.14147,
            30000: 0.96511,
            40000: 0.72286,
            50000: 0.57751,
            60000: 0.48255,
            80000: 0.36046,
            100000: 0.28875,
            120000: 0.24031,
            150000: 0.19186,
            200000: 0.14341
        }
    }
}

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


CrystalPosition = namedtuple('CrystalPosition', ['x', 'y', 'isolated', 'n_clusters', 'area_micrometer', 'area_pixel'])


def is_edge(prop):
    """Detects if a region touches the edge and is likely an artifact."""
    slc = prop._slice
    shape = prop._intensity_image.shape
    if (slc[0].start == 0 or slc[1].start == 0 or
        slc[0].stop == shape[0] or slc[1].stop == shape[1]):

        hist, _ = np.histogram(prop.intensity_image[prop.image])
        if np.sum(hist) // hist[0] < 2:
            return True
    return False


def autoscale(img, maxdim=512):
    """Resize image so its largest dimension is `maxdim`."""
    if maxdim:
        scale = float(maxdim) / max(img.shape)
        return ndimage.zoom(img, scale, order=1), scale
    return img, 1.0


def whiten(obs, check_finite=False):
    obs = _asarray_validated(obs, check_finite=check_finite)
    std_dev = np.std(obs, axis=0)
    std_dev[std_dev == 0] = 1.0
    return obs / std_dev, std_dev,



def segment_crystals(img, r=101, offset=25, footprint=5, remove_carbon_lacing=True):

    offset = offset / 255.0
    img = img * (1.0 / img.max())

    arr = img > filters.threshold_local(img, r, method='mean', offset=offset)
    arr = np.invert(arr)

    arr = morphology.remove_small_objects(arr, min_size=16, connectivity=0)
    arr = morphology.binary_closing(arr, morphology.disk(footprint))
    arr = morphology.binary_erosion(arr, morphology.disk(footprint))

    if remove_carbon_lacing:
        arr = morphology.remove_small_objects(arr, min_size=64, connectivity=0)
        arr = morphology.remove_small_holes(arr, area_threshold=1024, connectivity=0)

    arr = morphology.binary_dilation(arr, morphology.disk(footprint))
    bkg = np.invert(morphology.binary_dilation(arr, morphology.disk(footprint * 1)) | arr)
    markers = arr * 2 + bkg

    segmented = segmentation.random_walker(img, markers, beta=50, spacing=(5, 5), mode='bf')
    return arr, segmented.astype(int) - 1


def find_crystals(img, magnification, spread=20, plot=False, img_return=False, return_mask=False, **kwargs):
    img_scaled, scale = autoscale(img, maxdim=512)
    arr, seg = segment_crystals(img_scaled, **kwargs)
    labels, _ = ndimage.label(seg)
    props = measure.regionprops(labels, img_scaled)

    px = py = calibration['mag1']['pixelsize'][magnification] / 1000  # nm -> um
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

        ax.imshow(img_scaled, cmap="gray")
        ax.contour(seg, [0.5], linewidths=1.2, colors='yellow')
        if crystals:
            x, y = np.array([(c.x, c.y) for c in crystals]).T
            ax.scatter(y, x, color='red')
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        result = np.array(fig.canvas.renderer.buffer_rgba())

        plt.show()

        if img_return and return_mask:
            return crystals, result, seg
        if img_return:
            return crystals, result

    if return_mask:
        return crystals, seg

    return crystals


def save_step(image, out_path, cmap="gray"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
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


def visualize_steps(img, out_dir, r=101, offset=25, footprint=5, remove_carbon_lacing=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    offset = offset / 255.0
    img = img.astype(np.float32)
    img = img * (1.0 / img.max())

    save_step(img, out_dir / "01_original.png")

    arr = img > filters.threshold_local(img, r, method='mean', offset=offset)
    arr = np.invert(arr)
    save_step(arr, out_dir / "02_thresholded.png")

    arr = morphology.remove_small_objects(arr, min_size=16, connectivity=0)
    save_step(arr, out_dir / "03_small_objects_removed.png")

    arr = morphology.binary_closing(arr, morphology.disk(footprint))
    save_step(arr, out_dir / "04_closing.png")

    arr = morphology.binary_erosion(arr, morphology.disk(footprint))
    save_step(arr, out_dir / "05_erosion.png")

    if remove_carbon_lacing:
        arr = morphology.remove_small_objects(arr, min_size=64, connectivity=0)
        save_step(arr, out_dir / "06_remove_small_objects_2.png")

        arr = morphology.remove_small_holes(arr, area_threshold=1024, connectivity=0)
        save_step(arr, out_dir / "07_remove_small_holes.png")

    arr = morphology.binary_dilation(arr, morphology.disk(footprint))
    save_step(arr, out_dir / "08_dilation.png")

    bkg = np.invert(morphology.binary_dilation(arr, morphology.disk(footprint * 1)) | arr)
    save_step(bkg, out_dir / "09_background.png")

    markers = arr * 2 + bkg
    save_step(markers, out_dir / "10_markers.png", cmap="gray")

    segmented = segmentation.random_walker(img, markers, beta=50, spacing=(5, 5), mode='bf')
    segmented = segmented.astype(int) - 1
    save_step(segmented, out_dir / "11_random_walker.png", cmap="gray")

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

    return arr, segmented


def main(image_path, spread=20, plot=False, img_return=True, return_mask=False, debug_dir=None, **kwargs):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    img = np.array(Image.fromarray(img).resize((512, 512)))

    if debug_dir is not None:
        visualize_steps(img, debug_dir, **kwargs)

    start = time.time()
    result = find_crystals(
        img, magnification=25000,
        spread=spread,
        plot=plot,
        img_return=img_return,
        return_mask=return_mask,
        **kwargs
    )
    print(f"Processed in {time.time() - start:.2f} seconds")
    return result


if __name__ == "__main__":

    base_dir = "/home/anvy4548/projects/crystal-recognition/results_final/intamatic_all_images"

    test_image_path = r"/home/anvy4548/projects/montage_100im_orig/_png/mont_0042.png"
    debug_dir = os.path.join(base_dir, "debug_steps_IM", "mont_0042")

    test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    test_img = np.array(Image.fromarray(test_img).resize((512, 512)))

    visualize_steps(test_img, out_dir=debug_dir)


    sys.exit()

    base_dir = "/home/anvy4548/projects/crystal-recognition/results_final/intamatic_all_images"

    test_images_dir =  "/home/anvy4548/projects/crystal-recognition/training_test_all_images/"
    output_dir = os.path.join(base_dir, "processed")
    mask_dir = os.path.join(base_dir, "masks")
    coords_dir = os.path.join(base_dir, "coords")


    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(coords_dir, exist_ok=True)

    master_csv = os.path.join(coords_dir, "centroids_master.csv")

    for img_name in sorted(os.listdir(test_images_dir)):
        if not img_name.endswith(".png"):
            continue

        image_path = os.path.join(test_images_dir, img_name)

        crystals, result_image, mask = main(
            image_path,
            spread=20,
            plot=True,
            img_return=True,
            return_mask=True
        )

        base_name = os.path.splitext(img_name)[0]

        print(f"Processing: {image_path}")

        # save processed image
        output_path = os.path.join(output_dir, f"{base_name}.png")
        Image.fromarray(result_image).save(output_path)

        # save mask
        mask_path = os.path.join(mask_dir, f"{base_name}.png")
        mask_to_save = (mask.astype(np.uint8) * 255)
        Image.fromarray(mask_to_save).save(mask_path)

        print(f"Saved to: {base_dir}")
        print("-" * 30)