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


def find_crystals(img, spread=20, plot=False, img_return=False, return_mask=False, **kwargs):
    img_scaled, scale = autoscale(img, maxdim=256)
    arr, seg = segment_crystals(img_scaled, **kwargs)
    labels, _ = ndimage.label(seg)
    props = measure.regionprops(labels, img_scaled)

    crystals = []
    for prop in props:
        if is_edge(prop):
            continue

        area = prop.area
        origin = np.array(prop.bbox[0:2])
        nclust = int(area // spread) + 1

        if nclust > 1:
            coords = np.argwhere(prop.image)
            w, std = whiten(coords)
            centroids, _ = kmeans2(w, nclust, iter=20, minit='points')
            xy = centroids * std + origin
            crystals.extend([CrystalPosition(x, y, False, nclust, area, area) for x, y in xy])
        else:
            x, y = prop.centroid
            crystals.append(CrystalPosition(x, y, True, nclust, area, area))

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(img_scaled, cmap="gray")
        ax.contour(seg, [0.5], linewidths=1.2, colors='yellow')
        if crystals:
            x, y = np.array([(c.x, c.y) for c in crystals]).T
            ax.scatter(y, x, color='red')
        ax.set_axis_off()
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


def main(image_path, spread=20, plot=False, img_return=True, return_mask=False, **kwargs):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    img = np.array(Image.fromarray(img).resize((256, 256)))

    start = time.time()
    result = find_crystals(
        img,
        spread=spread,
        plot=plot,
        img_return=img_return,
        return_mask=return_mask,
        **kwargs
    )
    print(f"Processed in {time.time() - start:.2f} seconds")
    return result



if __name__ == "__main__":
    test_images_dir = r"/home/anvy4548/projects/montage_100im_orig/_png"
    output_dir = r"/home/anvy4548/projects/montage_100im_orig/processed_IM"
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = r"/home/anvy4548/projects/montage_100im_orig/masks_IM"
    os.makedirs(mask_dir, exist_ok=True)

    coords_dir = r"/home/anvy4548/projects/montage_100im_orig/coords_IM"
    os.makedirs(coords_dir, exist_ok=True)

    master_csv = os.path.join(coords_dir, "centroids_master.csv")

    for img_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, img_name)

        crystals, result_image, mask = main(
            image_path,
            spread=2500,
            plot=True,
            img_return=True,
            return_mask=True
        )

        base_name = os.path.splitext(img_name)[0]

        #per_image_csv = os.path.join(coords_dir, f"{base_name}_centroids.csv")
        #write_per_image_csv(crystals, img_name, per_image_csv)

        #append_to_master_csv(crystals, img_name, master_csv)

        print(image_path)


        # ---- save processed image ----
        i = 1
        while os.path.exists(os.path.join(output_dir, f"{base_name}_inst_r101_{i}.png")):
            i += 1

        output_path = os.path.join(output_dir, f"{base_name}_inst_r101_{i}.png")
        Image.fromarray(result_image).save(output_path)

        # ---- save mask ----
        mask_path = os.path.join(mask_dir, f"{base_name}.png")

        # Convert boolean/int mask to 0–255
        mask_to_save = (mask.astype(np.uint8) * 255)
        Image.fromarray(mask_to_save).save(mask_path)

        print(f"Saved mask to: {mask_path}")
        print("---" * 10)

