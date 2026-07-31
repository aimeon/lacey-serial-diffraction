from pathlib import Path
import cv2

src = Path("/media/anvy4548/T7/montage_100im_orig/grid_masks")
dst = Path("/home/anvy4548/projects/crystal-recognition/training_test_all_masks")
dst.mkdir(parents=True, exist_ok=True)

for p in src.iterdir():
    if p.suffix.lower() in [".png", ".jpg", ".tif", ".tiff"]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        inv = 255 - img
        cv2.imwrite(str(dst / p.name), inv)
