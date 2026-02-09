#!/usr/bin/env python3
import os
import csv
import glob
import shutil

# ----------------- CONFIG -----------------
new_folder = "unet20_50_noscaling"  # must match what you used when generating outputs

root = "/home/anvy4548/projects/crystal-recognition/test_images"
metrics_tsv = os.path.join(root, new_folder, "grid_metrics", "grid_metrics.tsv")

processed_images_dir = os.path.join(root, new_folder, "processed_images")

# Output
out_root = os.path.join(root, new_folder, "sorted_by_iou")
out_images = os.path.join(out_root, "processed_images_sorted")
out_metrics = os.path.join(out_root, "grid_metrics_sorted.txt")

# Sorting: True = best first
sort_desc = True

# If True, skip rows where result image can't be found
skip_missing_results = True
# ------------------------------------------


def newest_match(pattern: str):
    """Return newest file matching glob pattern, or None."""
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p))
    return matches[-1]


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_root, exist_ok=True)

    rows = []
    with open(metrics_tsv, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            iou = safe_float(r.get("IoU", ""))
            recall = safe_float(r.get("recall", ""))
            image = r.get("image", "")
            ref_mask = r.get("ref_mask", "")
            if iou is None or recall is None:
                continue  # skip non-numeric rows (incl. missing refs if they ever got written)
            rows.append({"image": image, "IoU": iou, "recall": recall})

    rows.sort(key=lambda r: r["IoU"], reverse=sort_desc)

    # Write sorted metrics
    with open(out_metrics, "w") as f:
        f.write("rank\tIoU\trecall\n")

        for idx, r in enumerate(rows, start=1):
            img = r["image"]
            base = os.path.splitext(os.path.basename(img))[0]

            # match your saved naming pattern
            pattern = os.path.join(processed_images_dir, f"{base}_inst_unet_after_random_walk_*.png")
            src_result = newest_match(pattern)

            if src_result is None and skip_missing_results:
                continue

            # rename by rank + IoU
            iou_str = f"{r['IoU']:.4f}"
            dst_name = f"{idx:04d}_IoU{iou_str}_{os.path.basename(src_result) if src_result else base + '.png'}"
            dst_path = os.path.join(out_images, dst_name)

            if src_result:
                shutil.copy2(src_result, dst_path)

            f.write(
                f"{idx}\t{r['IoU']:.6f}\t{r['recall']:.6f}\n"
            )

    print(f"Copied sorted images to: {out_images}")
    print(f"Wrote sorted metrics to: {out_metrics}")


if __name__ == "__main__":
    main()
