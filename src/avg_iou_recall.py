#!/usr/bin/env python3
import csv
import sys
import math

def safe_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def main(path):
    ious = []
    recalls = []

    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # expects columns: image, ref_mask, IoU, recall
        for r in reader:
            iou = safe_float(r.get("IoU", ""))
            rec = safe_float(r.get("recall", ""))
            if iou is None or rec is None:
                continue
            ious.append(iou)
            recalls.append(rec)

    n = len(ious)
    if n == 0:
        print("No numeric IoU/recall rows found.")
        sys.exit(1)

    mean_iou = sum(ious) / n
    mean_rec = sum(recalls) / n

    print(f"Rows used: {n}")
    print(f"Mean IoU:   {mean_iou:.6f}")
    print(f"Mean recall:{mean_rec:.6f}")

if __name__ == "__main__":
    metrics_tsv = (
        "/home/anvy4548/projects/crystal-recognition/"
        "test_images/unet20_50_noscaling/grid_metrics/grid_metrics.tsv"
    )
    main(metrics_tsv)

