import os
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

gt_dir = "path/to/ground_truth_masks"
pred_dir = "path/to/predicted_masks"

overlap_threshold = 0.1  # 10% overlap is typical for crystal picking

TP = FP = FN = 0

for fname in os.listdir(gt_dir):
    if not fname.lower().endswith(".png"):
        continue

    gt_path = os.path.join(gt_dir, fname)
    pred_path = os.path.join(pred_dir, fname)

    if not os.path.exists(pred_path):
        continue

    gt = np.array(Image.open(gt_path).convert("L")) > 0
    pred = np.array(Image.open(pred_path).convert("L")) > 0

    gt_labels = label(gt)
    pred_labels = label(pred)

    gt_regions = regionprops(gt_labels)
    pred_regions = regionprops(pred_labels)

    matched_gt = set()

    # Match predicted crystals to ground truth
    for pred_region in pred_regions:
        pred_mask = (pred_labels == pred_region.label)

        overlaps = []
        for i, gt_region in enumerate(gt_regions):
            gt_mask = (gt_labels == gt_region.label)
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            overlap = intersection / gt_region.area
            overlaps.append(overlap)

        if overlaps and max(overlaps) >= overlap_threshold:
            TP += 1
            matched_gt.add(np.argmax(overlaps))
        else:
            FP += 1

    # Ground-truth crystals not detected
    FN += len(gt_regions) - len(matched_gt)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Precision   : {precision:.3f}")
print(f"Sensitivity : {sensitivity:.3f}")
print(f"F1 score    : {f1:.3f}")
