import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from matplotlib.colors import ListedColormap

# ===========================
# Paths
# ===========================
img_dir  = "/home/anvy4548/projects/montage_100im_orig/_png"
gt_dir   = "/home/anvy4548/projects/montage_100im_orig/crystal_masks"
pred_dir = "/home/anvy4548/projects/montage_100im_orig/masks_IM"

fname = next(f for f in os.listdir(gt_dir) if f.lower().endswith("99.png"))
print("Using:", fname)

save_path = None  # e.g. "debug_match.png"

# ===========================
# Helpers
# ===========================
def load_gray(path):
    return np.array(Image.open(path).convert("L"))

def load_gt_mask(path):
    # GT: black = crystal
    return np.array(Image.open(path).convert("L")) == 0

def load_pred_mask(path):
    # Pred: white = crystal
    return np.array(Image.open(path).convert("L")) > 0

def resize_mask_to(mask_bool, target_shape):
    h, w = target_shape
    im = Image.fromarray(mask_bool.astype(np.uint8) * 255)
    im = im.resize((w, h), resample=Image.NEAREST)
    return np.array(im) > 0

def iou(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return inter / union if union > 0 else 0.0

# ===========================
# Load data
# ===========================
img_path  = os.path.join(img_dir, fname)
gt_path   = os.path.join(gt_dir, fname)
pred_path = os.path.join(pred_dir, fname)

bg = load_gray(img_path) if os.path.exists(img_path) else None
gt = load_gt_mask(gt_path)
pr = load_pred_mask(pred_path)

if pr.shape != gt.shape:
    pr = resize_mask_to(pr, gt.shape)

gt_lab = label(gt)
pr_lab = label(pr)

gt_regs = regionprops(gt_lab)
pr_regs = regionprops(pr_lab)

# Map GT label -> index
gt_label_to_index = {r.label: i for i, r in enumerate(gt_regs)}

# ===========================
# Matching: centroid-in-GT rule
# ===========================
matched_gt = set()
matches = []  # (pred_idx, gt_idx_or_None, IoU)

TP = FP = 0

for pi, pr_reg in enumerate(pr_regs):
    pred_mask = (pr_lab == pr_reg.label)

    cy, cx = pr_reg.centroid
    cy, cx = int(round(cy)), int(round(cx))

    # Out of bounds safety
    if cy < 0 or cy >= gt_lab.shape[0] or cx < 0 or cx >= gt_lab.shape[1]:
        FP += 1
        matches.append((pi, None, 0.0))
        continue

    gt_label = gt_lab[cy, cx]

    if gt_label > 0:
        gi = gt_label_to_index[gt_label]

        # enforce 1-to-1
        if gi in matched_gt:
            FP += 1
            matches.append((pi, None, 0.0))
        else:
            matched_gt.add(gi)
            TP += 1
            gt_mask = (gt_lab == gt_label)
            matches.append((pi, gi, iou(pred_mask, gt_mask)))
    else:
        FP += 1
        matches.append((pi, None, 0.0))

FN = len(gt_regs) - len(matched_gt)

# ===========================
# Visualization
# ===========================
fig, ax = plt.subplots(figsize=(10, 10))

if bg is not None:
    ax.imshow(bg, cmap="gray")
else:
    ax.imshow(np.zeros_like(gt), cmap="gray")

green = ListedColormap([(0, 1, 0, 1)])
red   = ListedColormap([(1, 0, 0, 1)])

gt_b = find_boundaries(gt, mode="outer")
pr_b = find_boundaries(pr, mode="outer")

# fills
ax.imshow(np.ma.masked_where(~gt, gt), cmap=green, alpha=0.15, interpolation="nearest")
ax.imshow(np.ma.masked_where(~pr, pr), cmap=red,   alpha=0.15, interpolation="nearest")

# outlines
ax.imshow(np.ma.masked_where(~gt_b, gt_b), cmap=green, alpha=0.9, interpolation="nearest")
ax.imshow(np.ma.masked_where(~pr_b, pr_b), cmap=red,   alpha=0.9, interpolation="nearest")

# label GT
for gi, r in enumerate(gt_regs):
    y, x = r.centroid
    ax.text(x, y, f"G{gi}", color="lime", fontsize=10, weight="bold")

# label predictions + lines
for pi, r in enumerate(pr_regs):
    y, x = r.centroid
    ax.text(x, y, f"P{pi}", color="red", fontsize=10, weight="bold")

    gi, ov = matches[pi][1], matches[pi][2]

    if gi is not None:
        gy, gx = gt_regs[gi].centroid
        ax.plot([x, gx], [y, gy], linewidth=1.5)
        ax.text((x+gx)/2, (y+gy)/2, f"IoU={ov:.2f}", fontsize=9, weight="bold")
    else:
        ax.text(x, y+10, "FP", color="red", fontsize=9)

ax.set_axis_off()
plt.tight_layout()

if save_path:
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

plt.show()

# ===========================
# Summary
# ===========================
print(f"GT objects   : {len(gt_regs)}")
print(f"Pred objects : {len(pr_regs)}")
print(f"TP={TP}  FP={FP}  FN={FN}")
