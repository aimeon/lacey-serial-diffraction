#!/usr/bin/env python3
import os
import csv
import glob
import shutil
import statistics as stats

import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
new_folder = "unet20_50_noscaling_training_only"  # must match what you used when generating outputs

root = "/home/anvy4548/projects/crystal-recognition/test_images"
metrics_tsv = os.path.join(root, new_folder, "grid_metrics", "grid_metrics.tsv")

processed_images_dir = os.path.join(root, new_folder, "processed_images")

# Output
out_root = os.path.join(root, new_folder, "sorted_by_iou")
out_images = os.path.join(out_root, "processed_images_sorted")
out_metrics = os.path.join(out_root, "grid_metrics_sorted.txt")

# NEW: IoU overview outputs
out_iou_table = os.path.join(out_root, "iou_across_all_images.tsv")
out_iou_rank_plot = os.path.join(out_root, "iou_by_rank.png")
out_iou_hist = os.path.join(out_root, "iou_hist.png")

# Sorting: True = best first
sort_desc = True

# If True, skip rows where result image can't be found
skip_missing_results = True

# How many to print in console for quick view
print_top_n = 20
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


def save_iou_plots(rows, out_rank_plot, out_hist):
    # IoU-by-rank (after sorting)
    ious = [r["IoU"] for r in rows]
    ranks = list(range(1, len(ious) + 1))

    plt.figure()
    plt.plot(ranks, ious)
    plt.xlabel("Rank (by IoU)")
    plt.ylabel("IoU")
    plt.title("IoU across all images (sorted)")
    plt.tight_layout()
    plt.savefig(out_rank_plot, dpi=200)
    plt.close()

    # Histogram of IoU
    plt.figure()
    plt.hist(ious, bins=30)
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.title("IoU distribution across all images")
    plt.tight_layout()
    plt.savefig(out_hist, dpi=200)
    plt.close()


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
            if iou is None or recall is None:
                continue
            rows.append({"image": image, "IoU": iou, "recall": recall})

    if not rows:
        raise RuntimeError(f"No numeric IoU/recall rows found in: {metrics_tsv}")

    # Sort by IoU
    rows.sort(key=lambda r: r["IoU"], reverse=sort_desc)

    # ---- NEW: show IoU across all images (console + files) ----
    all_ious = [r["IoU"] for r in rows]
    mean_iou = sum(all_ious) / len(all_ious)
    median_iou = stats.median(all_ious)
    min_iou = min(all_ious)
    max_iou = max(all_ious)

    print("\nIoU across all images")
    print(f"  N       : {len(all_ious)}")
    print(f"  mean    : {mean_iou:.6f}")
    print(f"  median  : {median_iou:.6f}")
    print(f"  min/max : {min_iou:.6f} / {max_iou:.6f}")

    # Print quick ranked preview
    print(f"\nTop {min(print_top_n, len(rows))} by IoU:")
    for i, r in enumerate(rows[:print_top_n], start=1):
        print(f"  {i:04d}  IoU={r['IoU']:.6f}  recall={r['recall']:.6f}  image={os.path.basename(r['image'])}")

    print(f"\nBottom {min(print_top_n, len(rows))} by IoU:")
    for i, r in enumerate(rows[-print_top_n:], start=max(1, len(rows) - print_top_n + 1)):
        print(f"  {i:04d}  IoU={r['IoU']:.6f}  recall={r['recall']:.6f}  image={os.path.basename(r['image'])}")

    # Save per-image IoU table
    with open(out_iou_table, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rank", "image", "IoU", "recall"])
        for idx, r in enumerate(rows, start=1):
            w.writerow([idx, r["image"], f"{r['IoU']:.6f}", f"{r['recall']:.6f}"])

    # Save plots
    save_iou_plots(rows, out_iou_rank_plot, out_iou_hist)
    print(f"\nWrote IoU table to: {out_iou_table}")
    print(f"Saved plot (IoU by rank) to: {out_iou_rank_plot}")
    print(f"Saved plot (IoU histogram) to: {out_iou_hist}")
    # ----------------------------------------------------------

    # Write sorted metrics + copy images (your existing behavior)
    with open(out_metrics, "w") as f:
        f.write("rank\tIoU\trecall\n")

        for idx, r in enumerate(rows, start=1):
            img = r["image"]
            base = os.path.splitext(os.path.basename(img))[0]

            pattern = os.path.join(processed_images_dir, f"{base}_inst_unet_after_random_walk_*.png")
            src_result = newest_match(pattern)

            if src_result is None and skip_missing_results:
                continue

            iou_str = f"{r['IoU']:.4f}"
            dst_name = f"{idx:04d}_IoU{iou_str}_{os.path.basename(src_result) if src_result else base + '.png'}"
            dst_path = os.path.join(out_images, dst_name)

            if src_result:
                shutil.copy2(src_result, dst_path)

            f.write(f"{idx}\t{r['IoU']:.6f}\t{r['recall']:.6f}\n")

    print(f"\nCopied sorted images to: {out_images}")
    print(f"Wrote sorted metrics to: {out_metrics}")


if __name__ == "__main__":
    main()