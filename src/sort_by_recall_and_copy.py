#!/usr/bin/env python3
import os
import csv
import glob
import shutil
import statistics as stats

import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
new_folder = "unet20_50_noscaling_training_only"

root = "/home/anvy4548/projects/crystal-recognition/test_images"
metrics_tsv = os.path.join(root, new_folder, "grid_metrics", "grid_metrics.tsv")

processed_images_dir = os.path.join(root, new_folder, "processed_images")

# Output
out_root = os.path.join(root, new_folder, "sorted_by_recall")
out_images = os.path.join(out_root, "processed_images_sorted")
out_metrics = os.path.join(out_root, "grid_metrics_sorted.txt")

# recall overview outputs
out_recall_table = os.path.join(out_root, "recall_across_all_images.tsv")
out_recall_rank_plot = os.path.join(out_root, "recall_by_rank.png")
out_recall_hist = os.path.join(out_root, "recall_hist.png")

# Sorting: True = best first
sort_desc = True

# If True, skip rows where result image can't be found
skip_missing_results = True

# How many to print in console for quick view
print_top_n = 20
# ------------------------------------------


def cm_to_inch(x):
    return x / 2.54


def set_iucr_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


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


def save_recall_plots(rows, out_rank_plot, out_hist):
    set_iucr_plot_style()

    recalls = [r["recall"] for r in rows]
    ranks = list(range(1, len(recalls) + 1))

    fig_w = cm_to_inch(8.85)

    blue = "#298c8c"  # matplotlib default blue

    # ---------- recall by rank ----------
    fig, ax = plt.subplots(figsize=(fig_w, cm_to_inch(6.0)))

    ax.plot(
        ranks,
        recalls,
        linewidth=1.2,
        color=blue
    )

    ax.set_xlabel("Rank by recall")
    ax.set_ylabel("Recall")
    ax.set_xlim(1, len(recalls))
    ax.set_ylim(0, 1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_rank_plot, dpi=600)
    plt.close(fig)

    # ---------- histogram ----------
    fig, ax = plt.subplots(figsize=(fig_w, cm_to_inch(6.0)))

    ax.hist(
        recalls,
        bins=30,
        color=blue,
        edgecolor="black",   # important for print
        linewidth=0.5
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_hist, dpi=600)
    plt.close(fig)


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

    # SORT BY RECALL
    rows.sort(key=lambda r: r["recall"], reverse=sort_desc)

    all_recalls = [r["recall"] for r in rows]
    mean_recall = sum(all_recalls) / len(all_recalls)
    median_recall = stats.median(all_recalls)
    min_recall = min(all_recalls)
    max_recall = max(all_recalls)

    print("\nRecall across all images")
    print(f"  N       : {len(all_recalls)}")
    print(f"  mean    : {mean_recall:.6f}")
    print(f"  median  : {median_recall:.6f}")
    print(f"  min/max : {min_recall:.6f} / {max_recall:.6f}")

    print(f"\nTop {min(print_top_n, len(rows))} by recall:")
    for i, r in enumerate(rows[:print_top_n], start=1):
        print(
            f"  {i:04d}  recall={r['recall']:.6f}  IoU={r['IoU']:.6f}  "
            f"image={os.path.basename(r['image'])}"
        )

    print(f"\nBottom {min(print_top_n, len(rows))} by recall:")
    for i, r in enumerate(rows[-print_top_n:], start=max(1, len(rows) - print_top_n + 1)):
        print(
            f"  {i:04d}  recall={r['recall']:.6f}  IoU={r['IoU']:.6f}  "
            f"image={os.path.basename(r['image'])}"
        )

    with open(out_recall_table, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rank", "image", "recall", "IoU"])
        for idx, r in enumerate(rows, start=1):
            w.writerow([idx, r["image"], f"{r['recall']:.6f}", f"{r['IoU']:.6f}"])

    save_recall_plots(rows, out_recall_rank_plot, out_recall_hist)
    print(f"\nWrote recall table to: {out_recall_table}")
    print(f"Saved plot (recall by rank) to: {out_recall_rank_plot}")
    print(f"Saved plot (recall histogram) to: {out_recall_hist}")

    with open(out_metrics, "w") as f:
        f.write("rank\trecall\tIoU\n")

        for idx, r in enumerate(rows, start=1):
            img = r["image"]
            base = os.path.splitext(os.path.basename(img))[0]

            pattern = os.path.join(
                processed_images_dir,
                f"{base}_inst_unet_after_random_walk_*.png"
            )
            src_result = newest_match(pattern)

            if src_result is None and skip_missing_results:
                continue

            recall_str = f"{r['recall']:.4f}"
            dst_name = (
                f"{idx:04d}_Recall{recall_str}_"
                f"{os.path.basename(src_result) if src_result else base + '.png'}"
            )
            dst_path = os.path.join(out_images, dst_name)

            if src_result:
                shutil.copy2(src_result, dst_path)

            f.write(f"{idx}\t{r['recall']:.6f}\t{r['IoU']:.6f}\n")

    print(f"\nCopied sorted images to: {out_images}")
    print(f"Wrote sorted metrics to: {out_metrics}")


if __name__ == "__main__":
    main()