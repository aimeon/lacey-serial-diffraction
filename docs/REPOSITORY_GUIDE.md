# Repository guide

This guide describes the role of each source file. Several scripts were
developed as research utilities and require local input and output paths to be
specified before use.

## Main workflows

| File | Purpose |
| --- | --- |
| `src/crystal_finder_from_instamatic.py` | Reference threshold-based crystal detection adapted from the Instamatic/SerialED workflow. |
| `src/crystal_finder_inst_and_unet.py` | Carbon-support segmentation followed by threshold-based crystal detection. This is the principal evaluated workflow. |
| `src/crystal_finder_inst_and_unet_inpainting.py` | Experimental variant that replaces masked carbon regions by inpainting. |
| `src/find_grid_unet.py` | Patch-wise U-Net inference and reconstruction of a full carbon-support mask. |

## Models

| File | Purpose |
| --- | --- |
| `src/models/unet_architecture.py` | U-Net definition used for training. |
| `src/models/loss_functions.py` | Jaccard, Dice, focal, binary-cross-entropy and combined loss functions. |
| `src/models/utils.py` | Paired image/mask loading, mask conversion and column-wise L2 normalization. |

## Preprocessing

| File | Purpose |
| --- | --- |
| `src/preprocessing/patchifier.py` | Matches images to masks, optionally resizes them and writes non-overlapping patches. |
| `src/preprocessing/augmentator.py` | Applies paired rotations/flips and image-only brightness/contrast augmentation. |
| `src/preprocessing/copy_and_invert_masks.py` | Utility for copying and inverting a set of grayscale masks. |
| `src/preprocessing/aug_test.ipynb` | Exploratory visualization of augmentation settings. |

The empty conversion/checking files are retained only to reflect utilities used
during development and do not contribute to the reported results.

## Evaluation

| File | Purpose |
| --- | --- |
| `src/evaluation/avg_iou_recall.py` | Summarizes IoU and recall values stored in a TSV file. |
| `src/evaluation/evaluation.py` | Object-level comparison of predicted crystal masks with manual reference masks. |
| `src/evaluation/metrics.py` | Alternative overlap-based crystal-level metric calculation. |
| `src/evaluation/sort_by_iou_and_copy.py` | Ranks segmentation results by IoU and creates plots/tables. |
| `src/evaluation/sort_by_recall_and_copy.py` | Ranks segmentation results by recall and creates plots/tables. |

## Notebooks

| File | Purpose |
| --- | --- |
| `src/notebooks/data_preparation.ipynb` | Patch extraction and augmentation. |
| `src/notebooks/simple_unet.ipynb` | Original U-Net training workflow. |
| `src/notebooks/simple_unet_imports_folded.ipynb` | Extended training and loss-function exploration. |
| `src/notebooks/test_models.ipynb` | Comparison of segmentation architectures. |

## Trained models

- `standard_aug_20epochs.keras`: model saved after the initial training run.
- `standard_aug_20_50epochs.keras`: model used in the reported integrated
  segmentation-assisted workflow.
