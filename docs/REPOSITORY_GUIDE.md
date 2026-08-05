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


## Complete workflow

The complete workflow consists of data preparation, model training, carbon-support prediction, segmentation-assisted crystal detection and evaluation.

### 1. Obtain the data

The training and evaluation data are available from Zenodo:

https://doi.org/10.5281/zenodo.20269439

The record contains two archives:

- `patches_for_training.zip` contains 10,272 paired 256 x 256 pixel TEM image patches and corresponding carbon-support masks after patch extraction and augmentation. These files can be used directly for model training without additional preprocessing or augmentation.
- `test_all_images.zip` contains the 100-image grid map and nine additional individual TEM images used to evaluate the complete workflow.

If the prepared patches are used directly, the patch-extraction step can be skipped.

### 2. Prepare image-mask patches

The data-preparation workflow is provided in:

```text
src/notebooks/data_preparation.ipynb
```

Before running the notebook, specify the directories containing:

- the original TEM images;
- the corresponding binary carbon-support masks;
- the output image patches;
- the output mask patches.

Image and mask files must have matching filenames.

The notebook uses:

```text
src/preprocessing/patchifier.py
src/preprocessing/augmentator.py
```

The 516 × 516 pixel images are resized to 512 × 512 pixels and divided into four non-overlapping 256 × 256 pixel patches. The 2048 × 2048 pixel images are divided directly into 256 × 256 pixel patches without rescaling.

Augmentation includes rotations, horizontal and vertical flips, and random brightness and contrast adjustments. Geometric transformations are applied to both images and masks, whereas brightness and contrast adjustments are applied only to the images.

The notebook can be opened and run directly from an IDE or Jupyter environment.

### 3. Train the U-Net model

The training workflow used for the reported model is provided in:

```text
src/notebooks/simple_unet_imports_folded.ipynb
```

Before running the notebook, specify the directories containing the prepared image patches and mask patches.

The notebook:

1. loads matching image-mask pairs;
2. reads the images as grayscale;
3. converts the masks to binary values;
4. normalizes the images column-wise using the L2 norm;
5. divides the data into training and validation subsets;
6. trains the U-Net model;
7. evaluates segmentation performance;
8. saves the trained Keras model.

The reusable model components are located in:

```text
src/models/unet_architecture.py
src/models/loss_functions.py
src/models/utils.py
```

The trained models used in the study are provided in:

```text
src/trained_models/
```

The model used in the integrated segmentation-assisted workflow is:

```text
src/trained_models/standard_aug_20_50epochs.keras
```

### 4. Predict the carbon-support mask

Patch-wise carbon-support prediction is implemented in:

```text
src/find_grid_unet.py
```

For each TEM image, the script:

1. divides the image into 256 × 256 pixel patches;
2. applies the same normalization used during training;
3. predicts a binary carbon-support mask for each patch;
4. reconstructs the complete carbon mask from the patch predictions.

The predicted carbon mask is then passed to the complete crystal-detection workflow.

### 5. Run segmentation-assisted crystal detection

The complete segmentation-assisted workflow is implemented in:

```text
src/crystal_finder_inst_and_unet.py
```

Before running the script, specify:

- the path to the trained Keras model;
- the directory containing the TEM images;
- the output directory for predicted carbon masks;
- the output directory for detected crystal masks;
- the output directory for visualizations;
- the output directory for target coordinates;
- the reference-mask directory, if segmentation metrics should be calculated.

The script can then be run directly from an IDE.

For each TEM image, the workflow:

1. loads the image;
2. predicts the lacey carbon-support mask;
3. applies adaptive local thresholding to identify candidate features;
4. applies the predicted carbon mask to suppress carbon-support features;
5. performs morphological filtering;
6. labels connected candidate regions;
7. assigns target positions to the detected regions;
8. saves the predicted masks, detected regions, target coordinates and optional visualizations.

In the evaluation reported in the study, one target position was assigned to each detected region.

### 6. Run the threshold-only reference workflow

The threshold-based reference workflow is provided in:

```text
src/crystal_finder_from_instamatic.py
```

Before running the script, specify the input-image and output directories in its configuration section.

This workflow performs crystal detection without applying the neural-network carbon mask. Its results can therefore be compared directly with those from the segmentation-assisted workflow.

The baseline workflow was adapted from the Instamatic source code:

https://github.com/instamatic-dev/instamatic

### 7. Evaluate carbon-support segmentation

Carbon-mask IoU and recall are calculated during the segmentation-assisted workflow and written to a TSV file.

Mean IoU and recall can be calculated using:

```text
src/evaluation/avg_iou_recall.py
```

Set the path to the generated `grid_metrics.tsv` file in the configuration section and run the script directly from the IDE.

Results can be organized and visualized using:

```text
src/evaluation/sort_by_iou_and_copy.py
src/evaluation/sort_by_recall_and_copy.py
```

These scripts produce ranked result sets, summary tables and distributions of IoU or recall.

### 8. Evaluate crystal detection

Crystal-level detection performance is evaluated using:

```text
src/evaluation/evaluation.py
```

Before running the script, specify:

- the directory containing the original TEM images;
- the directory containing the manually annotated crystal masks;
- the directory containing the predicted crystal masks;
- the output location for comparison images, if required.

A detected region is counted as a true positive when its selected target position falls inside a manually annotated crystal boundary. Target positions outside annotated crystals are counted as false positives, while annotated crystals containing no selected target position are counted as false negatives.

The resulting counts are used to calculate:

```text
sensitivity = TP / (TP + FN)
precision   = TP / (TP + FP)
```

The threshold-only and segmentation-assisted workflows should be evaluated using the same reference masks and matching criteria.