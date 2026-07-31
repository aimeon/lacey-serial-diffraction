# Neural-network masking of lacey carbon grids

Code and trained models for lacey carbon segmentation-assisted crystal detection in
transmission electron microscopy (TEM) images. The workflow uses a U-Net to
identify the lacey carbon support and applies the predicted carbon mask before
the original threshold-based crystal-detection procedure.

The method was developed for automated electron-diffraction workflows. Its
purpose is to reduce target positions assigned to carbon features while
preserving the existing threshold-based detection and target-selection
steps.

## Repository contents

```text
src/
├── crystal_finder_from_instamatic.py          # threshold-only reference workflow
├── crystal_finder_inst_and_unet.py            # segmentation-assisted workflow
├── crystal_finder_inst_and_unet_inpainting.py # experimental inpainting variant
├── find_grid_unet.py                           # patch-wise carbon-mask prediction
├── evaluation/                                 # segmentation and detection metrics
├── models/                                     # U-Net architecture, losses and data loading
├── notebooks/                                  # data preparation and model training
├── preprocessing/                              # patch extraction and augmentation
└── trained_models/                             # trained Keras models
```

More detail about individual files is provided in
[`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md).

## Installation

The code was developed with Python 3.10 and TensorFlow/Keras. Create the Conda
environment from the repository root:

```bash
conda env create -f environment.yml
conda activate lacey-carbon
```

The trained models are stored in `src/trained_models/`. The main model used in
the segmentation-assisted workflow is:

```text
src/trained_models/standard_aug_20_50epochs.keras
```

## Dataset

The dataset associated with this project is available from Zenodo:

https://doi.org/10.5281/zenodo.20269439

The record contains two archives:

- `patches_for_training.zip` contains 256 x 256 pixel TEM image patches used for model development and their corresponding manually annotated carbon-support masks.
- `test_all_images.zip` contains the 100-image grid map and nine additional individual TEM images used to evaluate the complete workflow.

The data are distributed separately and are not duplicated in this repository.


## Input data organization

Training requires paired grayscale TEM images and binary carbon-support masks.
Image and mask files must have the same filenames.

```text
data/
├── images/
│   ├── image_001.tif
│   └── image_002.tif
└── masks/
    ├── image_001.tif
    └── image_002.tif
```

Masks use `0` for the non-carbon pixels and a non-zero value for the carbon
pixels. During loading, masks are binarized and converted to values of 0 and 1.


## Data preparation

The preparation workflow is demonstrated in
`src/notebooks/data_preparation.ipynb` and uses functions from
`src/preprocessing/`.

The reported workflow used non-overlapping 256 x 256 pixel patches. Images of
516 x 516 pixels were resized to 512 x 512 pixels before patch extraction,
whereas 2048 x 2048 pixel images were divided directly without rescaling.
Images were read as grayscale. Training inputs were normalized column-wise
using the L2 norm implemented in `src/models/utils.py`.

The applied augmentation included:

- rotations by 0, 90, 180 or 270 degrees;
- independent horizontal and vertical flips;
- random brightness changes;
- random contrast changes.

Image and mask transformations are applied together for geometric operations.
Brightness and contrast changes are applied only to the images.

## Model training

Model training and comparison were performed in the notebooks under
`src/notebooks/`:

- `simple_unet.ipynb`: U-Net training workflow;
- `simple_unet_imports_folded.ipynb`: extended training and loss-function
  comparison;
- `test_models.ipynb`: comparison with alternative segmentation architectures.

Launch Jupyter from the repository root so that imports resolve consistently:

```bash
jupyter notebook
```

Open the required notebook and set the image and mask directories in its
configuration cell before running it.

## Segmentation-assisted crystal detection

`src/crystal_finder_inst_and_unet.py` implements the evaluated workflow:

1. load a TEM image;
2. divide it into 256 x 256 pixel patches;
3. predict the carbon-support mask with the trained U-Net;
4. apply the carbon mask to the thresholded image;
5. perform morphological filtering and connected-region labelling;
6. assign target positions to the detected regions;
7. optionally save masks, visualizations, coordinates and evaluation metrics.

The workflow scripts were developed and run in a Python environment with
dataset-specific paths specified in their configuration sections. To apply the
workflow to a new image set, update the input and output directories in the
`if __name__ == "__main__":` section of
`src/crystal_finder_inst_and_unet.py` and run the script from the `src`
directory.

The threshold-only reference workflow is provided in
`src/crystal_finder_from_instamatic.py`.

## Evaluation

The `src/evaluation/` directory contains scripts used for:

- carbon-mask intersection-over-union (IoU) and recall;
- crystal-level true-positive, false-positive and false-negative counting;
- precision and sensitivity calculation;
- sorting and visualizing results by IoU or recall.

The crystal-level evaluation uses the centroid-in-reference-mask rule described
in the associated manuscript: a detected region is counted as a true positive
when its selected centroid lies inside a manually annotated crystal region.

## Reproducibility notes

- Data paths are not stored in Git and must be supplied locally.
- The repository contains the trained Keras models used in the study.
- Complete per-image acquisition metadata were not retained in the exported
  PNG/TIFF training files.
- The inpainting workflow is experimental and was not the primary method
  evaluated in the manuscript.
- Physical pixel sizes were not normalized between microscope-detector
  configurations.

## Instamatic attribution

The baseline threshold-based crystal-detection workflow was adapted from
[Instamatic source code](https://github.com/instamatic-dev/instamatic).

The underlying SerialED workflow is described in:
- Smeets, S., Zou, X. & Wan, W. (2018).
  *Journal of Applied Crystallography*, **51**, 1262–1273.


## Citation

If you use this software, cite the associated article and the software record.
Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

If you use the accompanying images or masks, cite the dataset separately: 
https://doi.org/10.5281/zenodo.20269439


## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).
