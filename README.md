# IHCs Colocalizer V1.0

IHCs Colocalizer is a Python-based workflow for quantitative analysis of serial immunohistochemistry (IHC) images from the same biological sample. It supports region-of-interest (ROI) registration across adjacent tissue sections, H-DAB color deconvolution and thresholding, single-marker DAB quantification, and dual-marker spatial colocalization analysis.

The workflow is organized into four Python modules:

1. **ROI lasso, registration, and alignment**
2. **H-DAB color deconvolution and thresholding**
3. **Single-marker DAB quantification**
4. **Dual-marker colocalization analysis**

## Author

Zonghan Gan

## Workflow Overview

### Part 1 — ROI Lasso Interface and Registration

**File:** `Part 1 ROI lasso interface and registration.py`

This module processes serial IHC images from the same sample and establishes spatially corresponding ROIs across slides.

Main functions include:

- Interactive ROI selection on a reference image
- Optional loading of an existing ImageJ `.roi` file
- ROI projection to serial sections using VALIS registration
- Export of registered-but-not-aligned ROI images as `roiRaw`
- Homography-based alignment of projected ROIs and export as `roiCrop`
- Preservation of ROI geometry using RGBA transparency
- Recording of image scale information in `scale.csv`
- Export of ImageJ ROI files and alignment control points
- Batch merging of scale information across samples

The registration stage uses VALIS to project ROI coordinates between serial sections, followed by a four-point homography for local ROI alignment.

Typical outputs include:

```text
output_folder/
├── roi/
│   └── *.roi
├── roiRaw/
│   └── *-roiRaw.png
├── roi-crop/
│   └── *-roiCrop.png
└── scale.csv
```

`roiRaw` images retain the registered ROI without the final local homography alignment and are intended for value-based single-marker quantification.

`roiCrop` images contain the registered and locally aligned ROI and are intended for spatial colocalization analysis.

### Part 2 — H-DAB Color Deconvolution and Thresholding

**File:** `Part 2 colour deconvolution and thresholding.py`

This module performs H-DAB color deconvolution on RGBA ROI images using Fiji/ImageJ and generates threshold-based DAB-positive masks.

Main functions include:

- H-DAB color deconvolution through Fiji/ImageJ
- Export of nucleus and DAB channels
- Restoration of the original alpha channel after deconvolution
- Batch processing of marker-organized folder structures
- Generation of binary DAB-positive masks
- Application of masks while preserving RGBA transparency
- Mirroring of input folder structures
- Processing logs and summary statistics

The thresholding rule defines pixels within a selected DAB intensity range as positive.

Typical outputs include:

```text
output_root/
├── Nucleus/
│   └── <marker>/
├── DAB/
│   └── <marker>/
└── threshold/mask outputs
```

### Part 3 — Single-Marker DAB Quantification

**File:** `Part 3 Single marker analysis.py`

This module quantifies DAB-positive staining within individual IHC ROIs.

Thresholded RGBA images define positive regions using non-transparent alpha pixels, while the corresponding DAB ROI image defines the full analysis area.

Calculated measurements include:

- Positive area
- ROI area
- Percentage positivity (`POS_percent`)
- Minimum, mean, maximum, and integrated grayscale intensity
- Positive-island count
- Equivalent island diameter
- Island circularity
- Optical-density-based measurements:
  - `OD_from_meanI`
  - `AOD`
  - `IOD`
- `ODmean × POS_percent`
- `AOD × POS_percent`

The module also supports recursive image discovery, automatic threshold/DAB pairing, single-pair inspection, batch CSV export, and summary tables.

For quantitative single-marker analysis, the workflow is intended to use **registered-but-not-aligned `roiRaw` images**.

### Part 4 — Dual-Marker Colocalization Analysis

**File:** `Part 4 Dual marker colocalization.py`

This module quantifies spatial colocalization between two IHC markers using registered and aligned threshold-masked RGBA images.

Positive regions are identified from non-transparent alpha pixels within the corresponding ROI.

Calculated measurements include:

- Marker-positive areas
- Overlap area
- Directional colocalization ratios
- Jaccard index
- Dice coefficient
- Phi coefficient
- Global optical-density Pearson correlation
- Continuous weighted Jaccard index
- Tile-based Jaccard statistics
- Tile-based optical-density correlations
- Correlation between marker-positive area fractions across tiles

The module also supports OD-map reconstruction, folder-based batch processing, automatic matching of marker image triplets, optional common-area cropping for size mismatches, single-triplet inspection, and CSV export.

Dual-marker analysis is intended to use **registered and aligned `roiCrop` images**.

## Recommended Analysis Sequence

```text
Serial IHC images
        │
        ▼
Part 1
ROI selection → registration → ROI projection/alignment
        │
        ├──────────────► roiRaw ─────────► Part 2 ─────────► Part 3
        │                                  H-DAB              Single-marker
        │                                  processing         quantification
        │
        └──────────────► roiCrop ─────────► Part 2 ─────────► Part 4
                                           H-DAB              Dual-marker
                                           processing         colocalization
```

## Dependencies

The scripts use the following Python packages and external tools:

- Python 3.7 or later
- NumPy
- pandas
- OpenCV (`opencv-python`)
- Matplotlib
- Pillow
- SciPy
- scikit-image
- tifffile
- pyvips
- roifile
- VALIS
- pyimagej
- scyjava
- Fiji/ImageJ

Some modules require a working Java environment for Fiji/ImageJ and VALIS-related processing.

## Input Images

The registration module supports common pathology image formats including:

```text
.png
.tif
.tiff
.jpg
.jpeg
.bmp
```

The downstream analysis modules primarily operate on PNG images, especially RGBA PNG files in which alpha transparency is used to preserve ROI geometry and identify masked regions.

## File Naming

Several batch-processing functions depend on structured filenames.

A general naming pattern used by the workflow is:

```text
<slide_index>_<marker>_<sample_name>_<roiRaw_or_roiCrop>_...
```

Thresholded images may contain a suffix such as:

```text
_threshold220
```

Keep naming conventions consistent across corresponding marker and DAB images when using batch-processing functions.

## Usage

Each Python file contains callable functions together with example paths or execution blocks that can be adapted to a local directory structure.

Before running the workflow:

1. Install the required Python dependencies.
2. Install and configure Fiji/ImageJ where required.
3. Check input and output directory paths in each script.
4. Confirm the image scale (`um_per_pixel`) for the original pathology images.
5. Select an appropriate reference image for serial-section registration.
6. Adjust DAB threshold values according to staining and image-acquisition conditions.

Because the current scripts contain example local filesystem paths, replace these paths with paths appropriate to your own environment before execution.

## Notes

- `roiRaw` and `roiCrop` serve different analytical purposes and should not be treated interchangeably.
- RGBA transparency is used throughout the workflow to preserve ROI boundaries and distinguish excluded image regions.
- DAB threshold values should be selected consistently when samples are intended for direct comparison.
- Accurate serial-section registration is important for meaningful dual-marker colocalization measurements.
- Image preprocessing and acquisition conditions should be kept consistent across experimental groups whenever quantitative comparisons are performed.
