# MILK10k Skin Lesion Classification

## 1. Project overview

This project develops a computer-vision pipeline for classifying skin lesions
in the MILK10k dataset.

### Task

The primary prediction target is:

- Benign
- Malignant
- Indeterminate

The model uses lesion images as input. The two available image views are:

- dermoscopic images
- clinical images

The dataset contains 10,480 images corresponding to 5,240 lesions, with two
images per lesion.

The goal of Milestone 1 is to build a leak-free, reproducible data pipeline
covering data integrity, label strategy, quality checks, lesion-level splits,
preprocessing, augmentation, and DataLoaders.

## 2. Dataset

Dataset: MILK10k.

Number of images: 10,480.

Number of lesions: 5,240.

Each lesion has two images:

- one dermoscopic image
- one clinical image

The dataset source, license, and attribution follow the dataset
documentation supplied with the project.

Raw images are used locally but are not committed to GitHub.

## 3. Setup

Python version:

Python 3.13

Install dependencies:

```bash
pip install -r requirements.txt


## Dataset source, license, and attribution

MILK10k is used as the skin-lesion image dataset for this project. The project uses the supplied dataset metadata, ground-truth files, and 10,480 images (5,240 lesions, two images per lesion). Dataset attribution and licensing information are retained in the supplied metadata (`attribution` and `copyright_license`); raw images are not committed to the repository.

## Milestone goal

The milestone establishes an auditable, leakage-aware data pipeline: verified image integrity, label strategy, label-centered EDA, grouped stratified train/validation/test splits, preprocessing and augmentation, reproducible Dataset/DataLoaders, and documented class imbalance and limitations.

## Why the repository is organized this way

Reusable Python modules live in `src/` so data validation, EDA, splitting, preprocessing, transforms, and loading can be rerun independently. Generated tables and machine-readable configuration files live under `data/`, figures under `figures/`, and the milestone report under `report/`. This separates source code from generated outputs and keeps raw images outside version control while making every milestone artifact easy to locate and reproduce.

## Complete file/folder map

- `README.md` — project overview, setup, reproducibility commands, decisions, and repository organization.
- `SUBMISSION.md` — direct links to the files submitted for grading.
- `requirements.txt` — pinned Python dependencies for the project pipeline.
- `.gitignore` — prevents raw images and generated/cache files from being committed.
- `src/` — reusable project source code.
- `src/milk10k_utils.py` — shared dataset/path and split utilities.
- `src/eda.py` — label-centered exploratory analysis and figures.
- `src/data_quality.py` — metadata/image integrity and quality checks.
- `src/quality_check.py` — additional quality-control helpers.
- `src/split_data.py` — lesion-grouped stratified train/validation/test splitting.
- `src/preprocessing_config.py` — preprocessing resolution and view-handling configuration.
- `src/transforms.py` — training and deterministic evaluation transforms.
- `src/augmentation_figure.py` — B7 augmentation visualization.
- `src/dataset.py` — Dataset, DataLoaders, class weights, sampler, and weighted loss.
- `data/splits/` — train/validation/test split CSV files.
- `data/label_map.json` — diagnosis label-to-integer mapping.
- `data/class_weights.json` — training class weights.
- `data/normalization_stats.json` — resize and normalization statistics.
- `data/image_size_summary.csv` — B1 image dimension summary.
- `data/report/` — generated quality and augmentation tables.
- `figures/` — generated EDA, augmentation, and DataLoader figures.
- `report/` — milestone report.
- `homework_part_a.ipynb` — executed Part A notebook.
- `homework_part_a.pdf` — Part A PDF submission.

## Reproducibility

The split seed is `42`. The train/validation/test split is approximately 65%/15%/20% of lesions, with both images from each lesion kept in the same split. The split was generated on 2026-10-04. Raw images must be supplied locally under `data/milk10k/images/` and are excluded from version control.
