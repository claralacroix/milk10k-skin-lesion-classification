# MILK10k Skin Lesion Classification — Milestone 1 Report

## 1. Task and Dataset

This project develops a computer-vision pipeline for skin-lesion classification using the MILK10k dataset. The dataset contains 10,480 images representing 5,240 lesions, with two images per lesion: one dermoscopic image and one clinical close-up. The primary task uses three diagnosis_1 classes: Benign, Malignant, and Indeterminate.

The dataset is strongly imbalanced. Malignant lesions account for approximately 69.4% of lesions, Benign for 28.3%, and Indeterminate for 2.3%. The original 11-class diagnosis labels are retained for analysis and future modelling, but the main classification target is diagnosis_1. Indeterminate is retained as a separate class rather than being removed or merged.

## 2. Data Quality and Integrity

All 10,480 expected image files were present and readable. Image dimensions were consistent at 600 × 450 pixels. Every lesion has exactly two images, consisting of one dermoscopic and one clinical close-up image.

On the TRAIN split, the class counts are 1,924 Benign, 4,718 Malignant, and 168 Indeterminate, giving a maximum-to-minimum class imbalance ratio of 28.1:1 (4,718/168).
Several metadata fields contain substantial missingness. Highly incomplete or diagnostically informative fields are not used as model inputs. In particular, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_confirm_type, and concomitant_biopsy are excluded because they can introduce diagnostic leakage. Anatomical metadata with informative missingness is also not used as an unchecked model shortcut.

## 3. Splitting and Leakage Prevention

The data are split at lesion level so that the two images from a lesion can never appear in different partitions. A fixed random seed of 42 is used. The resulting split contains approximately 65% training lesions, 15% validation lesions, and 20% test lesions.

The 11-class distribution was checked across the splits, with the maximum class-proportion deviation kept below 0.2 percentage points. Reproducibility was also verified using the fixed seed.

## 4. Preprocessing and Augmentation

The selected model input resolution is 224 × 224 pixels. Since all original images are 600 × 450, 100% of images are larger than the selected model resolution.

Both dermoscopic and clinical views are retained as independent image samples, while their shared lesion identity is preserved during splitting and evaluation. Training uses horizontal and vertical flips, small rotations, and mild brightness, contrast, saturation, and hue changes. Evaluation preprocessing is deterministic and only resizes, converts to tensors, and normalizes images.

## 5. Class Imbalance

Class imbalance is handled during training using both inverse-frequency class weights and a WeightedRandomSampler. The resulting training class weights are approximately 1.18 for Benign, 0.48 for Malignant, and 13.51 for Indeterminate.

The very large Indeterminate weight reflects its small representation in the dataset. This approach is intended to reduce the tendency of a classifier to favour the majority Malignant class.

## 6. B8 Pipeline Validation

The Dataset implementation reads the predefined train, validation, and test split files and the label mapping. It loads images and returns an image tensor, integer label, and ISIC image identifier.

A batch-size of 16 was validated successfully, producing tensors of shape (16, 3, 224, 224) with float32 dtype. A 20-batch loading-time check required approximately 11.93 seconds. Training-batch class distribution and 16 transformed training examples were also visualized.

## 7. Main Decisions and Limitations

The main safeguards are lesion-level splitting, exclusion of diagnostically leaky metadata, deterministic evaluation preprocessing, and explicit handling of severe class imbalance. The dataset's clinical composition should nevertheless be interpreted cautiously because diagnostic prevalence may not represent routine clinical populations. In particular, biopsy-confirmed cases can be enriched for malignant disease, so performance estimates should not automatically be interpreted as real-world screening performance.

The next stage is model training and evaluation using the established lesion-level splits and preprocessing pipeline.
