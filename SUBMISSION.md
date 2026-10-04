Name: Clara Lacroix
Repository: https://github.com/claralacroix/milk10k-skin-lesion-classification

# Submission

## Part A

- [Part A notebook](homework_part_a.ipynb)
- [Part A PDF](homework_part_a.pdf)

## Part B — Documentation

- [README](README.md)
- [Requirements](requirements.txt)
- [Milestone 1 report](report/MILK10k_Milestone1_report.md)

## Part B — Source code

- [Shared utilities](src/milk10k_utils.py)
- [EDA](src/eda.py)
- [Data quality](src/data_quality.py)
- [Quality checks](src/quality_check.py)
- [Split generation](src/split_data.py)
- [Preprocessing configuration](src/preprocessing_config.py)
- [Transforms](src/transforms.py)
- [Augmentation figure](src/augmentation_figure.py)
- [Dataset and DataLoaders](src/dataset.py)

## Part B — Data/configuration outputs

- [Train split](data/splits/train.csv)
- [Validation split](data/splits/val.csv)
- [Test split](data/splits/test.csv)
- [Split summary](data/splits/split_summary.csv)
- [Label mapping](data/label_map.json)
- [Class weights](data/class_weights.json)
- [Normalization statistics](data/normalization_stats.json)
- [Image size summary](data/image_size_summary.csv)
- [Data quality report](report/data_quality_report.md)
- [Augmentation table](data/report/b7_augmentation_table.csv)

## Part B — Figures

- [Diagnosis 1 distribution](figures/diagnosis_1_distribution.png)
- [11-class distribution](figures/11_class_distribution_log.png)
- [11-class gallery](figures/11_class_gallery.png)
- [Diagnosis mapping](figures/dx_to_diagnosis1_mapping.csv)
- [Session 2 findings](figures/session2_findings_pipeline.csv)
- [B7 augmentation examples](figures/b7_augmentation_examples.png)
- [B8 train 20-batch histogram](figures/b8_train_20_batch_histogram.png)
- [B8 transformed images](figures/b8_16_transformed_images.png)

Raw dataset images are intentionally not included in the repository.
