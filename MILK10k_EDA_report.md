# MILK10k Exploratory Data Analysis

## 1. Clinical Task

MILK10k is a skin lesion image dataset containing dermoscopic images and associated metadata. The clinical task is to analyze the available image and metadata information to understand differences between lesion diagnosis groups and identify characteristics that may be useful for automated skin lesion classification.

## 2. Metadata Analysis

The dataset contains 10,480 images and 17 metadata columns. The metadata include demographic information, anatomical site, diagnostic information, image characteristics, and lesion identifiers.

Several metadata fields contain missing values. The largest amounts of missing data were found in `anatom_site_special` (98.03%), `diagnosis_4` (85.48%), `melanocytic` (77.18%), and `anatom_site_general` (37.33%). In contrast, fields such as sex, diagnosis_1, diagnosis_2, diagnosis_confirm_type, image_manipulation, image_type, and lesion_id had no meaningful missingness.

The target variable used for the analysis was `diagnosis_1`, grouped into Benign, Indeterminate, and Malignant. Chi-square tests showed statistically significant associations between the target and several categorical variables, including sex, anatomical site, concomitant biopsy, diagnosis confirmation type, and image manipulation (p < 0.001). `image_type` showed no association with the target.

Among the categorical variables, `anatom_site_special` had the highest Cramér's V (0.573), followed by `concomitant_biopsy` and `diagnosis_confirm_type` (0.311 each). However, the very high missingness of `anatom_site_special` limits its practical usefulness. The strong association of biopsy and diagnosis-confirmation variables may also reflect how diagnoses were established rather than independent visual characteristics, so these variables should be considered potential sources of bias or leakage.

For the numeric metadata, `age_approx` differed significantly between diagnosis groups (ANOVA, p < 0.001). The distribution indicated higher ages among malignant cases compared with benign cases.

Overall, anatomical information, age, and some diagnostic-process variables showed associations with the target, while `image_type` provided little information. Metadata associated with the diagnostic process should be treated carefully when designing a predictive model because it may not be available or appropriate at prediction time.

![Age distribution by diagnosis](figures/age_by_diagnosis.png)

*Figure 2. Distribution of approximate age across the diagnosis groups.*

## 3. Color and Histogram Analysis

To investigate whether simple image intensity and color characteristics differ between diagnosis groups, 20 images were sampled from each of the Benign, Indeterminate, and Malignant classes. Average grayscale histograms and RGB channel histograms were calculated for each group.

The grayscale histograms showed substantial overlap between the three diagnosis groups. Although the distributions had some differences in their shapes and peaks, there was no clear separation between classes. This suggests that grayscale intensity alone is unlikely to be sufficient for distinguishing the diagnosis groups.

The RGB histograms and per-image RGB mean and standard deviation showed some differences between classes, but there was also substantial overlap. Therefore, color and brightness may provide useful information to a model, but they are unlikely to be sufficient on their own.

These differences may be influenced by lesion characteristics, but also by lighting conditions, imaging equipment, image acquisition settings, skin tone, or anatomical-site differences. Consequently, a classification model should use additional visual information such as lesion shape, texture, and structure.

![Average grayscale histogram](figures/grayscale_histograms.png)

*Figure 3. Average grayscale pixel-intensity distributions for the three diagnosis groups.*

![Average RGB histograms](figures/rgb_histograms.png)

*Figure 4. Average red, green, and blue channel pixel-intensity distributions for the three diagnosis groups.*

![Diagnosis distribution by sex](figures/sex_by_diagnosis.png)

*Figure 5. Diagnosis distribution by sex.*

![Average RGB values by diagnosis](figures/rgb_mean_by_diagnosis.png)

*Figure 6. Average red, green, and blue pixel values by diagnosis group.*

## 4. Image Processing, Data Loading, and Visualization Design

For image preprocessing, a reusable function was implemented to resize images to 224 × 224 pixels, convert them to RGB or grayscale when requested, and normalize pixel values. A batch-processing function was also created to process multiple images while recording files that could not be processed. This makes the preprocessing pipeline reusable and robust to individual file errors.

The data loader processes images in batches rather than loading the entire dataset into memory. Batch size, image size, color mode, and normalization can be configured. A filtering function was included to identify locally available image files. Finally, visualization functions were implemented to display labelled image grids and summarize batch pixel distributions, supporting quick inspection and debugging of the image-processing pipeline.

## 5. Class Distribution

The dataset is imbalanced across the three diagnosis groups. There are 7,268 Malignant images, 2,966 Benign images, and 246 Indeterminate images. Therefore, Malignant images make up the largest group, while Indeterminate images are much less represented.

This imbalance should be considered when developing a classification model because a model trained directly on the available distribution may be more influenced by the majority class. Class-specific evaluation metrics and appropriate sampling or weighting strategies may therefore be important during model development.

![MILK10k class distribution](figures/class_distribution.png)

*Figure 1. Distribution of images across the three diagnosis groups.*

## 6. Conclusion

The exploratory analysis identified meaningful differences in both metadata and image characteristics across the diagnosis groups. Age and several categorical metadata variables were associated with the target, while some fields had substantial missingness or potential diagnostic-process bias. The color and intensity analysis showed some differences between classes but substantial overlap, indicating that these features alone are unlikely to provide complete separation. The reusable preprocessing, batch-loading, and visualization functions provide a foundation for subsequent computer vision modelling on the MILK10k dataset.

## 7. Reusable Code

The reusable computer vision functions are implemented in `src/milk10k_utils.py`. The module contains functions for single-image preprocessing, batch processing, data loading, filtering unavailable images, and displaying labelled image grids. The pipeline was tested on real MILK10k images, including a deliberate missing-file test to verify that processing failures are reported without stopping the entire batch.
