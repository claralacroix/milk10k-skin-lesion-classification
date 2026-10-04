# MILK10k Data-Quality Report

## Missing-value decisions

- **anatom_site_special**: 10274 missing — Leave NaN; not used as a model input because of very high missingness.
- **diagnosis_4**: 8958 missing — Leave NaN; exclude from model inputs because it is diagnostic information and may leak the label.
- **melanocytic**: 8088 missing — Leave NaN; not used as a model input because of high missingness and diagnostic relevance.
- **anatom_site_general**: 3912 missing — Leave NaN; do not use as a model input because missingness is informative.
- **diagnosis_3**: 158 missing — Leave NaN; column is not required as a model input.
- **age_approx**: 40 missing — Leave NaN; column is not required as a model input.

## Label and image consistency

- Metadata rows: 10480
- Unique lesions: 5240
- Every lesion has exactly 2 images: **Yes**
- Every lesion has one dermoscopic and one clinical image: **Yes**

## Image manipulation × diagnosis_1

| image_manipulation   |   Benign |   Indeterminate |   Malignant |
|:---------------------|---------:|----------------:|------------:|
| altered              |     4.48 |           17.48 |        2.19 |
| instrument only      |    95.52 |           82.52 |       97.81 |

## Image type × diagnosis_1

| image_type         |   Benign |   Indeterminate |   Malignant |
|:-------------------|---------:|----------------:|------------:|
| clinical: close-up |       50 |              50 |          50 |
| dermoscopic        |       50 |              50 |          50 |

## Shortcut conclusion

Image manipulation shows class-dependent differences and should not be used as a model input. Image type is balanced at 50% clinical and 50% dermoscopic within each diagnosis_1 class. Nevertheless, both images from the same lesion must remain in the same data split to prevent lesion-level leakage.

## Columns excluded from model inputs

- `diagnosis_2`
- `diagnosis_3`
- `diagnosis_4`
- `diagnosis_confirm_type`
- `concomitant_biopsy`
