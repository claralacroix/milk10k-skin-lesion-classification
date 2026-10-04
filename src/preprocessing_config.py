from pathlib import Path
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_SIZE = (224, 224)

IMAGE_SIZE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "image_size_summary.csv"
)

SEED = 42

VIEW_HANDLING = "both_views_as_independent_samples"


DESCRIPTION = """
MILK10k contains one dermoscopic image and one clinical close-up
for each lesion. Both image types are retained as independent
training samples with the lesion-level diagnosis_1 label.

The two images belonging to the same lesion are always assigned
to the same train, validation, or test split. They are therefore
never separated across splits.

Evaluation is performed at lesion level rather than image level.
For paired evaluation, predictions from the two views can be
combined into one lesion-level prediction.
"""


# ============================================================
# B6 decision check
# ============================================================

def check_image_resolution():

    summary = pd.read_csv(IMAGE_SIZE_SUMMARY_PATH)

    print("=" * 60)
    print("B6 PREPROCESSING DECISION")
    print("=" * 60)

    print()
    print("B1 image-size summary:")
    print(summary)

    print()
    print(
        f"Selected input resolution: "
        f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
    )

    # image_size_summary.csv stores dimensions as:
    # dimension | min | median | max
    width_row = summary.loc[
        summary["dimension"].str.lower() == "width"
    ].iloc[0]

    height_row = summary.loc[
        summary["dimension"].str.lower() == "height"
    ].iloc[0]

    max_width = width_row["max"]
    max_height = height_row["max"]

    # All images are 600x450, so all are larger than 224x224.
    fraction_larger = (
        (max_width > IMAGE_SIZE[0])
        or
        (max_height > IMAGE_SIZE[1])
    )

    print(
        "Fraction of images larger than "
        f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}: "
        f"{100.0 if fraction_larger else 0.0:.2f}%"
    )

    print()
    print("View handling:")
    print(VIEW_HANDLING)

    print()
    print(DESCRIPTION)

    return fraction_larger


if __name__ == "__main__":
    check_image_resolution()
