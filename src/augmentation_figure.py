from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "milk10k" / "images"
METADATA_PATH = PROJECT_ROOT / "data" / "milk10k" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)


# Seven medically reasonable augmentations
AUGMENTATIONS = [
    ("Horizontal flip", transforms.RandomHorizontalFlip(p=1.0)),
    ("Vertical flip", transforms.RandomVerticalFlip(p=1.0)),
    ("Rotation +15°", transforms.RandomRotation((15, 15))),
    ("Rotation -15°", transforms.RandomRotation((-15, -15))),
    ("Brightness", transforms.ColorJitter(brightness=0.2)),
    ("Contrast", transforms.ColorJitter(contrast=0.2)),
    ("Slight colour", transforms.ColorJitter(
        saturation=0.1,
        hue=0.02
    )),
]


def choose_image(metadata, diagnosis, seed=42):
    """Choose one available image for a diagnosis_1 class."""

    subset = metadata[
        metadata["diagnosis_1"] == diagnosis
    ].copy()

    subset = subset[
        subset["isic_id"].apply(
            lambda x: (
                IMAGE_DIR / f"{x}.jpg"
            ).exists()
        )
    ]

    if subset.empty:
        raise FileNotFoundError(
            f"No available image found for class {diagnosis}"
        )

    random.seed(seed)
    row = subset.sample(
        n=1,
        random_state=seed
    ).iloc[0]

    return row


def make_figure():
    metadata = pd.read_csv(METADATA_PATH)

    # Include the rare Indeterminate class.
    classes = [
        "Benign",
        "Malignant",
        "Indeterminate",
    ]

    fig, axes = plt.subplots(
        len(classes),
        8,
        figsize=(16, 6)
    )

    for row_idx, diagnosis in enumerate(classes):

        row = choose_image(
            metadata,
            diagnosis,
            seed=42 + row_idx
        )

        image_path = (
            IMAGE_DIR / f"{row['isic_id']}.jpg"
        )

        image = Image.open(image_path).convert("RGB")

        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(
            f"{diagnosis}\nOriginal"
        )
        axes[row_idx, 0].axis("off")

        for col_idx, (name, augmentation) in enumerate(
            AUGMENTATIONS,
            start=1
        ):
            augmented = augmentation(image)

            axes[row_idx, col_idx].imshow(
                augmented
            )
            axes[row_idx, col_idx].set_title(name)
            axes[row_idx, col_idx].axis("off")

    fig.suptitle(
        "MILK10k B7: Training Augmentations",
        fontsize=14
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "b7_augmentation_examples.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved augmentation figure to:\n{output_path}"
    )


if __name__ == "__main__":
    make_figure()
