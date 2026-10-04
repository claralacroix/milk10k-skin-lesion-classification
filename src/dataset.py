"""
B8 - MILK10k Dataset and DataLoaders.

Requirements covered:
- Reads train/val/test split CSVs
- Reads label_map.json
- Loads images
- Applies train-only augmentation
- Returns (image_tensor, label, isic_id)
- Reproducible random seeds
- Class weights
- WeightedRandomSampler
- Saves class_weights.json
- Missing-file errors
- Batch sanity checks
- Train-batch loading-time check
- 20-batch class histogram
- 16-image transformed-image visualization
"""

from pathlib import Path
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transforms import train_transform, eval_transform


# ---------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "milk10k" / "images"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
LABEL_MAP_PATH = PROJECT_ROOT / "data" / "label_map.json"
OUTPUT_DIR = PROJECT_ROOT / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
NUM_WORKERS = 0
SEED = 42


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Random seed set to {seed}")


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class MILK10kDataset(Dataset):

    def __init__(self, csv_path, label_map_path, image_dir,
                 transform=None):

        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {self.csv_path}"
            )

        if not Path(label_map_path).exists():
            raise FileNotFoundError(
                f"Label map not found: {label_map_path}"
            )

        self.data = pd.read_csv(self.csv_path)

        required_columns = {"isic_id", "diagnosis_1"}

        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"{self.csv_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        with open(label_map_path, "r") as f:
            label_map = json.load(f)

        self.label_map = label_map["diagnosis_1"]

        self.data["label"] = self.data["diagnosis_1"].map(
            self.label_map
        )

        if self.data["label"].isna().any():
            bad_labels = (
                self.data.loc[
                    self.data["label"].isna(),
                    "diagnosis_1"
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Unknown diagnosis_1 labels in {self.csv_path}: "
                f"{bad_labels}"
            )

        self.data["label"] = self.data["label"].astype(int)

        # Check every required image before training.
        missing_images = []

        for isic_id in self.data["isic_id"]:
            image_path = self.image_dir / f"{isic_id}.jpg"

            if not image_path.exists():
                missing_images.append(str(image_path))

        if missing_images:
            preview = missing_images[:10]

            raise FileNotFoundError(
                f"{len(missing_images)} image files are missing "
                f"from {self.csv_path}.\n"
                f"Examples:\n{preview}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        isic_id = row["isic_id"]
        label = int(row["label"])

        image_path = self.image_dir / f"{isic_id}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file missing: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, isic_id


# ---------------------------------------------------------
# Load label map
# ---------------------------------------------------------

def load_label_map():

    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)

    return label_map["diagnosis_1"]


# ---------------------------------------------------------
# Class weights
# ---------------------------------------------------------

def calculate_class_weights(dataset):

    labels = dataset.data["label"].to_numpy()

    class_ids = sorted(dataset.label_map.values())

    counts = np.bincount(
        labels,
        minlength=len(class_ids)
    )

    # Inverse-frequency weighting.
    weights = len(labels) / (
        len(class_ids) * np.maximum(counts, 1)
    )

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32
    )

    class_names = {
        value: key
        for key, value in dataset.label_map.items()
    }

    output = {}

    for class_id in class_ids:
        output[class_names[class_id]] = {
            "class_id": int(class_id),
            "count": int(counts[class_id]),
            "weight": float(weights[class_id]),
        }

    output_path = PROJECT_ROOT / "data" / "class_weights.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nClass weights:")
    print(json.dumps(output, indent=2))

    print(f"\nSaved class weights to:")
    print(output_path)

    return weights_tensor


# ---------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------


def make_weighted_loss(class_weights):
    """Create the class-weighted cross-entropy loss required for training."""
    return torch.nn.CrossEntropyLoss(weight=class_weights)

def make_weighted_sampler(dataset):

    labels = dataset.data["label"].to_numpy()

    class_counts = np.bincount(
        labels,
        minlength=len(dataset.label_map)
    )

    class_weights = 1.0 / np.maximum(class_counts, 1)

    sample_weights = class_weights[labels]

    sample_weights = torch.as_tensor(
        sample_weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


# ---------------------------------------------------------
# Create datasets and loaders
# ---------------------------------------------------------

def create_datasets_and_loaders():

    train_dataset = MILK10kDataset(
        SPLIT_DIR / "train.csv",
        LABEL_MAP_PATH,
        IMAGE_DIR,
        transform=train_transform,
    )

    val_dataset = MILK10kDataset(
        SPLIT_DIR / "val.csv",
        LABEL_MAP_PATH,
        IMAGE_DIR,
        transform=eval_transform,
    )

    test_dataset = MILK10kDataset(
        SPLIT_DIR / "test.csv",
        LABEL_MAP_PATH,
        IMAGE_DIR,
        transform=eval_transform,
    )

    train_sampler = make_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


# ---------------------------------------------------------
# Batch sanity check
# ---------------------------------------------------------

def batch_sanity_check(loader):

    images, labels, isic_ids = next(iter(loader))

    print("\nB8 BATCH SANITY CHECK")
    print("---------------------")
    print(f"Image shape: {tuple(images.shape)}")
    print(f"Image dtype: {images.dtype}")
    print(f"Image min: {images.min().item():.4f}")
    print(f"Image max: {images.max().item():.4f}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Number of IDs: {len(isic_ids)}")

    assert images.ndim == 4
    assert images.shape[1:] == (3, 224, 224)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert len(isic_ids) == images.shape[0]

    print("Batch sanity checks: PASSED")


# ---------------------------------------------------------
# Loading-time check
# ---------------------------------------------------------

def loading_time_check(loader):

    start = time.perf_counter()

    for _ in loader:
        pass

    elapsed = time.perf_counter() - start

    print("\nTRAIN LOADING TIME")
    print("------------------")
    print(f"20-batch/epoch loading time: {elapsed:.2f} seconds")

    return elapsed


# ---------------------------------------------------------
# 20-batch class histogram
# ---------------------------------------------------------

def plot_20_batch_histogram(loader):

    labels_seen = []

    for batch_index, (_, labels, _) in enumerate(loader):

        labels_seen.extend(labels.tolist())

        if batch_index == 19:
            break

    counts = np.bincount(
        labels_seen,
        minlength=3
    )

    label_names = ["Benign", "Malignant", "Indeterminate"]

    plt.figure(figsize=(7, 5))

    plt.bar(label_names, counts)

    plt.xlabel("Diagnosis")
    plt.ylabel("Number of samples")
    plt.title("B8: Class Distribution Across First 20 Training Batches")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "b8_train_20_batch_histogram.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved 20-batch histogram to:")
    print(output_path)


# ---------------------------------------------------------
# Visualize 16 transformed training images
# ---------------------------------------------------------

def unnormalize(images):

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    return images * std + mean


def visualize_16_images(loader, dataset):

    images, labels, isic_ids = next(iter(loader))

    images = images[:16]
    labels = labels[:16]

    images = unnormalize(images)
    images = images.clamp(0, 1)

    label_names = {
        value: key
        for key, value in dataset.label_map.items()
    }

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(10, 10)
    )

    for index, ax in enumerate(axes.flat):

        image = images[index].permute(1, 2, 0).numpy()

        ax.imshow(image)

        label_name = label_names[int(labels[index])]

        ax.set_title(
            f"{label_name}\n{isic_ids[index]}"
        )

        ax.axis("off")

    fig.suptitle(
        "B8: 16 Transformed Training Images",
        fontsize=14
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "b8_16_transformed_images.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nSaved 16-image visualization to:")
    print(output_path)


# ---------------------------------------------------------
# Main B8 test
# ---------------------------------------------------------

def main():

    set_seed(SEED)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = create_datasets_and_loaders()

    print("\nB8 DATASETS")
    print("-----------")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    # Save class weights.
    class_weights = calculate_class_weights(train_dataset)

    # Batch check.
    batch_sanity_check(train_loader)

    # Loading time.
    loading_time_check(train_loader)

    # Histogram.
    plot_20_batch_histogram(train_loader)

    # 16 transformed images.
    visualize_16_images(
        train_loader,
        train_dataset
    )

    print("\nB8 COMPLETE")


if __name__ == "__main__":
    main()

    
# B8 sanity check: instantiate the required class-weighted loss.
criterion = make_weighted_loss(class_weights)
print("Weighted CrossEntropyLoss:", criterion)
