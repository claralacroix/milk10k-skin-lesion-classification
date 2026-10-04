"""
B7 - Image preprocessing and augmentation pipeline.

All transforms use the B6 input resolution of 224x224.
"""

import random
import numpy as np
import torch

from PIL import Image
from torchvision import transforms


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# Training transform
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Geometric augmentation that preserves lesion identity.
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),

    # Mild colour/brightness variation.
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.02,
    ),

    transforms.ToTensor(),

    # ImageNet normalization for compatibility with common
    # pretrained CNN backbones.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Evaluation transform
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Determinism test
# ============================================================

def test_eval_transform_deterministic():
    """
    Apply eval_transform twice to the same image and verify that
    the resulting tensors are identical.
    """

    # Create a fixed synthetic RGB image.
    rng = np.random.default_rng(42)
    array = rng.integers(
        0,
        256,
        size=(450, 600, 3),
        dtype=np.uint8,
    )

    image = Image.fromarray(array, mode="RGB")

    # Reset seeds before each transformation.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    tensor_1 = eval_transform(image)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    tensor_2 = eval_transform(image)

    assert torch.equal(
        tensor_1,
        tensor_2,
    ), "eval_transform is not deterministic."

    print("B7 deterministic evaluation test: PASSED")
    print(f"Output shape: {tuple(tensor_1.shape)}")
    print(f"Output dtype: {tensor_1.dtype}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    test_eval_transform_deterministic()
