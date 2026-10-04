from pathlib import Path
from PIL import Image
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "milk10k"
IMAGE_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "metadata.csv"


def check_dataset_integrity():
    """Check all dataset images and save image dimension summary."""

    metadata = pd.read_csv(METADATA_FILE)

    print(f"Metadata rows: {len(metadata)}")

    missing_files = []
    unreadable_files = []
    widths = []
    heights = []

    for filename in metadata["isic_id"]:
        image_path = IMAGE_DIR / f"{filename}.jpg"

        if not image_path.exists():
            missing_files.append(filename)
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                widths.append(img.width)
                heights.append(img.height)

        except Exception:
            unreadable_files.append(filename)

    print(f"Missing image files: {len(missing_files)}")
    print(f"Unreadable image files: {len(unreadable_files)}")

    if missing_files:
        print("Missing files:")
        print(missing_files)

    if unreadable_files:
        print("Unreadable files:")
        print(unreadable_files)

    summary = pd.DataFrame(
        {
            "dimension": ["width", "height"],
            "min": [
                min(widths),
                min(heights),
            ],
            "median": [
                pd.Series(widths).median(),
                pd.Series(heights).median(),
            ],
            "max": [
                max(widths),
                max(heights),
            ],
        }
    )

    output_dir = PROJECT_ROOT / "data"
    output_file = output_dir / "image_size_summary.csv"

    summary.to_csv(output_file, index=False)

    print("\nImage size summary:")
    print(summary)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    check_dataset_integrity()
    