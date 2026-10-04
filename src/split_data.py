from pathlib import Path
from datetime import date

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# Configuration
# ============================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = PROJECT_ROOT / "data" / "milk10k" / "metadata.csv"
LESION_TABLE_PATH = PROJECT_ROOT / "notebooks" / "lesion_table.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

VAL_SIZE = 0.15
TEST_SIZE = 0.20


# ============================================================
# Split function
# ============================================================

def split_lesions(
    lesions: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int
):
    """
    Split lesions into train/validation/test.

    Splitting is performed at lesion level using
    StratifiedGroupKFold and the 11-class diagnosis label.

    Target:
        train ≈ 65%
        validation ≈ 15%
        test ≈ 20%
    """

    if val_size + test_size >= 1:
        raise ValueError(
            "val_size + test_size must be less than 1."
        )

    required_columns = {"lesion_id", "dx"}

    if not required_columns.issubset(lesions.columns):
        raise ValueError(
            f"lesions must contain columns: {required_columns}"
        )

    lesions = lesions.reset_index(drop=True)

    # --------------------------------------------------------
    # First split: 20% test / 80% train+validation
    # --------------------------------------------------------

    test_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    train_val_idx, test_idx = next(
        test_splitter.split(
            lesions[["lesion_id"]],
            lesions["dx"],
            lesions["lesion_id"]
        )
    )

    train_val = lesions.iloc[train_val_idx].reset_index(
        drop=True
    )

    test = lesions.iloc[test_idx].reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Second split:
    #
    # train_val = approximately 80% of the full dataset.
    #
    # 3/16 of train_val ≈ 15% of the full dataset.
    #
    # Therefore:
    # train ≈ 65%
    # val   ≈ 15%
    # test  = 20%
    # --------------------------------------------------------

    val_splitter = StratifiedGroupKFold(
        n_splits=16,
        shuffle=True,
        random_state=seed
    )

    folds = list(
        val_splitter.split(
            train_val[["lesion_id"]],
            train_val["dx"],
            train_val["lesion_id"]
        )
    )

    # Combine three folds for validation.
    val_idx = sorted(
        list(folds[0][1])
        + list(folds[1][1])
        + list(folds[2][1])
    )

    train_idx = sorted(
        set(range(len(train_val))) - set(val_idx)
    )

    train = train_val.iloc[train_idx].reset_index(
        drop=True
    )

    val = train_val.iloc[val_idx].reset_index(
        drop=True
    )

    return (
        train["lesion_id"].tolist(),
        val["lesion_id"].tolist(),
        test["lesion_id"].tolist()
    )


# ============================================================
# Checks
# ============================================================

def check_no_overlap(
    train_ids,
    val_ids,
    test_ids
):
    """Check that no lesion appears in multiple splits."""

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert train_set.isdisjoint(val_set), (
        "Lesion overlap detected between train and validation."
    )

    assert train_set.isdisjoint(test_set), (
        "Lesion overlap detected between train and test."
    )

    assert val_set.isdisjoint(test_set), (
        "Lesion overlap detected between validation and test."
    )

    print("Lesion overlap check: PASSED")


def check_two_images_per_lesion(
    metadata,
    lesion_ids,
    split_name
):
    """Check that every lesion has exactly two images."""

    subset = metadata[
        metadata["lesion_id"].isin(lesion_ids)
    ]

    counts = subset.groupby("lesion_id").size()

    assert (counts == 2).all(), (
        f"{split_name}: not every lesion has exactly 2 images."
    )

    assert len(counts) == len(set(lesion_ids)), (
        f"{split_name}: some lesions have no images."
    )

    print(
        f"{split_name}: every lesion has exactly 2 images"
    )


def class_proportions(
    lesions,
    lesion_ids,
    label_column="dx"
):
    """Return class proportions for a lesion split."""

    subset = lesions[
        lesions["lesion_id"].isin(lesion_ids)
    ]

    return subset[label_column].value_counts(
        normalize=True
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MILK10k lesion-level train/validation/test split")
    print("=" * 70)

    print(f"Seed: {SEED}")
    print(f"Validation target: {VAL_SIZE:.0%}")
    print(f"Test target: {TEST_SIZE:.0%}")
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    metadata = pd.read_csv(METADATA_PATH)
    lesions = pd.read_csv(LESION_TABLE_PATH)

    print(f"Metadata rows: {len(metadata)}")
    print(f"Lesion rows:   {len(lesions)}")
    print()

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    assert len(lesions) == 5240, (
        f"Expected 5240 lesions, found {len(lesions)}"
    )

    assert metadata["lesion_id"].nunique() == 5240, (
        "Metadata does not contain 5240 unique lesions."
    )

    assert lesions["lesion_id"].is_unique, (
        "lesion_table.csv must contain one row per lesion."
    )

    # --------------------------------------------------------
    # Create split
    # --------------------------------------------------------

    train_ids, val_ids, test_ids = split_lesions(
        lesions,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        seed=SEED
    )

    # --------------------------------------------------------
    # Check all lesions are assigned
    # --------------------------------------------------------

    all_ids = (
        set(train_ids)
        | set(val_ids)
        | set(test_ids)
    )

    assert len(all_ids) == len(lesions), (
        "Some lesions are missing from the splits."
    )

    assert (
        len(train_ids)
        + len(val_ids)
        + len(test_ids)
        == len(lesions)
    ), "Split sizes do not add up to the full dataset."

    check_no_overlap(
        train_ids,
        val_ids,
        test_ids
    )

    # --------------------------------------------------------
    # Split sizes
    # --------------------------------------------------------

    total_lesions = len(lesions)

    train_pct = len(train_ids) / total_lesions
    val_pct = len(val_ids) / total_lesions
    test_pct = len(test_ids) / total_lesions

    print()
    print("Split sizes:")

    print(
        f"Train: {len(train_ids):4d} "
        f"({train_pct:.2%})"
    )

    print(
        f"Val:   {len(val_ids):4d} "
        f"({val_pct:.2%})"
    )

    print(
        f"Test:  {len(test_ids):4d} "
        f"({test_pct:.2%})"
    )

    # Requested target: 65 / 15 / 20.
    # Allow ±1 percentage point.

    assert abs(train_pct - 0.65) <= 0.01, (
        "Train split is outside ±1 percentage point."
    )

    assert abs(val_pct - 0.15) <= 0.01, (
        "Validation split is outside ±1 percentage point."
    )

    assert abs(test_pct - 0.20) <= 0.01, (
        "Test split is outside ±1 percentage point."
    )

    print("Split-size check: PASSED")

    # --------------------------------------------------------
    # Exactly two images per lesion
    # --------------------------------------------------------

    check_two_images_per_lesion(
        metadata,
        train_ids,
        "train"
    )

    check_two_images_per_lesion(
        metadata,
        val_ids,
        "val"
    )

    check_two_images_per_lesion(
        metadata,
        test_ids,
        "test"
    )

    # --------------------------------------------------------
    # 11-class proportions
    # --------------------------------------------------------

    global_props = class_proportions(
        lesions,
        lesions["lesion_id"].tolist()
    )

    train_props = class_proportions(
        lesions,
        train_ids
    )

    val_props = class_proportions(
        lesions,
        val_ids
    )

    test_props = class_proportions(
        lesions,
        test_ids
    )

    proportion_table = pd.DataFrame({
        "global": global_props,
        "train": train_props,
        "val": val_props,
        "test": test_props
    }).fillna(0)

    print()
    print("11-class proportions:")
    print(proportion_table.round(4))

    deviations = (
        proportion_table[
            ["train", "val", "test"]
        ]
        .sub(
            proportion_table["global"],
            axis=0
        )
        .abs()
    )

    max_deviation = deviations.max().max()

    print()
    print(
        f"Maximum 11-class proportion deviation: "
        f"{max_deviation:.4f}"
    )

    assert max_deviation <= 0.01, (
        "An 11-class proportion deviates by more than "
        "1 percentage point."
    )

    print("11-class proportion check: PASSED")

    # --------------------------------------------------------
    # diagnosis_1 proportions
    # --------------------------------------------------------

    if "diagnosis_1" in lesions.columns:

        diagnosis_table = lesions.copy()

    else:

        diagnosis_lookup = (
            metadata[
                ["lesion_id", "diagnosis_1"]
            ]
            .drop_duplicates("lesion_id")
        )

        diagnosis_table = lesions.merge(
            diagnosis_lookup,
            on="lesion_id",
            how="left",
            validate="one_to_one"
        )

    def diagnosis_props(ids):

        subset = diagnosis_table[
            diagnosis_table["lesion_id"].isin(ids)
        ]

        return subset["diagnosis_1"].value_counts(
            normalize=True
        )

    diagnosis_proportions = pd.DataFrame({
        "global": diagnosis_props(
            diagnosis_table["lesion_id"].tolist()
        ),
        "train": diagnosis_props(train_ids),
        "val": diagnosis_props(val_ids),
        "test": diagnosis_props(test_ids)
    }).fillna(0)

    print()
    print("diagnosis_1 proportions:")
    print(diagnosis_proportions.round(4))

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    train_ids_2, val_ids_2, test_ids_2 = split_lesions(
        lesions,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        seed=SEED
    )

    assert set(train_ids_2) == set(train_ids), (
        "Train split is not reproducible."
    )

    assert set(val_ids_2) == set(val_ids), (
        "Validation split is not reproducible."
    )

    assert set(test_ids_2) == set(test_ids), (
        "Test split is not reproducible."
    )

    print()
    print("Reproducibility check: PASSED")

    # --------------------------------------------------------
    # Assign images AFTER lesion-level split
    # --------------------------------------------------------

    train_df = metadata[
        metadata["lesion_id"].isin(train_ids)
    ].copy()

    val_df = metadata[
        metadata["lesion_id"].isin(val_ids)
    ].copy()

    test_df = metadata[
        metadata["lesion_id"].isin(test_ids)
    ].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Sort for reproducibility.
    sort_columns = [
        "lesion_id",
        "image_type",
        "isic_id"
    ]

    train_df = train_df.sort_values(
        sort_columns
    ).reset_index(drop=True)

    val_df = val_df.sort_values(
        sort_columns
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        sort_columns
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Final image-level checks
    # --------------------------------------------------------

    assert len(train_df) == len(train_ids) * 2
    assert len(val_df) == len(val_ids) * 2
    assert len(test_df) == len(test_ids) * 2

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "val.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame({
        "split": [
            "train",
            "val",
            "test"
        ],
        "lesions": [
            len(train_ids),
            len(val_ids),
            len(test_ids)
        ],
        "images": [
            len(train_df),
            len(val_df),
            len(test_df)
        ],
        "proportion": [
            train_pct,
            val_pct,
            test_pct
        ]
    })

    summary_path = OUTPUT_DIR / "split_summary.csv"

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {summary_path}")

    print()
    print(f"Created on: {date.today()}")
    print(f"Seed: {SEED}")

    print()
    print("B5 checks completed successfully.")


if __name__ == "__main__":
    main()
    