from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "milk10k"
METADATA_FILE = DATA_DIR / "metadata.csv"
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(METADATA_FILE)

# ------------------------------------------------------------
# 1. Missing-value decisions
# ------------------------------------------------------------

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

decisions = {
    "anatom_site_special": "Leave NaN; not used as a model input because of very high missingness.",
    "diagnosis_4": "Leave NaN; exclude from model inputs because it is diagnostic information and may leak the label.",
    "melanocytic": "Leave NaN; not used as a model input because of high missingness and diagnostic relevance.",
    "anatom_site_general": "Leave NaN; do not use as a model input because missingness is informative.",
}

print("Missing-value decisions:")
for column, count in missing.items():
    decision = decisions.get(
        column,
        "Leave NaN; column is not required as a model input."
    )
    print(f"{column}: {count} missing -> {decision}")

# ------------------------------------------------------------
# 2. Exactly two images per lesion
# ------------------------------------------------------------

images_per_lesion = df.groupby("lesion_id").size()

print("\nImages per lesion:")
print(images_per_lesion.value_counts().sort_index())

assert (images_per_lesion == 2).all(), "Some lesions do not have exactly 2 images."

# ------------------------------------------------------------
# 3. One dermoscopic + one clinical image per lesion
# ------------------------------------------------------------

image_types_per_lesion = df.groupby("lesion_id")["image_type"].nunique()

print("\nNumber of distinct image types per lesion:")
print(image_types_per_lesion.value_counts().sort_index())

assert (image_types_per_lesion == 2).all(), (
    "Some lesions do not contain two distinct image types."
)

print("\nImage types:")
print(df["image_type"].value_counts())

# ------------------------------------------------------------
# 4. Suspicious shortcut: image manipulation
# ------------------------------------------------------------

manipulation = pd.crosstab(
    df["image_manipulation"],
    df["diagnosis_1"],
    normalize="columns"
) * 100

print("\nImage manipulation x diagnosis_1 (% within diagnosis):")
print(manipulation.round(2))

# ------------------------------------------------------------
# 5. Suspicious shortcut: image type
# ------------------------------------------------------------

image_type = pd.crosstab(
    df["image_type"],
    df["diagnosis_1"],
    normalize="columns"
) * 100

print("\nImage type x diagnosis_1 (% within diagnosis):")
print(image_type.round(2))

# ------------------------------------------------------------
# 6. Columns excluded because they leak diagnosis
# ------------------------------------------------------------

leakage_columns = [
    "diagnosis_2",
    "diagnosis_3",
    "diagnosis_4",
    "diagnosis_confirm_type",
    "concomitant_biopsy"
]

print("\nColumns excluded from model inputs because of leakage/bias:")
for column in leakage_columns:
    print("-", column)

# ------------------------------------------------------------
# 7. Save report
# ------------------------------------------------------------

report_file = REPORT_DIR / "data_quality_report.md"

with open(report_file, "w") as f:
    f.write("# MILK10k Data-Quality Report\n\n")

    f.write("## Missing-value decisions\n\n")
    for column, count in missing.items():
        decision = decisions.get(
            column,
            "Leave NaN; column is not required as a model input."
        )
        f.write(f"- **{column}**: {count} missing — {decision}\n")

    f.write("\n## Label and image consistency\n\n")
    f.write(
        f"- Metadata rows: {len(df)}\n"
        f"- Unique lesions: {df['lesion_id'].nunique()}\n"
        f"- Every lesion has exactly 2 images: **Yes**\n"
        f"- Every lesion has one dermoscopic and one clinical image: **Yes**\n"
    )

    f.write("\n## Image manipulation × diagnosis_1\n\n")
    f.write(manipulation.round(2).to_markdown())

    f.write("\n\n## Image type × diagnosis_1\n\n")
    f.write(image_type.round(2).to_markdown())

    f.write("\n\n## Shortcut conclusion\n\n")
    f.write(
        "Image manipulation shows class-dependent differences and should not "
        "be used as a model input. Image type is balanced at 50% clinical and "
        "50% dermoscopic within each diagnosis_1 class. Nevertheless, both "
        "images from the same lesion must remain in the same data split to "
        "prevent lesion-level leakage.\n"
    )

    f.write("\n## Columns excluded from model inputs\n\n")
    for column in leakage_columns:
        f.write(f"- `{column}`\n")

print(f"\nSaved report to: {report_file}")
