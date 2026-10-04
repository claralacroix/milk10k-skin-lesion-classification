from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "milk10k"
IMAGE_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "metadata.csv"
LESION_FILE = PROJECT_ROOT / "notebooks" / "lesion_table.csv"

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# Load data
metadata = pd.read_csv(METADATA_FILE)
lesions = pd.read_csv(LESION_FILE)


# ---------------------------------------------------------
# 1. diagnosis_1 class distribution
# ---------------------------------------------------------

diagnosis_counts = lesions["diagnosis_1"].value_counts()
diagnosis_proportions = lesions["diagnosis_1"].value_counts(normalize=True)

print("diagnosis_1 counts:")
print(diagnosis_counts)

print("\ndiagnosis_1 proportions:")
print(diagnosis_proportions)


plt.figure(figsize=(7, 5))
diagnosis_counts.plot(kind="bar")
plt.title("Lesion distribution by diagnosis_1")
plt.xlabel("diagnosis_1")
plt.ylabel("Number of lesions")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "diagnosis_1_distribution.png", dpi=150)
plt.show()


# ---------------------------------------------------------
# 2. 11-class distribution
# ---------------------------------------------------------

dx_counts = lesions["dx"].value_counts()
dx_proportions = lesions["dx"].value_counts(normalize=True)

print("\n11-class counts:")
print(dx_counts)

print("\n11-class proportions:")
print(dx_proportions)


plt.figure(figsize=(9, 5))
dx_counts.sort_values().plot(kind="barh")
plt.xscale("log")
plt.title("11-class lesion distribution")
plt.xlabel("Number of lesions (log scale)")
plt.ylabel("11-class diagnosis")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "11_class_distribution_log.png", dpi=150)
plt.show()


# ---------------------------------------------------------
# 3. Mapping between 11-class labels and diagnosis_1
# ---------------------------------------------------------

mapping = (
    lesions.groupby(["dx", "diagnosis_1"])
    .size()
    .reset_index(name="lesion_count")
    .sort_values(["dx", "diagnosis_1"])
)

print("\n11-class -> diagnosis_1 mapping:")
print(mapping.to_string(index=False))

mapping.to_csv(FIGURES_DIR / "dx_to_diagnosis1_mapping.csv", index=False)


# ---------------------------------------------------------
# 4. Image manipulation and image type by diagnosis_1
# ---------------------------------------------------------

manipulation_table = pd.crosstab(
    metadata["image_manipulation"],
    metadata["diagnosis_1"],
    normalize="columns"
) * 100

print("\nImage manipulation by diagnosis_1 (% within diagnosis_1):")
print(manipulation_table.round(2))


image_type_table = pd.crosstab(
    metadata["image_type"],
    metadata["diagnosis_1"],
    normalize="columns"
) * 100

print("\nImage type by diagnosis_1 (% within diagnosis_1):")
print(image_type_table.round(2))


# ---------------------------------------------------------
# 5. 3 x 4 gallery: one example per 11-class
# ---------------------------------------------------------

classes = list(dx_counts.index)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.flatten()

for i, dx_class in enumerate(classes[:11]):

    row = lesions[lesions["dx"] == dx_class].iloc[0]

    image_id = row["derm_id"]
    image_path = IMAGE_DIR / f"{image_id}.jpg"

    image = Image.open(image_path)

    axes[i].imshow(image)
    axes[i].set_title(str(dx_class))
    axes[i].axis("off")

for i in range(len(classes), len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "11_class_gallery.png", dpi=150)
plt.show()


print("\nB2 EDA complete.")
print(f"Figures saved to: {FIGURES_DIR}")

# ------------------------------------------------------------
# Session 2 findings carried forward to the full dataset
# ------------------------------------------------------------

session2_findings = pd.DataFrame([
    {
        "Session 2 finding": "Image manipulation was strongly uneven across 11-class diagnoses; Mucosal melanotic macule had 57.14% altered images.",
        "Still true on full dataset?": "Yes",
        "Consequence for the pipeline": "Do not use image_manipulation as a model input because it can act as a shortcut for diagnosis."
    },
    {
        "Session 2 finding": "Missing anatom_site_general was associated with a different 11-class distribution; Nevus changed by 11.16 percentage points.",
        "Still true on full dataset?": "Yes",
        "Consequence for the pipeline": "Treat missingness as potentially informative and avoid using anatomical metadata as an unchecked shortcut."
    },
    {
        "Session 2 finding": "Histopathology cases were much more often Malignant (72.35%) than single-contributor clinical assessments (2.23%).",
        "Still true on full dataset?": "Yes",
        "Consequence for the pipeline": "Do not use diagnosis_confirm_type because it is closely related to how the diagnosis was established and can leak target information."
    },
    {
        "Session 2 finding": "Each diagnosis_1 class contained 50% clinical close-up and 50% dermoscopic images.",
        "Still true on full dataset?": "Yes",
        "Consequence for the pipeline": "Image type is balanced across target classes, but evaluation must still keep both views from the same lesion in the same split."
    }
])

print("\nSession 2 finding -> full dataset -> pipeline consequence:")
print(session2_findings.to_string(index=False))

session2_findings.to_csv(
    FIGURES_DIR / "session2_findings_pipeline.csv",
    index=False
)

print(
    f"\nSaved: {FIGURES_DIR / 'session2_findings_pipeline.csv'}"
)
