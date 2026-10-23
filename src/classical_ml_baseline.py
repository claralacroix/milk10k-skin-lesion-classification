
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)
from skimage.feature import graycomatrix, graycoprops

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "milk10k" / "images"
OUT = ROOT / "report" / "classical_ml"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "lesion_table.csv")
df = df[df["diagnosis_1"].isin(["Benign", "Malignant"])].copy()
df = df.drop_duplicates(subset="lesion_id")
df["image_path"] = df["derm_id"].astype(str).map(
    lambda x: IMAGE_DIR / f"{x}.jpg"
)
df = df[df["image_path"].map(Path.exists)].reset_index(drop=True)

def extract_features(path):
    image = Image.open(path).convert("RGB").resize((128, 128))
    rgb = np.asarray(image)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)

    colour = []
    for channel in range(3):
        hist, _ = np.histogram(
            rgb[:, :, channel], bins=16, range=(0, 256), density=True
        )
        colour.extend(hist)

    texture = [
        gray.mean(), gray.std(),
        np.percentile(gray, 10), np.percentile(gray, 90)
    ]
    small = (gray[::4, ::4] // 16).astype(np.uint8)
    glcm = graycomatrix(
        small, distances=[1], angles=[0, np.pi / 4],
        levels=16, symmetric=True, normed=True
    )
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        texture.extend(graycoprops(glcm, prop).ravel())

    return np.asarray(colour), np.asarray(texture)

print("Extracting image features...")
colour_features, texture_features, valid_rows = [], [], []
for i, row in df.iterrows():
    try:
        colour, texture = extract_features(row["image_path"])
        colour_features.append(colour)
        texture_features.append(texture)
        valid_rows.append(i)
    except Exception as exc:
        print(f"Skipping {row['image_path'].name}: {exc}")

df = df.iloc[valid_rows].reset_index(drop=True)
colour_X = np.vstack(colour_features)
combined_X = np.hstack([colour_X, np.vstack(texture_features)])
y = df["diagnosis_1"].map({"Benign": 0, "Malignant": 1}).to_numpy()
groups = df["lesion_id"].to_numpy()

# One fixed test split shared by every experiment.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(combined_X, y, groups))

model_factories = {
    "Logistic Regression": lambda: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000)
    ),
    "SVM (RBF)": lambda: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        SVC(class_weight="balanced")
    ),
    "Random Forest": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    )
}

results = []
fitted = {}

for feature_name, X in [
    ("Colour only", colour_X),
    ("Colour + texture", combined_X)
]:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for model_name, factory in model_factories.items():
        name = f"{feature_name} — {model_name}"
        print("Training", name)
        model = factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = model

        results.append({
            "feature_set": feature_name,
            "model": model_name,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro")
        })

results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
results_df.to_csv(OUT / "model_comparison.csv", index=False)
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

best = results_df.iloc[0]
best_name = f"{best['feature_set']} — {best['model']}"
best_model = fitted[best_name]
best_X = colour_X if best["feature_set"] == "Colour only" else combined_X
best_pred = best_model.predict(best_X[test_idx])

print("\nBest configuration:", best_name)
print(classification_report(
    y[test_idx], best_pred, target_names=["Benign", "Malignant"]
))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y[test_idx], best_pred,
    display_labels=["Benign", "Malignant"], cmap="Blues", ax=ax
)
ax.set_title(best_name)
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=200)
plt.close(fig)

joblib.dump(
    {"model": best_model, "feature_set": best["feature_set"]},
    OUT / "best_model.joblib"
)
print("\nResults saved to", OUT)
