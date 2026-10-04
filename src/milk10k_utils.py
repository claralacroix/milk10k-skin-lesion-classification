import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def process_image(
    image_path,
    size=(224, 224),
    color_mode="RGB",
    normalization="minmax"
):
    image = Image.open(image_path)

    if color_mode == "RGB":
        image = image.convert("RGB")
    elif color_mode == "L":
        image = image.convert("L")
    else:
        raise ValueError("color_mode must be 'RGB' or 'L'")

    image = image.resize(size)

    image_array = np.array(image).astype(np.float32)

    if normalization == "minmax":
        image_array = image_array / 255.0

    elif normalization == "zscore":
        mean = image_array.mean()
        std = image_array.std()

        if std > 0:
            image_array = (image_array - mean) / std

    elif normalization != "none":
        raise ValueError(
            "normalization must be 'minmax', 'zscore', or 'none'"
        )

    info = {
        "final_shape": image_array.shape,
        "min_value": image_array.min(),
        "max_value": image_array.max(),
        "color_mode": color_mode,
        "normalization": normalization
    }

    return image_array, info


def process_image_batch(
    image_paths,
    size=(224, 224),
    color_mode="RGB",
    normalization="minmax"
):
    processed_images = []
    failures = []

    for image_path in image_paths:
        try:
            image_array, info = process_image(
                image_path,
                size=size,
                color_mode=color_mode,
                normalization=normalization
            )

            processed_images.append(image_array)

        except Exception as e:
            failures.append({
                "path": image_path,
                "reason": str(e)
            })

    return processed_images, failures


def image_data_loader(
    metadata_df,
    image_dir,
    batch_size=8,
    size=(224, 224),
    color_mode="RGB",
    normalization="minmax"
):
    image_paths = metadata_df["isic_id"].apply(
        lambda x: f"{image_dir}/{x}.jpg"
    ).tolist()

    labels = metadata_df["diagnosis_1"].tolist()

    for start in range(0, len(image_paths), batch_size):

        batch_paths = image_paths[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]

        batch_images = []

        for image_path in batch_paths:
            try:
                image_array, _ = process_image(
                    image_path,
                    size=size,
                    color_mode=color_mode,
                    normalization=normalization
                )

                batch_images.append(image_array)

            except Exception:
                continue

        if batch_images:
            yield np.stack(batch_images), batch_labels[:len(batch_images)]


def filter_available_images(metadata_df, image_dir):
    import os

    available_rows = []
    missing_files = []

    for _, row in metadata_df.iterrows():

        image_path = os.path.join(
            image_dir,
            f"{row['isic_id']}.jpg"
        )

        if os.path.exists(image_path):
            available_rows.append(row)

        else:
            missing_files.append(image_path)

    available_df = (
        metadata_df.iloc[0:0]
        if not available_rows
        else __import__("pandas").DataFrame(available_rows)
    )

    return available_df, missing_files


def show_image_grid(images, labels, n_images=8):
    n_images = min(n_images, len(images))

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 6)
    )

    axes = axes.flatten()

    for i in range(n_images):
        axes[i].imshow(images[i])
        axes[i].set_title(str(labels[i]))
        axes[i].axis("off")

    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
    