"""
SDNET2018 Crack Classification Dataset Pipeline.

Provides stratified splitting, class weight computation, image preprocessing,
training-only data augmentation, and high-performance tf.data input pipelines.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# Target image size for classification CNN
DEFAULT_IMG_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32

# SDNET2018 category mapping to binary classes
CRACKED_CATEGORIES = {"CD", "CP", "CW"}
UNCRACKED_CATEGORIES = {"UD", "UP", "UW"}


def scan_sdnet_dataset(dataset_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Scans SDNET2018 dataset directory and builds a catalog DataFrame.

    Category mapping:
      - CD, CP, CW -> Cracked (label: 1)
      - UD, UP, UW -> Uncracked (label: 0)

    Args:
        dataset_dir: Path to the SDNET2018 root directory.

    Returns:
        pd.DataFrame containing filepath, relative_path, category, and binary label.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"SDNET2018 dataset directory not found: {dataset_dir}")

    records = []
    # SDNET2018 is structured as D/(CD, UD), P/(CP, UP), W/(CW, UW)
    valid_exts = {".jpg", ".jpeg", ".png"}

    for struct_folder in sorted(dataset_dir.iterdir()):
        if not struct_folder.is_dir():
            continue
        for cat_folder in sorted(struct_folder.iterdir()):
            if not cat_folder.is_dir():
                continue
            cat_name = cat_folder.name.upper()
            if cat_name in CRACKED_CATEGORIES:
                label = 1
                class_name = "Cracked"
            elif cat_name in UNCRACKED_CATEGORIES:
                label = 0
                class_name = "Uncracked"
            else:
                continue

            for img_path in cat_folder.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                    records.append({
                        "filepath": str(img_path.resolve()),
                        "relative_path": str(img_path.relative_to(dataset_dir)),
                        "structure": struct_folder.name,
                        "category": cat_name,
                        "class_name": class_name,
                        "label": label
                    })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid images found in {dataset_dir}")

    return df


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Splits the dataset into train, validation, and test sets using stratified sampling.
    Guarantees no data leakage between splits and exact class proportion preservation.

    Args:
        df: Catalog DataFrame from scan_sdnet_dataset.
        train_ratio: Fraction of dataset for training (default 0.70).
        val_ratio: Fraction of dataset for validation (default 0.15).
        test_ratio: Fraction of dataset for testing (default 0.15).
        random_state: Seed for reproducibility.
        save_path: Optional CSV destination to persist split records.

    Returns:
        pd.DataFrame with an added 'split' column ('train', 'val', 'test').
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    df = df.copy()

    # Stratified first split: train vs temporary (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=random_state,
        shuffle=True
    )

    # Stratified second split: val vs test
    val_rel_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_rel_ratio,
        stratify=temp_df["label"],
        random_state=random_state,
        shuffle=True
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    full_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        full_split_df.to_csv(save_path, index=False)

    return full_split_df


def compute_classification_class_weights(train_labels: Union[np.ndarray, pd.Series]) -> Dict[int, float]:
    """
    Calculates balanced class weights for the binary classification task to mitigate
    the ~85% uncracked vs 15% cracked class imbalance.

    Formula: weight_c = total_samples / (n_classes * count_c)

    Args:
        train_labels: Sequence of binary training labels (0 and 1).

    Returns:
        Dict mapping class index (0 or 1) to weight float.
    """
    labels_arr = np.asarray(train_labels)
    classes = np.unique(labels_arr)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels_arr)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def _parse_and_preprocess_image(
    filepath: tf.Tensor,
    label: tf.Tensor,
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Reads image file from disk, decodes, resizes, and normalizes to [0, 1]."""
    img_raw = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)
    return img, label


def _augment_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Applies data augmentation to training images only.
    Geometric: Random horizontal flip, vertical flip, 90-deg rotation.
    Photometric: Random brightness and contrast variations.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random 90-degree rotations (0, 90, 180, 270)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)

    # Moderate photometric perturbations
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def build_tf_dataset(
    filepaths: Union[np.ndarray, list],
    labels: Union[np.ndarray, list],
    is_training: bool = False,
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle_buffer: int = 2048,
    seed: int = 42
) -> tf.data.Dataset:
    """
    Builds an optimized tf.data.Dataset pipeline for classification.

    Args:
        filepaths: Array or list of image file paths.
        labels: Array or list of integer labels (0 or 1).
        is_training: If True, shuffles and applies data augmentation.
        img_size: Target (height, width) dimensions.
        batch_size: Batch size.
        shuffle_buffer: Size of shuffle buffer for training.
        seed: Random seed for shuffling.

    Returns:
        Prefetched tf.data.Dataset yielding (batch_images, batch_labels).
    """
    paths_ds = tf.data.Dataset.from_tensor_slices(list(filepaths))
    labels_ds = tf.data.Dataset.from_tensor_slices(list(labels))
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(filepaths), shuffle_buffer), seed=seed)

    # Parse and resize
    dataset = dataset.map(
        lambda p, l: _parse_and_preprocess_image(p, l, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Apply training-only augmentation
    if is_training:
        dataset = dataset.map(_augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def get_classification_datasets(
    dataset_dir: Union[str, Path],
    splits_csv: Optional[Union[str, Path]] = None,
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    subset_fraction: Optional[float] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[int, float], Dict[str, dict]]:
    """
    High-level entry point: Loads or creates stratified splits and returns
    train, val, test tf.data.Datasets along with computed class weights.

    Args:
        dataset_dir: Root of SDNET2018 dataset.
        splits_csv: Optional path to load or cache the stratified split CSV.
        img_size: Target (height, width) tuple (default: 224x224).
        batch_size: Batch size (default: 32).
        train_ratio, val_ratio, test_ratio: Split ratios.
        random_state: Random seed for reproducibility.
        subset_fraction: Optional fraction (e.g., 0.1) for fast experimentation.

    Returns:
        (train_ds, val_ds, test_ds, class_weights, split_statistics)
    """
    dataset_dir = Path(dataset_dir)
    splits_df = None

    if splits_csv and Path(splits_csv).exists():
        splits_df = pd.read_csv(splits_csv)
    else:
        raw_df = scan_sdnet_dataset(dataset_dir)
        splits_df = create_stratified_splits(
            raw_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            save_path=splits_csv
        )

    # Optional stratified subset sampling for fast verification
    if subset_fraction and 0.0 < subset_fraction < 1.0:
        subsets = []
        for s in ["train", "val", "test"]:
            sub_df = splits_df[splits_df["split"] == s]
            _, sampled = train_test_split(
                sub_df,
                test_size=subset_fraction,
                stratify=sub_df["label"],
                random_state=random_state
            )
            subsets.append(sampled)
        splits_df = pd.concat(subsets, ignore_index=True)

    train_part = splits_df[splits_df["split"] == "train"]
    val_part = splits_df[splits_df["split"] == "val"]
    test_part = splits_df[splits_df["split"] == "test"]

    class_weights = compute_classification_class_weights(train_part["label"].values)

    train_ds = build_tf_dataset(
        train_part["filepath"].values,
        train_part["label"].values,
        is_training=True,
        img_size=img_size,
        batch_size=batch_size,
        seed=random_state
    )

    val_ds = build_tf_dataset(
        val_part["filepath"].values,
        val_part["label"].values,
        is_training=False,
        img_size=img_size,
        batch_size=batch_size
    )

    test_ds = build_tf_dataset(
        test_part["filepath"].values,
        test_part["label"].values,
        is_training=False,
        img_size=img_size,
        batch_size=batch_size
    )

    stats = {}
    for name, part in [("train", train_part), ("val", val_part), ("test", test_part)]:
        counts = part["label"].value_counts().to_dict()
        stats[name] = {
            "total": len(part),
            "uncracked": counts.get(0, 0),
            "cracked": counts.get(1, 0),
            "cracked_ratio": counts.get(1, 0) / max(len(part), 1)
        }

    return train_ds, val_ds, test_ds, class_weights, stats

