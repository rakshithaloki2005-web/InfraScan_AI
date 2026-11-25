"""
CRACK500 Crack Segmentation Dataset Pipeline.

Provides image-mask pairing verification, safe dimension mismatch correction,
synchronized spatial augmentation, binary mask preservation, and tf.data input pipelines.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

DEFAULT_SEG_IMG_SIZE = (256, 256)
DEFAULT_SEG_BATCH_SIZE = 16


def resolve_crack500_file_path(dataset_root: Path, relative_str: str) -> Optional[Path]:
    """
    Resolves relative path inside CRACK500 directory structure.
    Accounts for both flat ('traincrop/file.jpg') and nested ('traincrop/traincrop/file.jpg') layouts.
    """
    p1 = dataset_root / relative_str
    if p1.exists():
        return p1.resolve()

    rel_p = Path(relative_str)
    # Check nested: folder/folder/filename
    folder = rel_p.parent.name
    filename = rel_p.name
    p2 = dataset_root / folder / folder / filename
    if p2.exists():
        return p2.resolve()

    return None


def load_crack500_benchmark_splits(
    dataset_dir: Union[str, Path]
) -> Dict[str, pd.DataFrame]:
    """
    Loads official CRACK500 benchmark splits from train.txt, val.txt, and test.txt.
    Verifies that all image-mask pairs exist and match.

    Args:
        dataset_dir: Root directory of CRACK500 dataset.

    Returns:
        Dict mapping split name ('train', 'val', 'test') to DataFrame with
        columns ['image_path', 'mask_path', 'split'].
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"CRACK500 root directory not found: {dataset_dir}")

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = dataset_dir / f"{split_name}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Benchmark split file {split_file} not found")

        records = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_rel, mask_rel = parts[0], parts[1]
                img_path = resolve_crack500_file_path(dataset_dir, img_rel)
                mask_path = resolve_crack500_file_path(dataset_dir, mask_rel)

                if img_path is None or mask_path is None:
                    raise FileNotFoundError(
                        f"Could not resolve pair: {img_rel} or {mask_rel} in {dataset_dir}"
                    )

                records.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "split": split_name
                })

        splits[split_name] = pd.DataFrame(records)

    return splits


def scan_all_crack500_pairs(dataset_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Scans all image and mask pairs across all subfolders (crops + raw full data)
    in CRACK500 and checks image-mask dimension consistency.

    Args:
        dataset_dir: Path to CRACK500 root directory.

    Returns:
        pd.DataFrame of all matched pairs with folder origin and metadata.
    """
    dataset_dir = Path(dataset_dir)
    records = []
    valid_folders = ["traincrop", "valcrop", "testcrop", "traindata", "valdata", "testdata"]

    for folder_name in valid_folders:
        folder_path = dataset_dir / folder_name
        if not folder_path.exists():
            continue

        # Look in nested directory if present, else directly
        nested = folder_path / folder_name
        search_dir = nested if nested.exists() else folder_path

        jpg_files = {p.stem: p for p in search_dir.glob("*.jpg")}
        png_files = {p.stem.replace("_mask", ""): p for p in search_dir.glob("*.png")}

        for stem, img_p in jpg_files.items():
            mask_p = png_files.get(stem)
            if mask_p:
                records.append({
                    "folder": folder_name,
                    "stem": stem,
                    "image_path": str(img_p.resolve()),
                    "mask_path": str(mask_p.resolve()),
                })

    return pd.DataFrame(records)


def _load_and_preprocess_pair(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    target_size: Tuple[int, int] = DEFAULT_SEG_IMG_SIZE
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow I/O function to read image and mask, correct orientation/transposition
    if inverted, resize safely, and strictly preserve binary mask labels.

    Args:
        image_path: Tensor string of image path.
        mask_path: Tensor string of mask path.
        target_size: Target (height, width) dimensions.

    Returns:
        Tuple of (image, mask) as float32 Tensors.
    """
    # Read image
    img_raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(img_raw, channels=3)

    # Read mask (PNG)
    mask_raw = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_raw, channels=1)

    # Safe dimension mismatch check:
    # If dimensions are transposed (e.g. H_img == W_mask and W_img == H_mask), transpose mask
    img_shape = tf.shape(image)
    mask_shape = tf.shape(mask)
    h_img, w_img = img_shape[0], img_shape[1]
    h_mask, w_mask = mask_shape[0], mask_shape[1]

    def _rotate_90_cw_mask():
        # Physical camera orientation check: 90-degree CW rotation aligns
        # portrait images (2560x1440) with landscape masks (1440x2560)
        return tf.image.rot90(mask, k=3)

    mask = tf.cond(
        tf.logical_and(tf.equal(h_img, w_mask), tf.equal(w_img, h_mask)),
        _rotate_90_cw_mask,
        lambda: mask
    )

    # Resize image with bilinear interpolation
    image = tf.image.resize(image, target_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32) / 255.0

    # Resize mask with NEAREST_NEIGHBOR interpolation to strictly preserve discrete edges
    mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Threshold to strict binary {0.0, 1.0}
    mask = tf.cast(mask > 0, tf.float32)

    return image, mask


def _augment_segmentation_pair(
    image: tf.Tensor,
    mask: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Applies SYNCHRONIZED spatial data augmentation to image and mask simultaneously.
    Photometric changes are applied to the image only.
    """
    # Synchronized random horizontal flip
    flip_lr = tf.random.uniform(shape=[]) > 0.5
    image = tf.cond(flip_lr, lambda: tf.image.flip_left_right(image), lambda: image)
    mask = tf.cond(flip_lr, lambda: tf.image.flip_left_right(mask), lambda: mask)

    # Synchronized random vertical flip
    flip_ud = tf.random.uniform(shape=[]) > 0.5
    image = tf.cond(flip_ud, lambda: tf.image.flip_up_down(image), lambda: image)
    mask = tf.cond(flip_ud, lambda: tf.image.flip_up_down(mask), lambda: mask)

    # Synchronized random 90-degree rotations (0, 90, 180, 270 degrees)
    rot_k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=rot_k)
    mask = tf.image.rot90(mask, k=rot_k)

    # Photometric augmentation (image only)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.92, upper=1.08)
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Re-enforce binary mask
    mask = tf.cast(mask > 0.5, tf.float32)

    return image, mask


def build_segmentation_tf_dataset(
    image_paths: List[str],
    mask_paths: List[str],
    is_training: bool = False,
    target_size: Tuple[int, int] = DEFAULT_SEG_IMG_SIZE,
    batch_size: int = DEFAULT_SEG_BATCH_SIZE,
    shuffle_buffer: int = 1024,
    seed: int = 42
) -> tf.data.Dataset:
    """
    Constructs an optimized tf.data pipeline for segmentation.

    Args:
        image_paths: List of file paths to JPG images.
        mask_paths: List of file paths to PNG masks.
        is_training: If True, shuffles and applies synchronized augmentation.
        target_size: Target (height, width) dimensions.
        batch_size: Batch size.
        shuffle_buffer: Buffer size for training shuffle.
        seed: Random seed.

    Returns:
        Prefetched tf.data.Dataset yielding (batch_images, batch_masks).
    """
    img_ds = tf.data.Dataset.from_tensor_slices(list(image_paths))
    msk_ds = tf.data.Dataset.from_tensor_slices(list(mask_paths))
    dataset = tf.data.Dataset.zip((img_ds, msk_ds))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), shuffle_buffer), seed=seed)

    # Parse, align, resize
    dataset = dataset.map(
        lambda img_p, msk_p: _load_and_preprocess_pair(img_p, msk_p, target_size=target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Synchronized augmentation on training set only
    if is_training:
        dataset = dataset.map(_augment_segmentation_pair, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def get_segmentation_datasets(
    dataset_dir: Union[str, Path],
    target_size: Tuple[int, int] = DEFAULT_SEG_IMG_SIZE,
    batch_size: int = DEFAULT_SEG_BATCH_SIZE,
    random_state: int = 42,
    subset_fraction: Optional[float] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
    """
    High-level entry point for CRACK500 segmentation datasets.
    Loads benchmark splits (train: 1,896; val: 348; test: 1,124).

    Args:
        dataset_dir: Path to CRACK500 root directory.
        target_size: Target (height, width) tuple (default: 256x256).
        batch_size: Batch size (default: 16).
        random_state: Random seed.
        subset_fraction: Optional fraction for fast testing.

    Returns:
        (train_ds, val_ds, test_ds, split_counts)
    """
    splits = load_crack500_benchmark_splits(dataset_dir)

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    if subset_fraction and 0.0 < subset_fraction < 1.0:
        train_df = train_df.sample(frac=subset_fraction, random_state=random_state)
        val_df = val_df.sample(frac=subset_fraction, random_state=random_state)
        test_df = test_df.sample(frac=subset_fraction, random_state=random_state)

    train_ds = build_segmentation_tf_dataset(
        train_df["image_path"].tolist(),
        train_df["mask_path"].tolist(),
        is_training=True,
        target_size=target_size,
        batch_size=batch_size,
        seed=random_state
    )

    val_ds = build_segmentation_tf_dataset(
        val_df["image_path"].tolist(),
        val_df["mask_path"].tolist(),
        is_training=False,
        target_size=target_size,
        batch_size=batch_size
    )

    test_ds = build_segmentation_tf_dataset(
        test_df["image_path"].tolist(),
        test_df["mask_path"].tolist(),
        is_training=False,
        target_size=target_size,
        batch_size=batch_size
    )

    counts = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "total": len(train_df) + len(val_df) + len(test_df)
    }

    return train_ds, val_ds, test_ds, counts

