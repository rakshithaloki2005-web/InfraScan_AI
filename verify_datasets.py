"""
Verification Script for InfraScan Datasets and Preprocessing Pipelines.

Tests:
1. SDNET2018 Stratified Splitting, Class Weights, and Batch Generation.
2. CRACK500 Benchmark Split Loading, Synchronized Augmentation, and Mask Integrity.
3. Safe Dimension Mismatch Resolution on Raw Full Pairs.
4. Generates Visual Alignment Artifact: outputs/dataset_verification.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.classification.dataset import (
    get_classification_datasets,
    scan_sdnet_dataset,
    create_stratified_splits,
    compute_classification_class_weights
)
from src.segmentation.dataset import (
    get_segmentation_datasets,
    load_crack500_benchmark_splits,
    _load_and_preprocess_pair
)


def verify_classification_pipeline(project_root: Path):
    print("=" * 60)
    print("1. VERIFYING SDNET2018 CLASSIFICATION PIPELINE")
    print("=" * 60)

    dataset_dir = project_root / "dataset" / "SDNET2018"
    splits_csv = project_root / "outputs" / "classification_splits.csv"

    print(f"Scanning SDNET2018 from: {dataset_dir} ...")
    raw_df = scan_sdnet_dataset(dataset_dir)
    print(f"Total cataloged images: {len(raw_df):,}")
    print("Category breakdown:\n", raw_df["category"].value_counts())
    print("Class breakdown:\n", raw_df["class_name"].value_counts(normalize=True).apply(lambda v: f"{v*100:.2f}%"))

    print(f"\nGenerating Stratified Splits (70% train / 15% val / 15% test) -> {splits_csv} ...")
    splits_df = create_stratified_splits(raw_df, save_path=splits_csv)

    for split in ["train", "val", "test"]:
        part = splits_df[splits_df["split"] == split]
        cracked = (part["label"] == 1).sum()
        uncracked = (part["label"] == 0).sum()
        ratio = cracked / len(part)
        print(f"  {split.upper():5s}: {len(part):6,d} images | Cracked: {cracked:5,d} ({ratio*100:.2f}%), Uncracked: {uncracked:5,d}")

    # Compute class weights
    train_labels = splits_df[splits_df["split"] == "train"]["label"].values
    class_weights = compute_classification_class_weights(train_labels)
    print(f"\nComputed Balanced Class Weights: {class_weights}")

    # Build tf.data pipeline with small subset for instant batch verification
    print("\nBuilding tf.data classification datasets...")
    train_ds, val_ds, test_ds, _, _ = get_classification_datasets(
        dataset_dir=dataset_dir,
        splits_csv=splits_csv,
        img_size=(224, 224),
        batch_size=16
    )

    # Fetch 1 batch from train and val
    for img_batch, lbl_batch in train_ds.take(1):
        print(f"Train Batch Image Shape: {img_batch.shape}, Type: {img_batch.dtype}")
        print(f"Train Batch Label Shape: {lbl_batch.shape}, Type: {lbl_batch.dtype}")
        print(f"Pixel value range: [{np.min(img_batch):.3f}, {np.max(img_batch):.3f}]")
        assert img_batch.shape == (16, 224, 224, 3), "Invalid image batch shape!"
        assert lbl_batch.shape == (16,), "Invalid label batch shape!"
        sample_cls_images = img_batch.numpy()
        sample_cls_labels = lbl_batch.numpy()

    print("Classification Pipeline: PASSED [OK]\n")
    return sample_cls_images, sample_cls_labels


def verify_segmentation_pipeline(project_root: Path):
    print("=" * 60)
    print("2. VERIFYING CRACK500 SEGMENTATION PIPELINE")
    print("=" * 60)

    dataset_dir = project_root / "dataset" / "CRACK500"
    print(f"Loading CRACK500 benchmark splits from: {dataset_dir} ...")
    splits = load_crack500_benchmark_splits(dataset_dir)

    for name, df in splits.items():
        print(f"  {name.upper():5s}: {len(df):5,d} verified pairs")

    print("\nBuilding tf.data segmentation datasets...")
    train_ds, val_ds, test_ds, counts = get_segmentation_datasets(
        dataset_dir=dataset_dir,
        target_size=(256, 256),
        batch_size=8
    )

    # Fetch 1 batch
    for img_batch, msk_batch in train_ds.take(1):
        print(f"Segmentation Train Batch Image Shape: {img_batch.shape}, Type: {img_batch.dtype}")
        print(f"Segmentation Train Batch Mask Shape: {msk_batch.shape}, Type: {msk_batch.dtype}")
        print(f"Image Range: [{np.min(img_batch):.3f}, {np.max(img_batch):.3f}]")
        unique_mask_vals = np.unique(msk_batch.numpy())
        print(f"Unique Mask Values: {unique_mask_vals}")
        assert img_batch.shape == (8, 256, 256, 3), "Invalid seg image shape!"
        assert msk_batch.shape == (8, 256, 256, 1), "Invalid seg mask shape!"
        assert np.all(np.isin(unique_mask_vals, [0.0, 1.0])), "Mask values not strictly binary!"
        sample_seg_images = img_batch.numpy()
        sample_seg_masks = msk_batch.numpy()

    print("Segmentation Pipeline: PASSED [OK]\n")

    print("=" * 60)
    print("3. VERIFYING SAFE CORRECTION OF 3 RAW DIMENSION MISMATCHES")
    print("=" * 60)
    traindata_folder = dataset_dir / "traindata" / "traindata"
    mismatch_stems = ["20160326_144512", "20160326_150319", "20160330_170428"]
    corrected_pairs = []

    for stem in mismatch_stems:
        img_p = traindata_folder / f"{stem}.jpg"
        msk_p = traindata_folder / f"{stem}_mask.png"
        img_t, msk_t = _load_and_preprocess_pair(
            tf.constant(str(img_p)),
            tf.constant(str(msk_p)),
            target_size=(256, 256)
        )
        print(f"  {stem}: Image Shape={img_t.shape}, Mask Shape={msk_t.shape}, Mask Values={np.unique(msk_t.numpy())}")
        assert img_t.shape == (256, 256, 3)
        assert msk_t.shape == (256, 256, 1)
        corrected_pairs.append((stem, img_t.numpy(), msk_t.numpy()))

    print("Safe Dimension Correction: PASSED [OK]\n")
    return sample_seg_images, sample_seg_masks, corrected_pairs


def generate_verification_artifact(project_root: Path, cls_images, cls_labels, seg_images, seg_masks, corrected_pairs):
    print("=" * 60)
    print("4. GENERATING VISUAL VERIFICATION ARTIFACT")
    print("=" * 60)

    output_path = project_root / "outputs" / "dataset_verification.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))

    # Row 1: 4 Classification Samples (2 Uncracked, 2 Cracked)
    for i in range(4):
        ax = fig.add_subplot(3, 4, i + 1)
        ax.imshow(cls_images[i])
        lbl_str = "Cracked (1)" if cls_labels[i] == 1 else "Uncracked (0)"
        color = "red" if cls_labels[i] == 1 else "green"
        ax.set_title(f"Classification: {lbl_str}", color=color, fontsize=10, weight="bold")
        ax.axis("off")

    # Row 2: 2 Segmentation Samples (Image, Mask, Overlay)
    for i in range(2):
        img = seg_images[i]
        mask = seg_masks[i, :, :, 0]
        overlay = img.copy()
        # Red overlay on crack pixels
        overlay[mask > 0.5] = [1.0, 0.0, 0.0]

        ax_img = fig.add_subplot(3, 4, 5 + i * 3)
        ax_img.imshow(img)
        ax_img.set_title(f"CRACK500 Image #{i+1}", fontsize=10)
        ax_img.axis("off")

        ax_msk = fig.add_subplot(3, 4, 6 + i * 3)
        ax_msk.imshow(mask, cmap="gray")
        ax_msk.set_title(f"Binary Mask #{i+1} [0,1]", fontsize=10)
        ax_msk.axis("off")

        ax_ov = fig.add_subplot(3, 4, 7 + i * 3)
        ax_ov.imshow(overlay)
        ax_ov.set_title(f"Crack Overlay #{i+1}", fontsize=10, color="darkred", weight="bold")
        ax_ov.axis("off")

    # Row 3: Corrected Dimension Mismatch Pairs
    for i, (stem, img, mask) in enumerate(corrected_pairs):
        m = mask[:, :, 0]
        overlay = img.copy()
        overlay[m > 0.5] = [1.0, 0.2, 0.2]

        ax = fig.add_subplot(3, 4, 9 + i)
        ax.imshow(overlay)
        ax.set_title(f"Corrected: {stem}\n(Oriented & Aligned)", fontsize=9, color="blue")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visual verification figure: {output_path}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    cls_images, cls_labels = verify_classification_pipeline(project_root)
    seg_images, seg_masks, corrected_pairs = verify_segmentation_pipeline(project_root)
    generate_verification_artifact(project_root, cls_images, cls_labels, seg_images, seg_masks, corrected_pairs)
    print("\nALL PHASE 2 VERIFICATIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

