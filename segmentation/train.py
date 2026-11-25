"""
Training Script for InfraScan CRACK500 U-Net Segmentation Model.

Executes U-Net training, manages callbacks, saves best checkpoint weights,
evaluates on test dataset, and generates visual crack segmentation predictions.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from src.segmentation.dataset import get_segmentation_datasets
from src.segmentation.model import (
    build_unet,
    dice_coef,
    dice_loss,
    iou_metric,
    combined_bce_dice_loss
)


def plot_segmentation_history(history_dict: dict, save_path: Path):
    """Plots and saves loss, Dice coefficient, IoU, and pixel accuracy curves."""
    epochs_range = range(1, len(history_dict["loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs_range, history_dict["loss"], "b-o", label="Train Loss")
    axes[0, 0].plot(epochs_range, history_dict["val_loss"], "r--s", label="Val Loss")
    axes[0, 0].set_title("Combined BCE + Dice Loss", fontsize=12, weight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Dice Coefficient
    axes[0, 1].plot(epochs_range, history_dict["dice_coef"], "g-o", label="Train Dice")
    axes[0, 1].plot(epochs_range, history_dict["val_dice_coef"], "g--s", label="Val Dice")
    axes[0, 1].set_title("Dice Coefficient (F1 Score)", fontsize=12, weight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # IoU
    axes[1, 0].plot(epochs_range, history_dict["iou_metric"], "m-o", label="Train IoU")
    axes[1, 0].plot(epochs_range, history_dict["val_iou_metric"], "m--s", label="Val IoU")
    axes[1, 0].set_title("Intersection over Union (IoU)", fontsize=12, weight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("IoU")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Pixel Accuracy
    axes[1, 1].plot(epochs_range, history_dict["pixel_accuracy"], "b-o", label="Train Pixel Acc")
    axes[1, 1].plot(epochs_range, history_dict["val_pixel_accuracy"], "r--s", label="Val Pixel Acc")
    axes[1, 1].set_title("Pixel Accuracy", fontsize=12, weight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved segmentation training curves: {save_path}")


def evaluate_and_generate_visuals(
    model: keras.Model,
    test_ds,
    eval_dir: Path,
    num_samples: int = 4
) -> dict:
    """
    Evaluates U-Net on test dataset, computes mean IoU, Dice, Pixel Accuracy,
    and produces visual prediction comparisons (Image, GT, Prediction, Overlay).
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    print("\nEvaluating U-Net on test dataset...")

    ious = []
    dices = []
    pixel_accs = []
    precisions = []
    recalls = []

    sample_images = []
    sample_gts = []
    sample_preds = []

    for images, masks in test_ds:
        preds = model.predict(images, verbose=0)
        preds_bin = (preds >= 0.5).astype(np.float32)
        masks_np = masks.numpy().astype(np.float32)

        for i in range(len(images)):
            gt_f = masks_np[i].flatten()
            pr_f = preds_bin[i].flatten()

            intersection = np.sum(gt_f * pr_f)
            union = np.sum(gt_f) + np.sum(pr_f) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            dice = (2.0 * intersection + 1e-6) / (np.sum(gt_f) + np.sum(pr_f) + 1e-6)
            acc = np.mean(gt_f == pr_f)

            tp = np.sum((gt_f == 1) & (pr_f == 1))
            fp = np.sum((gt_f == 0) & (pr_f == 1))
            fn = np.sum((gt_f == 1) & (pr_f == 0))

            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)

            ious.append(iou)
            dices.append(dice)
            pixel_accs.append(acc)
            precisions.append(prec)
            recalls.append(rec)

            if len(sample_images) < num_samples and np.sum(gt_f) > 100:
                sample_images.append(images[i].numpy())
                sample_gts.append(masks_np[i, :, :, 0])
                sample_preds.append(preds_bin[i, :, :, 0])

    mean_iou = float(np.mean(ious))
    mean_dice = float(np.mean(dices))
    mean_acc = float(np.mean(pixel_accs))
    mean_prec = float(np.mean(precisions))
    mean_rec = float(np.mean(recalls))

    report_str = (
        "INFRASCAN CRACK SEGMENTATION REPORT (CRACK500 U-NET)\n"
        + "=" * 52 + "\n"
        + f"Test Image-Mask Pairs Evaluated: {len(ious)}\n"
        + f"Mean Intersection over Union (IoU): {mean_iou:.4f} ({mean_iou*100:.2f}%)\n"
        + f"Mean Dice Coefficient (F1 Score): {mean_dice:.4f} ({mean_dice*100:.2f}%)\n"
        + f"Mean Pixel Accuracy:               {mean_acc:.4f} ({mean_acc*100:.2f}%)\n"
        + f"Mean Pixel Precision:              {mean_prec:.4f} ({mean_prec*100:.2f}%)\n"
        + f"Mean Pixel Recall:                 {mean_rec:.4f} ({mean_rec*100:.2f}%)\n"
    )
    print("\n" + report_str)

    report_file = eval_dir / "segmentation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"Saved segmentation report: {report_file}")

    metrics = {
        "mean_iou": mean_iou,
        "mean_dice_coefficient": mean_dice,
        "pixel_accuracy": mean_acc,
        "precision": mean_prec,
        "recall": mean_rec,
        "test_pairs_evaluated": len(ious)
    }
    metrics_file = eval_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON: {metrics_file}")

    # Generate visual predictions figure
    fig, axes = plt.subplots(len(sample_images), 4, figsize=(16, 4 * len(sample_images)))
    if len(sample_images) == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(len(sample_images)):
        img = sample_images[i]
        gt = sample_gts[i]
        pred = sample_preds[i]

        overlay = img.copy()
        # Red highlight on predicted crack pixels
        overlay[pred > 0.5] = [1.0, 0.0, 0.0]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Test Image #{i+1}", fontsize=11)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title(f"Ground Truth #{i+1}", fontsize=11)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title(f"Predicted Mask #{i+1}", fontsize=11)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f"Predicted Overlay #{i+1}", fontsize=11, color="darkred", weight="bold")
        axes[i, 3].axis("off")

    plt.tight_layout()
    pred_path = eval_dir / "sample_predictions.png"
    plt.savefig(pred_path, dpi=150)
    plt.close()
    print(f"Saved prediction visual artifact: {pred_path}")

    return metrics


def train_segmentation_model(
    project_root: Path,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    subset_fraction: Optional[float] = None
):
    dataset_dir = project_root / "dataset" / "CRACK500"
    model_dir = project_root / "models" / "segmentation"
    eval_dir = project_root / "outputs" / "model_evaluation" / "segmentation"

    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STARTING CRACK500 U-NET SEGMENTATION TRAINING PIPELINE")
    print("=" * 60)
    print(f"Target Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    if subset_fraction:
        print(f"Subset Fraction: {subset_fraction:.2%}")

    train_ds, val_ds, test_ds, split_counts = get_segmentation_datasets(
        dataset_dir=dataset_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        subset_fraction=subset_fraction
    )
    print(f"Dataset split counts: {split_counts}")

    print("\nBuilding U-Net Architecture...")
    model = build_unet(
        input_shape=(256, 256, 3),
        initial_filters=16,
        dropout_rate=0.2,
        learning_rate=learning_rate
    )
    model.summary()

    best_model_path = model_dir / "best_unet.keras"
    training_log_path = eval_dir / "training_log.csv"

    # Define custom objects for reloadability
    custom_objects = {
        "combined_bce_dice_loss": combined_bce_dice_loss,
        "loss": combined_bce_dice_loss(),
        "dice_coef": dice_coef,
        "iou_metric": iou_metric
    }

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(str(training_log_path))
    ]

    print("\nStarting U-Net model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print("\nTraining completed.")
    plot_segmentation_history(
        history.history,
        eval_dir / "training_curves.png"
    )

    print(f"\nLoading best saved U-Net model from: {best_model_path} ...")
    best_model = keras.models.load_model(
        str(best_model_path),
        custom_objects=custom_objects
    )

    metrics = evaluate_and_generate_visuals(best_model, test_ds, eval_dir)
    print("\nPHASE 4 COMPLETE: CRACK500 U-NET MODEL TRAINED AND EVALUATED.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train CRACK500 U-Net Segmentation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--subset", type=float, default=None, help="Optional subset fraction (e.g. 0.3)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    train_segmentation_model(
        project_root=project_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_fraction=args.subset
    )


if __name__ == "__main__":
    main()
