"""
Training Script for InfraScan SDNET2018 Crack Classification Model.

Executes training, manages callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau),
applies class weights, saves best model weights, and outputs comprehensive training artifacts.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score
)
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from src.classification.dataset import get_classification_datasets
from src.classification.model import build_crack_classifier


def plot_training_history(history_dict: dict, save_path: Path):
    """Plots and saves loss, accuracy, precision, recall, and AUC progression."""
    epochs_range = range(1, len(history_dict["loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs_range, history_dict["loss"], "b-o", label="Train Loss")
    axes[0, 0].plot(epochs_range, history_dict["val_loss"], "r--s", label="Val Loss")
    axes[0, 0].set_title("Binary Cross-Entropy Loss", fontsize=12, weight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Accuracy
    axes[0, 1].plot(epochs_range, history_dict["accuracy"], "b-o", label="Train Acc")
    axes[0, 1].plot(epochs_range, history_dict["val_accuracy"], "r--s", label="Val Acc")
    axes[0, 1].set_title("Binary Accuracy", fontsize=12, weight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Precision & Recall
    axes[1, 0].plot(epochs_range, history_dict["precision"], "g-o", label="Train Precision")
    axes[1, 0].plot(epochs_range, history_dict["val_precision"], "g--s", label="Val Precision")
    axes[1, 0].plot(epochs_range, history_dict["recall"], "m-o", label="Train Recall")
    axes[1, 0].plot(epochs_range, history_dict["val_recall"], "m--s", label="Val Recall")
    axes[1, 0].set_title("Precision and Recall", fontsize=12, weight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # AUC
    axes[1, 1].plot(epochs_range, history_dict["auc"], "b-o", label="Train AUC")
    axes[1, 1].plot(epochs_range, history_dict["val_auc"], "r--s", label="Val AUC")
    axes[1, 1].set_title("ROC Area Under Curve (AUC)", fontsize=12, weight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves: {save_path}")


def evaluate_and_report(
    model: keras.Model,
    test_ds,
    eval_dir: Path
) -> dict:
    """
    Evaluates trained model on unseen test set, generates confusion matrix,
    classification report, and stores quantitative metrics.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("\nEvaluating model on test dataset...")
    y_true_list = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0).flatten()
        y_pred_probs.extend(preds)
        y_true_list.extend(labels.numpy().flatten())

    y_true = np.array(y_true_list, dtype=int)
    y_prob = np.array(y_pred_probs, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=["Uncracked (0)", "Cracked (1)"],
        digits=4
    )
    print("\n" + "=" * 50)
    print("TEST SET CLASSIFICATION REPORT")
    print("=" * 50)
    print(report_str)

    report_file = eval_dir / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("INFRASCAN CRACK CLASSIFICATION REPORT (SDNET2018)\n")
        f.write("=" * 50 + "\n")
        f.write(report_str)
        f.write(f"\nROC-AUC Score: {auc:.4f}\n")
    print(f"Saved classification report: {report_file}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Uncracked", "Cracked"]
    )
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title("Classification Confusion Matrix", fontsize=12, weight="bold")
    cm_path = eval_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot: {cm_path}")

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "total_test_samples": int(len(y_true))
    }
    metrics_file = eval_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON: {metrics_file}")

    return metrics


def train_classification_model(
    project_root: Path,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    subset_fraction: Optional[float] = None
):
    dataset_dir = project_root / "dataset" / "SDNET2018"
    splits_csv = project_root / "outputs" / "classification_splits.csv"
    model_dir = project_root / "models" / "classification"
    eval_dir = project_root / "outputs" / "model_evaluation" / "classification"

    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STARTING SDNET2018 CLASSIFICATION TRAINING PIPELINE")
    print("=" * 60)
    print(f"Target Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    if subset_fraction:
        print(f"Subset Fraction: {subset_fraction:.2%}")

    train_ds, val_ds, test_ds, class_weights, split_stats = get_classification_datasets(
        dataset_dir=dataset_dir,
        splits_csv=splits_csv,
        img_size=(224, 224),
        batch_size=batch_size,
        subset_fraction=subset_fraction
    )

    print(f"Class weights applied: {class_weights}")
    print(f"Split stats: {split_stats}")

    print("\nBuilding MobileNetV2 Transfer Learning Model...")
    model = build_crack_classifier(
        input_shape=(224, 224, 3),
        freeze_base=True,
        learning_rate=learning_rate
    )
    model.summary()

    best_model_path = model_dir / "best_classifier.keras"
    training_log_path = eval_dir / "training_log.csv"

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

    print("\nStarting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("\nTraining completed.")
    plot_training_history(
        history.history,
        eval_dir / "training_curves.png"
    )

    # Load best model weights for evaluation
    print(f"\nLoading best saved model from: {best_model_path} ...")
    best_model = keras.models.load_model(str(best_model_path))

    metrics = evaluate_and_report(best_model, test_ds, eval_dir)
    print("\nPHASE 3 COMPLETE: SDNET2018 CLASSIFICATION MODEL TRAINED AND EVALUATED.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train SDNET2018 Crack Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--subset", type=float, default=None, help="Optional subset fraction (e.g. 0.25)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    train_classification_model(
        project_root=project_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_fraction=args.subset
    )


if __name__ == "__main__":
    main()

