"""Training visualizations: curves, confusion matrix, sample grids, predictions."""

from __future__ import annotations

import os
import random
from typing import List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — prevents GUI pop-ups
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset


def plot_class_distribution(dataset, output_path: str, title: str = "Food-101 Class Distribution") -> None:
    """Save a bar chart of class distribution (assumes uniform Food-101 layout)."""
    num_classes = len(dataset.classes)
    samples_per_class = len(dataset) // num_classes
    plt.ioff()
    plt.figure(figsize=(20, 6))
    plt.bar(range(num_classes), [samples_per_class] * num_classes, color="steelblue", alpha=0.8)
    plt.xlabel("Food Classes", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(range(num_classes), dataset.classes, rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved class distribution to {output_path}")


def plot_sample_grid(dataset, output_path: str, n_samples: int = 16) -> None:
    """Save a grid of sample images with their class labels."""
    print("Generating sample image grid...")
    original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    rows = int(np.sqrt(n_samples))
    cols = (n_samples + rows - 1) // rows
    plt.ioff()
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten() if n_samples > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx < len(indices):
            img, label = dataset[indices[idx]]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = np.clip(std * img + mean, 0, 1)
            ax.imshow(img)
            ax.set_title(original_dataset.classes[label], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved sample grid to {output_path}")


def plot_training_curves(history: dict, model_name: str, output_dir: str) -> None:
    """Save loss curve and accuracy curve for a training run."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"{model_name} — Training Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, f"{model_name}_loss.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {loss_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_top1"], "b-", label="Train Top-1", linewidth=2)
    plt.plot(epochs, history["train_top3"], "b--", label="Train Top-3", linewidth=2)
    plt.plot(epochs, history["val_top1"], "r-", label="Val Top-1", linewidth=2)
    plt.plot(epochs, history["val_top3"], "r--", label="Val Top-3", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(f"{model_name} — Accuracy", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(output_dir, f"{model_name}_accuracy.png")
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy curve to {acc_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    output_path: str,
    model_name: str,
) -> None:
    """Save a confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    step = max(1, len(classes) // 20)
    plt.xticks(tick_marks[::step], [classes[i] for i in tick_marks[::step]], rotation=90, fontsize=6)
    plt.yticks(tick_marks[::step], [classes[i] for i in tick_marks[::step]], fontsize=6)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def save_sample_predictions(
    model,
    dataset,
    device: torch.device,
    class_names: List[str],
    output_path: str,
    n_samples: int = 5,
) -> None:
    """Write top-3 predictions for a sample of test images to a text file."""
    model.eval()
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Top-3 Predictions for Sample Test Images\n")
        f.write("=" * 60 + "\n\n")
        for idx in indices:
            img, true_label = dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                top3_prob, top3_idx = probs.topk(3, dim=1)
            f.write(f"Sample {idx}:\n")
            f.write(f"  True Label: {class_names[true_label]}\n")
            f.write("  Top-3 Predictions:\n")
            for i in range(3):
                f.write(
                    f"    {i + 1}. {class_names[top3_idx[0][i].item()]}: "
                    f"{top3_prob[0][i].item() * 100:.2f}%\n"
                )
            f.write("\n")

    print(f"Saved sample predictions to {output_path}")
