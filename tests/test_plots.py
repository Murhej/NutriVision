"""
Tests for src/visualization/plots.py

Checks that each plotting function:
  - completes without raising
  - produces the expected output file
  - output file is non-empty (>0 bytes)

All tests use tmp_path for isolation. Matplotlib is already configured with
the 'Agg' non-interactive backend in plots.py itself.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.visualization.plots import (
    plot_training_curves,
    plot_confusion_matrix,
    save_sample_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_history(n: int = 3) -> dict:
    return {
        "train_loss": [1.0 - 0.1 * i for i in range(n)],
        "train_top1": [50.0 + 5 * i for i in range(n)],
        "train_top3": [70.0 + 4 * i for i in range(n)],
        "val_top1":   [48.0 + 5 * i for i in range(n)],
        "val_top3":   [68.0 + 4 * i for i in range(n)],
    }


def _tiny_model(num_classes: int = 5) -> nn.Module:
    """Lightweight model: single linear layer usable on CPU."""
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, num_classes),
    )
    return model.eval()


def _dummy_dataset(n: int = 6, num_classes: int = 5):
    """Return a list-based dataset of (tensor, label) pairs."""
    class _DS:
        def __init__(self):
            self.data = [(torch.rand(3, 224, 224), i % num_classes) for i in range(n)]
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    return _DS()


# ---------------------------------------------------------------------------
# plot_training_curves
# ---------------------------------------------------------------------------

class TestPlotTrainingCurves:
    def test_creates_loss_png(self, tmp_path: Path):
        plot_training_curves(_minimal_history(), "resnet50", str(tmp_path))
        assert (tmp_path / "resnet50_loss.png").exists()

    def test_creates_accuracy_png(self, tmp_path: Path):
        plot_training_curves(_minimal_history(), "resnet50", str(tmp_path))
        assert (tmp_path / "resnet50_accuracy.png").exists()

    def test_output_files_are_nonempty(self, tmp_path: Path):
        plot_training_curves(_minimal_history(), "resnet50", str(tmp_path))
        assert (tmp_path / "resnet50_loss.png").stat().st_size > 0
        assert (tmp_path / "resnet50_accuracy.png").stat().st_size > 0

    def test_single_epoch_does_not_crash(self, tmp_path: Path):
        plot_training_curves(_minimal_history(n=1), "efficientnet_b0", str(tmp_path))
        assert (tmp_path / "efficientnet_b0_loss.png").exists()

    def test_many_epochs_do_not_crash(self, tmp_path: Path):
        plot_training_curves(_minimal_history(n=50), "resnet50", str(tmp_path))
        assert (tmp_path / "resnet50_loss.png").exists()


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------

class TestPlotConfusionMatrix:
    def test_creates_output_file(self, tmp_path: Path):
        classes = ["apple_pie", "pizza", "sushi"]
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 0, 2])
        out = tmp_path / "confusion.png"

        plot_confusion_matrix(y_true, y_pred, classes, str(out), "resnet50")

        assert out.exists()
        assert out.stat().st_size > 0

    def test_perfect_predictions_does_not_crash(self, tmp_path: Path):
        classes = ["a", "b", "c"]
        y = np.array([0, 1, 2, 0, 1, 2])
        out = tmp_path / "cm_perfect.png"

        plot_confusion_matrix(y, y, classes, str(out), "test_model")
        assert out.exists()

    def test_single_class_does_not_crash(self, tmp_path: Path):
        classes = ["only_class"]
        y = np.zeros(5, dtype=int)
        out = tmp_path / "cm_single.png"

        plot_confusion_matrix(y, y, classes, str(out), "test_model")
        assert out.exists()


# ---------------------------------------------------------------------------
# save_sample_predictions
# ---------------------------------------------------------------------------

class TestSaveSamplePredictions:
    def test_creates_output_file(self, tmp_path: Path):
        model = _tiny_model(num_classes=5)
        ds = _dummy_dataset(n=6, num_classes=5)
        class_names = [f"food_{i}" for i in range(5)]
        out = tmp_path / "preds.txt"

        save_sample_predictions(model, ds, torch.device("cpu"), class_names, str(out), n_samples=3)

        assert out.exists()

    def test_output_contains_true_label_header(self, tmp_path: Path):
        model = _tiny_model(num_classes=5)
        ds = _dummy_dataset(n=4, num_classes=5)
        class_names = [f"food_{i}" for i in range(5)]
        out = tmp_path / "preds.txt"

        save_sample_predictions(model, ds, torch.device("cpu"), class_names, str(out), n_samples=2)

        content = out.read_text(encoding="utf-8")
        assert "True Label" in content
        assert "Top-3 Predictions" in content

    def test_n_samples_larger_than_dataset_does_not_crash(self, tmp_path: Path):
        model = _tiny_model(num_classes=3)
        ds = _dummy_dataset(n=2, num_classes=3)
        class_names = ["a", "b", "c"]
        out = tmp_path / "preds.txt"

        save_sample_predictions(model, ds, torch.device("cpu"), class_names, str(out), n_samples=100)
        assert out.exists()
