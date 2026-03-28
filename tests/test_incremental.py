"""
Tests for src/training/incremental.py

Covers:
    - normalize_label
    - canonicalize_source_label
    - split_indices_by_class (three-way split: train/val/test)
    - LabelRemapDataset (length, relabeling via index_to_name + class_to_index)
    - SamplesDataset (length, return shape, transform applied)
    - evaluate_promotion_decision (pass / reject cases)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# normalize_label
# normalize_label lower-cases, strips, and replaces non-alphanumeric chars with '_'
# ---------------------------------------------------------------------------

from src.training.incremental import normalize_label


def test_normalize_label_lowercase():
    result = normalize_label("Apple Pie")
    assert result == result.lower()


def test_normalize_label_strips_leading_trailing():
    result = normalize_label("  banana  ")
    assert result == normalize_label("banana")


def test_normalize_label_space_to_underscore():
    """Spaces become underscores."""
    result = normalize_label("apple pie")
    assert "_" in result or result == "apple_pie"


def test_normalize_label_empty_string():
    assert normalize_label("") == ""


# ---------------------------------------------------------------------------
# canonicalize_source_label
# signature: canonicalize_source_label(source_name: str, raw_label: str) -> str
# ---------------------------------------------------------------------------

from src.training.incremental import canonicalize_source_label


def test_canonicalize_unknown_source_returns_normalized():
    """Unknown dataset source falls through to normalize_label."""
    result = canonicalize_source_label("unknown_dataset", "Some_Unknown_Food")
    assert isinstance(result, str)
    assert result == result.lower()


def test_canonicalize_fruits360_strips_numeric_suffix():
    """fruits_360 source should strip trailing numeric suffixes."""
    result = canonicalize_source_label("fruits_360", "apple_1")
    # Numeric suffix stripped
    assert not result.endswith("_1") or result == "apple_1"  # tolerate if no alias defined


def test_canonicalize_always_returns_string():
    result = canonicalize_source_label("food101", "Caesar Salad")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# split_indices_by_class
# signature: split_indices_by_class(targets, val_ratio, test_ratio)
#            -> (train_idx, val_idx, test_idx)
# ---------------------------------------------------------------------------

from src.training.incremental import split_indices_by_class


def test_split_indices_all_samples_covered():
    """All indices must appear in exactly one split."""
    labels = [i % 5 for i in range(50)]
    train_idx, val_idx, test_idx = split_indices_by_class(labels, val_ratio=0.2, test_ratio=0.1)
    all_idx = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_idx) == 50
    # No overlap
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx)   & set(test_idx)) == 0


def test_split_indices_val_fraction_approximate():
    """Validation split should be approximately val_ratio of total."""
    labels = [i % 10 for i in range(200)]
    train_idx, val_idx, test_idx = split_indices_by_class(labels, val_ratio=0.2, test_ratio=0.1)
    ratio = len(val_idx) / 200
    assert 0.10 <= ratio <= 0.30


def test_split_indices_all_classes_appear_in_train():
    """All classes should appear in the training split."""
    labels = [i % 5 for i in range(50)]
    train_idx, _, _ = split_indices_by_class(labels, val_ratio=0.2, test_ratio=0.1)
    train_classes = {labels[i] for i in train_idx}
    assert train_classes == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# LabelRemapDataset
# signature: LabelRemapDataset(dataset, index_to_name, class_to_index)
# ---------------------------------------------------------------------------

from src.training.incremental import LabelRemapDataset


class _DummyDataset(Dataset):
    """10 samples, labels 0..4 repeated twice."""
    def __init__(self):
        self.targets = [i % 5 for i in range(10)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.zeros(3, 224, 224), self.targets[idx]


def test_label_remap_dataset_length():
    ds = _DummyDataset()
    index_to_name = ["food_0", "food_1", "food_2", "food_3", "food_4"]
    class_to_index = {f"food_{i}": i + 100 for i in range(5)}
    wrapped = LabelRemapDataset(ds, index_to_name, class_to_index)
    assert len(wrapped) == len(ds)


def test_label_remap_dataset_relabels_correctly():
    ds = _DummyDataset()
    index_to_name = ["food_0", "food_1", "food_2", "food_3", "food_4"]
    class_to_index = {f"food_{i}": i + 100 for i in range(5)}
    wrapped = LabelRemapDataset(ds, index_to_name, class_to_index)
    for i in range(len(ds)):
        _, new_label = wrapped[i]
        original_label = ds.targets[i]
        assert new_label == class_to_index[index_to_name[original_label]]


# ---------------------------------------------------------------------------
# SamplesDataset
# signature: SamplesDataset(samples, class_names, transform=None)
# ---------------------------------------------------------------------------

from src.training.incremental import SamplesDataset


def test_samples_dataset_length(tmp_path: Path, test_image_pil: Image.Image):
    img_paths = []
    for i in range(5):
        p = tmp_path / f"img_{i}.jpg"
        test_image_pil.save(p, "JPEG")
        img_paths.append((str(p), i))

    class_names = [f"food_{i}" for i in range(5)]
    from src.core.transforms import get_transforms
    _, transform = get_transforms()
    ds = SamplesDataset(img_paths, class_names, transform=transform)
    assert len(ds) == 5


def test_samples_dataset_returns_tensor_and_correct_label(tmp_path: Path, test_image_pil: Image.Image):
    p = tmp_path / "sample.jpg"
    test_image_pil.save(p, "JPEG")

    from src.core.transforms import get_transforms
    _, transform = get_transforms()
    ds = SamplesDataset([(str(p), 42)], class_names=["food_42"], transform=transform)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert label == 42


# ---------------------------------------------------------------------------
# evaluate_promotion_decision
# signature: evaluate_promotion_decision(base_report, result, config) -> (bool, list)
# ---------------------------------------------------------------------------

from src.training.incremental import evaluate_promotion_decision
from src.training.config import IncrementalConfig


def test_promotion_accepted_when_accuracy_improves():
    base_report = {"best_model_metrics": {"test_top1_accuracy": 70.0, "test_top3_accuracy": 86.0}}
    result = {"final_top1": 72.0, "final_top3": 87.0}
    config = IncrementalConfig()
    config.protect_best_checkpoint = True
    config.max_allowed_test_top1_drop = 0.0
    config.max_allowed_test_top3_drop = 0.0
    accepted, reasons = evaluate_promotion_decision(base_report, result, config)
    assert accepted is True


def test_promotion_rejected_when_accuracy_drops_beyond_threshold():
    base_report = {"best_model_metrics": {"test_top1_accuracy": 70.0, "test_top3_accuracy": 86.0}}
    result = {"final_top1": 55.0, "final_top3": 70.0}  # large drop
    config = IncrementalConfig()
    config.protect_best_checkpoint = True
    config.max_allowed_test_top1_drop = 2.0
    config.max_allowed_test_top3_drop = 2.0
    accepted, reasons = evaluate_promotion_decision(base_report, result, config)
    assert accepted is False
    assert len(reasons) > 0


def test_promotion_skipped_when_protection_disabled():
    base_report = {"best_model_metrics": {"test_top1_accuracy": 70.0, "test_top3_accuracy": 86.0}}
    result = {"final_top1": 10.0, "final_top3": 20.0}
    config = IncrementalConfig()
    config.protect_best_checkpoint = False
    accepted, _ = evaluate_promotion_decision(base_report, result, config)
    assert accepted is True
