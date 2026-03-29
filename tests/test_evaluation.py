"""
Tests for src/evaluation/analyzer.py and src/core/device.py

Covers:
    - topk_accuracy (top-1, top-3, edge cases)
    - set_seed (reproducibility)
    - get_device (returns a torch.device)
    - analyze_per_class_performance (integration, output file written)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# topk_accuracy
# ---------------------------------------------------------------------------

from src.core.device import topk_accuracy


def test_topk_accuracy_top1_perfect():
    """Logits with argmax == target should give 100% top-1."""
    logits = torch.eye(5)
    targets = torch.arange(5)
    acc = topk_accuracy(logits, targets, k=1)
    assert abs(acc - 100.0) < 1e-3


def test_topk_accuracy_top1_zero():
    """Logits that always predict class 0 when targets are 1..4 → 0% top-1."""
    logits = torch.zeros(4, 5)
    logits[:, 0] = 10.0
    targets = torch.arange(1, 5)
    acc = topk_accuracy(logits, targets, k=1)
    assert abs(acc - 0.0) < 1e-3


def test_topk_accuracy_top3_perfect():
    """If the correct class always has the highest logit, top-3 accuracy is 100%."""
    n = 10
    logits = torch.zeros(n, 10)
    targets = torch.arange(n)
    for i in range(n):
        logits[i, targets[i]] = 5.0
    acc = topk_accuracy(logits, targets, k=3)
    assert abs(acc - 100.0) < 1e-3


def test_topk_accuracy_top3_partial():
    """When half the samples have the correct class in top-3 → 50%."""
    logits = torch.zeros(4, 5)
    logits[0, 0] = 10.0
    logits[1, 1] = 10.0
    logits[2, 4] = 10.0  # correct is 2, predicted 4
    logits[3, 4] = 10.0  # correct is 3, predicted 4
    targets = torch.tensor([0, 1, 2, 3])
    acc = topk_accuracy(logits, targets, k=3)
    assert abs(acc - 50.0) < 1e-3


def test_topk_accuracy_single_class():
    logits = torch.tensor([[5.0]])
    targets = torch.tensor([0])
    acc = topk_accuracy(logits, targets, k=1)
    assert abs(acc - 100.0) < 1e-3


def test_topk_accuracy_k_exceeds_num_classes_raises():
    """torch.topk with k > num_classes raises a RuntimeError — document the behaviour."""
    logits = torch.zeros(4, 2)   # only 2 classes
    targets = torch.zeros(4, dtype=torch.long)
    with pytest.raises(RuntimeError):
        topk_accuracy(logits, targets, k=3)


def test_topk_accuracy_batch_of_one():
    logits = torch.zeros(1, 5)
    logits[0, 3] = 10.0
    targets = torch.tensor([3])
    assert abs(topk_accuracy(logits, targets, k=1) - 100.0) < 1e-3


@pytest.mark.parametrize("k", [1, 3])
def test_topk_accuracy_returns_float(k):
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    result = topk_accuracy(logits, targets, k=k)
    assert isinstance(result, float)
    assert 0.0 <= result <= 100.0


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

from src.core.device import set_seed


def test_set_seed_reproducibility():
    set_seed(123)
    t1 = torch.randn(10)
    set_seed(123)
    t2 = torch.randn(10)
    assert torch.allclose(t1, t2)


def test_set_seed_different_seeds_differ():
    set_seed(1)
    t1 = torch.randn(100)
    set_seed(2)
    t2 = torch.randn(100)
    assert not torch.allclose(t1, t2)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

from src.core.device import get_device


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_type_is_valid():
    device = get_device()
    assert device.type in ("cuda", "cpu", "mps")


# ---------------------------------------------------------------------------
# analyze_per_class_performance (integration with mocks)
# ---------------------------------------------------------------------------

from src.evaluation.analyzer import analyze_per_class_performance
from tests.conftest import NUM_TEST_CLASSES, TEST_CLASS_NAMES


def test_analyze_per_class_writes_output_file(
    tmp_runs_dir: Path,
    tmp_outputs_dir: Path,
):
    """
    Verifies that analyze_per_class_performance:
      - loads the report from runs_dir
      - writes per_class_performance.json to outputs_dir
      - output contains a "classes" list

    External I/O is fully mocked:
      - Food101 dataset → tiny MagicMock
      - DataLoader → list of single-sample batches
      - build_dataset_index → minimal list
    """
    # Build a minimal dataset index so the length-equality check passes
    fake_dataset_index = [
        {"index": i, "label_index": i, "label_name": TEST_CLASS_NAMES[i], "image_path": f"/fake/{i}.jpg"}
        for i in range(NUM_TEST_CLASSES)
    ]

    # Mock the Food101 dataset
    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=NUM_TEST_CLASSES)
    mock_ds.classes = TEST_CLASS_NAMES

    # Mock DataLoader to yield (inputs, labels) tuples
    fake_batches = [
        (torch.randn(1, 3, 224, 224), torch.tensor([i % NUM_TEST_CLASSES]))
        for i in range(NUM_TEST_CLASSES)
    ]

    with (
        patch("src.evaluation.analyzer.Food101", return_value=mock_ds),
        patch("src.evaluation.analyzer.DataLoader", return_value=fake_batches),
        patch("src.evaluation.analyzer.build_dataset_index", return_value=fake_dataset_index),
    ):
        result = analyze_per_class_performance(
            device_name="cpu",
            runs_dir=tmp_runs_dir,
            outputs_dir=tmp_outputs_dir,
        )

    out_file = tmp_outputs_dir / "per_class_performance.json"
    assert out_file.exists(), "per_class_performance.json was not written"

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "classes" in data
    assert isinstance(data["classes"], list)
    assert data["model_name"] == "resnet50"


def test_analyze_raises_when_report_missing(tmp_path: Path):
    """If report.json is absent, load_report should raise FileNotFoundError."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    # No report.json created → should fail cleanly

    with pytest.raises(FileNotFoundError, match="report"):
        analyze_per_class_performance(
            device_name="cpu",
            runs_dir=runs_dir,
            outputs_dir=outputs_dir,
        )


def test_analyze_raises_when_best_model_pth_missing(
    tmp_runs_dir: Path,
    tmp_outputs_dir: Path,
):
    """If best_model.pth is absent, loading the checkpoint should fail."""
    pth = tmp_runs_dir / "best_model.pth"
    if pth.exists():
        pth.unlink()

    # Mock Food101 and DataLoader so the failure is specifically about the missing .pth
    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=NUM_TEST_CLASSES)
    mock_ds.classes = TEST_CLASS_NAMES

    fake_index = [
        {"index": i, "label_index": i, "label_name": TEST_CLASS_NAMES[i], "image_path": f"/x/{i}.jpg"}
        for i in range(NUM_TEST_CLASSES)
    ]

    with (
        patch("src.evaluation.analyzer.Food101", return_value=mock_ds),
        patch("src.evaluation.analyzer.DataLoader", return_value=[]),
        patch("src.evaluation.analyzer.build_dataset_index", return_value=fake_index),
    ):
        with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
            analyze_per_class_performance(
                device_name="cpu",
                runs_dir=tmp_runs_dir,
                outputs_dir=tmp_outputs_dir,
            )
