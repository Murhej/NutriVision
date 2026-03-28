"""
Tests for src/core/model.py

Covers:
    - build_model output shapes for all 7 architectures
    - load_checkpoint (weights_only=True)
    - forward_with_tta (1, 2, 4 views)
    - format_top3_predictions ordering
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.core.model import (
    SUPPORTED_MODELS,
    build_model,
    format_top3_predictions,
    forward_with_tta,
    load_checkpoint,
)

NUM_CLASSES = 101
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def test_build_model_resnet50_output_shape():
    model = build_model("resnet50", NUM_CLASSES).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, NUM_CLASSES)


def test_build_model_efficientnet_b0_output_shape():
    model = build_model("efficientnet_b0", NUM_CLASSES).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, NUM_CLASSES)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_build_all_supported_architectures(model_name: str):
    """Every supported architecture should produce (1, NUM_CLASSES) logits."""
    model = build_model(model_name, NUM_CLASSES).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, NUM_CLASSES), f"Shape mismatch for {model_name}"


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------

def test_load_checkpoint_weights_only(tmp_path: Path):
    """Checkpoint should be loadable with weights_only=True (no pickle exploit)."""
    model = build_model("resnet50", NUM_CLASSES)
    ckpt = tmp_path / "test_ckpt.pth"
    torch.save(model.state_dict(), ckpt)

    fresh = build_model("resnet50", NUM_CLASSES)
    loaded = load_checkpoint(fresh, ckpt, DEVICE)
    assert loaded is fresh  # in-place modification returns the same object

    # Verify weights match
    original_sd = model.state_dict()
    loaded_sd = loaded.state_dict()
    for key in original_sd:
        assert torch.allclose(original_sd[key].float(), loaded_sd[key].float())


def test_load_checkpoint_wrong_architecture_raises(tmp_path: Path):
    """Loading weights from one architecture into another should raise."""
    resnet = build_model("resnet50", NUM_CLASSES)
    ckpt = tmp_path / "resnet.pth"
    torch.save(resnet.state_dict(), ckpt)

    effnet = build_model("efficientnet_b0", NUM_CLASSES)
    with pytest.raises(RuntimeError):
        load_checkpoint(effnet, ckpt, DEVICE)


# ---------------------------------------------------------------------------
# forward_with_tta
# ---------------------------------------------------------------------------

def test_forward_with_tta_single_view(mock_model: nn.Module):
    """With tta_views=1, output equals single forward pass."""
    model = mock_model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        tta_out = forward_with_tta(model, x, DEVICE, tta_views=1)
        direct_out = model(x)
    assert torch.allclose(tta_out, direct_out, atol=1e-5)


def test_forward_with_tta_two_views_different_from_one(mock_model: nn.Module):
    """With tta_views=2, result should differ from single-view (unless weights are trivial)."""
    model = mock_model.eval()
    # Use a non-symmetric image so the horizontal flip matters
    x = torch.rand(1, 3, 224, 224)
    x[:, :, :, :112] = 0.0  # left half zero, right half random
    with torch.no_grad():
        out1 = forward_with_tta(model, x, DEVICE, tta_views=1)
        out2 = forward_with_tta(model, x, DEVICE, tta_views=2)
    # They should differ because flip changes input
    assert not torch.allclose(out1, out2, atol=1e-6)


def test_forward_with_tta_four_views_shape(mock_model: nn.Module):
    """Output shape should match (batch, num_classes) regardless of tta_views."""
    model = mock_model.eval()
    x = torch.randn(1, 3, 224, 224)
    num_classes = model.fc.out_features
    with torch.no_grad():
        out = forward_with_tta(model, x, DEVICE, tta_views=4)
    assert out.shape == (1, num_classes)


# ---------------------------------------------------------------------------
# format_top3_predictions
# ---------------------------------------------------------------------------

def test_format_predictions_top3_order():
    """Predictions should be rank-ordered by confidence (descending)."""
    n = 5
    logits = torch.zeros(1, n)
    logits[0, 2] = 10.0   # class 2 should be rank 1
    logits[0, 0] = 5.0    # class 0 should be rank 2
    logits[0, 4] = 2.0    # class 4 should be rank 3

    classes = [f"c{i}" for i in range(n)]
    preds = format_top3_predictions(logits, classes)

    assert len(preds) == 3
    assert preds[0]["rank"] == 1
    assert preds[0]["class"] == "c2"
    assert preds[1]["class"] == "c0"
    assert preds[2]["class"] == "c4"
    # Confidences should be descending
    assert preds[0]["confidence"] >= preds[1]["confidence"] >= preds[2]["confidence"]


def test_format_predictions_confidences_sum_near_100():
    """Top-3 probabilities (if dominant) should be most of the distribution."""
    logits = torch.tensor([[100.0, 80.0, 60.0, 0.0, 0.0]])
    classes = [f"c{i}" for i in range(5)]
    preds = format_top3_predictions(logits, classes)
    total = sum(p["confidence"] for p in preds)
    assert total > 95.0  # top-3 dominate


# ---------------------------------------------------------------------------
# build_model unknown name
# ---------------------------------------------------------------------------

def test_build_model_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("not_a_real_model", NUM_CLASSES)
