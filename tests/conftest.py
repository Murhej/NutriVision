"""
Shared pytest fixtures for the NutriVision test suite.

Fixtures:
    mock_model        — 5-class ResNet50 with random weights, CPU-only
    test_image_pil    — 224×224 random RGB PIL image
    test_image_bytes  — same image encoded as JPEG bytes
    tmp_runs_dir      — isolated temporary runs/ directory with a minimal report.json
    tmp_outputs_dir   — isolated temporary outputs/ directory
    api_client        — FastAPI TestClient with model/dataset state pre-patched
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# Model fixture
# ---------------------------------------------------------------------------

NUM_TEST_CLASSES = 5
TEST_CLASS_NAMES = [f"food_{i:02d}" for i in range(NUM_TEST_CLASSES)]


@pytest.fixture(scope="session")
def mock_model() -> nn.Module:
    """5-class ResNet50 with random weights, kept in eval mode on CPU."""
    from torchvision import models
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_TEST_CLASSES)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_image_pil() -> Image.Image:
    """224×224 random RGB PIL image for use in model and API tests."""
    random_pixels = [random.randint(0, 255) for _ in range(224 * 224 * 3)]
    return Image.frombytes("RGB", (224, 224), bytes(random_pixels))


@pytest.fixture(scope="session")
def test_image_bytes(test_image_pil: Image.Image) -> bytes:
    """The test image encoded as JPEG bytes (for upload simulation)."""
    buf = io.BytesIO()
    test_image_pil.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_runs_dir(tmp_path: Path, mock_model: nn.Module) -> Path:
    """
    Temporary runs/ directory containing:
      - best_model.pth  (saved from mock_model)
      - report.json     (minimal valid training report)
    """
    runs = tmp_path / "runs"
    runs.mkdir()

    # Save model weights
    torch.save(mock_model.state_dict(), runs / "best_model.pth")

    # Write a minimal report
    report: Dict = {
        "best_model_name": "resnet50",
        "class_names": TEST_CLASS_NAMES,
        "num_classes": NUM_TEST_CLASSES,
        "best_model_metrics": {
            "val_top1_accuracy": 71.0,
            "val_top3_accuracy": 86.0,
            "test_top1_accuracy": 70.5,
            "test_top3_accuracy": 85.5,
        },
        "config": {
            "eval_tta": False,
            "tta_num_views": 1,
        },
        "timestamp": "2025-01-01T00:00:00",
    }
    (runs / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return runs


@pytest.fixture()
def tmp_outputs_dir(tmp_path: Path) -> Path:
    """Temporary outputs/ directory (empty, tests can write into it)."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return outputs


# ---------------------------------------------------------------------------
# API client fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def api_client(tmp_runs_dir: Path, tmp_outputs_dir: Path, mock_model: nn.Module, test_image_pil: Image.Image):
    """
    FastAPI TestClient with the app's internal _state pre-loaded so that
    no real model files or datasets are needed.

    Uses starlette's TestClient (synchronous) which is safe for non-async tests.
    """
    from fastapi.testclient import TestClient
    from torchvision import transforms

    from src.api.app import _state, app

    # Build a minimal dataset index (5 classes, 1 image path each — pointing to
    # a real temp JPEG so /dataset/image/{index} can serve an actual file).
    import io

    image_dir = tmp_outputs_dir / "images"
    image_dir.mkdir()

    dataset_index = []
    for i, class_name in enumerate(TEST_CLASS_NAMES):
        img_path = image_dir / f"sample_{i:03d}.jpg"
        test_image_pil.save(img_path, format="JPEG")
        dataset_index.append({
            "index": i,
            "label_index": i,
            "label_name": class_name,
            "image_path": str(img_path),
        })

    # Override paths used by inference.py so they point to our tmp dirs
    import src.api.inference as _inf
    _inf.RUNS_DIR = tmp_runs_dir
    _inf.OUTPUTS_DIR = tmp_outputs_dir

    # Write the per-class performance cache so /analysis/per-class works
    perf_data = {
        "model_name": "resnet50",
        "total_classes": NUM_TEST_CLASSES,
        "total_test_samples": NUM_TEST_CLASSES,
        "overall_top1_accuracy": 70.0,
        "overall_top3_accuracy": 85.0,
        "classes": [
            {"class_name": cn, "class_index": i, "top1_accuracy": 70.0, "top3_accuracy": 85.0,
             "avg_confidence": 75.0, "top1_correct": 7, "top3_correct": 8, "total": 10}
            for i, cn in enumerate(TEST_CLASS_NAMES)
        ],
    }
    (tmp_outputs_dir / "per_class_performance.json").write_text(
        json.dumps(perf_data), encoding="utf-8"
    )

    # Patch _load_model_and_config so the lifespan doesn't overwrite our mock state
    with patch("src.api.inference._load_model_and_config", return_value={}):
        with TestClient(app) as client:
            # Set state AFTER lifespan (patch prevented real model loading)
            _state["model"] = mock_model
            _state["device"] = torch.device("cpu")
            _state["transform"] = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            _state["class_names"] = TEST_CLASS_NAMES
            _state["dataset_class_names"] = TEST_CLASS_NAMES
            _state["dataset"] = MagicMock()
            _state["dataset"].__len__ = MagicMock(return_value=len(dataset_index))
            _state["dataset_index"] = dataset_index
            _state["tta_views"] = 1
            yield client

    # Clean up state after the test
    _state.clear()
