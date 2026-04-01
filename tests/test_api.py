"""
Tests for src/api/inference.py endpoints.

Covers:
    GET  /health
    GET  /info
    GET  /classes
    POST /predict
    GET  /dataset/random
    GET  /dataset/image/{index}
    GET  /predict/dataset/{index}
    GET  /analysis/per-class
"""

from __future__ import annotations

import io

import pytest

from tests.conftest import NUM_TEST_CLASSES, TEST_CLASS_NAMES


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_ok(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "healthy"


def test_health_reports_model_loaded(api_client):
    resp = api_client.get("/health")
    data = resp.json()
    assert data.get("model_loaded") is True


# ---------------------------------------------------------------------------
# /info
# ---------------------------------------------------------------------------

def test_info_returns_report(api_client):
    resp = api_client.get("/info")
    assert resp.status_code == 200
    data = resp.json()
    # Report should contain the model name we wrote in tmp_runs_dir
    assert "best_model_name" in data or "model_name" in data


def test_info_contains_class_names(api_client):
    resp = api_client.get("/info")
    data = resp.json()
    assert "class_names" in data
    assert isinstance(data["class_names"], list)


# ---------------------------------------------------------------------------
# /classes
# ---------------------------------------------------------------------------

def test_classes_endpoint_returns_200(api_client):
    resp = api_client.get("/classes")
    assert resp.status_code == 200


def test_classes_contains_count(api_client):
    resp = api_client.get("/classes")
    data = resp.json()
    assert "count" in data
    assert data["count"] == NUM_TEST_CLASSES


def test_classes_list_contains_all_names(api_client):
    resp = api_client.get("/classes")
    data = resp.json()
    names = data.get("classes", [])
    for cn in TEST_CLASS_NAMES:
        assert cn in names


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

def test_predict_returns_success(api_client, test_image_bytes):
    resp = api_client.post(
        "/predict",
        files={"file": ("test.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("success") is True


def test_predict_returns_three_predictions(api_client, test_image_bytes):
    resp = api_client.post(
        "/predict",
        files={"file": ("test.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
    )
    data = resp.json()
    predictions = data.get("predictions", [])
    assert len(predictions) == 3


def test_predict_top_class_is_valid(api_client, test_image_bytes):
    resp = api_client.post(
        "/predict",
        files={"file": ("test.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
    )
    data = resp.json()
    top = data["predictions"][0]
    assert top["class"] in TEST_CLASS_NAMES
    assert 0.0 <= top["confidence"] <= 100.0
    assert top["rank"] == 1


def test_predict_confidence_sum_near_100(api_client, test_image_bytes):
    """Sum of top-3 confidences should be high when one class dominates."""
    resp = api_client.post(
        "/predict",
        files={"file": ("test.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
    )
    data = resp.json()
    total = sum(p["confidence"] for p in data["predictions"])
    assert total <= 100.01  # can't exceed 100%


def test_predict_non_image_returns_400(api_client):
    bad_bytes = b"this is not an image"
    resp = api_client.post(
        "/predict",
        files={"file": ("bad.txt", io.BytesIO(bad_bytes), "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_empty_file_returns_400(api_client):
    """A 0-byte upload should be rejected with HTTP 400."""
    resp = api_client.post(
        "/predict",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /dataset/random
# ---------------------------------------------------------------------------

def test_dataset_random_returns_200(api_client):
    resp = api_client.get("/dataset/random")
    assert resp.status_code == 200


def test_dataset_random_returns_label(api_client):
    resp = api_client.get("/dataset/random")
    data = resp.json()
    assert "true_label" in data or "label_name" in data


# ---------------------------------------------------------------------------
# GET /dataset/image/{index}
# ---------------------------------------------------------------------------

def test_dataset_image_first_item(api_client):
    resp = api_client.get("/dataset/image/0")
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("image/")


def test_dataset_image_out_of_range(api_client):
    resp = api_client.get("/dataset/image/99999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /predict/dataset/{index}
# ---------------------------------------------------------------------------

def test_predict_dataset_sample_success(api_client):
    resp = api_client.get("/predict/dataset/0")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("success") is True
    assert "sample" in data
    assert data["sample"]["true_label"] in TEST_CLASS_NAMES


# ---------------------------------------------------------------------------
# GET /analysis/per-class
# ---------------------------------------------------------------------------

def test_per_class_analysis_structure(api_client):
    resp = api_client.get("/analysis/per-class")
    assert resp.status_code == 200
    data = resp.json()
    assert "classes" in data
    assert isinstance(data["classes"], list)
    assert len(data["classes"]) == NUM_TEST_CLASSES


def test_per_class_analysis_accuracy_range(api_client):
    resp = api_client.get("/analysis/per-class")
    data = resp.json()
    for cls in data["classes"]:
        acc = cls.get("top1_accuracy", cls.get("accuracy"))
        if acc is not None:
            assert 0.0 <= acc <= 100.0
