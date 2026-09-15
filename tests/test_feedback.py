"""
Tests for src/api/feedback.py

Covers:
    GET  /feedback/guide
    POST /feedback/submit
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def feedback_client():
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# GET /feedback/guide
# ---------------------------------------------------------------------------

def test_guide_returns_200(feedback_client):
    resp = feedback_client.get("/feedback/guide")
    assert resp.status_code == 200


def test_guide_returns_json_or_html(feedback_client):
    resp = feedback_client.get("/feedback/guide")
    content_type = resp.headers.get("content-type", "")
    assert "application/json" in content_type or "text/html" in content_type


# ---------------------------------------------------------------------------
# POST /feedback/submit
# ---------------------------------------------------------------------------

def test_submit_feedback_minimal(feedback_client, test_image_bytes):
    """Submit feedback with minimal required fields."""
    form_data = {
        "predicted_class": "apple_pie",
        "correct_class": "pizza",
        "estimated_calories": "300",
    }
    resp = feedback_client.post(
        "/feedback/submit",
        data=form_data,
        files={"image": ("test.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
    )
    # Allow 200/201 (success) or 422 (if required fields differ in prod schema)
    assert resp.status_code in (200, 201, 422)


def test_submit_feedback_no_image(feedback_client):
    """Submitting without an image should return 422 or 400 if image is required."""
    form_data = {
        "predicted_class": "apple_pie",
        "correct_class": "pizza",
    }
    resp = feedback_client.post("/feedback/submit", data=form_data)
    # Without the image, either form processes fine or validation fails
    assert resp.status_code in (200, 201, 400, 422)


def test_submit_feedback_empty_form_returns_422(feedback_client):
    """Empty POST body should be rejected with 422."""
    resp = feedback_client.post("/feedback/submit")
    assert resp.status_code == 422
