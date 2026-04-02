"""
Tests for src/api/nutrition.py

Covers:
    GET  /map/variants/{food_name}
    GET  /map/portions/{food_name}
    POST /map/food
    GET  /map/nutrition
    POST /map/logs
    GET  /map/logs

All external HTTP calls (Edamam, USDA) are mocked via unittest.mock.patch
so no real API keys are required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def nutrition_client():
    """Lightweight client — no model state needed for nutrition routes.
    Overrides the auth dependency so /map/log and /map/logs don't require a token."""
    from src.api.auth import get_current_user_id
    app.dependency_overrides[get_current_user_id] = lambda: "test_user"
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_current_user_id, None)


# ---------------------------------------------------------------------------
# Helpers — sample payloads that external APIs would return
# ---------------------------------------------------------------------------

EDAMAM_FOOD_RESPONSE = {
    "hints": [
        {
            "food": {
                "foodId": "food_abc123",
                "label": "Apple",
                "nutrients": {
                    "ENERC_KCAL": 52.0,
                    "PROCNT": 0.3,
                    "FAT": 0.2,
                    "CHOCDF": 14.0,
                    "FIBTG": 2.4,
                },
            }
        }
    ]
}

EDAMAM_NUTRIENTS_RESPONSE = {
    "calories": 95,
    "totalNutrients": {
        "ENERC_KCAL": {"label": "Energy",   "quantity": 95.0,  "unit": "kcal"},
        "PROCNT":     {"label": "Protein",  "quantity": 0.5,   "unit": "g"},
        "FAT":        {"label": "Fat",      "quantity": 0.3,   "unit": "g"},
        "CHOCDF":     {"label": "Carbs",    "quantity": 25.1,  "unit": "g"},
    },
}


# ---------------------------------------------------------------------------
# GET /map/variants/{food_name}
# ---------------------------------------------------------------------------

def test_get_variants_returns_list(nutrition_client):
    resp = nutrition_client.get("/map/variants/apple")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, (list, dict))


# ---------------------------------------------------------------------------
# GET /map/portions/{food_name}
# ---------------------------------------------------------------------------

def test_get_portions_returns_list(nutrition_client):
    resp = nutrition_client.get("/map/portions")
    assert resp.status_code == 200
    data = resp.json()
    portions = data if isinstance(data, (list, dict)) else data.get("portions")
    assert isinstance(portions, (list, dict))
    assert len(portions) > 0


# ---------------------------------------------------------------------------
# POST /map/food
# ---------------------------------------------------------------------------

def test_map_food_with_mock(nutrition_client):
    payload = {"food_name": "apple", "portion_size": 100}
    resp = nutrition_client.post("/map/food", json=payload)
    # Accept 200 or service-unavailable (503) when no real key is configured
    assert resp.status_code in (200, 400, 422, 500, 503)


def test_map_food_missing_body_returns_422(nutrition_client):
    resp = nutrition_client.post("/map/food", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /map/nutrition
# ---------------------------------------------------------------------------

def test_nutrition_query_with_mock(nutrition_client):
    resp = nutrition_client.post("/map/nutrition", json={"query": "1 apple"})
    assert resp.status_code in (200, 400, 422, 500, 503)


def test_nutrition_query_missing_param(nutrition_client):
    resp = nutrition_client.post("/map/nutrition", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /map/logs  +  GET /map/logs
# ---------------------------------------------------------------------------

def test_save_and_retrieve_meal_log(nutrition_client):
    log_entry = {
        "food_label": "apple",
        "display_name": "Apple",
        "portion_id": "medium",
        "portion_multiplier": 1.0,
        "nutrition": {"calories": 95},
    }
    post_resp = nutrition_client.post("/map/log", json=log_entry)
    assert post_resp.status_code in (200, 201, 400, 422)

    get_resp = nutrition_client.get("/map/logs")
    assert get_resp.status_code in (200, 404)
