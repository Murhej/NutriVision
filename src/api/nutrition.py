"""
/map/* router — nutrition data, variants, portions, and meal logging.
All URL paths are identical to the original api_mapper.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.food_mapper import (
    FOOD_VARIANTS,
    _normalize_label,
    build_candidate_queries,
    build_external_api_queries,
    build_follow_up_questions,
    fetch_nutrition,
    get_last_nutrition_error,
    handle_unknown_food,
    scale_nutrition,
)

mapper_router = APIRouter(prefix="/map", tags=["Nutrition Mapping"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MEAL_LOG_PATH = OUTPUTS_DIR / "meal_logs.json"

PORTION_PRESETS = {
    "small": {"label": "Small", "multiplier": 0.68, "grams": 115, "ounces": 4},
    "medium": {"label": "Medium", "multiplier": 1.0, "grams": 170, "ounces": 6},
    "large": {"label": "Large", "multiplier": 1.32, "grams": 225, "ounces": 8},
    "extra_large": {"label": "Extra Large", "multiplier": 1.68, "grams": 285, "ounces": 10},
}


class FoodMappingRequest(BaseModel):
    food_label: str = Field(...)
    user_description: Optional[str] = Field(default=None)
    variants: Optional[Dict] = Field(default_factory=dict)
    portion_id: str = Field(default="medium")
    portion_multiplier: Optional[float] = Field(default=None)


class NutritionQueryRequest(BaseModel):
    query: str = Field(...)


class MealLogRequest(BaseModel):
    food_label: str
    display_name: Optional[str] = None
    comment: Optional[str] = None
    portion_id: str = "medium"
    portion_label: Optional[str] = None
    portion_multiplier: float = 1.0
    nutrition: Dict = Field(default_factory=dict)
    prediction: Optional[Dict] = None
    source: Optional[str] = None
    image_url: Optional[str] = None


def _resolve_portion(portion_id: str, multiplier: Optional[float]) -> Dict:
    preset = PORTION_PRESETS.get(portion_id, PORTION_PRESETS["medium"]).copy()
    if multiplier is not None:
        preset["multiplier"] = max(0.25, min(float(multiplier), 4.0))
    preset["id"] = portion_id if portion_id in PORTION_PRESETS else "medium"
    return preset


def _append_meal_log(entry: Dict) -> Dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if MEAL_LOG_PATH.exists():
        with open(MEAL_LOG_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []
    existing.append(entry)
    with open(MEAL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return entry


@mapper_router.get("/variants/{food_label}")
def get_variants(food_label: str):
    key = _normalize_label(food_label)
    questions = FOOD_VARIANTS.get(key, [])
    return {"food_label": food_label, "needs_variants": len(questions) > 0, "questions": questions}


@mapper_router.get("/portions")
def get_portion_presets():
    return {"portions": PORTION_PRESETS}


@mapper_router.post("/food")
def map_food(request: FoodMappingRequest):
    label = request.food_label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Food label cannot be empty.")

    key = _normalize_label(label)
    portion = _resolve_portion(request.portion_id, request.portion_multiplier)
    queries = build_candidate_queries(label, request.variants or {}, request.user_description)
    questions = FOOD_VARIANTS.get(key, [])

    for query in queries:
        nutrition = fetch_nutrition(query)
        if nutrition:
            return {
                "status": "found",
                "food_label": label,
                "display_name": label.replace("_", " "),
                "query_used": query,
                "queries_tried": queries,
                "variants_selected": request.variants or {},
                "questions": questions,
                "portion": portion,
                "base_nutrition": nutrition,
                "nutrition": scale_nutrition(nutrition, portion["multiplier"]),
            }

    last_error = get_last_nutrition_error()
    if last_error and last_error.get("kind") in {"network", "dependency", "auth", "rate_limit", "http_error"}:
        return {
            "status": "provider_error",
            "food_label": label,
            "display_name": label.replace("_", " "),
            "message": last_error.get("message") or "Could not reach the nutrition provider.",
            "provider": last_error.get("provider") or "nutrition_provider",
            "query_used": last_error.get("query"),
            "queries_tried": queries,
            "portion": portion,
            "questions": questions,
            "follow_up_questions": build_follow_up_questions(label, queries),
            "external_api_queries": build_external_api_queries(label, queries, request.user_description),
        }

    response = handle_unknown_food(request.user_description or label, attempted_queries=queries, user_description=request.user_description)
    response["queries_tried"] = queries
    response["portion"] = portion
    response["questions"] = questions
    return response


@mapper_router.post("/nutrition")
def get_nutrition_by_query(request: NutritionQueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    nutrition = fetch_nutrition(query)
    if nutrition:
        return {"status": "found", "nutrition": nutrition}

    last_error = get_last_nutrition_error()
    if last_error and last_error.get("kind") in {"network", "dependency", "auth", "rate_limit", "http_error"}:
        return {
            "status": "provider_error",
            "provider": last_error.get("provider") or "nutrition_provider",
            "message": last_error.get("message") or "Could not reach the nutrition provider.",
            "query_used": last_error.get("query") or query,
            "follow_up_questions": build_follow_up_questions(query, [query]),
            "external_api_queries": build_external_api_queries(query, [query]),
        }

    return {
        "status": "not_found",
        "message": f"No nutrition data found for '{query}'.",
        "follow_up_questions": build_follow_up_questions(query, [query]),
        "external_api_queries": build_external_api_queries(query, [query]),
        "tips": [
            "Try a more specific serving description",
            "Use common food names rather than dish names",
            "Search manually at https://www.edamam.com/",
        ],
    }


@mapper_router.post("/log")
def save_meal_log(request: MealLogRequest):
    if not request.food_label.strip():
        raise HTTPException(status_code=400, detail="Food label cannot be empty.")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "food_label": request.food_label,
        "display_name": request.display_name or request.food_label.replace("_", " "),
        "comment": request.comment,
        "portion_id": request.portion_id,
        "portion_label": request.portion_label or PORTION_PRESETS.get(request.portion_id, PORTION_PRESETS["medium"])["label"],
        "portion_multiplier": max(0.25, min(float(request.portion_multiplier), 4.0)),
        "nutrition": request.nutrition,
        "prediction": request.prediction,
        "source": request.source,
        "image_url": request.image_url,
    }
    return {"status": "saved", "entry": _append_meal_log(entry)}


@mapper_router.get("/logs")
def get_meal_logs(limit: int = 20):
    if not MEAL_LOG_PATH.exists():
        return {"entries": [], "count": 0}
    with open(MEAL_LOG_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    sliced = list(reversed(entries[-max(1, min(limit, 100)):]))
    return {"entries": sliced, "count": len(entries)}
