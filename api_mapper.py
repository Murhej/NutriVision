
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.food_mapper import (
    FOOD_VARIANTS,
    _normalize_label,
    build_query,
    fetch_nutrition,
    handle_unknown_food,
)

mapper_router = APIRouter(prefix="/map", tags=["Food Mapping"])


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------------------------

class FoodMappingRequest(BaseModel):
    """Client sends the AI-predicted food label + any variant answers."""
    food_label: str                      # e.g. "pizza"
    variants: Optional[dict] = {}        # e.g. {"size": "large", "crust": "thin crust", "type": "pepperoni", "slices": "2"}

class NutritionQueryRequest(BaseModel):
    """Direct natural-language nutrition query."""
    query: str                           # e.g. "2 slices large thin crust pepperoni pizza"

class UnknownFoodRequest(BaseModel):
    food_label: str
    user_description: Optional[str] = None  # what the user thinks it is


# ---------------------------------------------------------------------------
# ENDPOINT 1 — Get variant questions for a food (client polls this first)
# ---------------------------------------------------------------------------
@mapper_router.get("/variants/{food_label}")
def get_variants(food_label: str):
    """
    Returns the list of clarifying questions for a complex food.
    Returns an empty list for simple foods that don't need extra detail.

    Example:
      GET /map/variants/pizza
      → {"food_label": "pizza", "needs_variants": true, "questions": [...]}
    """
    key = _normalize_label(food_label)
    questions = FOOD_VARIANTS.get(key, [])
    return {
        "food_label":    food_label,
        "needs_variants": len(questions) > 0,
        "questions":     questions,
    }


# ---------------------------------------------------------------------------
# ENDPOINT 2 — Full food mapping (variant answers → nutrition)
# ---------------------------------------------------------------------------
@mapper_router.post("/food")
def map_food(request: FoodMappingRequest):
    """
    Main endpoint. Takes an AI-predicted food label + optional variant answers.

    Flow:
      1. If food has variants and none supplied → return the questions (client should ask user first)
      2. Build a precise NL query
      3. Fetch from Nutritionix
      4. If not found → return guidance

    Example request:
      {
        "food_label": "pizza",
        "variants": {
          "size":   "large (14\")",
          "crust":  "thin crust",
          "type":   "pepperoni",
          "slices": "2"
        }
      }
    """
    label = request.food_label.strip()
    key   = _normalize_label(label)

    # If this food has variants but the client sent none, return the questions
    if key in FOOD_VARIANTS and not request.variants:
        questions = FOOD_VARIANTS[key]
        return {
            "status":        "needs_variants",
            "food_label":    label,
            "message":       f"Please answer these questions so we can get accurate nutrition for '{label}':",
            "questions":     questions,
        }

    # Build query and fetch nutrition
    query     = build_query(label, request.variants or {})
    nutrition = fetch_nutrition(query)

    if nutrition:
        return {
            "status":             "found",
            "food_label":         label,
            "query_used":         query,
            "variants_selected":  request.variants,
            "nutrition":          nutrition,
        }

    # Not found — return guidance
    return handle_unknown_food(label)


# ---------------------------------------------------------------------------
# ENDPOINT 3 — Direct natural-language query (power-user / manual search)
# ---------------------------------------------------------------------------
@mapper_router.post("/nutrition")
def get_nutrition_by_query(request: NutritionQueryRequest):
    """
    Fetch nutrition for any free-text query string.

    Example:
      POST /map/nutrition
      {"query": "100g grilled salmon with olive oil"}
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    nutrition = fetch_nutrition(request.query)

    if nutrition:
        return {"status": "found", "nutrition": nutrition}

    return {
        "status":  "not_found",
        "message": f"No nutrition data found for '{request.query}'.",
        "tips": [
            "Try being more specific (e.g., 'grilled chicken breast 150g')",
            "Use common food names rather than dish names",
            "Search manually at https://www.nutritionix.com",
        ],
    }


# ---------------------------------------------------------------------------
# ENDPOINT 4 — Unknown food handler (user reports a food the AI can't identify)
# ---------------------------------------------------------------------------
@mapper_router.post("/unknown")
def report_unknown_food(request: UnknownFoodRequest):
    """
    Called when the AI model couldn't classify the food OR the API couldn't find it.
    If user provides a description, we try to query the API with that description first.

    Example:
      POST /map/unknown
      {"food_label": "some dish", "user_description": "Nigerian egusi soup with fufu"}
    """
    # If user described it, try the description first
    if request.user_description:
        nutrition = fetch_nutrition(request.user_description)
        if nutrition:
            return {
                "status":            "found_via_description",
                "original_label":    request.food_label,
                "query_used":        request.user_description,
                "nutrition":         nutrition,
            }

    # Still not found — return structured guidance
    guidance = handle_unknown_food(
        request.user_description or request.food_label
    )
    guidance["original_label"] = request.food_label
    return guidance