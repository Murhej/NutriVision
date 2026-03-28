"""
Nutrition query helpers for mapping predicted food labels to API-friendly text.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover - handled at runtime if dependency is missing
    requests = None


def _load_local_env() -> None:
    base_dir = Path(__file__).resolve().parent.parent.parent
    for candidate in (base_dir / ".env.local", base_dir / ".env"):
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env()


EDAMAM_APP_ID = os.getenv("EDAMAM_APP_ID", "7bf7dbd4")
EDAMAM_API_KEY = os.getenv("EDAMAM_API_KEY", "605c62a04add3ad790a5b95bd7c5a7bd")
EDAMAM_URL = "https://api.edamam.com/api/nutrition-data"
USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
_LAST_NUTRITION_ERROR: Optional[Dict] = None

FOOD_VARIANTS = {
    "pizza": [
        {"key": "size", "prompt": "What size?", "choices": ['personal (6")', 'small (10")', 'medium (12")', 'large (14")', 'extra-large (16")']},
        {"key": "crust", "prompt": "What type of crust?", "choices": ["thin crust", "regular", "thick/pan", "stuffed crust"]},
        {"key": "type", "prompt": "What kind of pizza?", "choices": ["cheese", "pepperoni", "margherita", "meat lovers", "veggie", "BBQ chicken", "hawaiian"]},
        {"key": "slices", "prompt": "How many slices?", "choices": ["1", "2", "3", "4"]},
    ],
    "burger": [
        {"key": "patty", "prompt": "What type of patty?", "choices": ["beef", "chicken", "turkey", "veggie/plant-based", "fish"]},
        {"key": "size", "prompt": "What size?", "choices": ["single patty", "double patty", "quarter-pound", "half-pound"]},
    ],
    "sandwich": [
        {"key": "bread", "prompt": "What type of bread?", "choices": ["white", "whole wheat", "sourdough", "wrap/tortilla", "bagel"]},
        {"key": "filling", "prompt": "What's the main filling?", "choices": ["turkey", "chicken", "tuna", "ham", "BLT", "veggie", "grilled cheese"]},
    ],
    "salad": [
        {"key": "type", "prompt": "What kind of salad?", "choices": ["caesar", "garden", "greek", "cobb", "chef"]},
        {"key": "dressing", "prompt": "Dressing?", "choices": ["none", "light/vinaigrette", "caesar", "ranch", "thousand island"]},
    ],
    "pasta": [
        {"key": "type", "prompt": "What type of pasta?", "choices": ["spaghetti bolognese", "carbonara", "alfredo", "marinara", "mac and cheese"]},
        {"key": "portion", "prompt": "Portion size?", "choices": ["small (~1 cup)", "medium (~1.5 cups)", "large (~2+ cups)"]},
    ],
    "sushi": [
        {"key": "type", "prompt": "What type of sushi?", "choices": ["nigiri", "maki roll", "hand roll", "sashimi", "specialty roll"]},
        {"key": "pieces", "prompt": "How many pieces?", "choices": ["2-3 pieces", "4-6 pieces", "8-10 pieces", "12+ pieces"]},
    ],
    "steak": [
        {"key": "cut", "prompt": "What cut?", "choices": ["sirloin", "ribeye", "filet mignon", "T-bone", "flank"]},
        {"key": "weight", "prompt": "Approximate weight?", "choices": ["4 oz (small)", "6 oz (medium)", "8 oz (large)", "12 oz+ (extra large)"]},
    ],
    "coffee": [
        {"key": "type", "prompt": "What type of coffee?", "choices": ["black coffee", "latte", "cappuccino", "flat white", "cold brew"]},
        {"key": "milk", "prompt": "Milk type?", "choices": ["none", "whole milk", "skim milk", "oat milk", "almond milk"]},
        {"key": "size", "prompt": "Size?", "choices": ["small (8 oz)", "medium (12 oz)", "large (16 oz)", "extra large (20 oz)"]},
    ],
}

NUTRITION_QUERY_ALIASES = {
    "rawon": [
        "1 bowl rawon beef soup",
        "1 bowl indonesian beef soup",
        "1 bowl black beef soup",
        "1 serving indonesian black beef soup",
        "1 serving beef soup",
        "1 serving beef stew",
    ],
    "rendang": [
        "1 serving beef rendang curry",
        "1 serving beef rendang",
        "1 serving indonesian beef curry",
        "1 serving slow cooked beef curry",
        "1 serving beef curry",
    ],
    "sate": [
        "2 skewers grilled chicken satay with peanut sauce",
        "2 skewers chicken satay",
        "1 serving chicken satay",
        "1 serving grilled meat skewers with peanut sauce",
        "1 serving satay",
    ],
    "soto": [
        "1 bowl soto chicken soup",
        "1 bowl chicken soup",
        "1 bowl indonesian chicken soup",
        "1 bowl turmeric chicken soup",
        "1 serving soup",
    ],
    "gado_gado": [
        "1 plate gado gado salad with peanut sauce",
        "1 serving vegetable salad with peanut sauce",
        "1 serving indonesian salad",
        "1 plate mixed vegetables with peanut sauce",
        "1 serving salad with peanut dressing",
    ],
    "mie_goreng": [
        "1 plate mie goreng",
        "1 serving fried noodles",
        "1 cup fried noodles",
        "1 cup stir fried noodles",
        "1 plate stir fried noodles",
        "1 plate indonesian fried noodles",
        "1 plate noodles with vegetables",
        "1 serving noodle dish",
    ],
    "nasi_goreng": [
        "1 plate nasi goreng",
        "1 serving fried rice",
        "1 cup fried rice",
        "1 cup egg fried rice",
        "1 cup chicken fried rice",
        "1 cup shrimp fried rice",
        "1 plate indonesian fried rice",
        "1 plate stir fried rice with egg",
        "1 plate chicken fried rice",
        "1 plate shrimp fried rice",
        "1 serving rice dish",
    ],
    "nasi_padang": [
        "1 plate nasi padang rice meal",
        "1 serving rice with beef curry",
        "1 plate rice with meat dishes",
        "1 plate rice with multiple side dishes",
        "1 serving rice meal",
    ],
    "ayam_goreng": [
        "1 piece indonesian fried chicken",
        "1 piece fried chicken",
        "1 serving fried chicken",
    ],
    "ikan_goreng": [
        "1 whole fried fish",
        "1 serving indonesian fried fish",
        "1 serving fried fish",
        "1 piece fried fish",
        "1 fillet fried fish",
        "1 serving fish fillet",
    ],
    "pho": [
        "1 bowl pho noodle soup",
        "1 bowl pho",
        "1 bowl beef noodle soup",
        "1 bowl vietnamese pho",
        "1 bowl vietnamese noodle soup",
    ],
}

TERM_ALIASES = {
    "ikan": "fish",
    "ayam": "chicken",
    "goreng": "fried",
    "nasi": "rice",
    "mie": "noodles",
    "sapi": "beef",
    "bakar": "grilled",
    "telur": "egg",
    "udang": "shrimp",
    "sayur": "vegetable",
}

GENERIC_UNKNOWN_QUESTIONS = [
    "Is it a full plate, bowl, single item, or package?",
    "About how much did you eat: 1 piece, 1 bowl, 1 plate, 1 cup, or in grams?",
    "What was the main ingredient: chicken, beef, fish, rice, noodles, vegetables, or something else?",
]

QUESTION_GROUPS = {
    "soup": [
        "Was it broth-based, creamy, or thick like stew?",
        "How much broth did you have: a cup, small bowl, or large bowl?",
        "Did you use chicken stock, beef stock, coconut milk, or another base?",
    ],
    "fried": [
        "Was it pan-fried, deep-fried, or air-fried?",
        "Did it have batter or breading?",
        "Was it one piece, one fillet, or a full plate portion?",
    ],
    "rice_noodle": [
        "Was the base mostly rice, noodles, or both?",
        "How many cups or spoonfuls did you eat?",
        "Did it include egg, chicken, beef, shrimp, vegetables, or sauce?",
    ],
    "salad": [
        "What dressing or sauce was added?",
        "Did it include protein like chicken, tofu, egg, or shrimp?",
        "Was it a side salad or a full meal salad?",
    ],
    "packaged": [
        "What brand or restaurant was it from?",
        "Do you have the package name or menu item name?",
        "Can you upload a photo of the nutrition facts label?",
    ],
}


def _normalize_label(label: str) -> str:
    normalized = label.lower().replace("-", "_").replace(" ", "_").strip("_")
    if normalized.endswith("s") and normalized[:-1] in FOOD_VARIANTS:
        normalized = normalized[:-1]
    return normalized


def _extract_first_number(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(text))
    return match.group(0) if match else None


def _clean_fragment(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip(" ,.")


def _translate_known_terms(label: str) -> str:
    parts = [TERM_ALIASES.get(part, part) for part in _normalize_label(label).split("_") if part]
    if len(parts) == 2 and parts[1] in {"fried", "grilled"}:
        parts = [parts[1], parts[0]]
    return _clean_fragment(" ".join(parts))


def _contains_any(text: str, tokens: List[str]) -> bool:
    return any(token in text for token in tokens)


def _set_last_nutrition_error(
    kind: str,
    message: str,
    query: Optional[str] = None,
    status_code: Optional[int] = None,
    provider: Optional[str] = None,
) -> None:
    global _LAST_NUTRITION_ERROR
    _LAST_NUTRITION_ERROR = {
        "kind": kind,
        "message": message,
        "query": query,
        "status_code": status_code,
        "provider": provider,
    }


def get_last_nutrition_error() -> Optional[Dict]:
    return _LAST_NUTRITION_ERROR.copy() if _LAST_NUTRITION_ERROR else None


def _infer_unknown_question_groups(label: str, attempted_queries: Optional[List[str]] = None) -> List[str]:
    normalized = _normalize_label(label)
    joined = " ".join([normalized, " ".join(attempted_queries or [])]).lower()
    groups: List[str] = []

    if _contains_any(joined, ["soup", "pho", "rawon", "soto", "stew", "broth"]):
        groups.append("soup")
    if _contains_any(joined, ["fried", "goreng", "satay", "sate", "crispy", "fillet"]):
        groups.append("fried")
    if _contains_any(joined, ["rice", "nasi", "noodle", "mie", "pasta"]):
        groups.append("rice_noodle")
    if _contains_any(joined, ["salad", "gado", "veggie", "vegetable"]):
        groups.append("salad")
    if _contains_any(joined, ["brand", "restaurant", "package", "packaged", "snack"]):
        groups.append("packaged")

    if not groups and len(normalized.split("_")) <= 2:
        groups.append("packaged")
    return groups


def build_external_api_queries(
    food_label: str,
    attempted_queries: Optional[List[str]] = None,
    user_description: Optional[str] = None,
) -> List[str]:
    normalized = _normalize_label(food_label)
    label = _clean_fragment(food_label.replace("_", " "))
    translated = _translate_known_terms(food_label)
    description = _clean_fragment(user_description)

    raw_queries: List[str] = []
    if description:
        raw_queries.extend(
            [
                description,
                f"1 serving {description}",
                f"{description} nutrition facts",
            ]
        )

    raw_queries.extend(NUTRITION_QUERY_ALIASES.get(normalized, []))
    if translated and translated.lower() != label.lower():
        raw_queries.extend(
            [
                f"1 serving {translated}",
                translated,
                f"{translated} nutrition facts",
            ]
        )

    raw_queries.extend(
        [
            f"1 serving {label}",
            label,
            f"{label} nutrition facts",
        ]
    )

    if attempted_queries:
        raw_queries.extend(attempted_queries[:4])

    deduped: List[str] = []
    seen = set()
    for query in raw_queries:
        normalized_query = _clean_fragment(query)
        if not normalized_query:
            continue
        key = normalized_query.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized_query)
    return deduped[:8]


def build_follow_up_questions(food_label: str, attempted_queries: Optional[List[str]] = None) -> List[str]:
    questions = list(GENERIC_UNKNOWN_QUESTIONS)
    for group in _infer_unknown_question_groups(food_label, attempted_queries):
        questions.extend(QUESTION_GROUPS[group])

    deduped: List[str] = []
    seen = set()
    for question in questions:
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(question)
    return deduped[:6]


def build_query(food_label: str, variants: Optional[Dict] = None) -> str:
    variants = variants or {}
    label = food_label.replace("_", " ").strip()

    if not variants:
        return f"1 serving {label}"

    parts: List[str] = []

    for qty_key in ("slices", "pieces", "portion"):
        if qty_key in variants:
            qty_value = str(variants[qty_key]).split()[0]
            parts.append(qty_value)

    for key, value in variants.items():
        if key in ("slices", "pieces", "portion"):
            continue
        parts.append(_clean_fragment(str(value).split("(")[0]))

    parts.append(label)
    return _clean_fragment(" ".join(parts))


def build_candidate_queries(
    food_label: str,
    variants: Optional[Dict] = None,
    user_description: Optional[str] = None,
) -> List[str]:
    """
    Build multiple natural-language candidates from most specific to most generic.
    """
    variants = variants or {}
    normalized_label = _normalize_label(food_label)
    label = _clean_fragment(food_label.replace("_", " "))
    translated_label = _translate_known_terms(food_label)
    description = _clean_fragment(user_description)
    type_value = _clean_fragment(str(variants.get("type", "")).split("(")[0])

    qty = None
    unit = None
    if variants.get("slices"):
        qty = _extract_first_number(str(variants["slices"])) or "1"
        unit = "slices"
    elif variants.get("pieces"):
        qty = _extract_first_number(str(variants["pieces"])) or "1"
        unit = "pieces"
    elif variants.get("portion"):
        qty = _extract_first_number(str(variants["portion"])) or "1"
        unit = "serving"

    raw_candidates: List[str] = []

    if description:
        raw_candidates.append(description)
        if not re.search(r"\b(serving|slice|slices|piece|pieces|cup|cups|oz|ounce|ounces|g|gram|grams)\b", description.lower()):
            raw_candidates.append(f"1 serving {description}")
        if label and label.lower() not in description.lower():
            raw_candidates.append(f"1 serving {description} with {label}")

    raw_candidates.extend(NUTRITION_QUERY_ALIASES.get(normalized_label, []))
    if translated_label and translated_label.lower() != label.lower():
        raw_candidates.extend(
            [
                f"1 serving {translated_label}",
                translated_label,
            ]
        )

    raw_candidates.append(build_query(label, variants))

    if qty and unit:
        raw_candidates.append(f"{qty} {unit} {label}")
    if qty and unit and type_value:
        raw_candidates.append(f"{qty} {unit} {type_value} {label}")
    if type_value:
        raw_candidates.append(f"1 serving {type_value} {label}")

    raw_candidates.extend(
        [
            f"1 serving {label}",
            label,
        ]
    )

    deduped: List[str] = []
    seen = set()
    for candidate in raw_candidates:
        normalized = _clean_fragment(candidate)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _aggregate_parsed_nutrients(data: Dict) -> Dict[str, float]:
    aggregate = {
        "ENERC_KCAL": 0.0,
        "PROCNT": 0.0,
        "CHOCDF": 0.0,
        "FAT": 0.0,
        "FIBTG": 0.0,
        "SUGAR": 0.0,
        "NA": 0.0,
        "CHOLE": 0.0,
    }
    for ingredient in data.get("ingredients", []) or []:
        for parsed in ingredient.get("parsed", []) or []:
            parsed_nutrients = parsed.get("nutrients", {}) or {}
            for key in aggregate:
                aggregate[key] += float((parsed_nutrients.get(key, {}) or {}).get("quantity", 0) or 0)
    return aggregate


USDA_NUTRIENT_NAME_MAP = {
    "calories": {
        "energy",
        "energy (atwater general factors)",
        "energy (atwater specific factors)",
    },
    "protein_g": {"protein"},
    "carbs_g": {"carbohydrate, by difference"},
    "fat_g": {"total lipid (fat)"},
    "fiber_g": {"fiber, total dietary"},
    "sugar_g": {"sugars, total including nlea", "sugars, total"},
    "sodium_mg": {"sodium, na"},
    "cholesterol_mg": {"cholesterol"},
}


def _extract_usda_value(food: Dict, nutrient_names: set[str]) -> float:
    for nutrient in food.get("foodNutrients", []) or []:
        name = str(nutrient.get("nutrientName") or "").strip().lower()
        if name not in nutrient_names:
            continue
        value = nutrient.get("value")
        if value is None:
            value = nutrient.get("amount")
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _fetch_nutrition_from_edamam(query: str) -> Optional[Dict]:
    params = {
        "app_id": EDAMAM_APP_ID,
        "app_key": EDAMAM_API_KEY,
        "ingr": query,
        "nutrition-type": "logging",
    }
    try:
        response = requests.get(EDAMAM_URL, params=params, timeout=8)
    except requests.RequestException as error:
        _set_last_nutrition_error("network", f"Could not reach Edamam: {error}", query=query, provider="edamam")
        return None
    if response.status_code != 200:
        message = f"Edamam returned HTTP {response.status_code}."
        if response.status_code in {401, 403}:
            kind = "auth"
            message = "Edamam rejected the request. Check EDAMAM_APP_ID and EDAMAM_API_KEY."
        elif response.status_code == 429:
            kind = "rate_limit"
            message = "Edamam rate limit reached."
        else:
            kind = "http_error"
        _set_last_nutrition_error(kind, message, query=query, status_code=response.status_code, provider="edamam")
        return None

    data = response.json()
    calories = float(data.get("calories") or 0.0)
    nutrients = data.get("totalNutrients", {}) or {}

    if calories <= 0:
        aggregate = _aggregate_parsed_nutrients(data)
        calories = aggregate["ENERC_KCAL"]
        if not nutrients:
            nutrients = {
                "PROCNT": {"quantity": aggregate["PROCNT"]},
                "CHOCDF": {"quantity": aggregate["CHOCDF"]},
                "FAT": {"quantity": aggregate["FAT"]},
                "FIBTG": {"quantity": aggregate["FIBTG"]},
                "SUGAR": {"quantity": aggregate["SUGAR"]},
                "NA": {"quantity": aggregate["NA"]},
                "CHOLE": {"quantity": aggregate["CHOLE"]},
            }

    if calories <= 0:
        _set_last_nutrition_error(
            "no_match",
            "Edamam did not return nutrition data for this query.",
            query=query,
            status_code=response.status_code,
            provider="edamam",
        )
        return None

    return {
        "food_name": query,
        "source": "edamam",
        "calories": round(calories, 1),
        "protein_g": round(float((nutrients.get("PROCNT", {}) or {}).get("quantity", 0) or 0), 1),
        "carbs_g": round(float((nutrients.get("CHOCDF", {}) or {}).get("quantity", 0) or 0), 1),
        "fat_g": round(float((nutrients.get("FAT", {}) or {}).get("quantity", 0) or 0), 1),
        "fiber_g": round(float((nutrients.get("FIBTG", {}) or {}).get("quantity", 0) or 0), 1),
        "sugar_g": round(float((nutrients.get("SUGAR", {}) or {}).get("quantity", 0) or 0), 1),
        "sodium_mg": round(float((nutrients.get("NA", {}) or {}).get("quantity", 0) or 0), 1),
        "cholesterol_mg": round(float((nutrients.get("CHOLE", {}) or {}).get("quantity", 0) or 0), 1),
    }


def _fetch_nutrition_from_usda(query: str) -> Optional[Dict]:
    payload = {
        "query": query,
        "pageSize": 5,
        "dataType": ["Foundation", "SR Legacy", "Survey (FNDDS)", "Branded"],
    }
    try:
        response = requests.post(
            f"{USDA_SEARCH_URL}?api_key={USDA_API_KEY}",
            json=payload,
            timeout=8,
        )
    except requests.RequestException as error:
        _set_last_nutrition_error("network", f"Could not reach USDA FoodData Central: {error}", query=query, provider="usda_fdc")
        return None
    if response.status_code != 200:
        message = f"USDA FoodData Central returned HTTP {response.status_code}."
        if response.status_code in {401, 403}:
            kind = "auth"
            message = "USDA FoodData Central rejected the request. Check USDA_API_KEY."
        elif response.status_code == 429:
            kind = "rate_limit"
            message = "USDA FoodData Central rate limit reached."
        else:
            kind = "http_error"
        _set_last_nutrition_error(kind, message, query=query, status_code=response.status_code, provider="usda_fdc")
        return None

    foods = (response.json() or {}).get("foods", []) or []
    if not foods:
        _set_last_nutrition_error(
            "no_match",
            "USDA FoodData Central did not return foods for this query.",
            query=query,
            status_code=response.status_code,
            provider="usda_fdc",
        )
        return None

    food = foods[0]
    calories = _extract_usda_value(food, USDA_NUTRIENT_NAME_MAP["calories"])
    if calories <= 0:
        _set_last_nutrition_error(
            "no_match",
            "USDA FoodData Central did not return usable nutrient data for this query.",
            query=query,
            status_code=response.status_code,
            provider="usda_fdc",
        )
        return None

    return {
        "food_name": str(food.get("description") or query),
        "source": "usda_fdc",
        "calories": round(calories, 1),
        "protein_g": round(_extract_usda_value(food, USDA_NUTRIENT_NAME_MAP["protein_g"]), 1),
        "carbs_g": round(_extract_usda_value(food, USDA_NUTRIENT_NAME_MAP["carbs_g"]), 1),
        "fat_g": round(_extract_usda_value(food, USDA_NUTRIENT_NAME_MAP["fat_g"]), 1),
        "fiber_g": round(_extract_usda_value(food, USDA_NUTRIENT_NAME_MAP["fiber_g"]), 1),
        "sugar_g": round(_extract_usda_value(food, USDA_NUTRIENT_NAME_MAP["sugar_g"]), 1),
        "sodium_mg": round(_extract_usda_value(food, USDA_NUTRIENT_NAME_MAP["sodium_mg"]), 1),
        "cholesterol_mg": round(_extract_usda_value(food, USDA_NUTRIENT_NAME_MAP["cholesterol_mg"]), 1),
    }


def fetch_nutrition(query: str) -> Optional[Dict]:
    global _LAST_NUTRITION_ERROR
    _LAST_NUTRITION_ERROR = None

    if not query:
        _set_last_nutrition_error("invalid_query", "Nutrition query was empty.", query=query)
        return None

    if requests is None:
        _set_last_nutrition_error("dependency", "The requests package is not installed, so the nutrition APIs cannot be called.", query=query)
        return None

    nutrition = _fetch_nutrition_from_edamam(query)
    if nutrition:
        return nutrition

    edamam_error = get_last_nutrition_error()
    nutrition = _fetch_nutrition_from_usda(query)
    if nutrition:
        return nutrition

    usda_error = get_last_nutrition_error()
    if edamam_error and usda_error:
        _set_last_nutrition_error(
            usda_error.get("kind") or "provider_error",
            f"{edamam_error.get('message')} USDA FoodData Central also failed: {usda_error.get('message')}",
            query=query,
            status_code=usda_error.get("status_code"),
            provider="edamam+usda_fdc",
        )
    elif edamam_error:
        _set_last_nutrition_error(
            edamam_error.get("kind") or "provider_error",
            edamam_error.get("message") or "Edamam failed.",
            query=edamam_error.get("query") or query,
            status_code=edamam_error.get("status_code"),
            provider=edamam_error.get("provider"),
        )
    elif usda_error:
        _set_last_nutrition_error(
            usda_error.get("kind") or "provider_error",
            usda_error.get("message") or "USDA FoodData Central failed.",
            query=usda_error.get("query") or query,
            status_code=usda_error.get("status_code"),
            provider=usda_error.get("provider"),
        )
    return None


def scale_nutrition(nutrition: Dict, multiplier: float) -> Dict:
    multiplier = max(0.25, min(float(multiplier or 1.0), 4.0))
    return {
        key: round(float(value) * multiplier, 1) if isinstance(value, (int, float)) else value
        for key, value in nutrition.items()
    }


def handle_unknown_food(
    food_label: str,
    attempted_queries: Optional[List[str]] = None,
    user_description: Optional[str] = None,
) -> Dict:
    label = _clean_fragment(food_label) or "this meal"
    external_api_queries = build_external_api_queries(label, attempted_queries, user_description)
    return {
        "status": "unknown_food",
        "food_label": label,
        "message": f"Could not find nutrition data for '{label}'.",
        "follow_up_questions": build_follow_up_questions(label, attempted_queries),
        "external_api_queries": external_api_queries,
        "next_best_query": external_api_queries[0] if external_api_queries else label,
        "suggestions": [
            "Describe the meal with ingredients or serving size",
            "Try a simpler label such as 'grilled chicken breast'",
            f"Search USDA manually: https://fdc.nal.usda.gov/fdc-app.html#/?query={label.replace(' ', '+')}",
        ],
    }
