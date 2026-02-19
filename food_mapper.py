
import os
from dotenv import load_dotenv
load_dotenv()
import requests
from typing import Optional

# Edamam API credentials
EDAMAM_APP_ID  = os.getenv("EDAMAM_APP_ID",  "7bf7dbd4")
EDAMAM_API_KEY = os.getenv("EDAMAM_API_KEY", "605c62a04add3ad790a5b95bd7c5a7bd")
EDAMAM_URL     = "https://api.edamam.com/api/nutrition-data"

FOOD_VARIANTS = {
    "pizza": [
        {"key": "size",   "prompt": "What size?",           "choices": ["personal (6\")", "small (10\")", "medium (12\")", "large (14\")", "extra-large (16\")"]},
        {"key": "crust",  "prompt": "What type of crust?",  "choices": ["thin crust", "regular", "thick/pan", "stuffed crust"]},
        {"key": "type",   "prompt": "What kind of pizza?",  "choices": ["cheese", "pepperoni", "margherita", "meat lovers", "veggie", "BBQ chicken", "hawaiian"]},
        {"key": "slices", "prompt": "How many slices?",     "choices": ["1", "2", "3", "4"]},
    ],
    "burger": [
        {"key": "patty", "prompt": "What type of patty?", "choices": ["beef", "chicken", "turkey", "veggie/plant-based", "fish"]},
        {"key": "size",  "prompt": "What size?",           "choices": ["single patty", "double patty", "quarter-pound", "half-pound"]},
    ],
    "sandwich": [
        {"key": "bread",   "prompt": "What type of bread?",    "choices": ["white", "whole wheat", "sourdough", "wrap/tortilla", "bagel"]},
        {"key": "filling", "prompt": "What's the main filling?", "choices": ["turkey", "chicken", "tuna", "ham", "BLT", "veggie", "grilled cheese"]},
    ],
    "salad": [
        {"key": "type",     "prompt": "What kind of salad?", "choices": ["caesar", "garden", "greek", "cobb", "chef"]},
        {"key": "dressing", "prompt": "Dressing?",           "choices": ["none", "light/vinaigrette", "caesar", "ranch", "thousand island"]},
    ],
    "pasta": [
        {"key": "type",    "prompt": "What type of pasta?",  "choices": ["spaghetti bolognese", "carbonara", "alfredo", "marinara", "mac and cheese"]},
        {"key": "portion", "prompt": "Portion size?",        "choices": ["small (~1 cup)", "medium (~1.5 cups)", "large (~2+ cups)"]},
    ],
    "sushi": [
        {"key": "type",   "prompt": "What type of sushi?", "choices": ["nigiri", "maki roll", "hand roll", "sashimi", "specialty roll"]},
        {"key": "pieces", "prompt": "How many pieces?",    "choices": ["2-3 pieces", "4-6 pieces", "8-10 pieces", "12+ pieces"]},
    ],
    "steak": [
        {"key": "cut",    "prompt": "What cut?",              "choices": ["sirloin", "ribeye", "filet mignon", "T-bone", "flank"]},
        {"key": "weight", "prompt": "Approximate weight?",    "choices": ["4 oz (small)", "6 oz (medium)", "8 oz (large)", "12 oz+ (extra large)"]},
    ],
    "coffee": [
        {"key": "type", "prompt": "What type of coffee?", "choices": ["black coffee", "latte", "cappuccino", "flat white", "cold brew"]},
        {"key": "milk", "prompt": "Milk type?",           "choices": ["none", "whole milk", "skim milk", "oat milk", "almond milk"]},
        {"key": "size", "prompt": "Size?",                "choices": ["small (8 oz)", "medium (12 oz)", "large (16 oz)", "extra large (20 oz)"]},
    ],
}

def _normalize_label(label: str) -> str:
    label = label.lower().replace("-", "_").replace(" ", "_")
    if label.endswith("s") and label[:-1] in FOOD_VARIANTS:
        label = label[:-1]
    return label

def prompt_user_for_variants(food_label: str) -> dict:
    key = _normalize_label(food_label)
    variants = FOOD_VARIANTS.get(key, [])
    answers = {}
    if not variants:
        return answers
    print(f"\n🍽️  We need a bit more info about your {food_label}:\n")
    for q in variants:
        print(f"  {q['prompt']}")
        for i, choice in enumerate(q["choices"], 1):
            print(f"    [{i}] {choice}")
        while True:
            raw = input("  Your choice (number or type your own): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(q["choices"]):
                answers[q["key"]] = q["choices"][int(raw) - 1]
                break
            elif raw:
                answers[q["key"]] = raw
                break
    return answers

def build_query(food_label: str, variants: dict) -> str:
    food_label = food_label.replace("_", " ")
    if not variants:
        return f"1 serving {food_label}"
    parts = []
    for qty_key in ("slices", "pieces", "portion"):
        if qty_key in variants:
            parts.append(variants[qty_key].split()[0])
    for key, val in variants.items():
        if key in ("slices", "pieces", "portion"):
            continue
        parts.append(val.split("(")[0].strip())
    parts.append(food_label)
    return " ".join(parts)

def fetch_nutrition(query: str) -> Optional[dict]:
    try:
        params = {
            "app_id":  EDAMAM_APP_ID,
            "app_key": EDAMAM_API_KEY,
            "ingr":    query,
            "nutrition-type": "logging",
        }
        resp = requests.get(EDAMAM_URL, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            calories = data.get("calories", 0)
            if calories and calories > 0:
                n = data.get("totalNutrients", {})
                return {
                    "food_name":      query,
                    "calories":       round(calories, 1),
                    "protein_g":      round(n.get("PROCNT", {}).get("quantity", 0), 1),
                    "carbs_g":        round(n.get("CHOCDF", {}).get("quantity", 0), 1),
                    "fat_g":          round(n.get("FAT",    {}).get("quantity", 0), 1),
                    "fiber_g":        round(n.get("FIBTG",  {}).get("quantity", 0), 1),
                    "sugar_g":        round(n.get("SUGAR",  {}).get("quantity", 0), 1),
                    "sodium_mg":      round(n.get("NA",     {}).get("quantity", 0), 1),
                    "cholesterol_mg": round(n.get("CHOLE",  {}).get("quantity", 0), 1),
                }
            return None
        print(f"⚠️  Edamam API error {resp.status_code}: {resp.text}")
        return None
    except requests.RequestException as e:
        print(f"⚠️  Network error: {e}")
        return None

def handle_unknown_food(food_label: str) -> dict:
    return {
        "status":      "unknown_food",
        "food_label":  food_label,
        "message":     f"Sorry, couldn't find nutrition data for '{food_label}'.",
        "suggestions": [
            "Try being more specific (e.g., 'grilled chicken breast 150g')",
            f"Search Edamam: https://www.edamam.com/",
            f"Search USDA: https://fdc.nal.usda.gov/fdc-app.html#/?query={food_label.replace(' ', '+')}",
        ],
    }

def map_food(food_label: str) -> dict:
    print(f"\n🔍 Looking up: {food_label}")
    variants  = prompt_user_for_variants(food_label)
    query     = build_query(food_label, variants)
    print(f"📡 Querying Edamam: '{query}'")
    nutrition = fetch_nutrition(query)
    if nutrition:
        nutrition["status"]            = "found"
        nutrition["query_used"]        = query
        nutrition["variants_selected"] = variants
        return nutrition
    return handle_unknown_food(food_label)

if __name__ == "__main__":
    import json
    print("TEST 1: apple")
    print(json.dumps(fetch_nutrition("1 medium apple"), indent=2))
    print("\nTEST 2: pizza")
    print(json.dumps(fetch_nutrition("2 slices pepperoni pizza"), indent=2))