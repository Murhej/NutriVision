"""
/feedback/* router — user image submissions for training review.
All URL paths are identical to the original feedback_api.py.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.food_mapper import fetch_nutrition
from src.api.auth import get_current_user_id

feedback_router = APIRouter(prefix="/feedback", tags=["User Feedback"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FEEDBACK_DIR = OUTPUTS_DIR / "training_submissions"
MANIFEST_PATH = FEEDBACK_DIR / "manifest.jsonl"

SHOT_GUIDE = [
    {"key": "top_image", "title": "Top View", "description": "Take one photo from above so the full plate is visible."},
    {"key": "side_image", "title": "Side View", "description": "Take one photo from the side to help estimate thickness and portion size."},
    {"key": "inside_image", "title": "Inside View", "description": "Take one photo showing the inside or cross-section of the meal."},
    {"key": "nutrition_label_image", "title": "Nutrition Facts", "description": "Optional: add a package label or menu nutrition photo when available."},
]


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    return cleaned or "meal"


def _compose_calorie_queries(
    meal_name: str,
    food_type: Optional[str],
    brand_name: Optional[str],
    portion_size: Optional[str],
    protein_type: Optional[str],
    protein_amount: Optional[str],
    added_items: Optional[str],
    added_amount: Optional[str],
    notes: Optional[str],
) -> List[str]:
    parts: List[str] = []
    if brand_name:
        parts.append(brand_name)
    if food_type and food_type.lower() not in meal_name.lower():
        parts.append(food_type)
    if portion_size:
        parts.append(portion_size)
    if protein_amount and protein_type:
        parts.append(f"{protein_amount} {protein_type}")
    elif protein_type:
        parts.append(protein_type)
    parts.append(meal_name)
    if added_items:
        parts.append(f"with {added_amount} {added_items}" if added_amount else f"with {added_items}")

    raw: List[str] = [" ".join(p for p in parts if p).strip()]
    if notes:
        raw.extend([f"{raw[0]} {notes}".strip(), notes])
    raw.extend([
        f"{portion_size} {brand_name} {meal_name}".strip(),
        f"{portion_size} {meal_name}".strip(),
        f"{brand_name} {meal_name}".strip(),
        f"1 serving {meal_name}".strip(),
        meal_name,
    ])

    deduped: List[str] = []
    seen: set[str] = set()
    for q in raw:
        normalized = re.sub(r"\s+", " ", q).strip(" ,.")
        if not normalized:
            continue
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def _estimate_feedback_nutrition(
    meal_name: str,
    food_type: Optional[str],
    brand_name: Optional[str],
    portion_size: Optional[str],
    protein_type: Optional[str],
    protein_amount: Optional[str],
    added_items: Optional[str],
    added_amount: Optional[str],
    notes: Optional[str],
) -> Dict:
    queries = _compose_calorie_queries(
        meal_name, food_type, brand_name, portion_size,
        protein_type, protein_amount, added_items, added_amount, notes,
    )
    for query in queries:
        nutrition = fetch_nutrition(query)
        if nutrition:
            return {"status": "found", "query_used": query, "queries_tried": queries, "nutrition": nutrition}
    return {"status": "not_found", "query_used": None, "queries_tried": queries, "nutrition": None, "message": "No nutrition estimate found."}


def _parse_nutrition_facts_text(nutrition_facts: Optional[str]) -> Optional[Dict]:
    text = str(nutrition_facts or "").strip()
    if not text:
        return None
    patterns = {
        "calories": r"calories?\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        "protein_g": r"protein\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?",
        "carbs_g": r"(?:carbs?|carbohydrates?)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?",
        "fat_g": r"fat\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?",
    }
    extracted = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            extracted[key] = round(float(m.group(1)), 1)
    if "calories" not in extracted:
        return None
    extracted.setdefault("protein_g", 0.0)
    extracted.setdefault("carbs_g", 0.0)
    extracted.setdefault("fat_g", 0.0)
    extracted["food_name"] = "User provided nutrition facts"
    return extracted


async def _save_upload(file: UploadFile, destination: Path) -> str:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail=f"Missing required image: {destination.stem}.")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"{file.filename} is not an image.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"{file.filename} is empty.")
    suffix = Path(file.filename).suffix or ".bin"
    actual = destination.with_suffix(suffix.lower())
    actual.write_bytes(content)
    return actual.name


async def _save_optional_upload(file: Optional[UploadFile], destination: Path) -> Optional[str]:
    if file is None or not getattr(file, "filename", ""):
        return None
    return await _save_upload(file, destination)


def _append_manifest(entry: Dict) -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


class CorrectionPayload(BaseModel):
    original_prediction: str = ""
    corrected_food_name: str
    serving_size: str = ""
    serving_unit: str = ""
    confidence: float = 0.0
    timestamp: str = ""
    has_top_image: bool = False
    has_side_image: bool = False
    has_inside_image: bool = False
    has_label_image: bool = False


@feedback_router.post("/corrections/submit")
async def submit_correction(
    payload: CorrectionPayload,
    user_id: str = Depends(get_current_user_id),
):
    entry = {
        "user_id": user_id,
        "timestamp": payload.timestamp or datetime.now(timezone.utc).isoformat(),
        "original_prediction": payload.original_prediction,
        "corrected_food_name": payload.corrected_food_name,
        "serving_size": payload.serving_size,
        "serving_unit": payload.serving_unit,
        "confidence": payload.confidence,
        "has_images": {
            "top": payload.has_top_image,
            "side": payload.has_side_image,
            "inside": payload.has_inside_image,
            "label": payload.has_label_image,
        },
    }
    corrections_file = OUTPUTS_DIR / "corrections.json"
    try:
        existing = json.loads(corrections_file.read_text(encoding="utf-8")) if corrections_file.exists() else []
    except Exception:
        existing = []
    existing.append(entry)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    corrections_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return {"status": "ok", "correction_id": f"{user_id}_{entry['timestamp']}"}


@feedback_router.get("/guide")
def get_feedback_guide():
    return {"shots": SHOT_GUIDE, "message": "Ask for three clear meal images, plus a nutrition label when available, so the sample can be reviewed for future training."}


@feedback_router.post("/submit")
async def submit_feedback(
    meal_name: str = Form(...),
    food_type: str = Form(""),
    brand_name: str = Form(""),
    portion_size: str = Form(...),
    protein_type: str = Form(""),
    protein_amount: str = Form(""),
    added_items: str = Form(""),
    added_amount: str = Form(""),
    nutrition_facts: str = Form(""),
    notes: str = Form(""),
    predicted_label: str = Form(""),
    prediction_context: str = Form(""),
    top_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    inside_image: UploadFile = File(...),
    nutrition_label_image: Optional[UploadFile] = File(None),
):
    meal_name = meal_name.strip()
    portion_size = portion_size.strip()
    if not meal_name:
        raise HTTPException(status_code=400, detail="Meal name is required.")
    if not portion_size:
        raise HTTPException(status_code=400, detail="Portion size is required.")

    timestamp = datetime.now(timezone.utc)
    submission_id = f"{timestamp.strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
    submission_dir = FEEDBACK_DIR / f"{_slugify(meal_name)}_{submission_id}"
    submission_dir.mkdir(parents=True, exist_ok=True)

    top_name = await _save_upload(top_image, submission_dir / "top_view.jpg")
    side_name = await _save_upload(side_image, submission_dir / "side_view.jpg")
    inside_name = await _save_upload(inside_image, submission_dir / "inside_view.jpg")
    nutrition_label_name = await _save_optional_upload(nutrition_label_image, submission_dir / "nutrition_label.jpg")

    nutrition_result = _estimate_feedback_nutrition(
        meal_name, food_type, brand_name, portion_size,
        protein_type, protein_amount, added_items, added_amount, notes,
    )
    parsed = _parse_nutrition_facts_text(nutrition_facts)
    if nutrition_result["status"] != "found" and parsed:
        nutrition_result = {
            "status": "found",
            "query_used": "nutrition_facts_text",
            "queries_tried": nutrition_result.get("queries_tried", []),
            "nutrition": parsed,
        }

    entry = {
        "submission_id": submission_id,
        "timestamp": timestamp.isoformat(),
        "meal_name": meal_name,
        "food_type": food_type,
        "brand_name": brand_name,
        "portion_size": portion_size,
        "protein_type": protein_type,
        "protein_amount": protein_amount,
        "added_items": added_items,
        "added_amount": added_amount,
        "nutrition_facts": nutrition_facts,
        "notes": notes,
        "predicted_label": predicted_label,
        "prediction_context": prediction_context,
        "images": {
            "top_image": top_name,
            "side_image": side_name,
            "inside_image": inside_name,
            "nutrition_label_image": nutrition_label_name,
        },
        "submission_dir": str(submission_dir.relative_to(BASE_DIR)),
        "nutrition_result": nutrition_result,
        "training_status": "queued_for_training_review",
    }

    with open(submission_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)
    _append_manifest(entry)

    return {
        "status": "saved",
        "message": "Correction saved for training review, queued in the training submissions folder, and calories were recalculated.",
        "entry": entry,
    }
