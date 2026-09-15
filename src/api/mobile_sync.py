"""
API routes for NutriVision Mobile App.
Provides backend-driven profile, feed, leaderboard, and calendar data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import re
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, EmailStr, Field

from src.api.auth import (
    get_current_user_id,
    load_users,
    save_users,
    username_exists,
    _hash_password,
    _password_is_strong,
)

sync_router = APIRouter(prefix="/api/mobile", tags=["Mobile Sync"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MEAL_LOG_PATH = OUTPUTS_DIR / "meal_logs.json"
PROFILE_IMAGE_DIR = BASE_DIR / "static" / "profile_images"

COUNTRY_CANADA = "canada"

GOAL_LABELS = {
    "lose": "Lose Weight",
    "maintain": "Maintain Weight",
    "gain": "Gain Weight",
    "muscle": "Gain Muscle",
}

GOAL_MACRO_SPLIT = {
    "lose": {"protein": 0.35, "carbs": 0.30, "fat": 0.35},
    "maintain": {"protein": 0.28, "carbs": 0.42, "fat": 0.30},
    "gain": {"protein": 0.27, "carbs": 0.48, "fat": 0.25},
    "muscle": {"protein": 0.33, "carbs": 0.42, "fat": 0.25},
}

ACHIEVEMENTS = [
    {
        "id": "1",
        "title": "First Scan",
        "emoji": "📸",
        "unlocked": False,
        "description": "Scanned your first meal",
        "category": "starter",
    },
    {
        "id": "2",
        "title": "7-Day Streak",
        "emoji": "🔥",
        "unlocked": False,
        "description": "Logged meals 7 days in a row",
        "category": "consistency",
    },
    {
        "id": "3",
        "title": "Macro Master",
        "emoji": "📏",
        "unlocked": False,
        "description": "Hit all macro goals in a day",
        "category": "precision",
    },
]

# Backfill competitors so leaderboard remains meaningful with few real users.
SEED_COMPETITORS = [
    {"id": "seed-1", "name": "Aisha Noor", "country": "Canada", "points": 640, "streak": 5, "totalScans": 11},
    {"id": "seed-2", "name": "Noah Clarke", "country": "Canada", "points": 730, "streak": 7, "totalScans": 13},
    {"id": "seed-3", "name": "Eli Thompson", "country": "United States", "points": 890, "streak": 8, "totalScans": 16},
    {"id": "seed-4", "name": "Maya Rivera", "country": "Mexico", "points": 560, "streak": 4, "totalScans": 10},
]


def load_meal_logs() -> List[Dict]:
    if not MEAL_LOG_PATH.exists():
        return []
    try:
        with open(MEAL_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def get_today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def parse_date(iso_str: str) -> str:
    return str(iso_str).split("T")[0]


def parse_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return "12:00 PM"


def get_user_record(user_id: str) -> dict:
    users = load_users()
    user = users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def get_user_profile(user_id: str) -> dict:
    user = get_user_record(user_id)
    return user.setdefault("profile", {})


def ensure_profile_defaults(user: dict) -> tuple[dict, bool]:
    profile = user.setdefault("profile", {})
    changed = False

    # Keep profile and account identity fields aligned.
    account_name = str(user.get("name") or "").strip()
    account_email = str(user.get("email") or "").strip().lower()

    if not str(profile.get("name") or "").strip() and account_name:
        profile["name"] = account_name
        changed = True

    if not str(profile.get("email") or "").strip() and account_email:
        profile["email"] = account_email
        changed = True

    if not str(profile.get("username") or "").strip():
        seed = "".join(ch for ch in (account_name or "user").lower() if ch.isalnum())[:18] or "user"
        username = seed
        suffix = 1
        users = load_users()
        while username_exists(users, username, exclude_user_id=user.get("user_id")):
            suffix += 1
            username = f"{seed[:14]}{suffix}"
        profile["username"] = username
        changed = True

    if not str(profile.get("initials") or "").strip():
        profile["initials"] = initials_for_name(profile.get("name") or account_name or "User")
        changed = True

    if "country" not in profile:
        profile["country"] = ""
        changed = True

    if not str(profile.get("goalType") or "").strip():
        profile["goalType"] = "maintain"
        changed = True

    if not str(profile.get("goal") or "").strip():
        profile["goal"] = GOAL_LABELS["maintain"]
        changed = True

    if not isinstance(profile.get("activityMultiplier"), (int, float)):
        profile["activityMultiplier"] = 1.2
        changed = True

    if not isinstance(profile.get("dailyCalorieGoal"), (int, float)):
        profile["dailyCalorieGoal"] = 2000
        changed = True

    if not isinstance(profile.get("proteinGoal"), (int, float)):
        profile["proteinGoal"] = 120
        changed = True

    if not isinstance(profile.get("carbsGoal"), (int, float)):
        profile["carbsGoal"] = 250
        changed = True

    if not isinstance(profile.get("fatGoal"), (int, float)):
        profile["fatGoal"] = 65
        changed = True

    if "settings" not in profile or not isinstance(profile.get("settings"), dict):
        profile["settings"] = {"darkMode": False, "notifications": True, "units": profile.get("unitSystem", "Metric")}
        changed = True

    return profile, changed


def normalize_phone(phone: str) -> str:
    return (phone or "").strip()


def is_valid_phone(phone: str) -> bool:
    if not phone:
        return True
    return bool(re.match(r"^\+?[0-9\-\s()]{7,20}$", phone))


def normalize_goal_type(goal_type: Optional[str], current_goal: str = "maintain") -> str:
    candidate = str(goal_type or "").strip().lower()
    if candidate in GOAL_LABELS:
        return candidate
    return current_goal if current_goal in GOAL_LABELS else "maintain"


def calculate_macro_targets(goal_type: str, calorie_goal: float) -> tuple[int, int, int]:
    split = GOAL_MACRO_SPLIT.get(goal_type, GOAL_MACRO_SPLIT["maintain"])
    protein = int(round((calorie_goal * split["protein"]) / 4))
    carbs = int(round((calorie_goal * split["carbs"]) / 4))
    fat = int(round((calorie_goal * split["fat"]) / 9))
    return max(20, protein), max(20, carbs), max(20, fat)


def initials_for_name(name: str) -> str:
    return "".join([part[0].upper() for part in (name or "").split() if part][:2]) or "U"


def get_user_logs(user_id: str) -> List[dict]:
    all_logs = load_meal_logs()
    return [log for log in all_logs if log.get("user_id") == user_id]


def compute_user_streak_and_scans(logs: List[dict]) -> tuple[int, int]:
    days_logged = set(parse_date(log.get("timestamp", "")) for log in logs if log.get("timestamp"))
    streak = len(days_logged)
    scans = len(logs)
    return streak, scans


def sum_nutrients(logs: List[dict]) -> dict:
    totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    for log in logs:
        nut = log.get("nutrition", {}) or {}
        totals["calories"] += float(nut.get("calories", 0) or 0)
        totals["protein_g"] += float(nut.get("protein_g", 0) or 0)
        totals["carbs_g"] += float(nut.get("carbs_g", 0) or 0)
        totals["fat_g"] += float(nut.get("fat_g", 0) or 0)
    return totals


def build_achievements(logs: List[dict], macro_master: bool) -> list[dict]:
    streak, _ = compute_user_streak_and_scans(logs)
    achievements = [a.copy() for a in ACHIEVEMENTS]
    if logs:
        achievements[0]["unlocked"] = True
    if streak >= 7:
        achievements[1]["unlocked"] = True
    if macro_master:
        achievements[2]["unlocked"] = True
    return achievements


def score_user(profile: dict, logs: List[dict]) -> int:
    streak, scans = compute_user_streak_and_scans(logs)
    unique_days = len({parse_date(log.get("timestamp", "")) for log in logs if log.get("timestamp")})
    consistency_bonus = 150 if unique_days >= 7 else 0
    return (streak * 100) + (scans * 50) + consistency_bonus


def summarize_macro_status(profile: dict, logs: List[dict]) -> tuple[bool, dict]:
    if not logs:
        return False, {}

    grouped: dict[str, list[dict]] = {}
    for log in logs:
        date_key = parse_date(log.get("timestamp", ""))
        if not date_key:
            continue
        grouped.setdefault(date_key, []).append(log)

    p_goal = float(profile.get("proteinGoal", 120) or 120)
    c_goal = float(profile.get("carbsGoal", 250) or 250)
    f_goal = float(profile.get("fatGoal", 65) or 65)

    day_stats = {}
    macro_master = False
    for day, day_logs in grouped.items():
        totals = sum_nutrients(day_logs)
        p_ratio = totals["protein_g"] / p_goal if p_goal else 0
        c_ratio = totals["carbs_g"] / c_goal if c_goal else 0
        f_ratio = totals["fat_g"] / f_goal if f_goal else 0
        day_stats[day] = {
            "protein": round(p_ratio, 2),
            "carbs": round(c_ratio, 2),
            "fat": round(f_ratio, 2),
        }
        if 0.85 <= p_ratio <= 1.15 and 0.85 <= c_ratio <= 1.15 and 0.85 <= f_ratio <= 1.15:
            macro_master = True

    return macro_master, day_stats


def build_feed_topics(profile: dict, logs: List[dict]) -> list[str]:
    topics = {"All"}
    goal = str(profile.get("goal", "")).lower()
    dietary = [str(x).lower() for x in profile.get("dietaryPreferences", [])]
    allergies = [str(x).lower() for x in profile.get("allergies", [])]
    activity = str(profile.get("activityLevel", "")).lower()

    if "lose" in goal:
        topics.add("Weight Loss")
    if "gain" in goal:
        topics.add("Weight Gain")
    if "maintain" in goal:
        topics.add("Maintenance")
    if activity:
        topics.add("Exercise")
    if dietary:
        topics.add("Dietary")
    if allergies:
        topics.add("Allergies")
    if logs:
        topics.add("Hydration")
        topics.add("Macros")
        topics.add("Recovery")

    return sorted(topics)


def build_feed_articles(profile: dict, logs: List[dict]) -> list[dict]:
    if len(logs) < 2:
        return []

    now_key = get_today_str()
    recent_logs = sorted(
        [log for log in logs if log.get("timestamp")],
        key=lambda l: l.get("timestamp", ""),
        reverse=True,
    )[:30]

    totals = sum_nutrients(recent_logs)
    day_count = len({parse_date(log.get("timestamp", "")) for log in recent_logs if log.get("timestamp")}) or 1

    cal_goal = float(profile.get("dailyCalorieGoal", 2000) or 2000)
    p_goal = float(profile.get("proteinGoal", 120) or 120)

    avg_cal = totals["calories"] / day_count
    avg_protein = totals["protein_g"] / day_count

    goal = str(profile.get("goal", "Maintain Weight"))
    dietary = profile.get("dietaryPreferences", []) or []
    allergies = profile.get("allergies", []) or []

    articles = []

    articles.append({
        "id": "goal-track",
        "topic": "All",
        "title": f"Nutrition strategy for {goal.lower()}",
        "excerpt": "Your daily plan adapts as your goals and meal logs change.",
        "reason": "Goal focus",
        "readTime": "4 min read",
        "priority": 90,
    })

    if avg_cal > cal_goal * 1.1:
        articles.append({
            "id": "cal-over",
            "topic": "Weight Loss" if "lose" in goal.lower() else "Maintenance",
            "title": "Calorie pacing tips to stay consistent",
            "excerpt": "Your recent intake trends above target. Small meal timing changes can improve adherence.",
            "reason": "Calorie behavior",
            "readTime": "5 min read",
            "priority": 95,
        })
    elif avg_cal < cal_goal * 0.85:
        articles.append({
            "id": "cal-under",
            "topic": "Weight Gain" if "gain" in goal.lower() else "Maintenance",
            "title": "How to increase calories without discomfort",
            "excerpt": "You are trending below your target intake. Add calorie-dense snacks around workouts.",
            "reason": "Calorie behavior",
            "readTime": "5 min read",
            "priority": 94,
        })

    if avg_protein < p_goal * 0.8:
        articles.append({
            "id": "protein-gap",
            "topic": "Macros",
            "title": "Protein timing tips for stronger recovery",
            "excerpt": "Your logs suggest a protein gap. Spread protein across 3-4 meals per day.",
            "reason": "Nutrient gap",
            "readTime": "4 min read",
            "priority": 93,
        })

    if dietary:
        articles.append({
            "id": "dietary-ideas",
            "topic": "Dietary",
            "title": f"{dietary[0]} meal ideas that fit your goals",
            "excerpt": "Curated meal patterns aligned to your dietary preferences.",
            "reason": "Dietary preference",
            "readTime": "6 min read",
            "priority": 88,
        })

    if allergies:
        articles.append({
            "id": "allergy-safe",
            "topic": "Allergies",
            "title": "Safe substitutions for allergy-aware meals",
            "excerpt": "Practical swaps to avoid trigger ingredients while keeping macro balance.",
            "reason": "Allergy profile",
            "readTime": "4 min read",
            "priority": 87,
        })

    articles.append({
        "id": f"hydration-{now_key}",
        "topic": "Hydration",
        "title": "Hydration reminders that match your routine",
        "excerpt": "Use meal times as hydration anchors to improve consistency.",
        "reason": "Habit support",
        "readTime": "3 min read",
        "priority": 80,
    })

    articles.append({
        "id": f"exercise-recovery-{now_key}",
        "topic": "Recovery",
        "title": "Post-workout nutrition for better recovery",
        "excerpt": "Pair carbs and protein after activity to replenish glycogen and repair muscle.",
        "reason": "Exercise habits",
        "readTime": "5 min read",
        "priority": 82,
    })

    deduped = {}
    for article in articles:
        deduped[article["id"]] = article

    return sorted(deduped.values(), key=lambda a: a["priority"], reverse=True)


class SettingsUpdate(BaseModel):
    darkMode: Optional[bool] = None
    notifications: Optional[bool] = None
    units: Optional[str] = None


class ProfilePatchRequest(BaseModel):
    fullName: Optional[str] = Field(default=None, max_length=120)
    username: Optional[str] = Field(default=None, min_length=3, max_length=32)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(default=None, max_length=32)
    country: Optional[str] = Field(default=None, max_length=64)
    goal: Optional[str] = Field(default=None, max_length=64)
    goalType: Optional[str] = Field(default=None, max_length=16)
    activityMultiplier: Optional[float] = None
    dailyCalorieGoal: Optional[float] = None
    proteinGoal: Optional[float] = None
    carbsGoal: Optional[float] = None
    fatGoal: Optional[float] = None
    nutrientTargets: Optional[Dict[str, float]] = None
    nutrientTargetModes: Optional[Dict[str, str]] = None
    dietaryPreferences: Optional[list[str]] = None
    allergies: Optional[list[str]] = None
    activityLevel: Optional[str] = None
    exerciseHabits: Optional[str] = None
    achievements: Optional[List[Dict]] = None
    settings: Optional[SettingsUpdate] = None


class AchievementsPatchRequest(BaseModel):
    achievements: List[Dict] = Field(default_factory=list)


class PasswordChangeRequest(BaseModel):
    currentPassword: str
    newPassword: str
    confirmPassword: str


@sync_router.get("/profile")
def get_profile(user_id: str = Depends(get_current_user_id)):
    logs = get_user_logs(user_id)
    users = load_users()
    user = users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    profile, changed = ensure_profile_defaults(user)
    if changed:
        save_users(users)

    profile = profile.copy()

    streak, total_scans = compute_user_streak_and_scans(logs)
    macro_master, _ = summarize_macro_status(profile, logs)

    profile["streak"] = streak
    profile["totalScans"] = total_scans
    profile["joinedDate"] = profile.get("memberSince", "Today")

    persisted_achievements = user.get("achievements")
    achievements_payload = (
        persisted_achievements
        if isinstance(persisted_achievements, list)
        else build_achievements(logs, macro_master)
    )

    return {
        "profile": profile,
        "achievements": achievements_payload,
    }


@sync_router.api_route("/profile", methods=["PATCH"])
def patch_profile(req: ProfilePatchRequest, user_id: str = Depends(get_current_user_id)):
    users = load_users()
    user = users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    profile = user.setdefault("profile", {})

    if req.fullName is not None:
        full_name = req.fullName.strip()
        if not full_name:
            raise HTTPException(status_code=400, detail="Full name cannot be empty")
        user["name"] = full_name
        profile["name"] = full_name
        profile["initials"] = initials_for_name(full_name)

    if req.username is not None:
        username = req.username.strip()
        if len(username) < 3:
            raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
        if username_exists(users, username, exclude_user_id=user_id):
            raise HTTPException(status_code=409, detail="Username already taken")
        profile["username"] = username

    if req.email is not None:
        email = req.email.lower().strip()
        for uid, candidate in users.items():
            if uid == user_id:
                continue
            if str(candidate.get("email", "")).lower() == email:
                raise HTTPException(status_code=409, detail="Email already in use")
        user["email"] = email
        profile["email"] = email

    if req.phone is not None:
        phone = normalize_phone(req.phone)
        if not is_valid_phone(phone):
            raise HTTPException(status_code=400, detail="Invalid phone number format")
        profile["phone"] = phone

    if req.country is not None:
        profile["country"] = req.country.strip()

    goal_type_changed = False
    calories_changed = False

    if req.goalType is not None:
        current_goal_type = str(profile.get("goalType") or "maintain").lower()
        next_goal_type = normalize_goal_type(req.goalType, current_goal=current_goal_type)
        if req.goalType.strip().lower() not in GOAL_LABELS:
            raise HTTPException(status_code=400, detail="Invalid goal type")
        profile["goalType"] = next_goal_type
        profile["goal"] = GOAL_LABELS[next_goal_type]
        goal_type_changed = next_goal_type != current_goal_type

    if req.goal is not None:
        profile["goal"] = req.goal.strip() or profile.get("goal", GOAL_LABELS["maintain"])

    if req.activityMultiplier is not None:
        activity = float(req.activityMultiplier)
        if activity < 1.1 or activity > 2.4:
            raise HTTPException(status_code=400, detail="Activity multiplier must be between 1.1 and 2.4")
        profile["activityMultiplier"] = round(activity, 2)

    if req.dailyCalorieGoal is not None:
        calories = float(req.dailyCalorieGoal)
        if calories < 1200 or calories > 6500:
            raise HTTPException(status_code=400, detail="Daily calorie goal must be between 1200 and 6500")
        profile["dailyCalorieGoal"] = int(round(calories))
        calories_changed = True

    if req.proteinGoal is not None:
        protein = float(req.proteinGoal)
        if protein < 20 or protein > 500:
            raise HTTPException(status_code=400, detail="Protein goal out of allowed range")
        profile["proteinGoal"] = int(round(protein))

    if req.carbsGoal is not None:
        carbs = float(req.carbsGoal)
        if carbs < 20 or carbs > 800:
            raise HTTPException(status_code=400, detail="Carbs goal out of allowed range")
        profile["carbsGoal"] = int(round(carbs))

    if req.fatGoal is not None:
        fat = float(req.fatGoal)
        if fat < 20 or fat > 350:
            raise HTTPException(status_code=400, detail="Fat goal out of allowed range")
        profile["fatGoal"] = int(round(fat))

    if req.nutrientTargets is not None:
        next_targets: Dict[str, float] = {}
        for key, value in req.nutrientTargets.items():
            val = float(value)
            if val <= 0:
                raise HTTPException(status_code=400, detail=f"Invalid nutrient target for {key}")
            next_targets[str(key)] = round(val, 3)
        profile["nutrientTargets"] = next_targets

    if req.nutrientTargetModes is not None:
        next_modes: Dict[str, str] = {}
        for key, value in req.nutrientTargetModes.items():
            mode = str(value).strip().lower()
            if mode not in {"auto", "custom"}:
                raise HTTPException(status_code=400, detail=f"Invalid nutrient target mode for {key}")
            next_modes[str(key)] = mode
        profile["nutrientTargetModes"] = next_modes

    if goal_type_changed or calories_changed:
        goal_type = normalize_goal_type(profile.get("goalType"), current_goal="maintain")
        calorie_goal = float(profile.get("dailyCalorieGoal") or 2000)
        auto_protein, auto_carbs, auto_fat = calculate_macro_targets(goal_type, calorie_goal)
        if req.proteinGoal is None:
            profile["proteinGoal"] = auto_protein
        if req.carbsGoal is None:
            profile["carbsGoal"] = auto_carbs
        if req.fatGoal is None:
            profile["fatGoal"] = auto_fat

    if req.dietaryPreferences is not None:
        profile["dietaryPreferences"] = [str(x).strip() for x in req.dietaryPreferences if str(x).strip()]

    if req.allergies is not None:
        profile["allergies"] = [str(x).strip() for x in req.allergies if str(x).strip()]

    if req.activityLevel is not None:
        profile["activityLevel"] = req.activityLevel.strip()

    if req.exerciseHabits is not None:
        profile["exerciseHabits"] = req.exerciseHabits.strip()

    if req.achievements is not None:
        user["achievements"] = [
            item for item in req.achievements if isinstance(item, dict) and item.get("id")
        ]

    if req.settings is not None:
        settings = profile.setdefault("settings", {})
        if req.settings.darkMode is not None:
            settings["darkMode"] = req.settings.darkMode
        if req.settings.notifications is not None:
            settings["notifications"] = req.settings.notifications
        if req.settings.units is not None:
            settings["units"] = req.settings.units
            profile["unitSystem"] = req.settings.units

    save_users(users)

    return {"status": "success", "profile": profile}


@sync_router.get("/achievements")
def get_achievements(user_id: str = Depends(get_current_user_id)):
    users = load_users()
    user = users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    logs = get_user_logs(user_id)
    profile = user.setdefault("profile", {})
    macro_master, _ = summarize_macro_status(profile, logs)

    persisted = user.get("achievements")
    achievements = persisted if isinstance(persisted, list) else build_achievements(logs, macro_master)

    return {
        "achievements": achievements,
    }


@sync_router.patch("/achievements")
def patch_achievements(req: AchievementsPatchRequest, user_id: str = Depends(get_current_user_id)):
    users = load_users()
    user = users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    cleaned = [item for item in req.achievements if isinstance(item, dict) and item.get("id")]
    user["achievements"] = cleaned
    save_users(users)

    return {
        "status": "success",
        "achievements": cleaned,
    }


@sync_router.post("/profile")
def post_profile_update(req: ProfilePatchRequest, user_id: str = Depends(get_current_user_id)):
    """
    POST update for profile compatibility (fallback when PATCH is not supported).
    Same behavior as PATCH endpoint.
    """
    return patch_profile(req, user_id)


@sync_router.post("/profile/avatar")
async def upload_profile_avatar(file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    users = load_users()
    user = users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    ext = Path(file.filename or "avatar.jpg").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    PROFILE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"{user_id}_{uuid.uuid4().hex}{ext}"
    dest_path = PROFILE_IMAGE_DIR / file_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with open(dest_path, "wb") as f:
        f.write(content)

    avatar_url = f"/static/profile_images/{file_name}"
    profile = user.setdefault("profile", {})
    profile["avatar"] = avatar_url
    save_users(users)

    return {"status": "success", "avatar": avatar_url}


@sync_router.post("/profile/change-password")
def change_password(req: PasswordChangeRequest, user_id: str = Depends(get_current_user_id)):
    users = load_users()
    user = users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if _hash_password(req.currentPassword) != user.get("password_hash"):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if req.newPassword != req.confirmPassword:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if not _password_is_strong(req.newPassword):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters and include uppercase, lowercase, number, and special character",
        )

    user["password_hash"] = _hash_password(req.newPassword)
    save_users(users)
    return {"status": "success"}


@sync_router.get("/feed")
def get_feed(
    topic: str = Query(default="All"),
    user_id: str = Depends(get_current_user_id),
):
    profile = get_user_profile(user_id)
    logs = get_user_logs(user_id)

    topics = build_feed_topics(profile, logs)
    articles = build_feed_articles(profile, logs)

    if topic and topic != "All":
        filtered_articles = [a for a in articles if a.get("topic") in {topic, "All"}]
    else:
        filtered_articles = articles

    has_enough_data = len(logs) >= 2

    return {
        "topics": topics,
        "activeTopic": topic,
        "hasEnoughData": has_enough_data,
        "preparationMessage": "We are preparing your personalized feed. Keep logging meals and goals.",
        "articles": filtered_articles,
        "signals": {
            "goal": profile.get("goal"),
            "country": profile.get("country", ""),
            "dietaryPreferences": profile.get("dietaryPreferences", []),
            "allergies": profile.get("allergies", []),
            "activityLevel": profile.get("activityLevel", ""),
            "exerciseHabits": profile.get("exerciseHabits", ""),
            "mealHistoryCount": len(logs),
            "scanHistoryCount": len(logs),
        },
    }


@sync_router.get("/leaderboard")
def get_leaderboard(
    scope: str = Query(default="worldwide", pattern="^(worldwide|canada)$"),
    user_id: str = Depends(get_current_user_id),
):
    users = load_users()
    logs = load_meal_logs()

    scoped_entries = []

    for uid, user in users.items():
        profile = user.get("profile", {})
        country = str(profile.get("country", "")).strip()

        if scope == "canada" and country.lower() != COUNTRY_CANADA:
            continue

        user_logs = [log for log in logs if log.get("user_id") == uid]
        streak, total_scans = compute_user_streak_and_scans(user_logs)
        points = score_user(profile, user_logs)

        scoped_entries.append(
            {
                "id": uid,
                "name": profile.get("name") or user.get("name") or "User",
                "username": profile.get("username", ""),
                "initials": profile.get("initials") or initials_for_name(profile.get("name") or user.get("name", "User")),
                "avatar": profile.get("avatar"),
                "country": country,
                "streak": streak,
                "totalScans": total_scans,
                "points": points,
                "isCurrentUser": uid == user_id,
            }
        )

    # Add seed competitors so scope differences are visible with sparse data.
    for seed in SEED_COMPETITORS:
        if scope == "canada" and str(seed.get("country", "")).lower() != COUNTRY_CANADA:
            continue
        scoped_entries.append(
            {
                "id": seed["id"],
                "name": seed["name"],
                "username": "",
                "initials": initials_for_name(seed["name"]),
                "avatar": None,
                "country": seed["country"],
                "streak": seed["streak"],
                "totalScans": seed["totalScans"],
                "points": seed["points"],
                "isCurrentUser": False,
            }
        )

    scoped_entries.sort(key=lambda x: x.get("points", 0), reverse=True)

    current_user_entry = None
    for idx, entry in enumerate(scoped_entries, start=1):
        entry["rank"] = idx
        if entry.get("isCurrentUser"):
            current_user_entry = entry

    return {
        "scope": scope,
        "users": scoped_entries,
        "currentUser": current_user_entry,
        "hasData": len(scoped_entries) > 0,
    }


@sync_router.get("/dashboard")
def get_dashboard(user_id: str = Depends(get_current_user_id)):
    user_logs = get_user_logs(user_id)
    today_str = get_today_str()

    today_logs = [log for log in user_logs if parse_date(log.get("timestamp", "")) == today_str]

    cals, protein, carbs, fat = 0.0, 0.0, 0.0, 0.0
    meals = []

    for i, log in enumerate(today_logs):
        nut = log.get("nutrition", {})
        cals += float(nut.get("calories", 0) or 0)
        protein += float(nut.get("protein_g", 0) or 0)
        carbs += float(nut.get("carbs_g", 0) or 0)
        fat += float(nut.get("fat_g", 0) or 0)

        meals.append(
            {
                "id": str(i),
                "name": log.get("display_name", "Meal"),
                "time": parse_time(log.get("timestamp", "")),
                "type": "Scanned Food",
                "calories": round(float(nut.get("calories", 0) or 0), 2),
                "protein": int(float(nut.get("protein_g", 0) or 0)),
                "carbs": int(float(nut.get("carbs_g", 0) or 0)),
                "fat": int(float(nut.get("fat_g", 0) or 0)),
                "image": log.get("image_url", None),
                "emoji": "🍽️",
            }
        )

    meals = meals[::-1]

    profile = get_user_profile(user_id)

    nutrition = {
        "calories": {"consumed": round(cals, 2), "goal": profile.get("dailyCalorieGoal", 2000)},
        "protein": {"consumed": int(protein), "goal": profile.get("proteinGoal", 120), "unit": "g"},
        "carbs": {"consumed": int(carbs), "goal": profile.get("carbsGoal", 250), "unit": "g"},
        "fat": {"consumed": int(fat), "goal": profile.get("fatGoal", 65), "unit": "g"},
        "water": {"consumed": 0, "goal": 8, "unit": "glasses"},
    }

    return {"nutrition": nutrition, "meals": meals}


@sync_router.get("/calendar")
def get_calendar(user_id: str = Depends(get_current_user_id)):
    user_logs = get_user_logs(user_id)
    profile = get_user_profile(user_id)
    cal_goal = profile.get("dailyCalorieGoal", 2000)

    data = {}
    for log in user_logs:
        d_str = parse_date(log.get("timestamp", ""))
        cals = float((log.get("nutrition", {}) or {}).get("calories", 0) or 0)
        if d_str not in data:
            data[d_str] = {"calories": 0, "goal": cal_goal, "status": "no-data"}
        data[d_str]["calories"] += cals

    for _, v in data.items():
        v["calories"] = round(v["calories"], 2)
        if v["calories"] <= v["goal"] + 100:
            v["status"] = "on-target"
        else:
            v["status"] = "over"

    today_str = get_today_str()
    today_logs = [log for log in user_logs if parse_date(log.get("timestamp", "")) == today_str]
    meals = []
    for i, log in enumerate(today_logs[::-1]):
        nut = log.get("nutrition", {})
        meals.append(
            {
                "id": str(i),
                "name": log.get("display_name", "Meal"),
                "time": parse_time(log.get("timestamp", "")),
                "type": "Scanned Food",
                "calories": round(float(nut.get("calories", 0) or 0), 2),
                "protein": int(float(nut.get("protein_g", 0) or 0)),
                "carbs": int(float(nut.get("carbs_g", 0) or 0)),
                "fat": int(float(nut.get("fat_g", 0) or 0)),
                "image": log.get("image_url", None),
                "emoji": "🍽️",
            }
        )

    return {"calendar_data": data, "meals": meals}
