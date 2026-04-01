"""
API routes for NutriVision Mobile App.
This serves the dynamic data generated from actual user meal logs.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional, Dict
from datetime import datetime
import json
from pathlib import Path
from pydantic import BaseModel

from src.api.auth import get_current_user_id, load_users, save_users

sync_router = APIRouter(prefix="/api/mobile", tags=["Mobile Sync"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MEAL_LOG_PATH = OUTPUTS_DIR / "meal_logs.json"

ACHIEVEMENTS = [
    {"id": "1", "title": "First Scan", "emoji": "📸", "unlocked": False, "description": "Scanned your first meal"},
    {"id": "2", "title": "7-Day Streak", "emoji": "🔥", "unlocked": False, "description": "Logged meals 7 days in a row"},
    {"id": "3", "title": "Macro Master", "emoji": "💪", "unlocked": False, "description": "Hit all macro goals in a day"},
]

LEADERBOARD_USERS = [
    {"id": "1", "name": "Sarah Miller", "initials": "SM", "streak": 45, "score": 9800, "avatar": None},
]

def load_meal_logs() -> List[Dict]:
    if not MEAL_LOG_PATH.exists():
        return []
    try:
        with open(MEAL_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def get_today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def parse_date(iso_str: str) -> str:
    return iso_str.split("T")[0]

def parse_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%I:%M %p").lstrip("0")
    except:
        return "12:00 PM"

def get_user_profile(user_id: str) -> dict:
    users = load_users()
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return users[user_id].get("profile", {})

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    goal: Optional[str] = None
    dailyCalorieGoal: Optional[int] = None
    proteinGoal: Optional[int] = None
    carbsGoal: Optional[int] = None
    fatGoal: Optional[int] = None
    unitSystem: Optional[str] = None


@sync_router.post("/profile")
def update_profile(req: ProfileUpdate, user_id: str = Depends(get_current_user_id)):
    users = load_users()
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
        
    prof = users[user_id]["profile"]
    if req.name is not None:
        prof["name"] = req.name
        prof["initials"] = "".join([part[0].upper() for part in req.name.split() if part][:2]) if req.name else "U"
    if req.goal is not None: prof["goal"] = req.goal
    if req.dailyCalorieGoal is not None: prof["dailyCalorieGoal"] = req.dailyCalorieGoal
    if req.proteinGoal is not None: prof["proteinGoal"] = req.proteinGoal
    if req.carbsGoal is not None: prof["carbsGoal"] = req.carbsGoal
    if req.fatGoal is not None: prof["fatGoal"] = req.fatGoal
    if req.unitSystem is not None: prof["unitSystem"] = req.unitSystem
    
    save_users(users)
    return {"status": "success", "profile": prof}


@sync_router.get("/profile")
def get_profile(user_id: str = Depends(get_current_user_id)):
    all_logs = load_meal_logs()
    user_logs = [log for log in all_logs if log.get("user_id") == user_id]
    
    days_logged = set(parse_date(log["timestamp"]) for log in user_logs)
    streak = len(days_logged)
    
    profile = get_user_profile(user_id).copy()
    profile["streak"] = streak
    profile["totalScans"] = len(user_logs)
    
    achievements = [a.copy() for a in ACHIEVEMENTS]
    if len(user_logs) > 0:
        achievements[0]["unlocked"] = True
    if streak >= 7:
        achievements[1]["unlocked"] = True
        
    return {
        "profile": profile,
        "achievements": achievements
    }

@sync_router.get("/dashboard")
def get_dashboard(user_id: str = Depends(get_current_user_id)):
    all_logs = load_meal_logs()
    user_logs = [log for log in all_logs if log.get("user_id") == user_id]
    today_str = get_today_str()
    
    today_logs = [log for log in user_logs if parse_date(log["timestamp"]) == today_str]
    
    cals, protein, carbs, fat = 0.0, 0.0, 0.0, 0.0
    meals = []
    
    for i, log in enumerate(today_logs):
        nut = log.get("nutrition", {})
        cals += float(nut.get("calories", 0))
        protein += float(nut.get("protein_g", 0))
        carbs += float(nut.get("carbs_g", 0))
        fat += float(nut.get("fat_g", 0))
        
        meals.append({
            "id": str(i),
            "name": log.get("display_name", "Meal"),
            "time": parse_time(log["timestamp"]),
            "type": "Scanned Food",
            "calories": round(float(nut.get("calories", 0)), 2),
            "protein": int(nut.get("protein_g", 0)),
            "carbs": int(nut.get("carbs_g", 0)),
            "fat": int(nut.get("fat_g", 0)),
            "image": log.get("image_url", None),
            "emoji": "🍽️"
        })
        
    meals = meals[::-1]
    
    profile = get_user_profile(user_id)
        
    nutrition = {
        "calories": {"consumed": round(cals, 2), "goal": profile.get("dailyCalorieGoal", 2000)},
        "protein": {"consumed": int(protein), "goal": profile.get("proteinGoal", 120), "unit": "g"},
        "carbs": {"consumed": int(carbs), "goal": profile.get("carbsGoal", 250), "unit": "g"},
        "fat": {"consumed": int(fat), "goal": profile.get("fatGoal", 65), "unit": "g"},
        "water": {"consumed": 0, "goal": 8, "unit": "glasses"}
    }
    
    return {
        "nutrition": nutrition,
        "meals": meals
    }

@sync_router.get("/leaderboard")
def get_leaderboard(user_id: str = Depends(get_current_user_id)):
    all_logs = load_meal_logs()
    user_logs = [log for log in all_logs if log.get("user_id") == user_id]
    days_logged = set(parse_date(log["timestamp"]) for log in user_logs)
    streak = len(days_logged)
    
    profile = get_user_profile(user_id)
    
    leaderboard = [u.copy() for u in LEADERBOARD_USERS]
    
    # Add current user dynamically
    leaderboard.append({
        "id": user_id,
        "name": profile.get("name", "You"),
        "initials": profile.get("initials", "U"),
        "streak": streak,
        "score": len(user_logs) * 100,
        "isCurrentUser": True,
        "avatar": None
    })
            
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    return {"users": leaderboard}

@sync_router.get("/calendar")
def get_calendar(user_id: str = Depends(get_current_user_id)):
    all_logs = load_meal_logs()
    user_logs = [log for log in all_logs if log.get("user_id") == user_id]
    profile = get_user_profile(user_id)
    cal_goal = profile.get("dailyCalorieGoal", 2000)
    
    data = {}
    for log in user_logs:
        d_str = parse_date(log["timestamp"])
        cals = float(log.get("nutrition", {}).get("calories", 0))
        if d_str not in data:
            data[d_str] = {"calories": 0, "goal": cal_goal, "status": "no-data"}
        data[d_str]["calories"] += cals
        
    for k, v in data.items():
        v["calories"] = round(v["calories"], 2)
        if v["calories"] <= v["goal"] + 100:
            v["status"] = "on-target"
        else:
            v["status"] = "over"
            
    today_str = get_today_str()
    today_logs = [log for log in user_logs if parse_date(log["timestamp"]) == today_str]
    meals = []
    for i, log in enumerate(today_logs[::-1]):
        nut = log.get("nutrition", {})
        meals.append({
            "id": str(i),
            "name": log.get("display_name", "Meal"),
            "time": parse_time(log["timestamp"]),
            "type": "Scanned Food",
            "calories": round(float(nut.get("calories", 0)), 2),
            "protein": int(nut.get("protein_g", 0)),
            "carbs": int(nut.get("carbs_g", 0)),
            "fat": int(nut.get("fat_g", 0)),
            "image": log.get("image_url", None),
            "emoji": "🍽️"
        })
    
    return {
        "calendar_data": data,
        "meals": meals
    }
