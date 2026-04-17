import json
import uuid
import hashlib
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

auth_router = APIRouter(prefix="/api/auth", tags=["Auth"])
security = HTTPBearer()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
USERS_FILE = OUTPUTS_DIR / "users.json"
MEAL_LOG_FILE = OUTPUTS_DIR / "meal_logs.json"

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _password_is_strong(password: str) -> bool:
    return bool(re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$", password or ""))

def load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_users(users_data: dict):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users_data, f, indent=2)


def load_meal_logs() -> list[dict]:
    if not MEAL_LOG_FILE.exists():
        return []
    try:
        with open(MEAL_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_meal_logs(logs: list[dict]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MEAL_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def build_initial_profile(name: str, email_key: str) -> dict:
    initials = "".join([part[0].upper() for part in name.split() if part][:2]) if name else "U"
    username_seed = "".join(ch for ch in name.lower() if ch.isalnum())[:18] or "user"
    return {
        "name": name,
        "username": username_seed,
        "avatar": None,
        "initials": initials,
        "email": email_key,
        "phone": "",
        "country": "",
        "goal": "Maintain Weight",
        "goalType": "maintain",
        "activityMultiplier": 1.2,
        "dietaryPreferences": [],
        "allergies": [],
        "activityLevel": "",
        "exerciseHabits": "",
        "dailyCalorieGoal": 2000,
        "proteinGoal": 120,
        "carbsGoal": 250,
        "fatGoal": 65,
        "unitSystem": "Metric",
        "memberSince": "Today",
        "settings": {
            "darkMode": False,
            "notifications": True,
            "units": "Metric",
        },
    }


def username_exists(users: dict, username: str, exclude_user_id: str | None = None) -> bool:
    username_key = (username or "").strip().lower()
    if not username_key:
        return False
    for uid, user in users.items():
        if exclude_user_id and uid == exclude_user_id:
            continue
        profile = user.get("profile", {})
        if str(profile.get("username", "")).strip().lower() == username_key:
            return True
    return False

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    token = credentials.credentials
    users = load_users()
    for uid, udata in users.items():
        if token in udata.get("tokens", []):
            return uid
    raise HTTPException(status_code=401, detail="Invalid or expired token")

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ChangePasswordRequest(BaseModel):
    currentPassword: str
    newPassword: str
    confirmPassword: str

@auth_router.post("/register")
def register_user(req: RegisterRequest):
    users = load_users()
    email_key = req.email.lower().strip()
    if any(u.get("email", "").lower() == email_key for u in users.values()):
        raise HTTPException(status_code=400, detail="User with this email already exists")
    if not _password_is_strong(req.password):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters and include uppercase, lowercase, number, and special character",
        )

    user_id = str(uuid.uuid4())
    token = str(uuid.uuid4())
    
    new_user = {
        "user_id": user_id,
        "name": req.name,
        "email": email_key,
        "password_hash": _hash_password(req.password),
        "tokens": [token],
        "profile": build_initial_profile(req.name, email_key),
    }
    users[user_id] = new_user
    save_users(users)
    return {"token": token, "user": {"id": user_id, "name": req.name, "email": email_key}}

@auth_router.post("/login")
def login_user(req: LoginRequest):
    users = load_users()
    email_key = req.email.lower().strip()
    pwd_hash = _hash_password(req.password)
    
    for uid, udata in users.items():
        if udata.get("email") == email_key and udata.get("password_hash") == pwd_hash:
            token = str(uuid.uuid4())
            udata.setdefault("tokens", []).append(token)
            save_users(users)
            return {"token": token, "user": {"id": uid, "name": udata["name"], "email": email_key}}
            
    raise HTTPException(status_code=401, detail="Invalid email or password")


@auth_router.get("/username-available")
def check_username_available(username: str, user_id: str = Depends(get_current_user_id)):
    users = load_users()
    available = not username_exists(users, username, exclude_user_id=user_id)
    return {"username": username, "available": available}


@auth_router.post("/logout")
def logout_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    users = load_users()
    for uid, user in users.items():
        tokens = user.get("tokens", [])
        if token in tokens:
            user["tokens"] = [t for t in tokens if t != token]
            save_users(users)
            return {"status": "success"}
    raise HTTPException(status_code=401, detail="Invalid or expired token")


@auth_router.post("/change-password")
def change_password(req: ChangePasswordRequest, user_id: str = Depends(get_current_user_id)):
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


@auth_router.delete("/delete-account")
def delete_account(user_id: str = Depends(get_current_user_id)):
    users = load_users()
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")

    del users[user_id]
    save_users(users)

    # Remove this user from leaderboard/activity by deleting logs.
    logs = load_meal_logs()
    filtered_logs = [log for log in logs if log.get("user_id") != user_id]
    if len(filtered_logs) != len(logs):
        save_meal_logs(filtered_logs)

    return {"status": "success", "deleted_user_id": user_id}
