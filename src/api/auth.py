import json
import uuid
import hashlib
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

auth_router = APIRouter(prefix="/api/auth", tags=["Auth"])
security = HTTPBearer()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
USERS_FILE = OUTPUTS_DIR / "users.json"

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

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
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@auth_router.post("/register")
def register_user(req: RegisterRequest):
    users = load_users()
    email_key = req.email.lower().strip()
    if any(u.get("email", "").lower() == email_key for u in users.values()):
        raise HTTPException(status_code=400, detail="User with this email already exists")

    user_id = str(uuid.uuid4())
    token = str(uuid.uuid4())
    
    new_user = {
        "user_id": user_id,
        "name": req.name,
        "email": email_key,
        "password_hash": _hash_password(req.password),
        "tokens": [token],
        "profile": {
            "name": req.name,
            "avatar": None,
            "initials": "".join([part[0].upper() for part in req.name.split() if part][:2]) if req.name else "U",
            "email": email_key,
            "goal": "Maintain Weight",
            "dailyCalorieGoal": 2000,
            "proteinGoal": 120,
            "carbsGoal": 250,
            "fatGoal": 65,
            "unitSystem": "Metric",
            "memberSince": "Today"
        }
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
