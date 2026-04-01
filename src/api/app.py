"""
FastAPI application factory.

All API state (model, device, transform, dataset, dataset_index)
lives in the _state dict — no module-level globals scattered around.

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
    python main.py serve
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.nutrition import mapper_router
from src.api.feedback import feedback_router
from src.api.inference import _load_model_and_config, register_routes
from src.api.mobile_sync import sync_router
from src.api.auth import auth_router

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = BASE_DIR / "static"

# Shared mutable state for this process.
# Using a plain dict keeps it testable — tests can mock _state keys.
_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load model artifacts once at startup."""
    try:
        _load_model_and_config()
        print("[OK] API ready!")
    except FileNotFoundError as exc:
        print(f"[WARN] Model not loaded: {exc}")
        print("       Run  python main.py train  to create a model first.")
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="NutriVision API",
        description="Food-101 Classification API for Smart Meal Logger",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.include_router(mapper_router)
    app.include_router(feedback_router)
    app.include_router(sync_router)
    app.include_router(auth_router)

    register_routes(app)

    return app


app = create_app()
