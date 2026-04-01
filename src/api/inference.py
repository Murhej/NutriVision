"""
Inference and dataset endpoints.

Endpoints preserved exactly as in original api.py:
    POST /predict
    GET  /predict/dataset/{index}
    GET  /dataset/random
    GET  /dataset/image/{index}
    GET  /classes
    GET  /info
    GET  /health
    GET  /plots/{plot_name}
    GET  /plots
    GET  /metrics
    GET  /predictions/sample/{model_name}
    GET  /analysis/per-class
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from src.core.model import build_dataset_index, build_model, forward_with_tta

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RUNS_DIR = BASE_DIR / "runs"
DATA_DIR = BASE_DIR / "data"


def _load_model_and_config() -> Dict:
    """Load the best trained model. Returns the report dict."""
    from src.core.transforms import get_transforms

    report_path = RUNS_DIR / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            "Training report not found. Run training first: python main.py train"
        )

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    model_name = report["best_model_name"]
    class_names = report.get("class_names") or []
    train_cfg = report.get("config", {})
    tta_views = int(train_cfg.get("tta_num_views", 1)) if train_cfg.get("eval_tta", False) else 1

    from src.api.app import _state
    _state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_transform = get_transforms()
    _state["transform"] = test_transform

    from torchvision.datasets import Food101
    dataset = Food101(root=str(DATA_DIR), split="test", download=False, transform=test_transform)
    _state["dataset_class_names"] = dataset.classes
    if not class_names:
        class_names = list(dataset.classes)
    _state["class_names"] = class_names
    _state["dataset"] = dataset
    _state["tta_views"] = tta_views

    dataset_index = build_dataset_index(DATA_DIR, "test", list(dataset.classes))
    if len(dataset_index) != len(dataset):
        raise RuntimeError(f"Dataset index mismatch: {len(dataset_index)} vs {len(dataset)}")
    _state["dataset_index"] = dataset_index

    model = build_model(model_name, len(class_names), pretrained=False)
    model.load_state_dict(torch.load(RUNS_DIR / "best_model.pth", map_location=_state["device"], weights_only=True))
    model = model.to(_state["device"]).eval()
    _state["model"] = model

    print(f"[OK] Model loaded: {model_name}")
    print(f"[OK] Classes: {len(class_names)}")
    print(f"[OK] Test dataset: {len(dataset)} images")
    return report


def _get_state():
    from src.api.app import _state
    return _state


def _require(key: str):
    s = _get_state()
    if not s.get(key):
        raise HTTPException(status_code=503, detail=f"API not ready: {key} not loaded.")
    return s[key]


def _get_dataset_sample(index: int) -> Dict:
    state = _get_state()
    dataset = state.get("dataset")
    dataset_index = state.get("dataset_index")
    if dataset is None or dataset_index is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    if index < 0 or index >= len(dataset):
        raise HTTPException(status_code=404, detail="Image index out of range")
    sample = dataset_index[index]
    if not Path(sample["image_path"]).exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return sample


def _predict_pil_image(image: Image.Image) -> Dict:
    state = _get_state()
    model = _require("model")
    transform = _require("transform")
    device = _require("device")
    class_names: List[str] = _require("class_names")
    tta_views: int = state.get("tta_views", 1)

    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = forward_with_tta(model, img_tensor, device, tta_views=tta_views)

    probs = torch.nn.functional.softmax(output, dim=1)
    top3_prob, top3_idx = probs.topk(3, dim=1)
    predictions = [
        {"rank": i + 1, "class": class_names[top3_idx[0][i].item()], "confidence": float(top3_prob[0][i].item() * 100)}
        for i in range(3)
    ]
    return {"success": True, "predictions": predictions, "image_size": list(image.size)}


def register_routes(app: FastAPI) -> None:
    """Attach all inference/dataset routes to the given FastAPI app."""

    @app.get("/")
    async def root():
        state = _get_state()
        return {
            "message": "NutriVision API — Food-101 Classification",
            "status": "ready" if state.get("model") else "model not loaded",
            "docs": "/docs",
        }

    @app.get("/health")
    async def health():
        state = _get_state()
        return {
            "status": "healthy",
            "model_loaded": state.get("model") is not None,
            "device": str(state.get("device")) if state.get("device") else None,
            "num_classes": len(state["class_names"]) if state.get("class_names") else None,
        }

    @app.get("/info")
    async def get_info():
        report_path = RUNS_DIR / "report.json"
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Training report not found")
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @app.get("/classes")
    async def get_classes():
        state = _get_state()
        class_names = state.get("class_names")
        if class_names is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        dataset_class_names = state.get("dataset_class_names") or class_names
        return {"classes": dataset_class_names, "count": len(dataset_class_names), "all_classes": class_names, "total_count": len(class_names)}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        _require("model")
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            return _predict_pil_image(image)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error processing image: {exc}")

    @app.get("/predict/dataset/{index}")
    async def predict_dataset_sample(index: int):
        sample = _get_dataset_sample(index)
        try:
            with Image.open(sample["image_path"]) as img:
                result = _predict_pil_image(img.convert("RGB"))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error processing dataset image: {exc}")
        result["sample"] = {
            "index": sample["index"],
            "true_label": sample["label_name"],
            "label_index": sample["label_index"],
            "image_url": f"/dataset/image/{sample['index']}",
        }
        return result

    @app.get("/dataset/random")
    async def get_random_dataset_image(split: str = "test", category: Optional[str] = None):
        state = _get_state()
        dataset = state.get("dataset")
        dataset_index = state.get("dataset_index")
        dataset_class_names = state.get("dataset_class_names")
        if dataset is None or dataset_index is None or dataset_class_names is None:
            raise HTTPException(status_code=503, detail="Dataset not loaded")
        if category:
            try:
                class_idx = list(dataset_class_names).index(category)
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Category not found: {category}")
            matching = [item["index"] for item in dataset_index if item["label_index"] == class_idx]
            if not matching:
                raise HTTPException(status_code=404, detail=f"No images for category: {category}")
            idx = random.choice(matching)
        else:
            idx = random.randint(0, len(dataset) - 1)
        sample = _get_dataset_sample(idx)
        return {"index": idx, "image_path": sample["image_path"], "true_label": sample["label_name"], "label_index": sample["label_index"]}

    @app.get("/dataset/image/{index}")
    async def get_dataset_image(index: int):
        return FileResponse(Path(_get_dataset_sample(index)["image_path"]), media_type="image/jpeg")

    @app.get("/plots/{plot_name}")
    async def get_plot(plot_name: str):
        plot_path = OUTPUTS_DIR / f"{plot_name}.png"
        if not plot_path.exists():
            raise HTTPException(status_code=404, detail=f"Plot not found: {plot_name}")
        return FileResponse(plot_path, media_type="image/png")

    @app.get("/plots")
    async def list_plots():
        if not OUTPUTS_DIR.exists():
            return {"plots": []}
        plots = [f.stem for f in OUTPUTS_DIR.glob("*.png")]
        return {"plots": plots, "count": len(plots)}

    @app.get("/metrics")
    async def get_metrics():
        metrics_path = OUTPUTS_DIR / "metrics.csv"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail="Metrics not found")
        import pandas as pd
        df = pd.read_csv(metrics_path)
        return {"metrics": df.to_dict(orient="records")}

    @app.get("/predictions/sample/{model_name}")
    async def get_sample_predictions(model_name: str):
        pred_path = OUTPUTS_DIR / f"sample_predictions_{model_name}.txt"
        if not pred_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample predictions not found for {model_name}")
        return {"content": pred_path.read_text(encoding="utf-8")}

    @app.get("/analysis/per-class")
    async def get_per_class_performance(force_recalculate: bool = False):
        cache_path = OUTPUTS_DIR / "per_class_performance.json"
        if cache_path.exists() and not force_recalculate:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        if not cache_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Per-class performance not yet computed. Run: python main.py evaluate",
            )
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
