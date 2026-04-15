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
from urllib.parse import quote, unquote

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from src.core.model import build_dataset_index, build_model, forward_with_tta
from src.training.incremental import normalize_label

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RUNS_DIR = BASE_DIR / "runs"
DATA_DIR = BASE_DIR / "data"


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_class_sample_index(data_dir: Path, class_names: List[str]) -> Dict[str, str]:
    """
    Build one representative image path per class name.

    Strategy (fast, no full directory walk):
    1. Exact lookup in Food-101 images/<class>/ folder.
    2. Targeted scan of each known extra-source training directory using prefix
       matching so canonicalised names like 'apple' are resolved from dirs named
       'Apple Braeburn 1' (normalises to 'apple_braeburn_1').
    3. Shallow scan of any *_incremental staging dirs as last resort.
    """
    sample_index: Dict[str, str] = {}
    unresolved: set = set(class_names)

    # Build exact and prefix lookup structures once.
    normalized_to_class: Dict[str, str] = {}
    for name in class_names:
        norm = normalize_label(name)
        if norm not in normalized_to_class:
            normalized_to_class[norm] = name
    # Sorted longest-first so more specific prefixes win.
    sorted_norms = sorted(normalized_to_class.keys(), key=len, reverse=True)

    def _grab_image(cls_dir: Path, cls: str) -> bool:
        """Pick the first image from cls_dir and record it. Returns True on success."""
        if cls not in unresolved:
            return False
        for item in cls_dir.iterdir():
            if item.is_file() and _is_image_file(item):
                sample_index[cls] = str(item)
                unresolved.discard(cls)
                return True
        return False

    def _match(norm_dir: str) -> Optional[str]:
        """Return the class name for a normalised directory name (exact or prefix)."""
        if norm_dir in normalized_to_class:
            return normalized_to_class[norm_dir]
        for norm_cls in sorted_norms:
            if norm_dir == norm_cls or norm_dir.startswith(norm_cls + "_"):
                return normalized_to_class[norm_cls]
        return None

    def _scan_split_dir(split_dir: Path) -> None:
        """Scan one level of class subdirs under split_dir."""
        if not split_dir.is_dir():
            return
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir() or not unresolved:
                continue
            cls = _match(normalize_label(cls_dir.name))
            if cls and cls in unresolved:
                _grab_image(cls_dir, cls)

    # ── Fast path 1: Food-101 images folder ──────────────────────────────────
    food101_images = data_dir / "food-101" / "images"
    for class_name in list(unresolved):
        cls_dir = food101_images / class_name
        if cls_dir.is_dir():
            _grab_image(cls_dir, class_name)

    if not unresolved:
        return sample_index

    # ── Fast path 2: Known extra-source training dirs ─────────────────────────
    # Import lazily to avoid hard coupling at module load time.
    try:
        from src.training.incremental import KNOWN_RAW_SOURCES, find_existing_source_root  # type: ignore
        for spec in KNOWN_RAW_SOURCES:
            if not unresolved:
                break
            source_root = find_existing_source_root(spec)
            if source_root is None:
                continue
            split_dirs: List[Path] = []
            for key in ("train_subdir", "test_subdir"):
                sd = spec.get(key)
                if sd and sd != ".":
                    split_dirs.append(source_root / sd)
            for fb in spec.get("fallback_train_subdirs", []):
                split_dirs.append(source_root / fb)
            if not split_dirs:
                split_dirs.append(source_root)
            for split_dir in split_dirs:
                _scan_split_dir(split_dir)
    except ImportError:
        pass

    # ── Fallback: shallow scan of staging/incremental dirs ───────────────────
    if unresolved:
        for inc_dir in data_dir.glob("*_incremental"):
            if not unresolved:
                break
            for split_dir in inc_dir.iterdir():
                if split_dir.is_dir():
                    _scan_split_dir(split_dir)

    return sample_index


def _resolve_class_sample_path(class_name: str, class_sample_index: Dict[str, str]) -> Optional[str]:
    """Resolve a class sample path with normalization and suffix fallback."""
    if class_name in class_sample_index:
        return class_sample_index[class_name]

    norm_requested = normalize_label(class_name)
    norm_to_path: Dict[str, str] = {}
    for key, path in class_sample_index.items():
        norm_key = normalize_label(key)
        if norm_key not in norm_to_path:
            norm_to_path[norm_key] = path

    # Exact normalized match.
    if norm_requested in norm_to_path:
        return norm_to_path[norm_requested]

    # Try dropping right-side suffixes: zucchini_green -> zucchini.
    parts = norm_requested.split("_")
    while len(parts) > 1:
        parts = parts[:-1]
        candidate = "_".join(parts)
        if candidate in norm_to_path:
            return norm_to_path[candidate]
    return None


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
    class_sample_index = _build_class_sample_index(DATA_DIR, list(class_names))
    _state["class_sample_index"] = class_sample_index
    _state["sample_classes"] = sorted(class_sample_index.keys())

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
    print(f"[OK] Class samples indexed: {len(class_sample_index)}")
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
    class_sample_index: Dict[str, str] = state.get("class_sample_index") or {}
    predictions = [
        {
            "rank": i + 1,
            "class": class_names[top3_idx[0][i].item()],
            "confidence": float(top3_prob[0][i].item() * 100),
            "sample_image_url": (
                f"/class/sample/{quote(class_names[top3_idx[0][i].item()], safe='')}"
                if class_names[top3_idx[0][i].item()] in class_sample_index else None
            ),
        }
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
        dataset_class_names = state.get("dataset_class_names") or []
        sample_classes = state.get("sample_classes") or []
        return {
            "classes": class_names,
            "count": len(class_names),
            "all_classes": class_names,
            "total_count": len(class_names),
            "dataset_classes": dataset_class_names,
            "dataset_count": len(dataset_class_names),
            "sample_classes": sample_classes,
            "sample_count": len(sample_classes),
        }

    @app.get("/class/sample/{class_name}")
    async def get_class_sample_image(class_name: str):
        state = _get_state()
        class_sample_index: Dict[str, str] = state.get("class_sample_index") or {}
        decoded = unquote(class_name)
        sample_path = _resolve_class_sample_path(decoded, class_sample_index)
        if not sample_path:
            raise HTTPException(status_code=404, detail=f"No sample image found for class: {decoded}")
        return FileResponse(Path(sample_path))

    @app.get("/predict/class-sample/{class_name}")
    async def predict_class_sample(class_name: str):
        state = _get_state()
        class_sample_index: Dict[str, str] = state.get("class_sample_index") or {}
        decoded = unquote(class_name)
        sample_path = _resolve_class_sample_path(decoded, class_sample_index)
        if not sample_path:
            raise HTTPException(status_code=404, detail=f"No sample image found for class: {decoded}")
        try:
            with Image.open(sample_path) as img:
                result = _predict_pil_image(img.convert("RGB"))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error processing class sample image: {exc}")
        result["sample"] = {
            "index": None,
            "true_label": decoded,
            "label_index": None,
            "image_url": f"/class/sample/{quote(decoded, safe='')}",
        }
        return result

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
                # Category may exist only in merged incremental classes; serve its sample image.
                class_sample_index: Dict[str, str] = state.get("class_sample_index") or {}
                sample_path = _resolve_class_sample_path(category, class_sample_index)
                if sample_path:
                    return {
                        "index": None,
                        "image_path": sample_path,
                        "true_label": category,
                        "label_index": None,
                        "image_url": f"/class/sample/{quote(category, safe='')}",
                    }
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
