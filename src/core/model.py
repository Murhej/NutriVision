"""
Single source of truth for model construction, checkpoint I/O,
dataset indexing, and shared inference utilities (TTA forward pass).

All other modules import from here — never define build_model or
forward_with_tta anywhere else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = [
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b4",
    "efficientnet_v2_s",
    "convnext_base",
    "mobilenet_v3_large",
    "vit_b_16",
]


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    Build a model architecture with a custom classification head.

    Args:
        model_name: One of SUPPORTED_MODELS.
        num_classes: Number of output classes.
        pretrained: Load ImageNet pretrained weights when True.
                    Use False for inference (loading saved checkpoints).
    """
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_base(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. Supported: {SUPPORTED_MODELS}"
        )

    return model


def classifier_head_keys(model_name: str) -> tuple[str, str]:
    """Return (weight_key, bias_key) for the classifier head state dict."""
    if model_name == "resnet50":
        return "fc.weight", "fc.bias"
    if model_name in {"efficientnet_b0", "efficientnet_b4", "efficientnet_v2_s", "mobilenet_v3_large"}:
        return "classifier.1.weight", "classifier.1.bias"
    if model_name == "convnext_base":
        return "classifier.2.weight", "classifier.2.bias"
    if model_name == "vit_b_16":
        return "heads.head.weight", "heads.head.bias"
    raise ValueError(f"Unknown model for head key lookup: {model_name}")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load a saved state dict into a model (weights_only=True for security)."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def load_report(runs_dir: Path) -> Dict:
    """Load runs/report.json and raise a clear error if it is missing."""
    report_path = runs_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Training report not found at {report_path}. "
            "Run training first: python main.py train"
        )
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------

def forward_with_tta(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    tta_views: int = 1,
    use_amp: bool = False,
) -> torch.Tensor:
    """
    Run a forward pass with optional flip-based test-time augmentation.

    Views generated (in order): original, horizontal flip, vertical flip, both.
    tta_views controls how many of these views are averaged.
    """
    if tta_views <= 1:
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=True):
                return model(inputs)
        return model(inputs)

    views = [
        inputs.contiguous(),
        torch.flip(inputs, dims=[3]).contiguous(),
    ]
    if tta_views >= 3:
        views.append(torch.flip(inputs, dims=[2]).contiguous())
    if tta_views >= 4:
        views.append(torch.flip(inputs, dims=[2, 3]).contiguous())

    num_views = min(tta_views, len(views))
    logits_sum: Optional[torch.Tensor] = None
    for view in views[:num_views]:
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(view)
        else:
            logits = model(view)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)

    return logits_sum / float(num_views)


def format_top3_predictions(
    output: torch.Tensor, class_names: List[str]
) -> List[Dict]:
    """Convert raw logits into a top-3 ranked prediction list."""
    probs = torch.nn.functional.softmax(output, dim=1)
    top3_prob, top3_idx = probs.topk(3, dim=1)
    return [
        {
            "rank": i + 1,
            "class": class_names[top3_idx[0][i].item()],
            "confidence": float(top3_prob[0][i].item() * 100),
        }
        for i in range(3)
    ]


# ---------------------------------------------------------------------------
# Dataset index (shared between API and evaluation)
# ---------------------------------------------------------------------------

def build_dataset_index(
    data_dir: Path, split: str, class_names: List[str]
) -> List[Dict]:
    """
    Build a stable, ordered index for Food-101 images using the metadata JSON.

    Returns a list of dicts with keys: index, label_index, label_name, image_path.
    """
    meta_path = data_dir / "food-101" / "meta" / f"{split}.json"
    images_dir = data_dir / "food-101" / "images"

    if not meta_path.exists():
        raise FileNotFoundError(f"Food-101 metadata not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_to_idx = {name: i for i, name in enumerate(sorted(metadata.keys()))}
    records = []
    idx = 0
    for class_label, rel_paths in metadata.items():
        class_idx = class_to_idx[class_label]
        for rel_path in rel_paths:
            image_path = images_dir.joinpath(*f"{rel_path}.jpg".split("/"))
            records.append(
                {
                    "index": idx,
                    "label_index": class_idx,
                    "label_name": class_names[class_idx],
                    "image_path": str(image_path),
                }
            )
            idx += 1
    return records
