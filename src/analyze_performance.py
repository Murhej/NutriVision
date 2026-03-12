"""
Evaluate the current best model on all Food-101 test images.

Outputs:
- overall Top-1 / Top-3 accuracy in the terminal
- per-class Top-1 / Top-3 metrics in outputs/per_class_performance.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import Food101
from tqdm import tqdm


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"
OUTPUTS_DIR = BASE_DIR / "outputs"


def build_dataset_index(split: str, class_names: List[str]) -> List[Dict]:
    meta_path = DATA_DIR / "food-101" / "meta" / f"{split}.json"
    images_dir = DATA_DIR / "food-101" / "images"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

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


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "convnext_base":
        model = models.convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    if model_name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {model_name}")


def forward_with_tta(model: nn.Module, inputs: torch.Tensor, device: torch.device, tta_views: int) -> torch.Tensor:
    views = [inputs]
    if tta_views > 1:
        views.append(torch.flip(inputs, dims=[3]))
    if tta_views > 2:
        views.append(torch.flip(inputs, dims=[2]))
    if tta_views > 3:
        views.append(torch.flip(inputs, dims=[2, 3]))

    logits_sum = None
    for view in views[: max(1, tta_views)]:
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(view)
        else:
            logits = model(view)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)
    return logits_sum / float(max(1, min(tta_views, len(views))))


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if platform.system() == "Windows" and torch.cuda.is_available():
        print("Windows CUDA eval can be unstable on this machine; using CPU for full-image analysis.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analyze_per_class_performance(device_name: str = "auto", batch_size: int | None = None) -> None:
    print("=" * 60)
    print("Food-101 Full Test Evaluation")
    print("=" * 60)

    report_path = RUNS_DIR / "report.json"
    if not report_path.exists():
        raise FileNotFoundError("Missing runs/report.json. Train a model first.")

    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    model_name = report["best_model_name"]
    class_names = report.get("class_names")
    tta_views = 1
    train_cfg = report.get("config", {})
    if train_cfg.get("eval_tta", False):
        tta_views = int(train_cfg.get("tta_num_views", 1))

    device = resolve_device(device_name)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"TTA views: {tta_views}")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = Food101(root=str(DATA_DIR), split="test", download=False, transform=transform)
    if not class_names:
        class_names = dataset.classes
    dataset_index = build_dataset_index("test", class_names)
    if len(dataset_index) != len(dataset):
        raise RuntimeError(f"Dataset index mismatch: {len(dataset_index)} vs {len(dataset)}")

    model = build_model(model_name, len(class_names))
    model.load_state_dict(torch.load(RUNS_DIR / "best_model.pth", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size or (32 if device.type == "cuda" else 16),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    class_top1_correct = [0] * len(class_names)
    class_top3_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    class_confidence = [[] for _ in range(len(class_names))]

    overall_top1_correct = 0
    overall_top3_correct = 0

    print(f"Evaluating {len(dataset)} images across {len(test_loader)} batches...")

    with torch.no_grad():
        progress = tqdm(test_loader, desc="Testing all images", unit="batch")
        for batch_idx, (inputs, labels) in enumerate(progress, 1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = forward_with_tta(model, inputs, device, tta_views)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            _, top3_predicted = torch.topk(outputs, k=min(3, outputs.shape[1]), dim=1)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                confidence = probs[i][pred].item()

                class_total[label] += 1
                class_confidence[label].append(confidence)

                if pred == label:
                    class_top1_correct[label] += 1
                    overall_top1_correct += 1
                if label in top3_predicted[i].tolist():
                    class_top3_correct[label] += 1
                    overall_top3_correct += 1

            seen = sum(class_total)
            progress.set_postfix(
                {
                    "top1": f"{(overall_top1_correct / seen * 100):.2f}%",
                    "top3": f"{(overall_top3_correct / seen * 100):.2f}%",
                }
            )

            if batch_idx % max(1, len(test_loader) // 4) == 0 or batch_idx == len(test_loader):
                seen = sum(class_total)
                print(
                    f"  Progress: {(batch_idx / len(test_loader) * 100):.0f}% | "
                    f"Top-1: {(overall_top1_correct / seen * 100):.2f}% | "
                    f"Top-3: {(overall_top3_correct / seen * 100):.2f}%"
                )

    first_index_for_label: Dict[int, int] = {}
    for item in dataset_index:
        label_idx = item["label_index"]
        if label_idx not in first_index_for_label:
            first_index_for_label[label_idx] = item["index"]

    results = []
    for i, class_name in enumerate(class_names):
        if class_total[i] == 0:
            continue
        top1_accuracy = (class_top1_correct[i] / class_total[i]) * 100
        top3_accuracy = (class_top3_correct[i] / class_total[i]) * 100
        avg_confidence = (sum(class_confidence[i]) / len(class_confidence[i])) * 100
        results.append(
            {
                "class_name": class_name,
                "class_index": i,
                "top1_accuracy": round(top1_accuracy, 2),
                "top3_accuracy": round(top3_accuracy, 2),
                "avg_confidence": round(avg_confidence, 2),
                "top1_correct": class_top1_correct[i],
                "top3_correct": class_top3_correct[i],
                "total": class_total[i],
                "sample_image_index": first_index_for_label.get(i),
            }
        )

    results.sort(key=lambda item: item["top1_accuracy"])

    output_data = {
        "model_name": model_name,
        "total_classes": len(results),
        "total_test_samples": len(dataset),
        "overall_top1_accuracy": round((overall_top1_correct / len(dataset)) * 100, 2),
        "overall_top3_accuracy": round((overall_top3_correct / len(dataset)) * 100, 2),
        "best_class": results[-1] if results else None,
        "worst_class": results[0] if results else None,
        "classes": results,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / "per_class_performance.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

    print("\nAnalysis complete!")
    print(f"Saved to: {output_path}")
    print("\nSummary:")
    print(f"  Overall Top-1: {output_data['overall_top1_accuracy']:.2f}%")
    print(f"  Overall Top-3: {output_data['overall_top3_accuracy']:.2f}%")
    if results:
        print(f"  Best class: {results[-1]['class_name']} ({results[-1]['top1_accuracy']:.1f}% Top-1)")
        print(f"  Worst class: {results[0]['class_name']} ({results[0]['top1_accuracy']:.1f}% Top-1)")

        print("\nTop 5 Best Recognized Foods:")
        for item in results[-5:][::-1]:
            print(
                f"  {item['class_name']:30s} "
                f"Top-1 {item['top1_accuracy']:5.1f}% | Top-3 {item['top3_accuracy']:5.1f}%"
            )

        print("\nTop 5 Worst Recognized Foods:")
        for item in results[:5]:
            print(
                f"  {item['class_name']:30s} "
                f"Top-1 {item['top1_accuracy']:5.1f}% | Top-3 {item['top3_accuracy']:5.1f}%"
            )
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the current best model on all Food-101 test images.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=0, help="Override eval batch size.")
    args = parser.parse_args()
    analyze_per_class_performance(
        device_name=args.device,
        batch_size=(args.batch_size if args.batch_size > 0 else None),
    )
