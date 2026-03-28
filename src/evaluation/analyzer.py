"""
Per-class Top-1 / Top-3 evaluation on the full Food-101 test set.

Outputs:
  - Console summary of overall accuracy and top/bottom 5 classes
  - outputs/per_class_performance.json

Entry point: python main.py evaluate
             python -m src.evaluation.analyzer
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from tqdm import tqdm

from src.core.model import build_model, build_dataset_index, load_report
from src.core.transforms import get_transforms

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"
OUTPUTS_DIR = BASE_DIR / "outputs"


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if platform.system() == "Windows" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analyze_per_class_performance(
    device_name: str = "auto",
    batch_size: Optional[int] = None,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = DATA_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Dict:
    print("=" * 60)
    print("Food-101 Full Test Evaluation")
    print("=" * 60)

    report = load_report(runs_dir)
    model_name = report["best_model_name"]
    class_names = report.get("class_names") or []

    train_cfg = report.get("config", {})
    tta_views = 1
    if train_cfg.get("eval_tta", False):
        tta_views = int(train_cfg.get("tta_num_views", 1))

    device = resolve_device(device_name)
    print(f"Model:  {model_name}")
    print(f"Device: {device}")
    print(f"TTA:    {tta_views} view(s)")

    _, test_transform = get_transforms()
    dataset = Food101(root=str(data_dir), split="test", download=False, transform=test_transform)
    if not class_names:
        class_names = list(dataset.classes)

    dataset_index = build_dataset_index(data_dir, "test", class_names)
    if len(dataset_index) != len(dataset):
        raise RuntimeError(f"Dataset index mismatch: {len(dataset_index)} vs {len(dataset)}")

    model = build_model(model_name, len(class_names), pretrained=False)
    model.load_state_dict(torch.load(runs_dir / "best_model.pth", map_location=device, weights_only=True))
    model = model.to(device).eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size or (32 if device.type == "cuda" else 16),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    n = len(class_names)
    class_top1 = [0] * n
    class_top3 = [0] * n
    class_total = [0] * n
    class_conf: List[List[float]] = [[] for _ in range(n)]
    overall_top1 = 0
    overall_top3 = 0

    print(f"Evaluating {len(dataset)} images across {len(loader)} batches...")
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", unit="batch")
        for batch_idx, (inputs, labels) in enumerate(pbar, 1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # TTA forward pass
            from src.core.model import forward_with_tta
            outputs = forward_with_tta(model, inputs, device, tta_views=tta_views, use_amp=(device.type == "cuda"))

            probs = torch.nn.functional.softmax(outputs, dim=1)
            top1_pred = outputs.argmax(dim=1)
            top3_pred = outputs.topk(k=min(3, outputs.shape[1]), dim=1).indices

            for i in range(len(labels)):
                lbl = labels[i].item()
                pred = top1_pred[i].item()
                conf = probs[i][pred].item()

                class_total[lbl] += 1
                class_conf[lbl].append(conf)
                if pred == lbl:
                    class_top1[lbl] += 1
                    overall_top1 += 1
                if lbl in top3_pred[i].tolist():
                    class_top3[lbl] += 1
                    overall_top3 += 1

            seen = sum(class_total)
            pbar.set_postfix({"top1": f"{overall_top1/seen*100:.2f}%", "top3": f"{overall_top3/seen*100:.2f}%"})

    first_img_for_label: Dict[int, int] = {}
    for item in dataset_index:
        if item["label_index"] not in first_img_for_label:
            first_img_for_label[item["label_index"]] = item["index"]

    results = []
    total_images = len(dataset)
    for i, class_name in enumerate(class_names):
        if class_total[i] == 0:
            continue
        results.append({
            "class_name": class_name,
            "class_index": i,
            "top1_accuracy": round(class_top1[i] / class_total[i] * 100, 2),
            "top3_accuracy": round(class_top3[i] / class_total[i] * 100, 2),
            "avg_confidence": round(sum(class_conf[i]) / len(class_conf[i]) * 100, 2),
            "top1_correct": class_top1[i],
            "top3_correct": class_top3[i],
            "total": class_total[i],
            "sample_image_index": first_img_for_label.get(i),
        })

    results.sort(key=lambda x: x["top1_accuracy"])

    output_data = {
        "model_name": model_name,
        "total_classes": len(results),
        "total_test_samples": total_images,
        "overall_top1_accuracy": round(overall_top1 / total_images * 100, 2),
        "overall_top3_accuracy": round(overall_top3 / total_images * 100, 2),
        "best_class": results[-1] if results else None,
        "worst_class": results[0] if results else None,
        "classes": results,
    }

    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "per_class_performance.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\nAnalysis complete!")
    print(f"  Overall Top-1: {output_data['overall_top1_accuracy']:.2f}%")
    print(f"  Overall Top-3: {output_data['overall_top3_accuracy']:.2f}%")
    if results:
        print(f"  Best class:  {results[-1]['class_name']} ({results[-1]['top1_accuracy']:.1f}%)")
        print(f"  Worst class: {results[0]['class_name']} ({results[0]['top1_accuracy']:.1f}%)")
        print("\nTop 5 Best:")
        for item in results[-5:][::-1]:
            print(f"  {item['class_name']:30s} Top-1 {item['top1_accuracy']:5.1f}%  Top-3 {item['top3_accuracy']:5.1f}%")
        print("\nTop 5 Worst:")
        for item in results[:5]:
            print(f"  {item['class_name']:30s} Top-1 {item['top1_accuracy']:5.1f}%  Top-3 {item['top3_accuracy']:5.1f}%")
    print(f"\nSaved to: {output_path}")
    print("=" * 60)
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the current best model on the full Food-101 test set.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    analyze_per_class_performance(
        device_name=args.device,
        batch_size=args.batch_size or None,
    )
