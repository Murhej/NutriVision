"""
Food-101 baseline training pipeline.

Entry point:
    python main.py train                        # interactive model selection
    python main.py train --models resnet50      # single model
    python main.py train --models resnet50,vit_b_16
    python main.py train --models all           # all configured models
    python main.py train --resume resnet50      # continue from last checkpoint
    python -m src.training.baseline
"""

from __future__ import annotations

import gc
import json
import os
import platform
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101

from src.core.device import get_device, log_environment_info, set_seed
from src.core.model import build_model
from src.core.transforms import get_transforms
from src.training.config import Config
from src.training.trainer import (
    _is_cuda_runtime_failure,
    _make_cpu_fallback_loader,
    train_model,
)
from src.visualization.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_sample_grid,
    plot_training_curves,
    save_sample_predictions,
)

# All models the pipeline knows about (in quality/speed order)
ALL_MODELS = [
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3_large",
    "efficientnet_b4",
    "efficientnet_v2_s",
    "convnext_base",
    "vit_b_16",
]


# ---------------------------------------------------------------------------
# Archive + promotion helpers
# ---------------------------------------------------------------------------

def _archive_model(
    weights_path: str,
    model_name: str,
    metrics: Dict,
    runs_dir: str,
) -> str:
    """
    Copy a completed model's weights to runs/archive/ with a timestamp so
    previous trained weights are never silently overwritten.
    Returns the archive path.
    """
    archive_dir = Path(runs_dir) / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = archive_dir / f"{model_name}_{ts}.pth"
    shutil.copy2(weights_path, dest)

    meta_path = dest.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "archived_at": datetime.now().isoformat(),
            "source": weights_path,
            **metrics,
        }, f, indent=2)

    print(f"  [Archive] {model_name} saved to {dest.name}")
    return str(dest)


def _promote_best_so_far(
    results: Dict,
    all_metrics: List[Dict],
    class_names: List[str],
    config: Config,
    runs_dir: str,
    start_time: float,
    models_requested: List[str],
) -> None:
    """
    After each model finishes:
      1. Compare all *finished* models from this run.
      2. Also compare against the existing best_model.pth (cross-run),
         reading its metrics from report.json.
      3. Only overwrite best_model.pth if the new winner actually beats
         the currently deployed model.
    """
    if not results:
        return

    # Winner from current run
    run_best_name = max(results, key=lambda k: (results[k]["best_val_top3"], results[k]["best_val_top1"]))
    run_best = results[run_best_name]
    run_best_weights = run_best.get("weights_path")
    if not run_best_weights or not os.path.exists(run_best_weights):
        return

    # Load existing deployed best (cross-run comparison)
    report_path = Path(runs_dir) / "report.json"
    existing_best_top3 = -1.0
    existing_best_top1 = -1.0
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            m = existing.get("best_model_metrics", {})
            existing_best_top3 = float(m.get("val_top3_accuracy", -1.0))
            existing_best_top1 = float(m.get("val_top1_accuracy", -1.0))
        except Exception:
            pass

    new_top3 = run_best["best_val_top3"]
    new_top1 = run_best["best_val_top1"]
    is_new_overall_best = (new_top3 > existing_best_top3) or (
        new_top3 == existing_best_top3 and new_top1 > existing_best_top1
    )

    if is_new_overall_best:
        best_model_path = Path(runs_dir) / "best_model.pth"
        shutil.copyfile(run_best_weights, best_model_path)
        flag = "[NEW BEST]" if existing_best_top3 >= 0 else "[FIRST]"
        print(f"  {flag} best_model.pth -> {run_best_name} "
              f"(Top-1={run_best['final_top1']:.2f}%, Top-3={new_top3:.2f}%)")
    else:
        print(f"  [No upgrade] {run_best_name} "
              f"(Top-3={new_top3:.2f}%) does not beat existing best (Top-3={existing_best_top3:.2f}%)")

    report = {
        "best_model_name": run_best_name if is_new_overall_best else existing.get("best_model_name", run_best_name),
        "class_names": class_names,
        "num_classes": len(class_names),
        "best_model_metrics": {
            "val_top1_accuracy": new_top1 if is_new_overall_best else existing_best_top1,
            "val_top3_accuracy": new_top3 if is_new_overall_best else existing_best_top3,
            "test_top1_accuracy": run_best["final_top1"] if is_new_overall_best else existing.get("best_model_metrics", {}).get("test_top1_accuracy", 0),
            "test_top3_accuracy": run_best["final_top3"] if is_new_overall_best else existing.get("best_model_metrics", {}).get("test_top3_accuracy", 0),
        },
        "current_run_metrics": all_metrics,
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "total_training_time_seconds": time.time() - start_time,
        "status": "in_progress" if len(results) < len(models_requested) else "complete",
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    """
    Download (if needed) and load Food-101 with a stratified train/val split.

    Returns (train_loader, val_loader, test_loader, test_dataset).
    """
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)

    train_transform, test_transform = get_transforms(config.randaugment_magnitude)

    print("Loading Food-101 dataset (auto-download if not present)...")
    train_dataset = Food101(root=config.data_dir, split="train", transform=train_transform, download=True)
    val_dataset_raw = Food101(root=config.data_dir, split="train", transform=test_transform, download=False)
    test_dataset = Food101(root=config.data_dir, split="test", transform=test_transform, download=False)

    print(f"Full dataset: {len(train_dataset)} train, {len(test_dataset)} test, {len(train_dataset.classes)} classes")

    if config.train_subset_size > 0 and config.train_subset_size < len(train_dataset):
        indices = np.random.choice(len(train_dataset), config.train_subset_size, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"Train subset: {len(train_dataset)} samples")

    if config.test_subset_size > 0 and config.test_subset_size < len(test_dataset):
        indices = np.random.choice(len(test_dataset), config.test_subset_size, replace=False)
        test_dataset = Subset(test_dataset, indices)
        print(f"Test subset: {len(test_dataset)} samples")

    # Stratified train/val split
    if isinstance(train_dataset, Subset):
        pool_indices = np.array(train_dataset.indices)
        base_ds = train_dataset.dataset
    else:
        pool_indices = np.arange(len(train_dataset))
        base_ds = train_dataset

    labels_array = np.array(base_ds._labels)
    class_to_indices: Dict[int, List[int]] = {}
    for idx in pool_indices:
        label = int(labels_array[int(idx)])
        class_to_indices.setdefault(label, []).append(int(idx))

    target_val_size = int(len(pool_indices) * config.val_split)
    target_val_size = max(1, min(target_val_size, len(pool_indices) - 1))
    max_safe_val = sum(max(0, len(v) - 1) for v in class_to_indices.values())
    if max_safe_val <= 0:
        raise ValueError("Cannot build val split: each class has only 1 sample. Increase train_subset_size.")
    if target_val_size > max_safe_val:
        target_val_size = max_safe_val

    class_train_indices: Dict[int, List[int]] = {}
    val_indices: List[int] = []
    for label, indices in class_to_indices.items():
        cls = np.array(indices, dtype=np.int64)
        np.random.shuffle(cls)
        if len(cls) == 1:
            class_train_indices[label] = cls.tolist()
            continue
        cls_val = min(int(len(cls) * config.val_split), len(cls) - 1)
        if cls_val > 0:
            val_indices.extend(cls[:cls_val].tolist())
            class_train_indices[label] = cls[cls_val:].tolist()
        else:
            class_train_indices[label] = cls.tolist()

    # Top-up val if needed
    if len(val_indices) < target_val_size:
        label_order = list(class_train_indices.keys())
        np.random.shuffle(label_order)
        made_progress = True
        while len(val_indices) < target_val_size and made_progress:
            made_progress = False
            for label in label_order:
                if len(val_indices) >= target_val_size:
                    break
                if len(class_train_indices[label]) > 1:
                    val_indices.append(class_train_indices[label].pop())
                    made_progress = True

    train_indices = [idx for v in class_train_indices.values() for idx in v]
    if not train_indices or not val_indices:
        raise ValueError("Failed to build train/val split. Increase train_subset_size or lower val_split.")

    train_dataset = Subset(base_ds, train_indices)
    val_dataset = Subset(val_dataset_raw, val_indices)
    print(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")

    is_windows = platform.system() == "Windows"
    use_cuda = torch.cuda.is_available()
    num_workers = max(0, int(config.num_workers))
    pin_memory = use_cuda
    persistent = num_workers > 0

    # Windows requires the spawn multiprocessing context for DataLoader workers,
    # but workers themselves work fine — we just cannot use fork.
    multiprocessing_context = "spawn" if (is_windows and num_workers > 0) else None

    loader_kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        if multiprocessing_context:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context

    print(f"DataLoader: num_workers={num_workers}, pin_memory={pin_memory}, "
          f"persistent_workers={persistent}"
          + (f", context=spawn" if multiprocessing_context else ""))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False, **loader_kwargs)

    print("=" * 60 + "\n")
    return train_loader, val_loader, test_loader, test_dataset


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def perform_eda(train_dataset, test_dataset, output_dir: str) -> None:
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    original_ds = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)} | Classes: {len(original_ds.classes)}")

    sample_img, sample_label = train_dataset[0]
    print(f"Sample image shape: {sample_img.shape}")
    print(f"Sample label: {sample_label} ({original_ds.classes[sample_label]})")

    plot_class_distribution(original_ds, os.path.join(output_dir, "class_distribution.png"))
    plot_sample_grid(train_dataset, os.path.join(output_dir, "sample_grid.png"))
    print("EDA complete\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(
    models_to_train: Optional[List[str]] = None,
    resume_model: Optional[str] = None,
) -> None:
    """
    Args:
        models_to_train: explicit list of model names, or None to use config defaults.
        resume_model:    single model name to resume from its last checkpoint, or
                         "all" to resume every model that has a checkpoint.
    """
    start_time = time.time()
    config = Config()
    set_seed(config.seed, enable_cudnn_benchmark=config.enable_cudnn_benchmark)
    log_environment_info()
    device = get_device()

    if device.type == "cuda":
        if platform.system() == "Windows" and config.disable_tf32_on_windows_cuda:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("Windows CUDA stability: TF32 disabled")
        else:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("GPU throughput: TF32 enabled")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.runs_dir, exist_ok=True)

    # Resolve which models to train
    if models_to_train is not None:
        selected = models_to_train
    else:
        selected = list(config.models_to_train or ALL_MODELS)

    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Train subset: {config.train_subset_size or 'full'}")
    print(f"Test subset:  {config.test_subset_size or 'full'}")
    print(f"Batch size:   {config.batch_size} (eval: {config.eval_batch_size})")
    print(f"Warmup:       {config.warmup_epochs} epochs | Fine-tune: {config.finetune_epochs} epochs")
    print(f"EMA: {config.use_ema} | AMP: {config.use_amp} | TTA: {config.eval_tta} ({config.tta_num_views} views)")
    print(f"Models ({len(selected)}): {', '.join(selected)}")
    if resume_model:
        print(f"Resume mode:  {resume_model}")
    print("=" * 60)

    train_loader, val_loader, test_loader, test_dataset = load_data(config)
    original_test = test_dataset.dataset if isinstance(test_dataset, Subset) else test_dataset
    class_names = original_test.classes

    if config.run_eda:
        perform_eda(train_loader.dataset, test_dataset, config.output_dir)

    results: Dict = {}
    all_metrics: List[Dict] = []
    failed_models: List[Dict] = []
    is_windows = platform.system() == "Windows"
    is_windows_cuda = device.type == "cuda" and is_windows
    cuda_runtime_broken = False
    unstable_windows_models = {"efficientnet_v2_s", "convnext_base"}
    cpu_train_loader: Optional[DataLoader] = None
    cpu_val_loader: Optional[DataLoader] = None
    cpu_test_loader: Optional[DataLoader] = None

    for model_name in selected:
        run_device = device
        if device.type == "cuda" and cuda_runtime_broken:
            run_device = torch.device("cpu")
            print(f"CUDA context unstable; running {model_name} on CPU.")
        elif model_name == "vit_b_16" and config.force_vit_cpu:
            run_device = torch.device("cpu")
            print("Device override: vit_b_16 forced to CPU.")

        if (
            run_device.type == "cuda"
            and is_windows_cuda
            and config.skip_unstable_windows_cuda_models
            and model_name in unstable_windows_models
        ):
            print(f"\nSkipping {model_name}: unstable on Windows CUDA.")
            failed_models.append({"model": model_name, "reason": "skipped_windows_cuda_unstable"})
            continue

        print(f"\n{'#' * 60}\n# MODEL: {model_name.upper()}\n{'#' * 60}")

        if run_device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except RuntimeError as err:
                if _is_cuda_runtime_failure(str(err).lower()):
                    cuda_runtime_broken = True
                    run_device = torch.device("cpu")
                else:
                    raise

        # Resolve checkpoint dir and resume flag for this model
        checkpoint_dir = Path(config.runs_dir) / "checkpoints" / model_name
        should_resume = (
            resume_model is not None
            and (resume_model == "all" or resume_model == model_name)
            and (checkpoint_dir / "best.pth").exists()
        )

        try:
            active_train = train_loader
            active_val = val_loader
            active_test = test_loader
            if device.type == "cuda" and run_device.type == "cpu":
                if cpu_train_loader is None:
                    cpu_train_loader = _make_cpu_fallback_loader(train_loader, shuffle=True)
                    cpu_val_loader = _make_cpu_fallback_loader(val_loader, shuffle=False)
                    cpu_test_loader = _make_cpu_fallback_loader(test_loader, shuffle=False)
                active_train = cpu_train_loader
                active_val = cpu_val_loader
                active_test = cpu_test_loader

            model = build_model(model_name, num_classes=len(class_names), pretrained=True)
            result = train_model(
                model, model_name,
                active_train, active_val, active_test,
                config, run_device,
                checkpoint_dir=checkpoint_dir,
                resume=should_resume,
            )
            results[model_name] = result

        except Exception as err:
            print(f"\nModel {model_name} failed: {err}")
            if run_device.type == "cuda" and _is_cuda_runtime_failure(str(err).lower()):
                cuda_runtime_broken = True
            failed_models.append({"model": model_name, "reason": str(err)})
            continue

        # Save visualizations
        print(f"\nGenerating visualizations for {model_name}...")
        plot_training_curves(result["history"], model_name, config.output_dir)
        plot_confusion_matrix(
            result["y_true"], result["y_pred"],
            class_names, os.path.join(config.output_dir, f"confusion_{model_name}.png"), model_name,
        )
        model_device = next(result["model"].parameters()).device
        save_sample_predictions(
            result["model"], test_dataset, model_device, class_names,
            os.path.join(config.output_dir, f"sample_predictions_{model_name}.txt"),
        )

        # Persist weights (non-destructive: keep previous run's file until this one is validated)
        weights_path = os.path.join(config.runs_dir, f"{model_name}_weights.pth")
        torch.save(result["model"].state_dict(), weights_path)
        result["weights_path"] = weights_path

        metrics_entry = {
            "model": model_name,
            "val_top1_accuracy": result["best_val_top1"],
            "val_top3_accuracy": result["best_val_top3"],
            "test_top1_accuracy": result["final_top1"],
            "test_top3_accuracy": result["final_top3"],
            "train_subset_size": config.train_subset_size or len(train_loader.dataset),
            "test_subset_size": config.test_subset_size or len(test_dataset),
            "trained_at": datetime.now().isoformat(),
        }

        # Archive: timestamped copy — never silently overwritten
        _archive_model(weights_path, model_name, metrics_entry, config.runs_dir)

        result["model"] = None
        gc.collect()
        if run_device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except RuntimeError as err:
                if _is_cuda_runtime_failure(str(err).lower()):
                    cuda_runtime_broken = True

        all_metrics.append(metrics_entry)

        # Promote best immediately — compares against existing deployed model too
        _promote_best_so_far(
            results, all_metrics, class_names, config,
            runs_dir=config.runs_dir, start_time=start_time,
            models_requested=selected,
        )

    # Save final metrics CSV
    pd.DataFrame(all_metrics).to_csv(os.path.join(config.output_dir, "metrics.csv"), index=False)
    if failed_models:
        print("\nFailed/Skipped models:")
        for item in failed_models:
            print(f"  - {item['model']}: {item['reason']}")

    if not results:
        print("No model finished successfully.")
        return

    best_name = max(results, key=lambda k: (results[k]["best_val_top3"], results[k]["best_val_top1"]))
    best = results[best_name]
    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining complete in {elapsed:.1f} min")
    print(f"  Best this run: {best_name} (Val Top-3={best['best_val_top3']:.2f}%, Top-1={best['final_top1']:.2f}%)")
    print(f"  Checkpoint:    {os.path.join(config.runs_dir, 'best_model.pth')}")
    print(f"  Report:        {os.path.join(config.runs_dir, 'report.json')}")
    print(f"  Archive:       {os.path.join(config.runs_dir, 'archive/')}")
    print(f"  Checkpoints:   {os.path.join(config.runs_dir, 'checkpoints/')}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
