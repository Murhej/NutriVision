"""
Core training engine: epoch loop, evaluation, EMA, mixed augmentation,
CUDA safety helpers, and the two-phase train_model() orchestrator.
"""

from __future__ import annotations

import json
import platform
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.device import topk_accuracy
from src.core.model import forward_with_tta
from src.training.config import Config


# ---------------------------------------------------------------------------
# CUDA safety helpers
# ---------------------------------------------------------------------------

_CUDA_ERROR_TOKENS = (
    "out of memory",
    "cuda out of memory",
    "cudnn_status",
    "cuda error",
    "illegal memory access",
    "illegal instruction",
    "unspecified launch failure",
    "device-side assert triggered",
    "cublas_status_alloc_failed",
)

_CUDA_CORRUPTION_TOKENS = (
    "illegal memory access",
    "illegal instruction",
    "unspecified launch failure",
    "device-side assert triggered",
)


def _is_cuda_runtime_failure(msg: str) -> bool:
    return any(t in msg for t in _CUDA_ERROR_TOKENS)


def _is_cuda_context_corruption(msg: str) -> bool:
    return any(t in msg for t in _CUDA_CORRUPTION_TOKENS)


def _safe_cuda_empty_cache() -> bool:
    if not torch.cuda.is_available():
        return True
    try:
        torch.cuda.empty_cache()
        return True
    except RuntimeError as err:
        if _is_cuda_runtime_failure(str(err).lower()):
            return False
        raise


def _make_cpu_fallback_loader(base_loader: DataLoader, shuffle: bool) -> DataLoader:
    return DataLoader(
        base_loader.dataset,
        batch_size=base_loader.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    checkpoint_dir: Path,
    model_name: str,
    global_epoch: int,
    state_dict: Dict,
    meta: Dict,
    is_best: bool = False,
) -> Path:
    """
    Persist a state dict + metadata JSON to checkpoint_dir.

    Always writes  epoch_NNN.pth / epoch_NNN.json.
    If is_best=True also copies to  best.pth / best.json.
    Returns the epoch checkpoint path.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = f"epoch_{global_epoch:03d}"
    weights_path = checkpoint_dir / f"{tag}.pth"
    meta_path    = checkpoint_dir / f"{tag}.json"

    torch.save(state_dict, weights_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({**meta, "checkpoint_file": str(weights_path)}, f, indent=2)

    if is_best:
        best_w = checkpoint_dir / "best.pth"
        best_m = checkpoint_dir / "best.json"
        import shutil
        shutil.copy2(weights_path, best_w)
        with open(best_m, "w", encoding="utf-8") as f:
            json.dump({**meta, "checkpoint_file": str(best_w)}, f, indent=2)

    return weights_path


def load_resume_checkpoint(checkpoint_dir: Path, model_name: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Load (state_dict, meta) from the best checkpoint for *model_name*.
    Returns (None, None) if no checkpoint exists.
    """
    best_w = checkpoint_dir / "best.pth"
    best_m = checkpoint_dir / "best.json"
    if not best_w.exists() or not best_m.exists():
        return None, None
    try:
        state_dict = torch.load(best_w, map_location="cpu", weights_only=True)
        with open(best_m, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"  [Resume] Loaded checkpoint: epoch {meta.get('global_epoch', '?')}  "
              f"val_top1={meta.get('val_top1', 0):.2f}%  val_top3={meta.get('val_top3', 0):.2f}%")
        return state_dict, meta
    except Exception as e:
        print(f"  [Resume] Failed to load checkpoint: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Backbone freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freeze all parameters except the classification head."""
    for param in model.parameters():
        param.requires_grad = False

    if model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name in ("efficientnet_b0", "efficientnet_b4", "efficientnet_v2_s", "convnext_base", "mobilenet_v3_large"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "vit_b_16":
        for param in model.heads.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model for freeze_backbone: {model_name}")


def unfreeze_last_blocks(model: nn.Module, model_name: str) -> None:
    """Unfreeze the last feature blocks and classification head for fine-tuning."""
    if model_name == "resnet50":
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name in ("efficientnet_b0", "efficientnet_b4"):
        for param in model.features[-4:].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "mobilenet_v3_large":
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "vit_b_16":
        for param in model.encoder.layers[-4:].parameters():
            param.requires_grad = True
        for param in model.encoder.ln.parameters():
            param.requires_grad = True
        for param in model.heads.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model for unfreeze_last_blocks: {model_name}")


def get_param_groups(model: nn.Module, model_name: str, base_lr: float):
    """Return optimizer param groups: backbone at lr/20, head at lr/5."""
    if model_name == "resnet50":
        head_params = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc.")]
    elif model_name in ("efficientnet_b0", "efficientnet_b4", "efficientnet_v2_s", "mobilenet_v3_large"):
        head_params = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.")]
    elif model_name == "convnext_base":
        head_params = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.")]
    elif model_name == "vit_b_16":
        head_params = list(model.heads.parameters())
        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("heads.")]
    else:
        raise ValueError(f"Unknown model_name for param groups: {model_name}")

    return [
        {"params": backbone_params, "lr": base_lr / 20},
        {"params": head_params, "lr": base_lr / 5},
    ]


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def initialize_ema_model(model: nn.Module, device: torch.device) -> nn.Module:
    ema = deepcopy(model).to(device)
    ema.eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema_model(ema: nn.Module, model: nn.Module, decay: float) -> None:
    one_minus = 1.0 - decay
    for ema_p, model_p in zip(ema.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(model_p.detach(), alpha=one_minus)
    for ema_b, model_b in zip(ema.buffers(), model.buffers()):
        ema_b.copy_(model_b.detach())


# ---------------------------------------------------------------------------
# Mixed augmentation (MixUp / CutMix)
# ---------------------------------------------------------------------------

def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    _, _, h, w = size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    return int(x1), int(y1), int(x2), int(y2)


def apply_mixed_augmentation(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return inputs, labels, labels, 1.0

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < 0.5)
    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        rand_index = torch.randperm(inputs.size(0), device=inputs.device)
        labels_a, labels_b = labels, labels[rand_index]
        x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)
        mixed = inputs.clone()
        mixed[:, :, y1:y2, x1:x2] = inputs[rand_index, :, y1:y2, x1:x2]
        patch_area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - patch_area / (inputs.size(-1) * inputs.size(-2))
        return mixed, labels_a, labels_b, lam

    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
    rand_index = torch.randperm(inputs.size(0), device=inputs.device)
    labels_a, labels_b = labels, labels[rand_index]
    mixed = lam * inputs + (1.0 - lam) * inputs[rand_index]
    return mixed, labels_a, labels_b, lam


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def is_better_score(
    current_top3: float,
    current_top1: float,
    best_top3: float,
    best_top1: float,
    min_delta: float = 0.0,
) -> bool:
    """Rank by Top-3 first, then Top-1."""
    if current_top3 > best_top3 + min_delta:
        return True
    if abs(current_top3 - best_top3) <= min_delta and current_top1 > best_top1 + min_delta:
        return True
    return False


def format_metric_delta(current: float, baseline: Optional[float]) -> str:
    if baseline is None:
        return f"{current:.2f}%"
    delta = current - baseline
    sign = "+" if delta >= 0 else ""
    return f"{current:.2f}% ({sign}{delta:.2f})"


# ---------------------------------------------------------------------------
# Per-batch forward with CUDA OOM retry
# ---------------------------------------------------------------------------

def _forward_with_chunk_retry(
    model: nn.Module,
    inputs: torch.Tensor,
    use_amp: bool,
    device: torch.device,
    use_tta: bool,
    tta_views: int,
) -> torch.Tensor:
    try:
        return forward_with_tta(
            model, inputs, device,
            tta_views=(tta_views if use_tta else 1),
            use_amp=use_amp,
        )
    except RuntimeError as err:
        err_msg = str(err).lower()
        if device.type != "cuda" or not _is_cuda_runtime_failure(err_msg) or inputs.size(0) <= 1:
            raise
        if _is_cuda_context_corruption(err_msg):
            raise
        mid = inputs.size(0) // 2
        _safe_cuda_empty_cache()
        left = _forward_with_chunk_retry(model, inputs[:mid], use_amp, device, use_tta, tta_views)
        right = _forward_with_chunk_retry(model, inputs[mid:], use_amp, device, use_tta, tta_views)
        return torch.cat((left, right), dim=0)


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    gradient_accumulation_steps: int = 1,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_augmentation_prob: float = 0.0,
    max_grad_norm: float = 0.0,
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.0,
) -> Tuple[float, float, float]:
    """Train for one epoch. Returns (avg_loss, top1_acc%, top3_acc%)."""
    model.train()
    running_loss = 0.0
    top1_correct = 0.0
    top3_correct = 0.0
    total = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Training", leave=False)
    last_batch_idx = 0
    for batch_idx, (inputs, labels) in enumerate(pbar):
        last_batch_idx = batch_idx
        non_blocking = device.type == "cuda"
        inputs = inputs.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        use_mix = (
            mix_augmentation_prob > 0.0
            and (mixup_alpha > 0.0 or cutmix_alpha > 0.0)
            and np.random.rand() < mix_augmentation_prob
        )
        if use_mix:
            inputs, labels_a, labels_b, lam = apply_mixed_augmentation(inputs, labels, mixup_alpha, cutmix_alpha)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        use_amp = scaler is not None and device.type == "cuda"
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs)
                raw_loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
                loss = raw_loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                if ema_model is not None:
                    update_ema_model(ema_model, model, ema_decay)
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = model(inputs)
            raw_loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
            loss = raw_loss / gradient_accumulation_steps
            loss.backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if ema_model is not None:
                    update_ema_model(ema_model, model, ema_decay)
                optimizer.zero_grad(set_to_none=True)

        running_loss += raw_loss.item() * inputs.size(0)
        logits = outputs.float()
        top1 = topk_accuracy(logits, labels_a, k=1)
        top3 = topk_accuracy(logits, labels_a, k=3)
        if lam < 1.0:
            top1 = lam * top1 + (1.0 - lam) * topk_accuracy(logits, labels_b, k=1)
            top3 = lam * top3 + (1.0 - lam) * topk_accuracy(logits, labels_b, k=3)
        top1_correct += top1 * inputs.size(0) / 100
        top3_correct += top3 * inputs.size(0) / 100
        total += inputs.size(0)
        pbar.set_postfix({"loss": raw_loss.item(), "top1": f"{top1:.2f}%", "top3": f"{top3:.2f}%"})

    # Flush leftover gradients
    if gradient_accumulation_steps > 1 and (last_batch_idx + 1) % gradient_accumulation_steps != 0:
        if scaler is not None:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                update_ema_model(ema_model, model, ema_decay)
        else:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, ema_decay)
        optimizer.zero_grad(set_to_none=True)

    return running_loss / total, 100 * top1_correct / total, 100 * top3_correct / total


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False,
    use_amp: bool = False,
    use_tta: bool = False,
    tta_views: int = 2,
) -> Tuple:
    """Evaluate on a loader. Returns (top1%, top3%) or (top1%, top3%, y_true, y_pred)."""
    model.eval()
    top1_correct = 0.0
    top3_correct = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

            outputs = _forward_with_chunk_retry(model, inputs, use_amp, device, use_tta, tta_views)
            logits = outputs.float()
            top1 = topk_accuracy(logits, labels, k=1)
            top3 = topk_accuracy(logits, labels, k=3)
            bs = inputs.size(0)
            top1_correct += (top1 / 100.0) * bs
            top3_correct += (top3 / 100.0) * bs
            total += bs

            if return_predictions:
                all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

            pbar.set_postfix({"top1": f"{top1:.2f}%", "top3": f"{top3:.2f}%"})

    if device.type == "cuda":
        torch.cuda.synchronize()

    top1_acc = 100.0 * (top1_correct / total)
    top3_acc = 100.0 * (top3_correct / total)

    if return_predictions:
        return top1_acc, top3_acc, np.array(all_labels), np.array(all_preds)
    return top1_acc, top3_acc


def evaluate_with_retries(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False,
    use_amp: bool = False,
    use_tta: bool = False,
    tta_views: int = 2,
    cpu_fallback_model: Optional[nn.Module] = None,
) -> Tuple:
    """Evaluate with CUDA-safe fallback: retries without AMP/TTA, then falls back to CPU."""
    try:
        return evaluate(model, loader, device, return_predictions, use_amp, use_tta, tta_views)
    except RuntimeError as err:
        err_msg = str(err).lower()
        if device.type != "cuda" or not _is_cuda_runtime_failure(err_msg):
            raise
        if cpu_fallback_model is None:
            cpu_fallback_model = deepcopy(model).to(torch.device("cpu")).eval()
        cpu_loader = _make_cpu_fallback_loader(loader, shuffle=False)

        if _is_cuda_context_corruption(err_msg):
            print("CUDA eval failed; falling back to CPU evaluation...")
            return evaluate(cpu_fallback_model, cpu_loader, torch.device("cpu"), return_predictions, False, False, 1)

        print("CUDA eval failed; retrying without AMP/TTA...")
        if _safe_cuda_empty_cache():
            try:
                return evaluate(model, loader, device, return_predictions, False, False, 1)
            except RuntimeError as retry_err:
                if not _is_cuda_runtime_failure(str(retry_err).lower()):
                    raise

        print("Falling back to CPU evaluation...")
        return evaluate(cpu_fallback_model, cpu_loader, torch.device("cpu"), return_predictions, False, False, 1)


# ---------------------------------------------------------------------------
# Two-phase train_model orchestrator
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Config,
    device: torch.device,
    comparison_metrics: Optional[Dict[str, float]] = None,
    checkpoint_dir: Optional[Path] = None,
    resume: bool = False,
) -> Dict:
    """
    Train a model with warmup (head only) + fine-tuning (last blocks) phases.

    Args:
        checkpoint_dir: directory to write per-epoch checkpoints + best.pth/best.json.
        resume:         if True, load best.pth from checkpoint_dir and continue.

    Returns a dict with: model, history, best_val_top1, best_val_top3, final_top1, final_top3, y_true, y_pred.
    """
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name.upper()}")
    print(f"{'=' * 60}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    ema_model = initialize_ema_model(model, device) if config.use_ema else None

    is_windows = platform.system() == "Windows"
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    if device.type == "cuda" and is_windows and model_name == "vit_b_16":
        amp_enabled = False
        print(f"  Windows CUDA stability: AMP disabled for {model_name}")

    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    if scaler:
        print("  AMP enabled for faster training")
    if ema_model is not None:
        print(f"  EMA enabled (decay={config.ema_decay})")
    if comparison_metrics:
        print(
            f"  Baseline to beat: "
            f"Top-1 {comparison_metrics.get('test_top1_accuracy', 0):.2f}% | "
            f"Top-3 {comparison_metrics.get('test_top3_accuracy', 0):.2f}%"
        )

    history: Dict = {"train_loss": [], "train_top1": [], "train_top3": [], "val_top1": [], "val_top3": []}
    best_state_dict = deepcopy((ema_model or model).state_dict())
    best_top3 = -1.0
    best_top1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stop_training = False
    cuda_runtime_failed = False
    global_epoch = 0
    train_start_time = datetime.now()

    # --- Resume from checkpoint ---
    resume_from_global_epoch = 0
    if resume and checkpoint_dir is not None:
        ckpt_state, ckpt_meta = load_resume_checkpoint(checkpoint_dir, model_name)
        if ckpt_state is not None:
            (ema_model or model).load_state_dict(ckpt_state)
            model.load_state_dict(ckpt_state)
            resume_from_global_epoch = int(ckpt_meta.get("global_epoch", 0))
            best_top3 = float(ckpt_meta.get("best_val_top3", -1.0))
            best_top1 = float(ckpt_meta.get("best_val_top1", -1.0))
            best_epoch = resume_from_global_epoch
            best_state_dict = ckpt_state
            history = ckpt_meta.get("history", history)
            global_epoch = resume_from_global_epoch
            print(f"  Resuming from global epoch {resume_from_global_epoch}")

    def _run_epoch(phase_epoch: int, phase_total: int) -> Optional[Tuple[float, float, float]]:
        nonlocal amp_enabled, scaler, stop_training, cuda_runtime_failed
        print(f"\nEpoch {phase_epoch + 1}/{phase_total}")
        try:
            return train_epoch(
                model, train_loader, criterion, optimizer, device,
                scaler=scaler,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                mixup_alpha=config.mixup_alpha,
                cutmix_alpha=config.cutmix_alpha,
                mix_augmentation_prob=config.mix_augmentation_prob,
                max_grad_norm=config.max_grad_norm,
                ema_model=ema_model,
                ema_decay=config.ema_decay,
            )
        except RuntimeError as err:
            err_msg = str(err).lower()
            if amp_enabled and ("stream_mismatch" in err_msg or "cudnn_status_bad_param" in err_msg):
                print("AMP backward failed; retrying epoch without AMP")
                amp_enabled = False
                scaler = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                return train_epoch(
                    model, train_loader, criterion, optimizer, device,
                    scaler=None,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    mixup_alpha=config.mixup_alpha,
                    cutmix_alpha=config.cutmix_alpha,
                    mix_augmentation_prob=config.mix_augmentation_prob,
                    max_grad_norm=config.max_grad_norm,
                    ema_model=ema_model,
                    ema_decay=config.ema_decay,
                )
            if device.type == "cuda" and _is_cuda_runtime_failure(err_msg):
                print("CUDA runtime failure; stopping training for this model.")
                stop_training = True
                cuda_runtime_failed = True
                return None
            raise

    def _record_and_check(train_loss: float, train_top1: float, train_top3: float) -> bool:
        """Record epoch metrics, update best checkpoint, check early stopping."""
        nonlocal best_top3, best_top1, best_epoch, epochs_without_improvement, stop_training, best_state_dict

        eval_model = ema_model if ema_model is not None else model
        cpu_fallback = (
            deepcopy(eval_model).to(torch.device("cpu")).eval()
            if device.type == "cuda" and is_windows
            else None
        )
        val_top1, val_top3 = evaluate_with_retries(
            eval_model, val_loader, device,
            use_amp=amp_enabled,
            use_tta=config.val_tta,
            tta_views=config.tta_num_views,
            cpu_fallback_model=cpu_fallback,
        )

        history["train_loss"].append(train_loss)
        history["train_top1"].append(train_top1)
        history["train_top3"].append(train_top3)
        history["val_top1"].append(val_top1)
        history["val_top3"].append(val_top3)

        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Train Top-1: {train_top1:.2f}% | Train Top-3: {train_top3:.2f}%"
        )
        print(f"  Val Top-1: {val_top1:.2f}% | Val Top-3: {val_top3:.2f}%")
        if comparison_metrics:
            print(
                f"  Val vs baseline: "
                f"Top-1 {format_metric_delta(val_top1, comparison_metrics.get('val_top1_accuracy'))} | "
                f"Top-3 {format_metric_delta(val_top3, comparison_metrics.get('val_top3_accuracy'))}"
            )

        new_best = is_better_score(val_top3, val_top1, best_top3, best_top1, config.min_improvement_delta)
        if new_best:
            best_top3 = val_top3
            best_top1 = val_top1
            best_epoch = global_epoch
            best_state_dict = deepcopy(eval_model.state_dict())
            epochs_without_improvement = 0
            print("  -> New best checkpoint saved")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s)")

        # --- Persist checkpoint ---
        if checkpoint_dir is not None:
            elapsed = (datetime.now() - train_start_time).total_seconds()
            ckpt_meta = {
                "model_name": model_name,
                "global_epoch": global_epoch,
                "train_loss": round(train_loss, 6),
                "train_top1": round(train_top1, 4),
                "train_top3": round(train_top3, 4),
                "val_top1": round(val_top1, 4),
                "val_top3": round(val_top3, 4),
                "best_val_top1": round(best_top1, 4),
                "best_val_top3": round(best_top3, 4),
                "is_best": new_best,
                "history": history,
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": round(elapsed, 1),
            }
            save_checkpoint(
                checkpoint_dir, model_name, global_epoch,
                eval_model.state_dict(), ckpt_meta, is_best=new_best,
            )

        if new_best:
            pass  # already handled above
        elif config.early_stopping_patience > 0 and epochs_without_improvement >= config.early_stopping_patience:
            print("  -> Early stopping triggered")
            stop_training = True
            return False
        return True

    # --- Phase 1: warmup (head only) ---
    if config.warmup_epochs > 0:
        warmup_start = 0
        if resume_from_global_epoch > 0:
            warmup_start = min(resume_from_global_epoch, config.warmup_epochs)
        if warmup_start < config.warmup_epochs:
            skipped = f" (resuming from epoch {warmup_start})" if warmup_start > 0 else ""
            print(f"\nPhase 1: Warmup ({config.warmup_epochs} epoch(s)) — training head only{skipped}")
            freeze_backbone(model, model_name)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate, weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.warmup_epochs)
            # Advance scheduler to match skipped epochs
            for _ in range(warmup_start):
                scheduler.step()
            for epoch in range(warmup_start, config.warmup_epochs):
                result = _run_epoch(epoch, config.warmup_epochs)
                if result is None or stop_training:
                    break
                global_epoch += 1
                _record_and_check(*result)
                scheduler.step()
                if stop_training:
                    break
        else:
            print(f"\nPhase 1: Warmup — all {config.warmup_epochs} epoch(s) already completed, skipping")
            global_epoch = max(global_epoch, config.warmup_epochs)

    # --- Phase 2: fine-tuning (last blocks) ---
    if config.finetune_epochs > 0 and not stop_training:
        finetune_start = 0
        if resume_from_global_epoch > config.warmup_epochs:
            finetune_start = min(resume_from_global_epoch - config.warmup_epochs, config.finetune_epochs)
        if finetune_start < config.finetune_epochs:
            skipped = f" (resuming from epoch {finetune_start})" if finetune_start > 0 else ""
            print(f"\nPhase 2: Fine-tuning ({config.finetune_epochs} epoch(s)) — last blocks{skipped}")
            unfreeze_last_blocks(model, model_name)
            param_groups = get_param_groups(model, model_name, config.learning_rate)
            optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.finetune_epochs)
            for _ in range(finetune_start):
                scheduler.step()
            for epoch in range(finetune_start, config.finetune_epochs):
                result = _run_epoch(epoch, config.finetune_epochs)
                if result is None or stop_training:
                    break
                global_epoch += 1
                _record_and_check(*result)
                scheduler.step()
                if stop_training:
                    break
        else:
            print(f"\nPhase 2: Fine-tuning — all {config.finetune_epochs} epoch(s) already completed, skipping")

    if best_epoch > 0:
        model.load_state_dict(best_state_dict)
        if ema_model is not None:
            ema_model.load_state_dict(best_state_dict)
        print(f"\nRestored best checkpoint from epoch {best_epoch} (Top-3={best_top3:.2f}%, Top-1={best_top1:.2f}%)")

    if cuda_runtime_failed and device.type == "cuda":
        raise RuntimeError("CUDA runtime failure during training")

    # --- Final test evaluation ---
    eval_model = ema_model if ema_model is not None else model
    print("\nFinal evaluation on test set...")
    eval_use_tta = bool(config.eval_tta)
    eval_tta_views = int(config.tta_num_views)
    if device.type == "cuda" and is_windows and eval_use_tta:
        print("Windows CUDA safety: disabling TTA for final test evaluation")
        eval_use_tta = False
        eval_tta_views = 1

    cpu_fallback = (
        deepcopy(eval_model).to(torch.device("cpu")).eval()
        if device.type == "cuda" and is_windows
        else None
    )
    try:
        final_top1, final_top3, y_true, y_pred = evaluate_with_retries(
            eval_model, test_loader, device,
            return_predictions=True,
            use_amp=bool(amp_enabled and not eval_use_tta),
            use_tta=eval_use_tta,
            tta_views=eval_tta_views,
            cpu_fallback_model=cpu_fallback,
        )
    except RuntimeError as err:
        if device.type == "cuda" and _is_cuda_runtime_failure(str(err).lower()):
            print("CUDA eval failed; retrying on CPU...")
            fallback = cpu_fallback or deepcopy(eval_model).to(torch.device("cpu")).eval()
            cpu_loader = _make_cpu_fallback_loader(test_loader, shuffle=False)
            final_top1, final_top3, y_true, y_pred = evaluate(
                fallback, cpu_loader, torch.device("cpu"), return_predictions=True
            )
            eval_model = fallback
        else:
            raise

    print(f"\nTraining complete!")
    print(f"  Final Test Top-1: {final_top1:.2f}%")
    print(f"  Final Test Top-3: {final_top3:.2f}%")
    if comparison_metrics:
        print(
            f"  Test vs baseline: "
            f"Top-1 {format_metric_delta(final_top1, comparison_metrics.get('test_top1_accuracy'))} | "
            f"Top-3 {format_metric_delta(final_top3, comparison_metrics.get('test_top3_accuracy'))}"
        )

    return {
        "model": eval_model,
        "history": history,
        "best_val_top1": best_top1,
        "best_val_top3": best_top3,
        "final_top1": final_top1,
        "final_top3": final_top3,
        "y_true": y_true,
        "y_pred": y_pred,
    }
