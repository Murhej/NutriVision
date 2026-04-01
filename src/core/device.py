"""Device detection, seeding, and environment utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42, enable_cudnn_benchmark: bool = True) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("cuDNN benchmark enabled for faster GPU training")
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Deterministic mode enabled (slower but reproducible)")


def get_device() -> torch.device:
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {total_memory:.2f} GB")
        torch.cuda.empty_cache()
        print("  GPU cache cleared")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        print("  WARNING: Training on CPU will be significantly slower!")
    return device


def log_environment_info() -> None:
    """Print relevant PyTorch and CUDA environment information."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT INFO")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    """Compute top-k accuracy as a percentage."""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()
