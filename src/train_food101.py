"""
Food-101 Classification Training Pipeline
Single entrypoint script for end-to-end model training and evaluation

Usage: python -m src.train_food101
"""

import os
import json
import gc
import shutil
import time
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms, models
from torchvision.datasets import Food101



import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import (
    set_seed, get_device, log_environment_info,
    topk_accuracy, plot_class_distribution, plot_sample_grid,
    plot_training_curves, plot_confusion_matrix, save_sample_predictions
)


@dataclass
class Config:
    """Configuration for training pipeline"""
    # Data settings
    train_subset_size: int = 0    # 0 for full dataset (75,750) | 15000 = balanced | 2000 = quick test
    test_subset_size: int = 0       # 0 for full dataset (25,250) | 3000 = balanced | 500 = quick test
    val_split: float = 0.1          # Fraction of training split used for validation
    eval_batch_size: int = 16  # Safer default for heavy backbones on smaller GPUs

    # Training settings
    batch_size: int = 32          # Automatically adjusted for GPU/CPU
    num_workers: int = 4          # Automatically adjusted for GPU/CPU
    windows_stable_dataloader: bool = True  # Safer default on Windows + CUDA
    skip_unstable_windows_cuda_models: bool = True
    force_vit_gpu_on_windows: bool = True
    prioritize_gpu_stable_models: bool = True
    disable_tf32_on_windows_cuda: bool = True
    
    # Model settings - add/remove models as needed
    models_to_train: List[str] = None  # Will be set in __post_init__
    
    # Training phases
    warmup_epochs: int = 3      # Train only classification head (backbone frozen)
    finetune_epochs: int = 10   # Fine-tune last blocks (0 to skip)
    
    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    use_amp: bool = True        # Mixed precision training (faster on GPU)
    gradient_accumulation_steps: int = 2  # Simulate larger batch sizes
    max_grad_norm: float = 1.0  # 0 or negative disables clipping
    
    # Regularization
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_augmentation_prob: float = 0.30
    randaugment_magnitude: int = 7
    use_ema: bool = True
    ema_decay: float = 0.9997
    eval_tta: bool = True
    val_tta: bool = False
    tta_num_views: int = 2
    
     # Training control
    early_stopping_patience: int = 5
    min_improvement_delta: float = 0.0
    fast_mode: bool = False
    run_eda: bool = True
    # Performance
    enable_cudnn_benchmark: bool = True  # True for speed, False for reproducibility
    
    # Misc
    seed: int = 42
    data_dir: str = './data'
    output_dir: str = './outputs'
    runs_dir: str = './runs'
    
    def __post_init__(self):
        import platform
        if not (0.0 < self.val_split < 1.0):
            raise ValueError(f"val_split must be between 0 and 1, got {self.val_split}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.eval_batch_size < 1:
            raise ValueError(f"eval_batch_size must be >= 1, got {self.eval_batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        
        if self.models_to_train is None:
            # GPU-first ordering: run CUDA-stable models before CPU-only fallback models.
            self.models_to_train = [ 'vit_b_16','resnet50', 'efficientnet_b0']

        if self.tta_num_views < 1:
            raise ValueError(f"tta_num_views must be >= 1, got {self.tta_num_views}")

        # Fast mode trims optional work to shorten wall-clock runtime.
        if self.fast_mode:
            self.run_eda = False
            if len(self.models_to_train) > 1:
                self.models_to_train = [self.models_to_train[0]]

        if (
            torch.cuda.is_available()
            and platform.system() == 'Windows'
            and self.skip_unstable_windows_cuda_models
        ):
            unstable_windows_models = {'efficientnet_v2_s', 'convnext_base'}
            removed_models = [m for m in self.models_to_train if m in unstable_windows_models]
            if removed_models:
                self.models_to_train = [m for m in self.models_to_train if m not in unstable_windows_models]
                print(
                    "Windows CUDA stability: skipping unstable models on this stack: "
                    + ", ".join(removed_models)
                )
        if (
            torch.cuda.is_available()
            and platform.system() == 'Windows'
            and self.prioritize_gpu_stable_models
        ):
            # Run known-risk models last so GPU-stable models can still finish first.
            risky_windows_cuda_models = {'vit_b_16'}
            stable_first = [m for m in self.models_to_train if m not in risky_windows_cuda_models]
            risky_last = [m for m in self.models_to_train if m in risky_windows_cuda_models]
            reordered = stable_first + risky_last
            if reordered != self.models_to_train:
                self.models_to_train = reordered
                print(
                    "Windows CUDA stability: reordered models to run GPU-stable models first "
                    f"({', '.join(self.models_to_train)})"
                )
        if (
            torch.cuda.is_available()
            and platform.system() == 'Windows'
            and len(self.models_to_train) > 1
            and self.models_to_train[0] == 'vit_b_16'
        ):
            self.models_to_train = [
                m for m in self.models_to_train if m != 'vit_b_16'
            ] + ['vit_b_16']
            print("Windows CUDA stability: moved vit_b_16 to the end to protect GPU context.")
        # Auto-adjust settings for GPU vs CPU
        if torch.cuda.is_available():
            heavy_models = {'vit_b_16', 'resnet50', 'efficientnet_b0'}
            has_heavy_model = any(m in heavy_models for m in self.models_to_train)
            try:
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                print(f"GPU VRAM detected: {total_memory_gb:.2f} GB")
            except Exception:
                total_memory_gb = None
            # GPU optimizations
            if self.batch_size <= 32 and not has_heavy_model:
                # Larger batch size for GPU efficiency
                self.batch_size = 64
            elif self.batch_size <= 32 and has_heavy_model:
                # Heavier backbones are more likely to OOM/crash on smaller GPUs.
                self.batch_size = 16

            # Low-VRAM guardrails to reduce eval/train crashes on heavy models.
            if total_memory_gb is not None and total_memory_gb <= 8.0:
                self.batch_size = min(self.batch_size, 8 if has_heavy_model else 16)
                self.eval_batch_size = min(self.eval_batch_size, 8 if has_heavy_model else 16)
                self.use_ema = False
                self.eval_tta = False
                print("VRAM safety: reduced batch sizes and disabled EMA/TTA")

            if has_heavy_model:
                self.eval_batch_size = min(self.eval_batch_size, max(8, self.batch_size))
            elif self.eval_batch_size <= 16:
                self.eval_batch_size = max(self.batch_size, 32)
            if self.num_workers == 4:  # If using default
                # Windows multiprocessing is problematic - use fewer workers
                if platform.system() == 'Windows':
                    self.num_workers = 2 if self.windows_stable_dataloader else 4
                else:
                    self.num_workers = 8
            if (
                platform.system() == 'Windows'
                and self.windows_stable_dataloader
                and self.enable_cudnn_benchmark
            ):
                # Prevent known cuDNN stream mismatch instability on some Windows GPU stacks.
                self.enable_cudnn_benchmark = False
                print("Windows CUDA stability: forcing cuDNN benchmark OFF")
            print(
                f" GPU detected: Using batch_size={self.batch_size}, "
                f"eval_batch_size={self.eval_batch_size}, num_workers={self.num_workers}"
            )
            if self.use_amp:
                print(" Mixed precision training (AMP) enabled for faster GPU training")
        else:
            # CPU optimizations
            if self.batch_size > 16:
                self.batch_size = 16
            if self.num_workers > 4:
                self.num_workers = 0 if platform.system() == 'Windows' else 4
            self.use_amp = False  # AMP not beneficial on CPU
            print(f" CPU detected: Using batch_size={self.batch_size}, num_workers={self.num_workers}")


def get_transforms(config: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and test transforms with ImageNet normalization
    
    Returns:
        Tuple of (train_transform, test_transform)
    """
    # ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=config.randaugment_magnitude),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.2)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    
    return train_transform, test_transform

    

def load_data(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader, Food101]:
    """
    Load Food-101 dataset with automatic download
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, test_dataset)
    """
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)
    
    train_transform, test_transform = get_transforms(config)
    
    # Download and load datasets
    print("Loading Food-101 dataset (will download if not present)...")
    train_dataset = Food101(
        root=config.data_dir,
        split='train',
        transform=train_transform,
        download=True
    )
    val_dataset = Food101(
        root=config.data_dir,
        split='train',
        transform=test_transform,
        download=False
    )
    
    test_dataset = Food101(
        root=config.data_dir,
        split='test',
        transform=test_transform,
        download=False  # Already downloaded
    )
    
    print(f" Full dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    print(f" Number of classes: {len(train_dataset.classes)}")
    
    # Create subsets if specified
    if config.train_subset_size > 0 and config.train_subset_size < len(train_dataset):
        indices = np.random.choice(len(train_dataset), config.train_subset_size, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f" Using train subset: {len(train_dataset)} samples")
    
    if config.test_subset_size > 0 and config.test_subset_size < len(test_dataset):
        indices = np.random.choice(len(test_dataset), config.test_subset_size, replace=False)
        test_dataset = Subset(test_dataset, indices)
        print(f" Using test subset: {len(test_dataset)} samples")
    
    # Create explicit train/validation split from train split only
    if isinstance(train_dataset, Subset):
        train_pool_indices = np.array(train_dataset.indices)
        base_train_dataset = train_dataset.dataset
    else:
        train_pool_indices = np.arange(len(train_dataset))
        base_train_dataset = train_dataset

    labels_array = np.array(base_train_dataset._labels)
    class_to_indices = {}
    for idx in train_pool_indices:
        label = int(labels_array[int(idx)])
        class_to_indices.setdefault(label, []).append(int(idx))

    target_val_size = int(len(train_pool_indices) * config.val_split)
    if target_val_size == 0 and len(train_pool_indices) > 1:
        target_val_size = 1
    if target_val_size >= len(train_pool_indices):
        target_val_size = len(train_pool_indices) - 1
    if target_val_size <= 0:
        raise ValueError(
            f"Not enough samples ({len(train_pool_indices)}) for val_split={config.val_split}. "
            "Increase train_subset_size or reduce val_split."
        )

    # Maximum validation size that still leaves at least one train sample per class.
    max_safe_val_size = sum(max(0, len(indices) - 1) for indices in class_to_indices.values())
    if max_safe_val_size <= 0:
        raise ValueError(
            "Cannot build a safe validation split: each selected class has only one sample. "
            "Increase train_subset_size."
        )
    if target_val_size > max_safe_val_size:
        print(
            f"Warning: requested val size {target_val_size} is too large for class-safe split; "
            f"reducing to {max_safe_val_size}."
        )
        target_val_size = max_safe_val_size

    class_train_indices = {}
    val_indices = []
    for label, indices in class_to_indices.items():
        cls_indices = np.array(indices, dtype=np.int64)
        np.random.shuffle(cls_indices)
        if len(cls_indices) == 1:
            class_train_indices[label] = cls_indices.tolist()
            continue

        cls_val_size = int(len(cls_indices) * config.val_split)
        cls_val_size = min(cls_val_size, len(cls_indices) - 1)
        if cls_val_size > 0:
            val_indices.extend(cls_indices[:cls_val_size].tolist())
            class_train_indices[label] = cls_indices[cls_val_size:].tolist()
        else:
            class_train_indices[label] = cls_indices.tolist()

    # Top up val samples while keeping at least one train sample per class.
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

    train_indices = []
    for indices in class_train_indices.values():
        train_indices.extend(indices)

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError(
            "Failed to construct a valid train/val split. "
            "Increase train_subset_size or lower val_split."
        )

    train_dataset = Subset(base_train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    train_classes = len({int(labels_array[i]) for i in train_indices})
    val_classes = len({int(labels_array[i]) for i in val_indices})
    print(
        f"Train/Val split created: {len(train_dataset)} train, {len(val_dataset)} val "
        f"(classes: train={train_classes}, val={val_classes})"
    )

    # Use configured worker settings and GPU-friendly transfer options.
    import platform
    loader_num_workers = max(0, int(config.num_workers))
    use_pin_memory = torch.cuda.is_available()
    use_persistent_workers = loader_num_workers > 0

    if (
        platform.system() == 'Windows'
        and torch.cuda.is_available()
        and config.windows_stable_dataloader
    ):
        loader_num_workers = 0
        use_pin_memory = False
        use_persistent_workers = False
        print("Windows stability mode: num_workers=0, pin_memory=False, persistent_workers=False")


    loader_kwargs = {
        'num_workers': loader_num_workers,
        'pin_memory': use_pin_memory,
        'persistent_workers': use_persistent_workers,
    }
    if loader_num_workers > 0:
        loader_kwargs['prefetch_factor'] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        **loader_kwargs
        )

    
    print("="*60 + "\n")
    return train_loader, val_loader, test_loader, test_dataset


def perform_eda(train_dataset, test_dataset, output_dir: str):
    """
    Perform exploratory data analysis and save visualizations
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        output_dir: Directory to save plots
    """
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Basic integrity checks
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Get original dataset for class info
    if isinstance(train_dataset, Subset):
        original_dataset = train_dataset.dataset
    else:
        original_dataset = train_dataset
    
    print(f"Number of classes: {len(original_dataset.classes)}")
    
    # Check sample shape
    sample_img, sample_label = train_dataset[0]
    print(f"Sample image shape: {sample_img.shape}")
    print(f"Sample label: {sample_label} ({original_dataset.classes[sample_label]})")
    
    # Plot class distribution (use full dataset to avoid slow iteration)
    print("   Generating class distribution plot (full dataset)...")
    plot_class_distribution(
        original_dataset,  # Use original dataset for fast plotting
        os.path.join(output_dir, 'class_distribution.png'),
        title='Food-101 Train Class Distribution (Full Dataset)'
    )
    
    # Plot sample grid
    plot_sample_grid(
        train_dataset,
        os.path.join(output_dir, 'sample_grid.png'),
        n_samples=16
    )
    
    print(" EDA complete")
    print("="*60 + "\n")


def create_model(model_name: str, num_classes: int = 101, pretrained: bool = True) -> nn.Module:
    """
    Create a pretrained model with custom head
    
    Args:
        model_name: Name of the model ('resnet50', 'efficientnet_b0', etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model with modified final layer
    """
    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    

    elif model_name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    
    return model


def freeze_backbone(model: nn.Module, model_name: str):
    """Freeze all layers except the final classification head"""
    if model_name == 'resnet50':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    
    elif model_name in ('efficientnet_b0', 'efficientnet_b4', 'efficientnet_v2_s'):
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name == 'convnext_base':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name == 'mobilenet_v3_large':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name == 'vit_b_16':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.parameters():
            param.requires_grad = True
    
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def unfreeze_last_blocks(model: nn.Module, model_name: str):
    """Unfreeze last block(s) for fine-tuning"""
    if model_name == 'resnet50':
        # Unfreeze deeper residual stages and head.
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    
    elif model_name in ('efficientnet_b0', 'efficientnet_b4'):
        # Unfreeze deeper high-level feature blocks.
        for param in model.features[-4:].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True


    elif model_name == 'mobilenet_v3_large':
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True


    elif model_name == 'vit_b_16':
        for param in model.encoder.layers[-4:].parameters():
            param.requires_grad = True
        for param in model.encoder.ln.parameters():
            param.requires_grad = True
        for param in model.heads.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown model: {model_name}")


def is_better_score(current_top3: float, current_top1: float,
                    best_top3: float, best_top1: float,
                    min_delta: float = 0.0) -> bool:
    """Model ranking: prioritize Top-3, then Top-1."""
    if current_top3 > best_top3 + min_delta:
        return True
    if abs(current_top3 - best_top3) <= min_delta and current_top1 > best_top1 + min_delta:
        return True
    return False


def _is_cuda_runtime_failure(err_msg: str) -> bool:
    """Return True for common CUDA/cuDNN runtime failures worth retrying."""
    cuda_error_tokens = (
        "out of memory",
        "cuda out of memory",
        "cudnn_status",
        "cuda error",
        "cuda runtime failure",
        "illegal memory access",
        "illegal instruction",
        "unspecified launch failure",
        "device-side assert triggered",
        "cublas_status_alloc_failed",
    )
    return any(token in err_msg for token in cuda_error_tokens)


def _forward_with_chunk_retry(
    model: nn.Module,
    inputs: torch.Tensor,
    use_amp: bool,
    device: torch.device,
    use_tta: bool,
    tta_views: int,
) -> torch.Tensor:
    """Run inference and recursively split a batch when CUDA memory errors happen."""
    try:
        return _forward_with_tta(
            model,
            inputs,
            use_amp=use_amp,
            device=device,
            tta_views=(tta_views if use_tta else 1),
        )
    except RuntimeError as err:
        err_msg = str(err).lower()
        if device.type != "cuda" or not _is_cuda_runtime_failure(err_msg) or inputs.size(0) <= 1:
            raise
        mid = inputs.size(0) // 2
        if mid <= 0:
            raise
        torch.cuda.empty_cache()
        left = _forward_with_chunk_retry(
            model,
            inputs[:mid],
            use_amp=use_amp,
            device=device,
            use_tta=use_tta,
            tta_views=tta_views,
        )
        right = _forward_with_chunk_retry(
            model,
            inputs[mid:],
            use_amp=use_amp,
            device=device,
            use_tta=use_tta,
            tta_views=tta_views,
        )
        return torch.cat((left, right), dim=0)


def evaluate_with_retries(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False,
    use_amp: bool = False,
    use_tta: bool = False,
    tta_views: int = 2,
) -> Tuple:
    """Evaluate with a CUDA-safe retry that disables AMP/TTA on failure."""
    try:
        return evaluate(
            model,
            loader,
            device,
            return_predictions=return_predictions,
            use_amp=use_amp,
            use_tta=use_tta,
            tta_views=tta_views,
        )
    except RuntimeError as err:
        err_msg = str(err).lower()
        if device.type != "cuda" or not _is_cuda_runtime_failure(err_msg):
            raise
        print("CUDA eval failed; retrying with AMP/TTA disabled...")
        torch.cuda.empty_cache()
        return evaluate(
            model,
            loader,
            device,
            return_predictions=return_predictions,
            use_amp=False,
            use_tta=False,
            tta_views=1,
        )


def initialize_ema_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Create a non-trainable EMA copy of the model."""
    ema_model = deepcopy(model).to(device)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model: Optional[nn.Module], model: nn.Module, decay: float):
    """Update EMA model parameters and keep non-parameter buffers in sync."""
    if ema_model is None:
        return
    one_minus_decay = 1.0 - decay
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(model_param.detach(), alpha=one_minus_decay)
    for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(model_buffer.detach())


def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    """Generate CutMix box coordinates for an input tensor shape [B, C, H, W]."""
    _, _, h, w = size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    return int(x1), int(y1), int(x2), int(y2)


def apply_mixed_augmentation(inputs: torch.Tensor, labels: torch.Tensor,
                             mixup_alpha: float, cutmix_alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp or CutMix and return (inputs, labels_a, labels_b, lam).
    If augmentation is disabled, returns original inputs with lam=1.
    """
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return inputs, labels, labels, 1.0

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < 0.5)
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        rand_index = torch.randperm(inputs.size(0), device=inputs.device)
        labels_a = labels
        labels_b = labels[rand_index]
        x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)
        mixed_inputs = inputs.clone()
        mixed_inputs[:, :, y1:y2, x1:x2] = inputs[rand_index, :, y1:y2, x1:x2]

        patch_area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - (patch_area / (inputs.size(-1) * inputs.size(-2)))
        return mixed_inputs, labels_a, labels_b, float(lam)

    lam = np.random.beta(mixup_alpha, mixup_alpha)
    rand_index = torch.randperm(inputs.size(0), device=inputs.device)
    labels_a = labels
    labels_b = labels[rand_index]
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[rand_index]
    return mixed_inputs, labels_a, labels_b, float(lam)



def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, 
                scaler: torch.cuda.amp.GradScaler = None,
                gradient_accumulation_steps: int = 1,
                mixup_alpha: float = 0.0,
                cutmix_alpha: float = 0.0,
                mix_augmentation_prob: float = 0.0,
                max_grad_norm: float = 0.0,
                ema_model: Optional[nn.Module] = None,
                ema_decay: float = 0.0) -> Tuple[float, float, float]:
    """
    Train for one epoch with optional mixed precision and gradient accumulation
    
    Args:
        model: Model to train
        loader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision training (None to disable AMP)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        mixup_alpha: MixUp alpha parameter (0 to disable)
        cutmix_alpha: CutMix alpha parameter (0 to disable)
        mix_augmentation_prob: Probability of applying MixUp/CutMix to each batch
        max_grad_norm: Gradient clipping norm (0 to disable)
        ema_model: Optional EMA model updated after each optimizer step
        ema_decay: EMA decay factor
    
    Returns:
        Tuple of (avg_loss, top1_acc, top3_acc)
    """
    model.train()
    running_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    total = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for batch_idx, (inputs, labels) in enumerate(pbar):
        # Non-blocking transfer for async GPU copy
        non_blocking = device.type == "cuda"
        inputs = inputs.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        
        use_mix_augmentation = (
            mix_augmentation_prob > 0.0
            and (mixup_alpha > 0.0 or cutmix_alpha > 0.0)
            and np.random.rand() < mix_augmentation_prob
        )
        if use_mix_augmentation:
            inputs, labels_a, labels_b, lam = apply_mixed_augmentation(
                inputs, labels, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha
            )
        else:
            labels_a, labels_b, lam = labels, labels, 1.0
        
        # Mixed precision training
        use_amp = scaler is not None and device.type == "cuda"

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):

                outputs = model(inputs)
                raw_loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
                loss = raw_loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
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
         # Calculate accuracies in fp32 for stability.
        logits = outputs.float()
        top1_a = topk_accuracy(logits, labels_a, k=1)
        top3_a = topk_accuracy(logits, labels_a, k=3)
        if lam < 1.0:
            top1_b = topk_accuracy(logits, labels_b, k=1)
            top3_b = topk_accuracy(logits, labels_b, k=3)
            top1 = lam * top1_a + (1.0 - lam) * top1_b
            top3 = lam * top3_a + (1.0 - lam) * top3_b
        else:
            top1, top3 = top1_a, top3_a
        
        top1_correct += top1 * inputs.size(0) / 100
        top3_correct += top3 * inputs.size(0) / 100
        total += inputs.size(0)
        
        pbar.set_postfix({'loss': raw_loss.item(), 
                          'top1': f'{top1:.2f}%', 'top3': f'{top3:.2f}%'})
    
    # Handlesleftover gradients if total batches
    # is not divisible by gradient_accumlation_steps
    if gradient_accumulation_steps > 1:
        if (batch_idx + 1)% gradient_accumulation_steps !=0:
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
     
    avg_loss = running_loss / total
    top1_acc = 100 * top1_correct / total
    top3_acc = 100 * top3_correct / total
    
    return avg_loss, top1_acc, top3_acc


def _forward_with_tta(model: nn.Module, inputs: torch.Tensor,
                      use_amp: bool, device: torch.device,
                      tta_views: int = 1) -> torch.Tensor:
    if tta_views <= 1:
        if use_amp:
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                return model(inputs)
        return model(inputs)

    # IMPORTANT: make views contiguous
    views = [inputs.contiguous()]
    views.append(torch.flip(inputs, dims=[3]).contiguous())  # horizontal flip

    if tta_views >= 3:
        views.append(torch.flip(inputs, dims=[2]).contiguous())  # vertical flip
    if tta_views >= 4:
        views.append(torch.flip(inputs, dims=[2, 3]).contiguous())  # both flips

    max_views = min(tta_views, len(views))
    logits_sum = None
    for i in range(max_views):
        view = views[i]
        if use_amp:
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(view)
        else:
            logits = model(view)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)

    return logits_sum / float(max_views)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             return_predictions: bool = False, use_amp: bool = False,
             use_tta: bool = False, tta_views: int = 2) -> Tuple:
    """
    Evaluate model on validation/test set with optional mixed precision + TTA
    """
    model.eval()
    top1_correct = 0.0
    top3_correct = 0.0
    total = 0

    all_preds = []
    all_labels = []

    # inference_mode is faster + uses less memory than no_grad
    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        for inputs, labels in pbar:
            non_blocking = device.type == "cuda"
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            outputs = _forward_with_chunk_retry(
                model,
                inputs,
                use_amp=use_amp,
                device=device,
                use_tta=use_tta,
                tta_views=tta_views,
            )

            # make sure metrics are computed in fp32 for stability
            logits = outputs.float()

            top1 = topk_accuracy(logits, labels, k=1)
            top3 = topk_accuracy(logits, labels, k=3)

            bs = inputs.size(0)
            top1_correct += (top1 / 100.0) * bs
            top3_correct += (top3 / 100.0) * bs
            total += bs

            if return_predictions:
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

            pbar.set_postfix({"top1": f"{top1:.2f}%", "top3": f"{top3:.2f}%"})

    # (Optional) helps surface CUDA issues at the right spot during debugging
    if device.type == "cuda":
        torch.cuda.synchronize()

    top1_acc = 100.0 * (top1_correct / total)
    top3_acc = 100.0 * (top3_correct / total)

    if return_predictions:
        return top1_acc, top3_acc, np.array(all_labels), np.array(all_preds)
    return top1_acc, top3_acc


def _make_cpu_fallback_loader(base_loader: DataLoader, shuffle: bool) -> DataLoader:
    """Rebuild loader for CPU-only fallback after CUDA runtime failures."""
    return DataLoader(
        base_loader.dataset,
        batch_size=base_loader.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

def get_param_groups(model: nn.Module, model_name: str, base_lr: float):
    if model_name == "resnet50":
        head_params = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and not n.startswith("fc.")]

    elif model_name in ("efficientnet_b0", "efficientnet_b4", "efficientnet_v2_s", "mobilenet_v3_large"):
        head_params = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and not n.startswith("classifier.")]
    elif model_name == "convnext_base":
        head_params = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and not n.startswith("classifier.")]
    elif model_name == "vit_b_16":
        head_params = list(model.heads.parameters())
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and not n.startswith("heads.")]

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return [
        {"params": backbone_params, "lr": base_lr / 20},
        {"params": head_params, "lr": base_lr / 5},
    ]

def train_model(model: nn.Module, model_name: str, train_loader: DataLoader,
                val_loader: DataLoader, test_loader: DataLoader,
                config: Config, device: torch.device) -> Dict:
    """
    Train a model with two-stage fine-tuning
    
    Args:
        model: Model to train
        model_name: Name of the model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration object
        device: Device to train on
    
    Returns:
        Dictionary with training history and final metrics
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name.upper()}")
    print(f"{'='*60}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    ema_model = initialize_ema_model(model, device) if config.use_ema else None
    import platform
    amp_enabled = bool(config.use_amp and device.type == 'cuda')
    if (
        device.type == 'cuda'
        and platform.system() == 'Windows'
        and model_name == 'vit_b_16'
    ):
        amp_enabled = False
        print(f"  -> Windows CUDA stability: AMP disabled for {model_name}")
    
    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if amp_enabled else None

    if scaler:
        print("   Using automatic mixed precision (AMP) for faster training")
    
    if ema_model is not None:
        print(f"  -> EMA enabled (decay={config.ema_decay})")

    history = {
        'train_loss': [],
        'train_top1': [],
        'train_top3': [],
        'val_top1': [],
        'val_top3': []
    }
    best_state_dict = deepcopy((ema_model or model).state_dict())
    best_top3 = -1.0
    best_top1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stop_training = False
    cuda_runtime_failed = False
    global_epoch = 0
    
    # Phase 1: Warmup - train only head
    if config.warmup_epochs > 0:
        print(f"\n Phase 1: Warmup ({config.warmup_epochs} epoch(s)) - Training head only")
        freeze_backbone(model, model_name)
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.warmup_epochs
        )

        for epoch in range(config.warmup_epochs):
            print(f"\nEpoch {epoch+1}/{config.warmup_epochs}")
            try:
                train_loss, train_top1, train_top3 = train_epoch(
                    model, train_loader, criterion, optimizer, device,
                    scaler=scaler,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    mixup_alpha=config.mixup_alpha,
                    cutmix_alpha=config.cutmix_alpha,
                    mix_augmentation_prob=config.mix_augmentation_prob,
                    max_grad_norm=config.max_grad_norm,
                    ema_model=ema_model,
                    ema_decay=config.ema_decay
                )
            except RuntimeError as err:
                err_msg = str(err).lower()
                if amp_enabled and ("stream_mismatch" in err_msg or "cudnn_status_bad_param" in err_msg):
                    print("AMP backward failed; retrying this epoch with AMP disabled for stability")
                    amp_enabled = False
                    scaler = None
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    train_loss, train_top1, train_top3 = train_epoch(
                        model, train_loader, criterion, optimizer, device,
                        scaler=None,
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        mixup_alpha=config.mixup_alpha,
                        cutmix_alpha=config.cutmix_alpha,
                        mix_augmentation_prob=config.mix_augmentation_prob,
                        max_grad_norm=config.max_grad_norm,
                        ema_model=ema_model,
                        ema_decay=config.ema_decay
                    )
                elif device.type == 'cuda' and _is_cuda_runtime_failure(err_msg):
                    print("CUDA runtime failure during warmup; stopping further epochs for this model.")
                    stop_training = True
                    cuda_runtime_failed = True
                    break
                else:
                    raise
            eval_model = ema_model if ema_model is not None else model
            val_top1, val_top3 = evaluate_with_retries(
                eval_model, val_loader, device,
                use_amp=amp_enabled,
                use_tta=config.val_tta,
                tta_views=config.tta_num_views
            )
            global_epoch += 1
            
            history['train_loss'].append(train_loss)
            history['train_top1'].append(train_top1)
            history['train_top3'].append(train_top3)
            history['val_top1'].append(val_top1)
            history['val_top3'].append(val_top3)
            
            print(f"  Train Loss: {train_loss:.4f} | "
                  f"Train Top-1: {train_top1:.2f}% | Train Top-3: {train_top3:.2f}%")
            print(f"  Val Top-1: {val_top1:.2f}% | Val Top-3: {val_top3:.2f}%")
    
            scheduler.step()

            if is_better_score(
                current_top3=val_top3,
                current_top1=val_top1,
                best_top3=best_top3,
                best_top1=best_top1,
                min_delta=config.min_improvement_delta
            ):
                best_top3 = val_top3
                best_top1 = val_top1
                best_epoch = global_epoch
                best_state_dict = deepcopy(eval_model.state_dict())
                epochs_without_improvement = 0
                print("  -> New best checkpoint saved")
            else:
                epochs_without_improvement += 1
                print(f"  -> No improvement for {epochs_without_improvement} epoch(s)")
                if (
                    config.early_stopping_patience > 0
                    and epochs_without_improvement >= config.early_stopping_patience
                ):
                    print("  -> Early stopping triggered")
                    stop_training = True
                    break

    # Phase 2: Fine-tuning - train last block(s)
    if config.finetune_epochs > 0 and not stop_training:
        print(f"\n Phase 2: Fine-tuning ({config.finetune_epochs} epoch(s)) - Training last blocks")
        unfreeze_last_blocks(model, model_name)
        
        param_groups = get_param_groups(model, model_name, base_lr=config.learning_rate)

        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config.weight_decay
        )

        print(f"   Fine-tune param groups: "
            f"backbone={sum(p.numel() for p in param_groups[0]['params']):,} params, "
            f"head={sum(p.numel() for p in param_groups[1]['params']):,} params")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.finetune_epochs
        )

        
        for epoch in range(config.finetune_epochs):
            print(f"\nEpoch {epoch+1}/{config.finetune_epochs}")
            try:
                train_loss, train_top1, train_top3 = train_epoch(
                    model, train_loader, criterion, optimizer, device,
                    scaler=scaler,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    mixup_alpha=config.mixup_alpha,
                    cutmix_alpha=config.cutmix_alpha,
                    mix_augmentation_prob=config.mix_augmentation_prob,
                    max_grad_norm=config.max_grad_norm,
                    ema_model=ema_model,
                    ema_decay=config.ema_decay
                )
            except RuntimeError as err:
                err_msg = str(err).lower()
                if amp_enabled and ("stream_mismatch" in err_msg or "cudnn_status_bad_param" in err_msg):
                    print("AMP backward failed; retrying this epoch with AMP disabled for stability")
                    amp_enabled = False
                    scaler = None
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    train_loss, train_top1, train_top3 = train_epoch(
                        model, train_loader, criterion, optimizer, device,
                        scaler=None,
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        mixup_alpha=config.mixup_alpha,
                        cutmix_alpha=config.cutmix_alpha,
                        mix_augmentation_prob=config.mix_augmentation_prob,
                        max_grad_norm=config.max_grad_norm,
                        ema_model=ema_model,
                        ema_decay=config.ema_decay
                    )
                elif device.type == 'cuda' and _is_cuda_runtime_failure(err_msg):
                    print("CUDA runtime failure during fine-tuning; stopping further epochs for this model.")
                    stop_training = True
                    cuda_runtime_failed = True
                    break
                else:
                    raise
            eval_model = ema_model if ema_model is not None else model
            val_top1, val_top3 = evaluate_with_retries(
                eval_model, val_loader, device,
                use_amp=amp_enabled,
                use_tta=config.val_tta,
                tta_views=config.tta_num_views
            )
            global_epoch += 1
            history['train_loss'].append(train_loss)
            history['train_top1'].append(train_top1)
            history['train_top3'].append(train_top3)
            history['val_top1'].append(val_top1)
            history['val_top3'].append(val_top3)
            
            print(f"  Train Loss: {train_loss:.4f} | "
                  f"Train Top-1: {train_top1:.2f}% | Train Top-3: {train_top3:.2f}%")
            print(f"  Val Top-1: {val_top1:.2f}% | Val Top-3: {val_top3:.2f}%")
    
            scheduler.step()

            if is_better_score(
                current_top3=val_top3,
                current_top1=val_top1,
                best_top3=best_top3,
                best_top1=best_top1,
                min_delta=config.min_improvement_delta
            ):
                best_top3 = val_top3
                best_top1 = val_top1
                best_epoch = global_epoch
                best_state_dict = deepcopy(eval_model.state_dict())
                epochs_without_improvement = 0
                print("  -> New best checkpoint saved")
            else:
                epochs_without_improvement += 1
                print(f"  -> No improvement for {epochs_without_improvement} epoch(s)")
                if (
                    config.early_stopping_patience > 0
                    and epochs_without_improvement >= config.early_stopping_patience
                ):
                    print("  -> Early stopping triggered")
                    break

    if best_epoch > 0:
        model.load_state_dict(best_state_dict)
        if ema_model is not None:
            ema_model.load_state_dict(best_state_dict)
        print(f"\n-> Restored best checkpoint from epoch {best_epoch} (Top-3={best_top3:.2f}%, Top-1={best_top1:.2f}%)")

    if cuda_runtime_failed and device.type == 'cuda':
        raise RuntimeError("CUDA runtime failure during training for this model")

    # Final evaluation with predictions
    eval_model = ema_model if ema_model is not None else model
    print("\n Final evaluation on test set...")
    import platform
    eval_use_tta = bool(config.eval_tta)
    eval_tta_views = int(config.tta_num_views)
    if device.type == "cuda" and platform.system() == "Windows" and eval_use_tta:
        print("Windows CUDA safety: disabling TTA for final test evaluation")
        eval_use_tta = False
        eval_tta_views = 1

    safe_eval_amp = bool(amp_enabled and (not eval_use_tta))  # disable AMP when TTA is enabled

    cpu_fallback_model = None
    if device.type == "cuda" and platform.system() == "Windows":
        # Prepare fallback model before CUDA eval in case CUDA context becomes unstable.
        cpu_fallback_model = deepcopy(eval_model).to(torch.device("cpu")).eval()

    try:
        final_top1, final_top3, y_true, y_pred = evaluate_with_retries(
            eval_model, test_loader, device,
            return_predictions=True,
            use_amp=safe_eval_amp,
            use_tta=eval_use_tta,
            tta_views=eval_tta_views
        )
    except RuntimeError as err:
        err_msg = str(err).lower()
        if device.type == "cuda" and _is_cuda_runtime_failure(err_msg):
            print("CUDA evaluation failed; retrying final test evaluation on CPU without AMP/TTA...")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            fallback_model = cpu_fallback_model if cpu_fallback_model is not None else deepcopy(eval_model).to(torch.device("cpu")).eval()
            cpu_test_loader = _make_cpu_fallback_loader(test_loader, shuffle=False)
            final_top1, final_top3, y_true, y_pred = evaluate(
                fallback_model, cpu_test_loader, torch.device("cpu"),
                return_predictions=True,
                use_amp=False,
                use_tta=False,
                tta_views=1
            )
            eval_model = fallback_model
        else:
            raise
    print(f"\n Training complete!")
    print(f"  Final Test Top-1 Accuracy: {final_top1:.2f}%")
    print(f"  Final Test Top-3 Accuracy: {final_top3:.2f}%")
    print("="*60)
    
    return {
        'model': eval_model,
        'history': history,
        'best_val_top1': best_top1,
        'best_val_top3': best_top3,
        'final_top1': final_top1,
        'final_top3': final_top3,
        'y_true': y_true,
        'y_pred': y_pred
    }


def main():
    """Main training pipeline"""
    start_time = time.time()
    
    # Initialize
    config = Config()
    set_seed(config.seed, enable_cudnn_benchmark=config.enable_cudnn_benchmark)
    log_environment_info()
    device = get_device()
    if device.type == 'cuda':
        import platform
        if platform.system() == 'Windows' and config.disable_tf32_on_windows_cuda:
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("Windows CUDA stability: TF32 disabled for matmul/cuDNN")
        else:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("GPU throughput mode: TF32 matmul/cuDNN enabled")
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.runs_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Train subset size: {config.train_subset_size if config.train_subset_size > 0 else 'Full dataset'}")
    print(f"Test subset size: {config.test_subset_size if config.test_subset_size > 0 else 'Full dataset'}")
    print(f"Validation split: {config.val_split:.2f}")
    print(f"Batch size: {config.batch_size}")
    print(f"Eval batch size: {config.eval_batch_size}")
    print(f"Warmup epochs: {config.warmup_epochs}")
    print(f"Finetune epochs: {config.finetune_epochs}")
    print(f"MixUp alpha: {config.mixup_alpha}")
    print(f"Mix augmentation prob: {config.mix_augmentation_prob}")
    print(f"RandAugment magnitude: {config.randaugment_magnitude}")
    print(f"Max grad norm: {config.max_grad_norm}")
    print(f"EMA enabled: {config.use_ema} (decay={config.ema_decay})")
    print(f"Validation TTA: {config.val_tta}")
    print(f"Test TTA: {config.eval_tta} (views={config.tta_num_views})")
    print(f"Windows stable dataloader: {config.windows_stable_dataloader}")
    print(f"Skip unstable Windows CUDA models: {config.skip_unstable_windows_cuda_models}")
    print(f"Force vit_b_16 GPU on Windows: {config.force_vit_gpu_on_windows}")
    print(f"Prioritize GPU-stable models: {config.prioritize_gpu_stable_models}")
    print(f"Disable TF32 on Windows CUDA: {config.disable_tf32_on_windows_cuda}")
    print(f"Early stopping patience: {config.early_stopping_patience}")
    print(f"Fast mode: {config.fast_mode}")
    print(f"Run EDA: {config.run_eda}")
    print(f"Models to train: {', '.join(config.models_to_train)}")
    if 0 < config.train_subset_size < 1000:
        print(
            "WARNING: train_subset_size is very small; validation can stay at 0% and early stopping may trigger too soon. "
            "Use >=2000 for meaningful metrics on Food-101."
        )
    print("="*60)
    
    
    # Load data
    train_loader, val_loader, test_loader, test_dataset = load_data(config)
    cpu_train_loader = None
    cpu_val_loader = None
    cpu_test_loader = None
    
    # Get original dataset for EDA and class names
    if isinstance(test_dataset, Subset):
        original_test_dataset = test_dataset.dataset
    else:
        original_test_dataset = test_dataset
    
    class_names = original_test_dataset.classes
    
    train_dataset = train_loader.dataset
    if config.run_eda:
        perform_eda(train_dataset, test_dataset, config.output_dir)
    else:
        print("\nSkipping EDA (run_eda=False)")
     
    # Train all models
    results = {}
    all_metrics = []
    failed_models = []
    import platform
    is_windows_cuda = device.type == 'cuda' and platform.system() == 'Windows'
    cuda_runtime_broken = False
    unstable_windows_models = {'efficientnet_v2_s', 'convnext_base'}
    
    for model_name in config.models_to_train:
        run_device = device
        if device.type == 'cuda' and cuda_runtime_broken:
            run_device = torch.device('cpu')
            print(f"CUDA context is unstable; running {model_name} on CPU fallback.")
        elif (
            is_windows_cuda
            and model_name == 'vit_b_16'
            and not config.force_vit_gpu_on_windows
        ):
            run_device = torch.device('cpu')
            print("Windows CUDA stability: running vit_b_16 on CPU to avoid illegal memory access.")
        elif is_windows_cuda and model_name == 'vit_b_16':
            print("Windows CUDA stability: forcing vit_b_16 to run on GPU as requested.")

        if (
            run_device.type == 'cuda'
            and is_windows_cuda
            and config.skip_unstable_windows_cuda_models
            and model_name in unstable_windows_models
        ):
            print(
                f"\n-> Skipping {model_name}: marked unstable on Windows CUDA "
                "(set skip_unstable_windows_cuda_models=False to force-run)."
            )
            failed_models.append({'model': model_name, 'reason': 'skipped_windows_cuda_unstable'})
            continue

        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name.upper()}")
        print(f"{'#'*60}")

        if run_device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except RuntimeError as cache_err:
                cache_msg = str(cache_err).lower()
                if _is_cuda_runtime_failure(cache_msg):
                    cuda_runtime_broken = True
                    run_device = torch.device('cpu')
                    print("CUDA cache clear failed; switching remaining training to CPU fallback.")
                else:
                    raise

        try:
            active_train_loader = train_loader
            active_val_loader = val_loader
            active_test_loader = test_loader
            if device.type == 'cuda' and run_device.type == 'cpu':
                if cpu_train_loader is None:
                    cpu_train_loader = _make_cpu_fallback_loader(train_loader, shuffle=True)
                    cpu_val_loader = _make_cpu_fallback_loader(val_loader, shuffle=False)
                    cpu_test_loader = _make_cpu_fallback_loader(test_loader, shuffle=False)
                active_train_loader = cpu_train_loader
                active_val_loader = cpu_val_loader
                active_test_loader = cpu_test_loader

            # Create and train model
            model = create_model(model_name, num_classes=len(class_names))
            result = train_model(
                model,
                model_name,
                active_train_loader,
                active_val_loader,
                active_test_loader,
                config,
                run_device,
            )
        
            results[model_name] = result
        except Exception as err:
            print(f"\n-> Model {model_name} failed: {err}")
            if run_device.type == 'cuda' and _is_cuda_runtime_failure(str(err).lower()):
                cuda_runtime_broken = True
                print("Detected fatal CUDA runtime failure; remaining models will run on CPU fallback.")
            failed_models.append({'model': model_name, 'reason': str(err)})
            continue
        
        # Save training curves
        print(f"\n Generating visualizations for {model_name}...")
        plot_training_curves(result['history'], model_name, config.output_dir)
        
        # Save confusion matrix
        cm_path = os.path.join(config.output_dir, f'confusion_{model_name}.png')
        plot_confusion_matrix(
            result['y_true'], result['y_pred'],
            class_names, cm_path, model_name
        )
        
        # Save sample predictions
        pred_path = os.path.join(config.output_dir, f'sample_predictions_{model_name}.txt')
        model_device = next(result['model'].parameters()).device
        save_sample_predictions(
            result['model'], test_dataset, model_device,
            class_names, pred_path, n_samples=5
        )

        # Save model weights now and release model object to avoid VRAM buildup.
        weights_path = os.path.join(config.runs_dir, f'{model_name}_weights.pth')
        torch.save(result['model'].state_dict(), weights_path)
        result['weights_path'] = weights_path
        result['model'] = None
        gc.collect()
        if run_device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except RuntimeError as cache_err:
                cache_msg = str(cache_err).lower()
                if _is_cuda_runtime_failure(cache_msg):
                    cuda_runtime_broken = True
                    print("CUDA cache clear failed after model run; remaining models will use CPU fallback.")
                else:
                    raise
        
        # Collect metrics
        all_metrics.append({
            'model': model_name,
            'val_top1_accuracy': result['best_val_top1'],
            'val_top3_accuracy': result['best_val_top3'],
            'test_top1_accuracy': result['final_top1'],
            'test_top3_accuracy': result['final_top3'],
            'train_subset_size': config.train_subset_size if config.train_subset_size > 0 else len(train_dataset),
            'test_subset_size': config.test_subset_size if config.test_subset_size > 0 else len(test_dataset)
        })
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(config.output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    if failed_models:
        print("\nFailed/Skipped models:")
        for item in failed_models:
            print(f"  - {item['model']}: {item['reason']}")
    print(f"\n Saved metrics to {metrics_path}")
    
    # Select best model by validation score (prioritize top-3, then top-1)
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    if not results:
        print("No model finished successfully. Adjust config (smaller batch/models) and rerun.")
        return
    
    best_model_name = max(
        results.keys(),
        key=lambda k: (results[k]['best_val_top3'], results[k]['best_val_top1'])
    )
    
    best_result = results[best_model_name]
    
    print(f" Best model: {best_model_name}")
    print(f"  Best Val Top-3: {best_result['best_val_top3']:.2f}%")
    print(f"  Best Val Top-1: {best_result['best_val_top1']:.2f}%")
    print(f"  Final Test Top-3: {best_result['final_top3']:.2f}%")
    print(f"  Final Test Top-1: {best_result['final_top1']:.2f}%")
    
    # Save best model
    best_model_path = os.path.join(config.runs_dir, 'best_model.pth')
    best_weights_path = best_result.get('weights_path')
    if best_weights_path is None or not os.path.exists(best_weights_path):
        raise FileNotFoundError(f"Best model weights not found for {best_model_name}")
    shutil.copyfile(best_weights_path, best_model_path)
    print(f" Saved best model weights to {best_model_path}")
    
    # Save report
    report = {
        'best_model_name': best_model_name,
        'best_model_metrics': {
            'val_top1_accuracy': best_result['best_val_top1'],
            'val_top3_accuracy': best_result['best_val_top3'],
            'test_top1_accuracy': best_result['final_top1'],
            'test_top3_accuracy': best_result['final_top3']
        },
        'all_models_metrics': all_metrics,
        'config': asdict(config),
        'timestamp': datetime.now().isoformat(),
        'total_training_time_seconds': time.time() - start_time
    }
    
    report_path = os.path.join(config.runs_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f" Saved training report to {report_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f"\nDataset sizes used:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    print(f"\nModel Performance:")
    for model_name, result in results.items():
        print(f"  {model_name}:")
        print(f"    Top-1 Accuracy: {result['final_top1']:.2f}%")
        print(f"    Top-3 Accuracy: {result['final_top3']:.2f}%")
    
    print(f"\nBest Model: {best_model_name}")
    print(f"  Reason: Highest Validation Top-3 Accuracy ({best_result['best_val_top3']:.2f}%)")
    
    print(f"\nArtifacts saved to:")
    print(f"  Visualizations & Metrics: {config.output_dir}/")
    print(f"  Best Model & Report: {config.runs_dir}/")
    
    print(f"\nTotal training time: {(time.time() - start_time)/60:.2f} minutes")
    print("="*60)


if __name__ == '__main__':
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()

