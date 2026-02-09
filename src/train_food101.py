"""
Food-101 Classification Training Pipeline
Single entrypoint script for end-to-end model training and evaluation

Usage: python -m src.train_food101
"""

import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

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
    train_subset_size: int = 15000  # 0 for full dataset (75,750) | 15000 = balanced | 2000 = quick test
    test_subset_size: int = 3000    # 0 for full dataset (25,250) | 3000 = balanced | 500 = quick test
    
    # Training settings
    batch_size: int = 32           # Automatically adjusted for GPU/CPU
    num_workers: int = 4           # Automatically adjusted for GPU/CPU
    
    # Model settings - add/remove models as needed
    models_to_train: List[str] = None  # Will be set in __post_init__
    
    # Training phases
    warmup_epochs: int = 2      # Train only classification head (backbone frozen)
    finetune_epochs: int = 3    # Fine-tune last blocks (0 to skip)
    
    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    use_amp: bool = True        # Mixed precision training (faster on GPU)
    gradient_accumulation_steps: int = 1  # Simulate larger batch sizes
    
    # Performance
    enable_cudnn_benchmark: bool = True  # True for speed, False for reproducibility
    
    # Misc
    seed: int = 42
    data_dir: str = './data'
    output_dir: str = './outputs'
    runs_dir: str = './runs'
    
    def __post_init__(self):
        import platform
        
        if self.models_to_train is None:
            self.models_to_train = ['resnet50', 'efficientnet_b0']
        
        # Auto-adjust settings for GPU vs CPU
        if torch.cuda.is_available():
            # GPU optimizations
            if self.batch_size == 32:  # If using default
                # Larger batch size for GPU efficiency
                self.batch_size = 64
            if self.num_workers == 4:  # If using default
                # Windows multiprocessing is problematic - use fewer workers
                if platform.system() == 'Windows':
                    self.num_workers = 0  # Disable multiprocessing on Windows
                else:
                    self.num_workers = 8
            print(f"✓ GPU detected: Using batch_size={self.batch_size}, num_workers={self.num_workers}")
            if self.use_amp:
                print("✓ Mixed precision training (AMP) enabled for faster GPU training")
        else:
            # CPU optimizations
            if self.batch_size > 16:
                self.batch_size = 16
            if self.num_workers > 4:
                self.num_workers = 0 if platform.system() == 'Windows' else 4
            self.use_amp = False  # AMP not beneficial on CPU
            print(f"✓ CPU detected: Using batch_size={self.batch_size}, num_workers={self.num_workers}")


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and test transforms with ImageNet normalization
    
    Returns:
        Tuple of (train_transform, test_transform)
    """
    # ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, test_transform


def load_data(config: Config) -> Tuple[DataLoader, DataLoader, Food101]:
    """
    Load Food-101 dataset with automatic download
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (train_loader, test_loader, test_dataset)
    """
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)
    
    train_transform, test_transform = get_transforms()
    
    # Download and load datasets
    print("Loading Food-101 dataset (will download if not present)...")
    train_dataset = Food101(
        root=config.data_dir,
        split='train',
        transform=train_transform,
        download=True
    )
    
    test_dataset = Food101(
        root=config.data_dir,
        split='test',
        transform=test_transform,
        download=False  # Already downloaded
    )
    
    print(f"✓ Full dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    print(f"✓ Number of classes: {len(train_dataset.classes)}")
    
    # Create subsets if specified
    if config.train_subset_size > 0 and config.train_subset_size < len(train_dataset):
        indices = np.random.choice(len(train_dataset), config.train_subset_size, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"✓ Using train subset: {len(train_dataset)} samples")
    
    if config.test_subset_size > 0 and config.test_subset_size < len(test_dataset):
        indices = np.random.choice(len(test_dataset), config.test_subset_size, replace=False)
        test_dataset = Subset(test_dataset, indices)
        print(f"✓ Using test subset: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=config.num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=config.num_workers > 0
    )
    
    print("="*60 + "\n")
    return train_loader, test_loader, test_dataset


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
    print("  → Generating class distribution plot (full dataset)...")
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
    
    print("✓ EDA complete")
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
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
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
    
    elif model_name == 'efficientnet_b0':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, model_name: str):
    """Unfreeze last block(s) for fine-tuning"""
    if model_name == 'resnet50':
        # Unfreeze layer4 and fc
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    
    elif model_name == 'efficientnet_b0':
        # Unfreeze last few blocks
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, 
                scaler: torch.cuda.amp.GradScaler = None,
                gradient_accumulation_steps: int = 1) -> Tuple[float, float, float]:
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
    
    Returns:
        Tuple of (avg_loss, top1_acc, top3_acc)
    """
    model.train()
    running_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for batch_idx, (inputs, labels) in enumerate(pbar):
        # Non-blocking transfer for async GPU copy
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        running_loss += loss.item() * inputs.size(0) * gradient_accumulation_steps
        
        # Calculate accuracies (outside autocast for stability)
        with torch.cuda.amp.autocast(enabled=False):
            top1 = topk_accuracy(outputs.float(), labels, k=1)
            top3 = topk_accuracy(outputs.float(), labels, k=3)
        
        top1_correct += top1 * inputs.size(0) / 100
        top3_correct += top3 * inputs.size(0) / 100
        total += inputs.size(0)
        
        pbar.set_postfix({'loss': loss.item() * gradient_accumulation_steps, 
                         'top1': f'{top1:.2f}%', 'top3': f'{top3:.2f}%'})
    
    avg_loss = running_loss / total
    top1_acc = 100 * top1_correct / total
    top3_acc = 100 * top3_correct / total
    
    return avg_loss, top1_acc, top3_acc


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             return_predictions: bool = False, use_amp: bool = False) -> Tuple:
    """
    Evaluate model on validation/test set with optional mixed precision
    
    Args:
        model: Model to evaluate
        loader: Data loader
        device: Device to evaluate on
        return_predictions: Whether to return predictions for confusion matrix
        use_amp: Use automatic mixed precision for faster evaluation
    
    Returns:
        If return_predictions: (top1_acc, top3_acc, y_true, y_pred)
        Otherwise: (top1_acc, top3_acc)
    """
    model.eval()
    top1_correct = 0
    top3_correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Evaluating', leave=False)
        for inputs, labels in pbar:
            # Non-blocking transfer for async GPU copy
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Use AMP for faster inference
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            top1 = topk_accuracy(outputs, labels, k=1)
            top3 = topk_accuracy(outputs, labels, k=3)
            
            top1_correct += top1 * inputs.size(0) / 100
            top3_correct += top3 * inputs.size(0) / 100
            total += inputs.size(0)
            
            if return_predictions:
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'top1': f'{top1:.2f}%', 'top3': f'{top3:.2f}%'})
    
    top1_acc = 100 * top1_correct / total
    top3_acc = 100 * top3_correct / total
    
    if return_predictions:
        return top1_acc, top3_acc, np.array(all_labels), np.array(all_preds)
    else:
        return top1_acc, top3_acc


def train_model(model: nn.Module, model_name: str, train_loader: DataLoader,
                test_loader: DataLoader, config: Config, device: torch.device) -> Dict:
    """
    Train a model with two-stage fine-tuning
    
    Args:
        model: Model to train
        model_name: Name of the model
        train_loader: Training data loader
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
    criterion = nn.CrossEntropyLoss()
    
    # Initialize GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if (config.use_amp and device.type == 'cuda') else None
    if scaler:
        print("  → Using automatic mixed precision (AMP) for faster training")
    
    history = {
        'train_loss': [],
        'train_top1': [],
        'train_top3': [],
        'val_top1': [],
        'val_top3': []
    }
    
    # Phase 1: Warmup - train only head
    if config.warmup_epochs > 0:
        print(f"\n→ Phase 1: Warmup ({config.warmup_epochs} epoch(s)) - Training head only")
        freeze_backbone(model, model_name)
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        for epoch in range(config.warmup_epochs):
            print(f"\nEpoch {epoch+1}/{config.warmup_epochs}")
            train_loss, train_top1, train_top3 = train_epoch(
                model, train_loader, criterion, optimizer, device,
                scaler=scaler, gradient_accumulation_steps=config.gradient_accumulation_steps
            )
            val_top1, val_top3 = evaluate(model, test_loader, device, use_amp=config.use_amp)
            
            history['train_loss'].append(train_loss)
            history['train_top1'].append(train_top1)
            history['train_top3'].append(train_top3)
            history['val_top1'].append(val_top1)
            history['val_top3'].append(val_top3)
            
            print(f"  Train Loss: {train_loss:.4f} | "
                  f"Train Top-1: {train_top1:.2f}% | Train Top-3: {train_top3:.2f}%")
            print(f"  Val Top-1: {val_top1:.2f}% | Val Top-3: {val_top3:.2f}%")
    
    # Phase 2: Fine-tuning - train last block(s)
    if config.finetune_epochs > 0:
        print(f"\n→ Phase 2: Fine-tuning ({config.finetune_epochs} epoch(s)) - Training last blocks")
        unfreeze_last_blocks(model, model_name)
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate / 10,  # Lower learning rate for fine-tuning
            weight_decay=config.weight_decay
        )
        
        for epoch in range(config.finetune_epochs):
            print(f"\nEpoch {epoch+1}/{config.finetune_epochs}")
            train_loss, train_top1, train_top3 = train_epoch(
                model, train_loader, criterion, optimizer, device,
                scaler=scaler, gradient_accumulation_steps=config.gradient_accumulation_steps
            )
            val_top1, val_top3 = evaluate(model, test_loader, device, use_amp=config.use_amp)
            
            history['train_loss'].append(train_loss)
            history['train_top1'].append(train_top1)
            history['train_top3'].append(train_top3)
            history['val_top1'].append(val_top1)
            history['val_top3'].append(val_top3)
            
            print(f"  Train Loss: {train_loss:.4f} | "
                  f"Train Top-1: {train_top1:.2f}% | Train Top-3: {train_top3:.2f}%")
            print(f"  Val Top-1: {val_top1:.2f}% | Val Top-3: {val_top3:.2f}%")
    
    # Final evaluation with predictions
    print("\n→ Final evaluation on test set...")
    final_top1, final_top3, y_true, y_pred = evaluate(
        model, test_loader, device, return_predictions=True, use_amp=config.use_amp
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Final Test Top-1 Accuracy: {final_top1:.2f}%")
    print(f"  Final Test Top-3 Accuracy: {final_top3:.2f}%")
    print("="*60)
    
    return {
        'model': model,
        'history': history,
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
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.runs_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Train subset size: {config.train_subset_size if config.train_subset_size > 0 else 'Full dataset'}")
    print(f"Test subset size: {config.test_subset_size if config.test_subset_size > 0 else 'Full dataset'}")
    print(f"Batch size: {config.batch_size}")
    print(f"Warmup epochs: {config.warmup_epochs}")
    print(f"Finetune epochs: {config.finetune_epochs}")
    print(f"Models to train: {', '.join(config.models_to_train)}")
    print("="*60)
    
    # Load data
    train_loader, test_loader, test_dataset = load_data(config)
    
    # Get original dataset for EDA and class names
    if isinstance(test_dataset, Subset):
        original_test_dataset = test_dataset.dataset
    else:
        original_test_dataset = test_dataset
    
    class_names = original_test_dataset.classes
    
    # Perform EDA
    train_dataset = train_loader.dataset
    perform_eda(train_dataset, test_dataset, config.output_dir)
    
    # Train all models
    results = {}
    all_metrics = []
    
    for model_name in config.models_to_train:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name.upper()}")
        print(f"{'#'*60}")
        
        # Create and train model
        model = create_model(model_name, num_classes=len(class_names))
        result = train_model(model, model_name, train_loader, test_loader, config, device)
        
        results[model_name] = result
        
        # Save training curves
        print(f"\n→ Generating visualizations for {model_name}...")
        plot_training_curves(result['history'], model_name, config.output_dir)
        
        # Save confusion matrix
        cm_path = os.path.join(config.output_dir, f'confusion_{model_name}.png')
        plot_confusion_matrix(
            result['y_true'], result['y_pred'],
            class_names, cm_path, model_name
        )
        
        # Save sample predictions
        pred_path = os.path.join(config.output_dir, f'sample_predictions_{model_name}.txt')
        save_sample_predictions(
            result['model'], test_dataset, device,
            class_names, pred_path, n_samples=5
        )
        
        # Collect metrics
        all_metrics.append({
            'model': model_name,
            'test_top1_accuracy': result['final_top1'],
            'test_top3_accuracy': result['final_top3'],
            'train_subset_size': config.train_subset_size if config.train_subset_size > 0 else len(train_dataset),
            'test_subset_size': config.test_subset_size if config.test_subset_size > 0 else len(test_dataset)
        })
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(config.output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Saved metrics to {metrics_path}")
    
    # Select best model (prioritize top-3, then top-1)
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    best_model_name = max(
        results.keys(),
        key=lambda k: (results[k]['final_top3'], results[k]['final_top1'])
    )
    
    best_result = results[best_model_name]
    
    print(f"✓ Best model: {best_model_name}")
    print(f"  Top-3 Accuracy: {best_result['final_top3']:.2f}%")
    print(f"  Top-1 Accuracy: {best_result['final_top1']:.2f}%")
    
    # Save best model
    best_model_path = os.path.join(config.runs_dir, 'best_model.pth')
    torch.save(best_result['model'].state_dict(), best_model_path)
    print(f"✓ Saved best model weights to {best_model_path}")
    
    # Save report
    report = {
        'best_model_name': best_model_name,
        'best_model_metrics': {
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
    print(f"✓ Saved training report to {report_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f"\nDataset sizes used:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    print(f"\nModel Performance:")
    for model_name, result in results.items():
        print(f"  {model_name}:")
        print(f"    Top-1 Accuracy: {result['final_top1']:.2f}%")
        print(f"    Top-3 Accuracy: {result['final_top3']:.2f}%")
    
    print(f"\nBest Model: {best_model_name}")
    print(f"  Reason: Highest Top-3 Accuracy ({best_result['final_top3']:.2f}%)")
    
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
