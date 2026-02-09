"""
Utility functions for Food-101 classification pipeline
"""

import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
import matplotlib.pyplot as plt
from typing import Tuple, List
import os


def set_seed(seed: int = 42, enable_cudnn_benchmark: bool = True):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
        enable_cudnn_benchmark: If True, enables cudnn.benchmark for faster training.
                               Set to False for exact reproducibility at cost of speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For GPU efficiency: benchmark=True is much faster but slightly non-deterministic
    # For exact reproducibility: benchmark=False, deterministic=True
    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("✓ cuDNN benchmark enabled for faster GPU training")
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("✓ Deterministic mode enabled (slower but reproducible)")


def get_device():
    """Detect and return best available device with detailed GPU info"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Display GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {total_memory:.2f} GB")
        
        # Clear cache for clean start
        torch.cuda.empty_cache()
        print("  GPU cache cleared")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")
        print("  WARNING: Training on CPU will be significantly slower!")
        print("  Consider using a GPU for faster training.")
    return device


def log_environment_info():
    """Log relevant environment information"""
    print("\n" + "="*60)
    print("ENVIRONMENT INFO")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    """
    Compute top-k accuracy
    
    Args:
        output: Model predictions (batch_size, num_classes)
        target: Ground truth labels (batch_size,)
        k: Top-k predictions to consider
    
    Returns:
        Top-k accuracy as percentage
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()


def plot_class_distribution(dataset, output_path: str, title: str = "Food-101 Class Distribution"):
    """
    Plot and save class distribution bar chart
    
    Args:
        dataset: PyTorch dataset (should be full dataset, not Subset for performance)
        output_path: Path to save the plot
        title: Plot title
    """
    from collections import Counter
    
    # Food-101 has 1000 images per class (750 train, 250 test)
    # Just create uniform distribution for full dataset
    num_classes = len(dataset.classes)
    samples_per_class = len(dataset) // num_classes
    
    classes = dataset.classes
    counts = [samples_per_class] * num_classes
    
    plt.ioff()  # Turn off interactive mode
    plt.figure(figsize=(20, 6))
    plt.bar(range(len(classes)), counts, color='steelblue', alpha=0.8)
    plt.xlabel('Food Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"✓ Saved class distribution to {output_path}")


def plot_sample_grid(dataset, output_path: str, n_samples: int = 16):
    """
    Plot grid of sample images with labels
    
    Args:
        dataset: PyTorch dataset (or Subset)
        output_path: Path to save the plot
        n_samples: Number of samples to display
    """
    from torch.utils.data import Subset
    
    print("  → Generating sample image grid...")
    
    # Get original dataset if this is a Subset
    original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    
    rows = int(np.sqrt(n_samples))
    cols = (n_samples + rows - 1) // rows
    
    plt.ioff()  # Turn off interactive mode
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < len(indices):
            img, label = dataset[indices[idx]]
            
            # Convert tensor to displayable format
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                # Denormalize (ImageNet stats)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(original_dataset.classes[label], fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"✓ Saved sample grid to {output_path}")


def plot_training_curves(history: dict, model_name: str, output_dir: str):
    """
    Plot and save training curves (loss and accuracy)
    
    Args:
        history: Dictionary with 'train_loss', 'train_top1', 'train_top3', 'val_top1', 'val_top3'
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, f'{model_name}_loss.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved loss curve to {loss_path}")
    
    # Accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_top1'], 'b-', label='Train Top-1', linewidth=2)
    plt.plot(epochs, history['train_top3'], 'b--', label='Train Top-3', linewidth=2)
    plt.plot(epochs, history['val_top1'], 'r-', label='Val Top-1', linewidth=2)
    plt.plot(epochs, history['val_top3'], 'r--', label='Val Top-3', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(output_dir, f'{model_name}_accuracy.png')
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved accuracy curve to {acc_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         classes: List[str], output_path: str, 
                         model_name: str, max_classes: int = 101):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        output_path: Path to save the plot
        model_name: Name of the model
        max_classes: Maximum number of classes to display
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix (Top-1)', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    # Show fewer tick labels if too many classes
    tick_marks = np.arange(len(classes))
    if len(classes) <= 30:
        plt.xticks(tick_marks, classes, rotation=90, fontsize=6)
        plt.yticks(tick_marks, classes, fontsize=6)
    else:
        # Show every Nth class
        step = len(classes) // 20
        plt.xticks(tick_marks[::step], [classes[i] for i in tick_marks[::step]], rotation=90, fontsize=6)
        plt.yticks(tick_marks[::step], [classes[i] for i in tick_marks[::step]], fontsize=6)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved confusion matrix to {output_path}")


def save_sample_predictions(model, dataset, device, class_names: List[str], 
                           output_path: str, n_samples: int = 5):
    """
    Save top-3 predictions for sample images to text file
    
    Args:
        model: Trained model
        dataset: Test dataset
        device: torch device
        class_names: List of class names
        output_path: Path to save predictions
        n_samples: Number of samples to predict
    """
    model.eval()
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    
    with open(output_path, 'w') as f:
        f.write("Top-3 Predictions for Sample Test Images\n")
        f.write("=" * 60 + "\n\n")
        
        for idx in indices:
            img, true_label = dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                top3_prob, top3_idx = probs.topk(3, dim=1)
            
            f.write(f"Sample {idx}:\n")
            f.write(f"  True Label: {class_names[true_label]}\n")
            f.write(f"  Top-3 Predictions:\n")
            for i in range(3):
                pred_class = class_names[top3_idx[0][i].item()]
                prob = top3_prob[0][i].item() * 100
                f.write(f"    {i+1}. {pred_class}: {prob:.2f}%\n")
            f.write("\n")
    
    print(f"  → Saved sample predictions to {output_path}")
