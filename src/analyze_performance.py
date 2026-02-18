"""
Pre-calculate per-class performance metrics
Run this after training to generate performance analysis cache
"""

import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import Food101
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"
OUTPUTS_DIR = BASE_DIR / "outputs"


def build_dataset_index(split: str, class_names):
  
    """Build Food-101 index from metadata to avoid private torchvision fields."""
    meta_path = DATA_DIR / "food-101" / "meta" / f"{split}.json"
    images_dir = DATA_DIR / "food-101" / "images"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r") as f:
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


def forward_with_tta(model: nn.Module, inputs: torch.Tensor, device: torch.device,
                     tta_views: int = 1) -> torch.Tensor:
    """Inference helper with optional flip-based test-time augmentation."""
    views = [inputs]
    if tta_views > 1:
        views.append(torch.flip(inputs, dims=[3]))
    if tta_views > 2:
        views.append(torch.flip(inputs, dims=[2]))
    if tta_views > 3:
        views.append(torch.flip(inputs, dims=[2, 3]))

    max_views = min(max(1, int(tta_views)), len(views))
    logits_sum = None
    for i in range(max_views):
        view = views[i]
        if device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', enabled=True):
                logits = model(view)
        else:
            logits = model(view)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)
    return logits_sum / float(max_views)


def analyze_per_class_performance():
    """Analyze and save per-class performance metrics"""
    
    print("="*60)
    print("Per-Class Performance Analysis")
    print("="*60)
    
    # Load report to get model info
    report_path = RUNS_DIR / "report.json"
    if not report_path.exists():
        print("❌ Error: Training report not found.")
        print("   Please run training first: python -m src.train_food101")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
 
    best_model_name = report['best_model_name']
    train_cfg = report.get("config", {})
    tta_views = int(train_cfg.get("tta_num_views", 1)) if train_cfg.get("eval_tta", False) else 1
    print(f"\n📊 Analyzing: {best_model_name}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load dataset
    print("\n📥 Loading Food-101 test dataset...")
    dataset = Food101(root=str(DATA_DIR), split='test', download=False, transform=transform)
    class_names = dataset.classes
    dataset_index = build_dataset_index(split="test", class_names=class_names)
    if len(dataset_index) != len(dataset):
        raise RuntimeError(
            f"Dataset index size mismatch: metadata={len(dataset_index)} dataset={len(dataset)}"
        )
    print(f"✓ Loaded {len(dataset)} test images, {len(class_names)} classes")
    
    # Create model
    print(f"\n🤖 Loading {best_model_name} model...")
    if best_model_name == 'resnet50':
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
    elif best_model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    elif best_model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    elif best_model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    elif best_model_name == 'convnext_base':
        model = models.convnext_base(weights=None)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, len(class_names))
    elif best_model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, len(class_names))
    else:
        print(f"❌ Unknown model: {best_model_name}")
        return
    
    # Load trained weights
    model_path = RUNS_DIR / "best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"TTA views: {tta_views}")
    print("✓ Model loaded")
    
    # Create dataloader
    test_loader = DataLoader(
        dataset,
        batch_size=64 if torch.cuda.is_available() else 16,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Track per-class stats
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    class_confidence = [[] for _ in range(len(class_names))]
    
    print(f"\n🔍 Evaluating {len(dataset)} images across {len(test_loader)} batches...")
    
    total_batches = len(test_loader)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Analyzing per-class performance", unit="batch")
        for batch_idx, (inputs, labels) in enumerate(pbar, 1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
                       
            # Predict
            outputs = forward_with_tta(model, inputs, device=device, tta_views=tta_views)
            
            
            # Get probabilities and predictions
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Update per-class stats
            batch_correct = 0
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                confidence = probs[i][pred].item()
                
                class_total[label] += 1
                class_confidence[label].append(confidence)
                
                if pred == label:
                    class_correct[label] += 1
                    batch_correct += 1
            
            # Update progress bar with current batch accuracy
            batch_acc = (batch_correct / len(labels)) * 100
            progress_pct = (batch_idx / total_batches) * 100
            pbar.set_postfix({
                'batch_acc': f'{batch_acc:.1f}%',
                'progress': f'{progress_pct:.0f}%'
            })
            
            # Print milestone updates (every 25%)
            if batch_idx % max(1, total_batches // 4) == 0 or batch_idx == total_batches:
                overall_correct = sum(class_correct)
                overall_total = sum(class_total)
                overall_acc = (overall_correct / overall_total * 100) if overall_total > 0 else 0
                print(f"\n  Progress: {progress_pct:.0f}% | Overall Accuracy: {overall_acc:.2f}% ({overall_correct}/{overall_total})")
    
    # Compile results
    print("\n📊 Compiling results...")
    results = []
    first_index_for_label = {}
    for item in dataset_index:
        label_idx = item["label_index"]
        if label_idx not in first_index_for_label:
            first_index_for_label[label_idx] = item["index"]
    
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            accuracy = (class_correct[i] / class_total[i]) * 100
            avg_confidence = sum(class_confidence[i]) / len(class_confidence[i]) * 100
            
            sample_idx = first_index_for_label.get(i)
            
            results.append({
                "class_name": class_name,
                "class_index": i,
                "accuracy": round(accuracy, 2),
                "avg_confidence": round(avg_confidence, 2),
                "correct": class_correct[i],
                "total": class_total[i],
                "sample_image_index": sample_idx
            })
    
    # Sort by accuracy (worst to best)
    results.sort(key=lambda x: x['accuracy'])
    
    # Save to file
    output_data = {
        "model_name": best_model_name,
        "total_classes": len(results),
        "total_test_samples": len(dataset),
        "best_class": results[-1] if results else None,
        "worst_class": results[0] if results else None,
        "classes": results
    }
    
    output_path = OUTPUTS_DIR / "per_class_performance.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Saved to: {output_path}")
    print(f"\n📈 Summary:")
    print(f"   Total classes: {len(results)}")
    print(f"   Best class: {results[-1]['class_name']} ({results[-1]['accuracy']:.1f}%)")
    print(f"   Worst class: {results[0]['class_name']} ({results[0]['accuracy']:.1f}%)")
    
    # Show top 5 best and worst
    print(f"\n🏆 Top 5 Best Recognized Foods:")
    for item in results[-5:][::-1]:
        print(f"   {item['class_name']:30s} {item['accuracy']:5.1f}%  ({item['correct']}/{item['total']})")
    
    print(f"\n⚠️  Top 5 Worst Recognized Foods:")
    for item in results[:5]:
        print(f"   {item['class_name']:30s} {item['accuracy']:5.1f}%  ({item['correct']}/{item['total']})")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    analyze_per_class_performance()
