"""
FastAPI server for Food-101 classification model
Provides endpoints for inference and visualization
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import Food101
from PIL import Image
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="NutriVision API",
    description="Food-101 Classification API for Smart Meal Logger",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
device = None
class_names = None
transform = None
dataset = None

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RUNS_DIR = BASE_DIR / "runs"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def load_model_and_config():
    """Load the best trained model and configuration"""
    global model, device, class_names, transform, dataset
    
    # Load report to get model info
    report_path = RUNS_DIR / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            "Training report not found. Please run training first: python -m src.train_food101"
        )
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    best_model_name = report['best_model_name']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on: {device}")
    
    # Setup transform (same as test transform during training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset for class names and analysis (with transforms)
    dataset = Food101(root=str(DATA_DIR), split='test', download=False, transform=transform)
    class_names = dataset.classes
    
    # Create model architecture
    if best_model_name == 'resnet50':
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
    elif best_model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    else:
        raise ValueError(f"Unknown model: {best_model_name}")
    
    # Load trained weights
    model_path = RUNS_DIR / "best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded: {best_model_name}")
    print(f"✓ Classes: {len(class_names)}")
    print(f"✓ Test dataset loaded: {len(dataset)} images")
    return report


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        report = load_model_and_config()
        print("✓ API ready!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run training first: python -m src.train_food101")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NutriVision API - Food-101 Classification",
        "status": "ready" if model is not None else "model not loaded",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "num_classes": len(class_names) if class_names else None
    }


@app.get("/info")
async def get_info():
    """Get training information and model details"""
    report_path = RUNS_DIR / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Training report not found")
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    return report


@app.get("/classes")
async def get_classes():
    """Get list of all food classes"""
    if class_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"classes": class_names, "count": len(class_names)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict food class from uploaded image
    Returns top-3 predictions with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform and predict
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(img_tensor)
            else:
                output = model(img_tensor)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            top3_prob, top3_idx = probs.topk(3, dim=1)
        
        # Format results
        predictions = []
        for i in range(3):
            predictions.append({
                "rank": i + 1,
                "class": class_names[top3_idx[0][i].item()],
                "confidence": float(top3_prob[0][i].item() * 100)
            })
        
        return {
            "success": True,
            "predictions": predictions,
            "image_size": image.size
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.get("/dataset/random")
async def get_random_dataset_image(split: str = "test", category: str = None):
    """
    Get a random image from the dataset
    Optionally filter by category name
    Returns image path and true label
    """
    if dataset is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    # If category specified, filter by it
    if category:
        # Find the class index for this category
        try:
            class_idx = class_names.index(category)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Category not found: {category}")
        
        # Find all indices for this class
        matching_indices = [i for i, label in enumerate(dataset._labels) if label == class_idx]
        
        if not matching_indices:
            raise HTTPException(status_code=404, detail=f"No images found for category: {category}")
        
        # Pick random from matching indices
        idx = random.choice(matching_indices)
    else:
        # Get random sample from entire dataset
        idx = random.randint(0, len(dataset) - 1)
    
    img_path, label = dataset._image_files[idx], dataset._labels[idx]
    
    return {
        "index": idx,
        "image_path": str(img_path),
        "true_label": class_names[label],
        "label_index": label
    }


@app.get("/dataset/image/{index}")
async def get_dataset_image(index: int, split: str = "test"):
    """Get specific image from dataset by index"""
    if dataset is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    if index < 0 or index >= len(dataset):
        raise HTTPException(status_code=404, detail="Image index out of range")
    
    # Get raw image path (not the transformed tensor)
    img_path = dataset._image_files[index]
    
    # Return the actual image file
    full_path = DATA_DIR / img_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    return FileResponse(full_path, media_type="image/jpeg")


@app.get("/plots/{plot_name}")
async def get_plot(plot_name: str):
    """
    Get training plots/visualizations
    Available: class_distribution, sample_grid, resnet50_loss, resnet50_accuracy,
               efficientnet_b0_loss, efficientnet_b0_accuracy, confusion_resnet50, etc.
    """
    plot_path = OUTPUTS_DIR / f"{plot_name}.png"
    
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail=f"Plot not found: {plot_name}")
    
    return FileResponse(plot_path, media_type="image/png")


@app.get("/plots")
async def list_plots():
    """List all available plots"""
    if not OUTPUTS_DIR.exists():
        return {"plots": []}
    
    plots = [f.stem for f in OUTPUTS_DIR.glob("*.png")]
    return {"plots": plots, "count": len(plots)}


@app.get("/metrics")
async def get_metrics():
    """Get model comparison metrics"""
    metrics_path = OUTPUTS_DIR / "metrics.csv"
    
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")
    
    import pandas as pd
    df = pd.read_csv(metrics_path)
    
    return {
        "metrics": df.to_dict(orient='records')
    }


@app.get("/predictions/sample/{model_name}")
async def get_sample_predictions(model_name: str):
    """Get sample predictions text file"""
    pred_path = OUTPUTS_DIR / f"sample_predictions_{model_name}.txt"
    
    if not pred_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample predictions not found for {model_name}")
    
    with open(pred_path, 'r') as f:
        content = f.read()
    
    return {"content": content}


@app.get("/analysis/per-class")
async def get_per_class_performance(force_recalculate: bool = False):
    """
    Get per-class performance metrics
    Loads from cached file if available, otherwise calculates on-the-fly
    
    Args:
        force_recalculate: If True, recalculates even if cache exists
    """
    cache_path = OUTPUTS_DIR / "per_class_performance.json"
    
    # Try to load from cache first
    if cache_path.exists() and not force_recalculate:
        print("✓ Loading per-class performance from cache...")
        with open(cache_path, 'r') as f:
            data = json.load(f)
        return data
    
    # If no cache or force recalculate, compute it
    if model is None or dataset is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    print("⚠️  Cache not found or recalculation requested.")
    print("💡 Tip: Run 'python src/analyze_performance.py' to pre-calculate and save results")
    print("Computing per-class performance (this may take a moment)...")
    
    from torch.utils.data import DataLoader
    
    # Create test dataloader
    test_loader = DataLoader(
        dataset,
        batch_size=64 if torch.cuda.is_available() else 16,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Track per-class stats
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    class_confidence = [[] for _ in range(len(class_names))]
    
    # Import tqdm for progress bar
    from tqdm import tqdm
    
    model.eval()
    total_batches = len(test_loader)
    
    print(f"Processing {len(dataset)} images across {total_batches} batches...")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Analyzing per-class performance", unit="batch")
        for batch_idx, (inputs, labels) in enumerate(pbar, 1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Predict
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
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
            
            # Update progress bar
            batch_acc = (batch_correct / len(labels)) * 100
            progress_pct = (batch_idx / total_batches) * 100
            pbar.set_postfix({
                'batch_acc': f'{batch_acc:.1f}%',
                'progress': f'{progress_pct:.0f}%'
            })
            
            # Print milestone updates to API logs (every 25%)
            if batch_idx % max(1, total_batches // 4) == 0 or batch_idx == total_batches:
                overall_correct = sum(class_correct)
                overall_total = sum(class_total)
                overall_acc = (overall_correct / overall_total * 100) if overall_total > 0 else 0
                print(f"  Progress: {progress_pct:.0f}% | Overall Accuracy: {overall_acc:.2f}% ({overall_correct}/{overall_total})")
    
    # Compile results
    results = []
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            accuracy = (class_correct[i] / class_total[i]) * 100
            avg_confidence = sum(class_confidence[i]) / len(class_confidence[i]) * 100
            
            # Get a sample image for this class
            sample_idx = None
            for idx, label in enumerate(dataset._labels):
                if label == i:
                    sample_idx = idx
                    break
            
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
    
    # Get model name from report
    report_path = RUNS_DIR / "report.json"
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Save to cache for next time
    output_data = {
        "model_name": report['best_model_name'],
        "total_classes": len(results),
        "total_test_samples": len(dataset),
        "best_class": results[-1] if results else None,
        "worst_class": results[0] if results else None,
        "classes": results
    }
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Results cached to: {cache_path}")
    except Exception as e:
        print(f"⚠️  Could not save cache: {e}")
    
    print(f"✓ Per-class analysis complete for {len(results)} classes")
    
    return output_data


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting NutriVision API Server")
    print("="*60)
    print("\nAPI Docs: http://localhost:8000/docs")
    print("Frontend: http://localhost:8000/static/index.html")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
