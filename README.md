# 🍽️ NutriVision: Smart Meal Logger Using AI Image Classification

**Capstone Project 2026** | Food-101 Dataset | PyTorch | ResNet50 & EfficientNet-B0

Train and deploy deep learning models for food image classification with top-3 predictions for human-in-the-loop meal logging.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Training](#-training)
- [Web Application](#-web-application)
- [Understanding Metrics](#-understanding-metrics)
- [Configuration](#-configuration)
- [GPU Optimization](#-gpu-optimization)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)

---

## ✨ Features

### 🤖 Machine Learning Pipeline
- ✅ **Multi-Model Training**: ResNet50 & EfficientNet-B0 with automatic comparison
- ✅ **GPU-Optimized**: 2-3x faster with Automatic Mixed Precision (AMP)
- ✅ **Transfer Learning**: ImageNet pre-trained models with two-stage fine-tuning
- ✅ **Top-3 Predictions**: Optimized for human-in-the-loop confirmation
- ✅ **Reproducible**: Fixed seeds, deterministic training, environment logging
- ✅ **Automatic Dataset Handling**: Downloads Food-101 on first run

### 🌐 Interactive Web Application
- ✅ **Real-time Classification**: Upload images or test with dataset samples
- ✅ **Category Selection**: Choose specific foods to test (all 101 classes)
- ✅ **Training Visualizations**: Loss curves, accuracy plots, confusion matrices
- ✅ **Per-Class Performance**: See which foods the model recognizes best/worst
- ✅ **Beautiful Modern UI**: Responsive design with confidence visualization
- ✅ **Carousel Navigation**: Swipe through training insights with arrows

### 📊 Comprehensive Analysis
- ✅ **Accuracy Metrics**: Top-1 (71.73%) and Top-3 (86.33%)
- ✅ **Confusion Matrix**: Visual representation of classification errors
- ✅ **Sample Predictions**: Detailed examples with ground truth comparison
- ✅ **Performance Insights**: 101 food categories ranked by recognition accuracy

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd NutriVision
pip install -r requirements.txt
```

### 2. (Optional) Check GPU Setup

```bash
python check_gpu.py
```

### 3. Train the Model

```bash
python -m src.train_food101
```

**That's it!** The script will:
- Download Food-101 dataset (first run only, ~5GB)
- Train ResNet50 and EfficientNet-B0
- Generate visualizations and metrics
- Save the best model to `runs/best_model.pth`

**Training Time:**
- GPU (with AMP): ~10-20 minutes for balanced config
- CPU: ~2-3 hours (use subset mode)

### 4. Launch Web Application

```bash
# Pre-calculate per-class performance (recommended, ~30-60 seconds)
python src/analyze_performance.py

# Start the web server
python src/api.py
```

Open: **http://localhost:8000/static/index.html**

---

## 📦 Installation

### Requirements
- Python 3.8+
- CUDA 11.8 or 12.1 (for GPU)
- 8GB+ RAM (16GB+ recommended)
- 10GB free disk space

### Step-by-Step

**1. Create virtual environment (recommended):**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**2. Install PyTorch with CUDA:**
```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only (not recommended)
pip install torch torchvision
```

**3. Install other dependencies:**
```bash
pip install -r requirements.txt
```

**4. Verify GPU (optional but recommended):**
```bash
python check_gpu.py
```

You should see:
```
✓ CUDA Available: True
✓ GPU Device: NVIDIA GeForce RTX 3060
✓ GPU Memory: 12.00 GB
```

---

## 🎓 Training

### Default Training (Balanced Mode)

```bash
python -m src.train_food101
```

**Default Configuration:**
- 15,000 training images (subset)
- 3,000 test images (subset)
- 2 warmup epochs + 3 fine-tuning epochs
- Batch size: 64 (GPU) or 16 (CPU)
- Mixed precision: Enabled
- Expected accuracy: Top-1 ~72%, Top-3 ~86%
- Time: ~10-20 minutes on GPU

### Full Dataset Training

Edit `src/train_food101.py` → `Config` class:

```python
train_subset_size: int = 0  # Use all 75,750 images
test_subset_size: int = 0   # Use all 25,250 images
warmup_epochs: int = 3
finetune_epochs: int = 7
```

Then run:
```bash
python -m src.train_food101
```

**Expected Performance:**
- Top-1 accuracy: 75-85%
- Top-3 accuracy: 90-95%
- Time: ~1-2 hours on modern GPU

### Quick Test Mode

For rapid iteration and testing:

```python
train_subset_size: int = 2000
test_subset_size: int = 500
warmup_epochs: int = 1
finetune_epochs: int = 0
```

Time: ~1-2 minutes on GPU

### Configuration Options

```python
@dataclass
class Config:
    # Data
    train_subset_size: int = 15000  # 0 = full dataset
    test_subset_size: int = 3000
    
    # Training
    batch_size: int = 64  # Auto-adjusted for hardware
    warmup_epochs: int = 2
    finetune_epochs: int = 3
    learning_rate: float = 0.001
    
    # Models
    models_to_train: List[str] = ['resnet50', 'efficientnet_b0']
    
    # GPU Optimization
    use_amp: bool = True  # 2-3x faster
    gradient_accumulation_steps: int = 1
    enable_cudnn_benchmark: bool = True
    
    # Reproducibility
    seed: int = 42
    num_workers: int = 8  # Auto-adjusted for platform
```

---

## 🌐 Web Application

### Starting the Server

```bash
# Step 1: Pre-calculate analytics (optional but recommended)
python src/analyze_performance.py

# Step 2: Start API server
python src/api.py
```

Server starts at: **http://localhost:8000**

### Access Points

- **Web UI**: http://localhost:8000/static/index.html
- **API Docs**: http://localhost:8000/docs (interactive Swagger UI)
- **Help Page**: http://localhost:8000/static/help.html

### Features

#### 📸 Food Recognition
1. **Upload Your Images**
   - Drag & drop or click to upload
   - Supports JPG, PNG, WebP
   - Instant top-3 predictions with confidence scores

2. **Test with Dataset**
   - 🎲 Random image from any category
   - 🎯 Select specific food category (all 101 foods)
   - See ground truth vs predictions

3. **Results Display**
   - Top-3 predictions with confidence bars
   - Color-coded by rank (🥇 🥈 🥉)
   - Validation indicator (✅ correct / ❌ incorrect)

#### 📊 Training Insights
- **Carousel Navigation**: Swipe through plots with arrows
- **Model Metrics**: Best model, Top-1, Top-3 accuracy
- **Training Curves**: Loss and accuracy over epochs
- **Confusion Matrices**: Visual classification performance
- **Sample Images**: Example dataset images
- **Class Distribution**: Balanced 1000 images per class

#### 🎯 Per-Class Performance
- **101 Food Categories**: Each with accuracy score
- **Visual Cards**: Sample image, accuracy bar, stats
- **Sortable**: Worst-first, best-first, alphabetical
- **Color Coded**: Green (>80%), Blue (60-80%), Orange (40-60%), Red (<40%)
- **Interactive**: Click to open sample image

### API Endpoints

#### Prediction
```bash
# Upload image for classification
curl -X POST "http://localhost:8000/predict" -F "file=@pizza.jpg"

# Get random dataset image
curl "http://localhost:8000/dataset/random"

# Get image from specific category
curl "http://localhost:8000/dataset/random?category=pizza"
```

#### Information
```bash
# Model info
curl "http://localhost:8000/info"

# All 101 food classes
curl "http://localhost:8000/classes"

# Per-class performance
curl "http://localhost:8000/analysis/per-class"
```

#### Visualizations
```bash
# List available plots
curl "http://localhost:8000/plots"

# Get specific plot
curl "http://localhost:8000/plots/resnet50_loss" -o loss_curve.png
```

### Production Deployment

```bash
# Multiple workers
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# HTTPS
uvicorn src.api:app --host 0.0.0.0 --port 443 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

---

## 📚 Understanding Metrics

### 🎯 Top-1 Accuracy (71.73%)

**Definition:** Percentage of times the **#1 prediction** is exactly correct.

**Example:**
```
Image: Pizza
Model's #1 prediction: pizza ✅ → Counts!
Model's #1 prediction: flatbread ❌ → Doesn't count
```

Your model's #1 guess is correct **7 out of 10 times**.

### 🏆 Top-3 Accuracy (86.33%)

**Definition:** Percentage of times the correct answer appears **anywhere in top 3**.

**Example:**
```
Image: Sushi

Scenario 1:
#1: sushi ← CORRECT IN TOP 3 ✅
#2: sashimi
#3: tuna roll

Scenario 2:
#1: sashimi
#2: sushi ← CORRECT IN TOP 3 ✅
#3: tuna roll

Scenario 3:
#1: sashimi
#2: tuna roll
#3: sushi ← CORRECT IN TOP 3 ✅

Scenario 4:
#1: sashimi
#2: tuna roll
#3: salmon ← NOT IN TOP 3 ❌
```

Your model includes the right answer in top 3 **9 out of 10 times**.

### 💡 Why Top-3 Matters

For your **Smart Meal Logger** with human confirmation:

- ✅ Users see 3 options, not just 1
- ✅ 86% chance the right food is visible
- ✅ Quick selection instead of manual entry
- ✅ Much better UX than single prediction
- ✅ Handles ambiguous foods (e.g., pizza vs flatbread)

**Bottom line:** Top-3 accuracy is your most important metric for user experience!

### 📊 Per-Class Accuracy

Shows **Top-1 accuracy for each specific food**.

Example: **Pizza: 90%** means:
- When model sees a pizza image
- It predicts "pizza" as #1
- 90% of the time

Use this to find:
- ✅ Best recognized foods (confidence builders)
- ⚠️ Worst recognized foods (might need more training)
- 🤔 Confused pairs (similar-looking foods)

---

## ⚙️ Configuration

### Training Modes

#### 🚀 Fast Testing (Default)
```python
train_subset_size: int = 2000
test_subset_size: int = 500
warmup_epochs: int = 1
finetune_epochs: int = 0
```
**Time:** ~1-2 minutes | **Use:** Quick iteration

#### ⚖️ Balanced (Recommended)
```python
train_subset_size: int = 15000
test_subset_size: int = 3000
warmup_epochs: int = 2
finetune_epochs: int = 3
```
**Time:** ~10-20 minutes | **Accuracy:** Top-3 ~86%

#### 🎯 Full Training
```python
train_subset_size: int = 0  # All 75,750 images
test_subset_size: int = 0   # All 25,250 images
warmup_epochs: int = 3
finetune_epochs: int = 7
```
**Time:** ~1-2 hours | **Accuracy:** Top-3 ~90-95%

#### 💾 Memory Constrained (4-6GB GPU)
```python
batch_size: int = 32
gradient_accumulation_steps: int = 2
use_amp: bool = True  # CRITICAL
```

#### 🔬 Research Quality (Reproducible)
```python
use_amp: bool = False
enable_cudnn_benchmark: bool = False
seed: int = 42
```

#### 🖥️ CPU Training
```python
train_subset_size: int = 2000
batch_size: int = 16
num_workers: int = 4
warmup_epochs: int = 1
```

---

## 🚀 GPU Optimization

### Automatic Features (Zero Config)

The pipeline **automatically optimizes** for your hardware:

#### 1. Mixed Precision Training (AMP)
- ⚡ **2-3x faster** training on modern GPUs
- 💾 **40-50% less memory** usage
- 🎯 **Same accuracy** as full precision
- Auto-enabled on CUDA GPUs

**How it works:**
```python
with torch.cuda.amp.autocast():
    outputs = model(inputs)  # Uses FP16
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
```

#### 2. cuDNN Benchmark Mode
- ⚡ **5-10% faster** convolutions
- 🎯 Finds optimal algorithm for your GPU
- Auto-enabled (adds 1-2 min startup)

#### 3. Optimized Data Pipeline
- **Pin Memory**: 2-3x faster CPU→GPU transfer
- **Non-blocking Transfer**: Overlaps transfer & computation
- **Auto-scaled Workers**: 8 (GPU) vs 4 (CPU)
- **Persistent Workers**: Reuses worker processes

#### 4. Adaptive Batch Sizing
- **GPU**: batch_size=64 (better utilization)
- **CPU**: batch_size=16 (memory efficient)
- **Auto-adjusted** based on hardware

#### 5. Gradient Accumulation
```python
gradient_accumulation_steps: int = 2
# Effective batch = 64 * 2 = 128
```
Simulates larger batches without extra memory.

### Performance Benchmarks

**Training Speed (ResNet50, 15K images, 5 epochs):**
- 💻 CPU: ~2-3 hours
- 🎮 GPU (no AMP): ~20-30 minutes
- ⚡ GPU (with AMP): ~10-15 minutes

**Memory Usage:**
- Without AMP: ~6-8 GB GPU
- With AMP: ~3-4 GB GPU

**Full Dataset (75K images, 10 epochs):**
- GPU (AMP): ~1-2 hours
- CPU: ~8-12 hours (not recommended)

### Manual Control

```python
# Disable AMP (if issues)
use_amp: bool = False

# Reduce batch size (if OOM)
batch_size: int = 32

# Use gradient accumulation
gradient_accumulation_steps: int = 2

# Disable cuDNN benchmark (for reproducibility)
enable_cudnn_benchmark: bool = False

# Adjust workers
num_workers: int = 4
```

---

## 📁 Project Structure

```
NutriVision/
├── src/
│   ├── __init__.py
│   ├── train_food101.py         # Main training pipeline ⭐
│   ├── api.py                   # FastAPI web server ⭐
│   ├── analyze_performance.py   # Per-class analysis
│   └── utils.py                 # Helper functions
├── static/
│   ├── index.html              # Web UI
│   ├── help.html               # Metrics guide
│   ├── styles.css              # Styling
│   └── app.js                  # Frontend logic
├── data/
│   └── food-101/               # Dataset (auto-downloaded, gitignored)
├── outputs/
│   ├── *.png                   # Training visualizations
│   ├── metrics.csv             # Model comparison
│   ├── per_class_performance.json  # Cached analytics
│   └── sample_predictions_*.txt
├── runs/
│   ├── best_model.pth          # Best model weights ⭐
│   └── report.json             # Training report ⭐
├── check_gpu.py                # GPU verification
├── requirements.txt            # Dependencies
├── .gitignore
└── README.md                   # This file
```

### Key Files

- **`src/train_food101.py`**: Complete training pipeline
- **`src/api.py`**: Web server + API endpoints
- **`runs/best_model.pth`**: Trained model (generated)
- **`runs/report.json`**: Metrics & config (generated)
- **`check_gpu.py`**: GPU diagnostics

---

## 🐛 Troubleshooting

### Training Issues

#### Out of Memory (OOM)

1. **Enable AMP** (should be on by default):
   ```python
   use_amp: bool = True
   ```

2. **Reduce batch size**:
   ```python
   batch_size: int = 32  # or 16
   ```

3. **Use gradient accumulation**:
   ```python
   batch_size: int = 16
   gradient_accumulation_steps: int = 4  # Effective = 64
   ```

4. **Use smaller subset**:
   ```python
   train_subset_size: int = 5000
   ```

#### Slow Training

1. **Verify GPU is used**:
   ```bash
   python check_gpu.py
   ```
   Look for: "✓ Using GPU: [GPU Name]"

2. **Check GPU utilization**:
   ```bash
   # In another terminal
   nvidia-smi -l 1
   ```
   Should show 80-100% GPU usage

3. **Increase batch size** (if memory allows):
   ```python
   batch_size: int = 128
   ```

4. **Increase workers**:
   ```python
   num_workers: int = 12
   ```

#### Dataset Download Fails

- Check internet connection
- Manually download: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- Extract to `data/food-101/`

#### Windows Multiprocessing Issues

Script hangs at data loading:
```python
num_workers: int = 0  # Auto-set on Windows
```

### Web Application Issues

#### "Model not loaded" Error

```bash
# 1. Make sure training completed
python -m src.train_food101

# 2. Check files exist
ls runs/best_model.pth
ls runs/report.json
```

#### Plots Not Showing

- Refresh browser (Ctrl+F5)
- Check `outputs/` folder has PNG files
- Open browser console (F12) for errors

#### Port Already in Use

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Or change port in src/api.py
uvicorn.run(app, port=8080)
```

#### Categories Not Loading

- Check API server is running
- Visit http://localhost:8000/classes manually
- Check browser console for errors

#### Slow Predictions

- First prediction loads model (~2-3 seconds)
- Subsequent predictions are fast (<100ms)
- GPU automatically used if available

---

## 🔧 Advanced Usage

### Adding New Models

1. Add to config:
```python
models_to_train: List[str] = ['resnet50', 'efficientnet_b0', 'vit_b_16']
```

2. Implement in `create_model()`:
```python
elif model_name == 'vit_b_16':
    model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
```

### Custom Dataset

Replace Food-101 with your own:

```python
# In load_data()
train_dataset = YourCustomDataset(
    root=DATA_DIR,
    split='train',
    transform=train_transform
)
```

### Export to ONNX

```python
import torch

model = torch.load('runs/best_model.pth')
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

### Batch Prediction

```python
import torch
from PIL import Image

images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
batch = torch.stack([transform(Image.open(img)) for img in images])

with torch.no_grad():
    outputs = model(batch.to(device))
    predictions = outputs.topk(3, dim=1)
```

---

## 📊 Outputs

### Training Artifacts

#### `outputs/` Directory

**Visualizations:**
- `class_distribution.png` - Dataset balance chart
- `sample_grid.png` - 16 example images
- `resnet50_loss.png` - Loss curves
- `resnet50_accuracy.png` - Accuracy curves
- `efficientnet_b0_loss.png`
- `efficientnet_b0_accuracy.png`
- `confusion_resnet50.png` - Confusion matrix
- `confusion_efficientnet_b0.png`

**Metrics:**
- `metrics.csv` - Model comparison table
- `sample_predictions_resnet50.txt` - Examples
- `sample_predictions_efficientnet_b0.txt`
- `per_class_performance.json` - Cached analytics

#### `runs/` Directory

- `best_model.pth` - Best model weights (load with `torch.load()`)
- `report.json` - Complete training report

**Example report.json:**
```json
{
  "best_model_name": "resnet50",
  "best_model_metrics": {
    "test_top1_accuracy": 71.73,
    "test_top3_accuracy": 86.33
  },
  "config": {
    "train_subset_size": 15000,
    "test_subset_size": 3000,
    "batch_size": 64,
    "epochs_total": 5
  },
  "timestamp": "2026-02-09T17:30:45"
}
```

---

## 📚 References

### Dataset
- **Food-101**: [Bossard et al., ECCV 2014](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- 101 food categories
- 101,000 images (1,000 per class)
- 75,750 training images
- 25,250 test images

### Models
- **ResNet50**: [He et al., CVPR 2016](https://arxiv.org/abs/1512.03385)
- **EfficientNet**: [Tan & Le, ICML 2019](https://arxiv.org/abs/1905.11946)

### Techniques
- **Transfer Learning**: [Pan & Yang, 2010](https://ieeexplore.ieee.org/document/5288526)
- **Mixed Precision**: [Micikevicius et al., 2018](https://arxiv.org/abs/1710.03740)

---

## 📄 License

This project is for academic purposes as part of a capstone project.

---

## 👥 Team

**NutriVision Team**
- Course: COMP385 - AI Capstone Project
- Institution: Centennial College
- Academic Year: 2026

---

## 🎓 Acknowledgments

- Food-101 dataset creators
- PyTorch team
- FastAPI framework
- Course instructors and advisors

---

## 🏗️ Project Foundation

**ML Pipeline & Web Application Baseline built by:**

**Maksym Ostanin** ([@angaga2011](https://github.com/angaga2011))
- Complete training pipeline with GPU optimization
- FastAPI backend architecture
- Interactive web application UI
- Comprehensive documentation

*Student of Software Engineering Technology - AI at Centennial College*

GitHub: [github.com/angaga2011](https://github.com/angaga2011)

---

## 📞 Support

**Issues?**
1. Check [Troubleshooting](#-troubleshooting) section
2. Review [API Docs](http://localhost:8000/docs)
3. Check browser console (F12) for frontend issues
4. Review terminal logs for backend issues

**Questions?**
- Open an issue on GitHub
- Contact team members
- Review the help page: http://localhost:8000/static/help.html

---

<div align="center">

**🍽️ NutriVision - Smart Meal Logger Using AI**

*Making food logging as easy as taking a photo*

[Documentation](#) • [Web App](#-web-application) • [Training](#-training) • [API](#-api-documentation)

</div>
