# Model Integration Guide

## Your New Model

You have provided the `inc_r3_from_baseline_fast` model weights. This is an incremental learning model trained on Food-101 classes.

### Integration Steps

#### Step 1: Copy Model File
First, ensure the model file is in an accessible location:
```bash
# If it's in Downloads
cp ~/Downloads/inc_r3_from_baseline_fast.pth ./runs/inc_r3_model/
```

#### Step 2: Run Integration Script
Use the provided integration script to automatically integrate the model:

```bash
# Basic usage (assumes ResNet50 architecture)
python integrate_model.py --model-path runs/inc_r3_model/inc_r3_from_baseline_fast.pth

# With custom architecture name
python integrate_model.py \
    --model-path runs/inc_r3_model/inc_r3_from_baseline_fast.pth \
    --model-arch resnet50

# Without backing up the old model
python integrate_model.py \
    --model-path runs/inc_r3_model/inc_r3_from_baseline_fast.pth \
    --no-backup
```

#### Step 3: Restart Backend
After integration, restart the FastAPI server:
```bash
python main.py serve
# or
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

#### Step 4: Verify
The server will load the new model on startup. Check the console output:
```
[OK] Model loaded: resnet50
[OK] Classes: 101
[OK] API ready!
```

### What the Script Does

The `integrate_model.py` script:

1. **Backs up current model** - Saves `best_model.pth` and `report.json` to `runs/archive/backup_TIMESTAMP/`
2. **Loads your model** - Supports various PyTorch model formats
3. **Extracts weights** - Converts to state_dict format compatible with the backend
4. **Updates configuration** - Modifies `report.json` with new model info
5. **Saves weights** - Stores as `runs/best_model.pth`

### Rolling Back

If you need to restore the previous model:

```bash
# Find latest backup
ls -lt runs/archive | head -5

# Restore from backup (replace TIMESTAMP)
cp runs/archive/backup_20260416_120000/best_model.pth runs/best_model.pth
cp runs/archive/backup_20260416_120000/report.json runs/report.json

# Restart server
python main.py serve
```

### Troubleshooting

#### Model Loading Error: "API not ready: model not loaded"

This error appears when:
- Model file doesn't exist at the expected path
- Model format is incompatible
- report.json is missing or malformed

**Solution:**
```bash
# Check if best_model.pth exists
ls -la runs/best_model.pth

# Check report.json
cat runs/report.json

# Re-run integration
python integrate_model.py --model-path <your_model_path>
```

#### Permission Denied on Windows

If you get permission errors:
```powershell
# Run as administrator or
# Close any open file handles to the model files
```

#### Model Architecture Mismatch

If the backend expects a different architecture:
```bash
# Update report.json manually:
# Change "best_model_name" to match your model's architecture
# E.g., "resnet50", "efficientnet_b4", "vit_b_16", etc.
```

### Supported Architectures

The backend supports these model architectures (check `src/core/model.py`):
- `resnet50` (default)
- `efficientnet_b0`
- `efficientnet_b4`
- `mobilenet_v3_large`
- `vit_b_16` (Vision Transformer)

### Model Requirements

Your model must:
- Be a PyTorch model (`.pth`, `.pt`, or directory format)
- Output 101 class predictions (Food-101 dataset)
- Be compatible with the standard transforms (224x224 images, ImageNet normalization)

### Fast Integration for Development

If you just want to test:
```bash
# Quick swap (no architecture change)
1. Copy your model to runs/best_model.pth
2. Restart the server
3. Report.json will use the existing architecture

# Or use the script with defaults:
python integrate_model.py --model-path your_model.pth --no-backup
```

---

**Need Help?**

Check the backend logs:
```bash
python main.py serve 2>&1 | tee server.log
```

The logs will show exactly why the model failed to load.
