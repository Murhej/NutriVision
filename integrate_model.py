"""
Model Integration Script

This script helps integrate the new inc_r3_from_baseline_fast model into the backend.

Usage:
    python integrate_model.py --model-path /path/to/inc_r3_from_baseline_fast
    python integrate_model.py --model-path /path/to/inc_r3_from_baseline_fast --model-arch resnet50

The script will:
 1. Load the new model weights
 2. Convert to compatible format if needed
 3. Update report.json with new model info
 4. Backup the old model
"""

import json
import shutil
import torch
import argparse
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"

def backup_current_model():
    """Backup current best_model.pth"""
    best_model = RUNS_DIR / "best_model.pth"
    report_file = RUNS_DIR / "report.json"
    
    if best_model.exists():
        timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = RUNS_DIR / "archive" / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(best_model, backup_dir / "best_model.pth")
        if report_file.exists():
            shutil.copy2(report_file, backup_dir / "report.json")
        
        print(f"✓ Backed up current model to {backup_dir}")
        return str(backup_dir)
    return None

def load_model_safely(model_path: str):
    """Safely load model from various formats"""
    path = Path(model_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    print(f"Loading model from: {path}")
    print(f"Is directory: {path.is_dir()}")
    
    try:
        # Try standard torch.load
        model = torch.load(str(path), map_location='cpu', weights_only=False)
        print(f"✓ Successfully loaded model")
        print(f"  Type: {type(model)}")
        return model
    except Exception as e:
        print(f"✗ Failed to load with torch.load: {e}")
        
        # Try alternative approaches
        try:
            if path.is_dir():
                # Try TorchScript if it's a directory
                if (path / "__pycache__" / "__init__.so").exists():
                    model = torch.jit.load(str(path))
                    print(f"✓ Loaded as TorchScript model")
                    return model
        except:
            pass
        
        raise ValueError(f"Could not load model from {model_path}")

def extract_state_dict(model):
    """Extract state_dict from various model formats"""
    if isinstance(model, dict):
        # Already a state dict
        return model
    elif hasattr(model, 'state_dict'):
        return model.state_dict()
    else:
        raise ValueError(f"Cannot extract state_dict from {type(model)}")

def integrate_model(
    model_path: str,
    model_arch: str = "resnet50",
    num_classes: int = 101,
    backup: bool = True
):
    """
    Integrate the new model into the backend.
    
    Args:
        model_path: Path to the new model weights
        model_arch: Model architecture name (resnet50, efficientnet_b4, etc.)
        num_classes: Number of classes
        backup: Whether to backup current model
    """
    print(f"Integrating model: {model_arch} ({num_classes} classes)")
    print(f"Model path: {model_path}")
    
    # Backup current model
    if backup:
        backup_current_model()
    
    # Load new model
    print("\nLoading new model...")
    model = load_model_safely(model_path)
    
    # Extract state dict
    print("Extracting state dict...")
    state_dict = extract_state_dict(model)
    
    print(f"State dict keys: {len(state_dict)}")
    print(f"First few keys: {list(state_dict.keys())[:3]}")
    
    # Save state dict as best_model.pth
    print(f"\nSaving to {RUNS_DIR / 'best_model.pth'}...")
    torch.save(state_dict, RUNS_DIR / "best_model.pth")
    print("✓ Model weights saved")
    
    # Update report.json
    print("\nUpdating report.json...")
    report_path = RUNS_DIR / "report.json"
    
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
    else:
        # Create new report
        report = {
            "class_names": [f"class_{i}" for i in range(num_classes)],
            "config": {"tta_num_views": 1, "eval_tta": False}
        }
    
    # Update model info
    old_model = report.get("best_model_name", "unknown")
    report["best_model_name"] = model_arch
    report["num_classes"] = num_classes
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report updated")
    print(f"  Old model: {old_model}")
    print(f"  New model: {model_arch}")
    print(f"  Classes: {report.get('num_classes', len(report.get('class_names', [])))}")
    
    print("\n✓✓✓ Model integration complete!")
    print("\nNext steps:")
    print("  1. Restart the backend server")
    print("  2. Test with a sample image upload")
    print("\nTo restore the previous model:")
    backup_dir = RUNS_DIR / "archive"
    if backup_dir.exists():
        latest = sorted(backup_dir.glob("backup_*"))[-1] if any(backup_dir.glob("backup_*")) else None
        if latest:
            print(f"  cp {latest}/best_model.pth {RUNS_DIR}/best_model.pth")
            print(f"  cp {latest}/report.json {RUNS_DIR}/report.json")


def main():
    parser = argparse.ArgumentParser(description="Integrate a new model into NutriVision backend")
    parser.add_argument("--model-path", required=True, help="Path to the new model file or directory")
    parser.add_argument("--model-arch", default="resnet50", help="Model architecture name")
    parser.add_argument("--num-classes", type=int, default=101, help="Number of food classes")
    parser.add_argument("--no-backup", action="store_true", help="Don't backup current model")
    
    args = parser.parse_args()
    
    try:
        integrate_model(
            model_path=args.model_path,
            model_arch=args.model_arch,
            num_classes=args.num_classes,
            backup=not args.no_backup
        )
    except Exception as e:
        print(f"\n✗ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
