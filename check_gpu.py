"""
Quick script to check GPU availability and configuration
Run this before training to verify your setup
"""

import torch
import sys

print("="*60)
print("GPU CONFIGURATION CHECK")
print("="*60)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    # GPU details
    print(f"\nNumber of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Multi-processors: {props.multi_processor_count}")
        
        # Current memory usage
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.3f} GB")
        print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1e9:.3f} GB")
    
    # Test AMP support
    print("\n" + "="*60)
    print("AUTOMATIC MIXED PRECISION (AMP) CHECK")
    print("="*60)
    
    try:
        # Quick AMP test
        device = torch.device('cuda:0')
        x = torch.randn(8, 3, 224, 224, device=device)
        
        with torch.cuda.amp.autocast():
            y = torch.nn.functional.conv2d(x, torch.randn(64, 3, 3, 3, device=device))
        
        print("✅ AMP is supported and working!")
        print("   Your GPU supports mixed precision training for 2-3x speedup")
        
        # Check compute capability for tensor cores
        if props.major >= 7:  # Tensor Cores available on Volta (7.0) and newer
            print("✅ Tensor Cores available (Compute Capability >= 7.0)")
            print("   You'll get maximum speedup from AMP!")
        elif props.major >= 6:
            print("⚠️  Tensor Cores not available (Compute Capability 6.x)")
            print("   AMP will still work but with smaller speedup")
        else:
            print("⚠️  Older GPU detected (Compute Capability < 6.0)")
            print("   Consider disabling AMP for this GPU")
            
    except Exception as e:
        print(f"❌ AMP test failed: {e}")
        print("   Consider setting use_amp=False in config")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR YOUR SETUP")
    print("="*60)
    
    total_mem_gb = props.total_memory / 1e9
    
    if total_mem_gb >= 8:
        print(f"✅ {total_mem_gb:.0f} GB GPU - Excellent for training!")
        print("   Recommended config:")
        print("   - batch_size: 64-128")
        print("   - use_amp: True")
        print("   - Full dataset training: ✅ Recommended")
    elif total_mem_gb >= 6:
        print(f"✅ {total_mem_gb:.0f} GB GPU - Good for training!")
        print("   Recommended config:")
        print("   - batch_size: 32-64")
        print("   - use_amp: True")
        print("   - Full dataset training: ✅ Possible")
    elif total_mem_gb >= 4:
        print(f"⚠️  {total_mem_gb:.0f} GB GPU - Sufficient for training")
        print("   Recommended config:")
        print("   - batch_size: 16-32")
        print("   - use_amp: True (important!)")
        print("   - gradient_accumulation_steps: 2-4")
    else:
        print(f"⚠️  {total_mem_gb:.0f} GB GPU - Limited")
        print("   Recommended config:")
        print("   - batch_size: 8-16")
        print("   - use_amp: True (critical!)")
        print("   - gradient_accumulation_steps: 4-8")
        print("   - train_subset_size: 2000-5000")

else:
    print("\n❌ No GPU detected!")
    print("\nTraining will run on CPU (much slower).")
    print("Expected training time: 8-10 minutes/epoch for subset, hours for full dataset")
    print("\nTo use GPU:")
    print("  1. Install CUDA-enabled PyTorch:")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("  2. Ensure you have an NVIDIA GPU with CUDA drivers installed")
    print("  3. Verify with: nvidia-smi")

print("\n" + "="*60)
print("READY TO TRAIN!")
print("="*60)
print("\nRun: python -m src.train_food101")
print()
