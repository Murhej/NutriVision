import torch
import sys
from pathlib import Path

try:
    # Load model directly from Downloads
    model_path = Path(r'c:\Users\murhe\Downloads\inc_r3_from_baseline_fast')
    
    print(f'Attempting to load from: {model_path}')
    print(f'Path exists: {model_path.exists()}')
    
    # List contents
    if model_path.exists():
        print(f'Contents: {list(model_path.iterdir())}')
    
    # Try loading
    m = torch.load(str(model_path), map_location='cpu', weights_only=False)
    print('✓ Model loaded successfully')
    print('Type:', type(m))
    
    if hasattr(m, 'state_dict'):
        print('  ✓ Has state_dict method')
        try:
            sd = m.state_dict()
            keys = list(sd.keys())
            print('  State dict keys (first 5):', keys[:5])
            print('  Total state dict keys:', len(keys))
        except Exception as e:
            print(f'  ✗ state_dict error: {e}')
    
    if isinstance(m, dict):
        keys = list(m.keys())
        print('  ✓ Is a dict')
        print('  Dict keys (first 5):', keys[:5])
        print('  Total keys:', len(keys))
    
    print('\n✓ Model is compatible!')
    sys.exit(0)
except Exception as e:
    print('✗ Error:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
