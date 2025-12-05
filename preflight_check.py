#!/usr/bin/env python3
"""
Pre-flight check script to validate all components are ready for experiments.
Run this after activating your environment to ensure everything works.
"""
import sys
import importlib

def check_imports():
    """Check that all required packages can be imported."""
    print("=" * 60)
    print("CHECKING IMPORTS")
    print("=" * 60)
    
    required = ['cv2', 'numpy', 'torch', 'torchvision', 'matplotlib']
    failed = []
    
    for pkg in required:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {pkg:15s} {version}")
        except ImportError as e:
            print(f"✗ {pkg:15s} FAILED: {e}")
            failed.append(pkg)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Activate your environment: conda activate isd_fixed")
        return False
    
    print("\n✓ All imports successful!")
    return True

def check_project_structure():
    """Check that all required files and directories exist."""
    print("\n" + "=" * 60)
    print("CHECKING PROJECT STRUCTURE")
    print("=" * 60)
    
    from pathlib import Path
    
    required_files = [
        'run.py',
        'scripts/run_batch.py',
        'algorithms/retinex.py',
        'model/unet_models2.py',
        'requirements.txt',
        'TUNING_GUIDE.md'
    ]
    
    required_dirs = [
        'data/images',
        'data/sr_maps',
        'model',
        'scripts',
        'algorithms',
        'results'
    ]
    
    failed_files = []
    failed_dirs = []
    
    for fpath in required_files:
        p = Path(fpath)
        if p.exists():
            print(f"✓ {fpath}")
        else:
            print(f"✗ {fpath} MISSING")
            failed_files.append(fpath)
    
    for dpath in required_dirs:
        p = Path(dpath)
        if p.exists() and p.is_dir():
            print(f"✓ {dpath}/")
        else:
            print(f"✗ {dpath}/ MISSING")
            failed_dirs.append(dpath)
    
    if failed_files or failed_dirs:
        print(f"\n❌ Missing components")
        return False
    
    print("\n✓ All files present!")
    return True

def check_algorithms():
    """Check that algorithm implementations can be imported and have correct signatures."""
    print("\n" + "=" * 60)
    print("CHECKING ALGORITHM IMPLEMENTATIONS")
    print("=" * 60)
    
    try:
        from algorithms.retinex import (
            baseline_retinex,
            spectral_ratio_retinex,
            apply_spectral_ratio_color_correction,
            normalize_sr_map
        )
        print("✓ baseline_retinex imported")
        print("✓ spectral_ratio_retinex imported")
        print("✓ apply_spectral_ratio_color_correction imported")
        print("✓ normalize_sr_map imported")
        
        # Check function signatures
        import inspect
        
        sig = inspect.signature(baseline_retinex)
        params = list(sig.parameters.keys())
        assert 'image' in params and 'iterations' in params and 'sigma' in params
        print("✓ baseline_retinex signature correct")
        
        sig = inspect.signature(spectral_ratio_retinex)
        params = list(sig.parameters.keys())
        assert 'image' in params and 'sr_map' in params and 'iterations' in params
        print("✓ spectral_ratio_retinex signature correct")
        
        print("\n✓ All algorithms ready!")
        return True
        
    except Exception as e:
        print(f"✗ Algorithm check failed: {e}")
        return False

def check_model_files():
    """Check model checkpoint status."""
    print("\n" + "=" * 60)
    print("CHECKING MODEL FILES")
    print("=" * 60)
    
    from pathlib import Path
    
    checkpoint = Path('model/UNET_run_x10_01_last_model.pth')
    
    if not checkpoint.exists():
        print(f"✗ Model checkpoint not found: {checkpoint}")
        print("\n⚠️  You need to download the model weights (528MB)")
        print("   Options:")
        print("   1. brew install git-lfs && git lfs pull")
        print("   2. Request remote server access from professor")
        return False
    
    size_mb = checkpoint.stat().st_size / (1024 * 1024)
    
    if size_mb < 100:
        print(f"✗ Model checkpoint is only {size_mb:.1f}MB (expected ~503MB)")
        print("   This is likely a Git LFS pointer, not the actual weights")
        print("\n⚠️  Download the actual model:")
        print("   brew install git-lfs && git lfs pull")
        return False
    
    print(f"✓ Model checkpoint found: {checkpoint}")
    print(f"✓ Size: {size_mb:.1f}MB")
    print("\n✓ Model ready!")
    return True

def check_data():
    """Check that data directories have images."""
    print("\n" + "=" * 60)
    print("CHECKING DATA")
    print("=" * 60)
    
    from pathlib import Path
    
    image_dir = Path('data/images')
    sr_map_dir = Path('data/sr_maps')
    
    tif_files = list(image_dir.glob('*.tif'))
    sr_maps = list(sr_map_dir.glob('*_isd.tiff'))
    
    print(f"Images found: {len(tif_files)}")
    print(f"SR maps found: {len(sr_maps)}")
    
    if len(tif_files) == 0:
        print("\n⚠️  No .tif images in data/images/")
        print("   Add your test images to data/images/ before running experiments")
        return False
    
    print(f"\n✓ Data ready! ({len(tif_files)} images)")
    return True

def main():
    print("\n" + "=" * 60)
    print("SPECTRAL-RATIO RETINEX PRE-FLIGHT CHECK")
    print("=" * 60)
    
    checks = [
        ("Imports", check_imports),
        ("Project Structure", check_project_structure),
        ("Algorithms", check_algorithms),
        ("Model Files", check_model_files),
        ("Data", check_data)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} check crashed: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - READY FOR EXPERIMENTS!")
        print("=" * 60)
        print("\nRun your first experiment:")
        print("  conda activate isd_fixed")
        print("  export PYTHONPATH=$PWD:$PYTHONPATH")
        print("  python scripts/run_batch.py --use-model --retinex --baseline-retinex")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES ABOVE")
        print("=" * 60)
        
        # Give specific guidance
        if not results.get("Model Files", True):
            print("\n⚠️  CRITICAL: Model weights needed")
            print("   → Install git-lfs: brew install git-lfs")
            print("   → Download model: git lfs pull")
            print("   OR")
            print("   → Ask professor for remote server access")
        
        if not results.get("Imports", True):
            print("\n⚠️  Environment not activated")
            print("   → Run: conda activate isd_fixed")
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
