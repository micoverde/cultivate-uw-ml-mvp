#!/usr/bin/env python3
"""
Quick test to verify model loading works correctly
"""

import sys
from pathlib import Path

# Add src to path like the API does
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing model path resolution...")
print(f"Current directory: {Path.cwd()}")
print(f"Script location: {__file__}")

# Test path resolution like the API does
main_py = Path(__file__).parent / "src" / "api" / "main.py"
print(f"\nSimulating main.py at: {main_py}")

# Calculate models_dir like the API does
models_dir = main_py.parent.parent.parent / "models"
print(f"Computed models_dir: {models_dir}")
print(f"Models directory exists: {models_dir.exists()}")

if models_dir.exists():
    pkl_files = sorted(models_dir.glob("*.pkl"))
    print(f"\nFound {len(pkl_files)} .pkl files:")
    for f in pkl_files[-5:]:
        print(f"  - {f.name}")

    # Check for ensemble models
    ensemble_models = sorted(models_dir.glob("ensemble_*.pkl"))
    print(f"\nFound {len(ensemble_models)} ensemble models")
    if ensemble_models:
        latest = max(ensemble_models, key=lambda p: p.stat().st_mtime)
        print(f"Most recent: {latest.name}")

    # Check for classic models
    classic_models = sorted(models_dir.glob("classic_*.pkl"))
    print(f"Found {len(classic_models)} classic models")
    if classic_models:
        latest = max(classic_models, key=lambda p: p.stat().st_mtime)
        print(f"Most recent: {latest.name}")

    # Try loading with joblib
    try:
        import joblib
        print("\n✓ joblib is available")

        # Try loading the ensemble model
        ensemble_latest = models_dir / "ensemble_latest.pkl"
        if ensemble_latest.exists():
            print(f"\n✓ Found ensemble_latest.pkl symlink")
            try:
                classifier = joblib.load(ensemble_latest)
                print(f"✓ Successfully loaded ensemble classifier")
                print(f"  Type: {type(classifier)}")
                print(f"  Has 'ensemble' attribute: {hasattr(classifier, 'ensemble')}")
                print(f"  Has 'scaler' attribute: {hasattr(classifier, 'scaler')}")
                print(f"  Has 'models' attribute: {hasattr(classifier, 'models')}")
            except Exception as e:
                print(f"✗ Failed to load: {e}")
        else:
            print(f"✗ ensemble_latest.pkl not found (symlink broken?)")

    except ImportError:
        print("✗ joblib not available - run: pip install joblib")
else:
    print("✗ Models directory not found!")

print("\n" + "="*60)
print("Testing fallback path resolution (Docker-like)...")

# Simulate Docker path resolution
docker_main = Path("/app/src/api/main.py")
docker_models = docker_main.parent.parent.parent / "models"
print(f"Docker models path would resolve to: {docker_models}")

# But our fallback should check /app/models directly
app_models = Path("/app/models")
print(f"Docker fallback checks: {app_models}")

if app_models.exists():
    print(f"✓ {app_models} exists")
else:
    print(f"✗ {app_models} does not exist (expected in Docker container)")
