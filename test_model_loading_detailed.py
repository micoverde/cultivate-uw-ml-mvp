#!/usr/bin/env python3
"""
Reproduce the model loading error
"""

import sys
from pathlib import Path
import joblib

print("=" * 70)
print("REPRODUCING MODEL LOADING ERROR")
print("=" * 70)

# Simulate the exact code path from main.py
models_dir = Path(__file__).parent / "models"

print(f"\n1. Initial models_dir: {models_dir}")
print(f"   Exists: {models_dir.exists()}")

# Fallback logic from the fix
if not models_dir.exists():
    docker_models = Path("/app/models")
    if docker_models.exists():
        models_dir = docker_models
        print(f"   Fallback to /app/models: SUCCESS")
    else:
        alt_models = Path(__file__).parent / ".." / ".." / ".." / "models"
        if alt_models.exists():
            models_dir = alt_models.resolve()
            print(f"   Fallback to alt_models: SUCCESS")

print(f"\n2. Final models_dir: {models_dir}")
print(f"   Exists: {models_dir.exists()}")

if models_dir.exists():
    available_files = sorted(models_dir.glob("*.pkl"))
    print(f"\n3. Available .pkl files: {len(available_files)}")
    for f in available_files[-5:]:
        print(f"   - {f.name}")

# Try to load ensemble_latest.pkl
print(f"\n4. Loading ensemble_latest.pkl...")
ensemble_path = models_dir / "ensemble_latest.pkl"
print(f"   Path: {ensemble_path}")
print(f"   Exists: {ensemble_path.exists()}")
print(f"   Is symlink: {ensemble_path.is_symlink()}")

if ensemble_path.is_symlink():
    target = ensemble_path.resolve()
    print(f"   Symlink target: {target}")
    print(f"   Target exists: {target.exists()}")

try:
    classifier = joblib.load(ensemble_path)
    print(f"   ✓ Successfully loaded!")
    print(f"   Type: {type(classifier)}")
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    print(f"   Error code: {e.errno if hasattr(e, 'errno') else 'N/A'}")

# Try the fallback: find any ensemble_*.pkl
print(f"\n5. Fallback: Looking for any ensemble_*.pkl...")
ensemble_candidates = sorted(
    models_dir.glob("ensemble_*.pkl"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
print(f"   Found {len(ensemble_candidates)} candidates")

if ensemble_candidates:
    ensemble_path = ensemble_candidates[0]
    print(f"   Most recent: {ensemble_path.name} (mtime: {ensemble_path.stat().st_mtime})")
    try:
        classifier = joblib.load(ensemble_path)
        print(f"   ✓ Successfully loaded!")
        print(f"   Type: {type(classifier)}")
    except Exception as e:
        print(f"   ✗ FAILED: {type(e).__name__}: {e}")
else:
    print(f"   ✗ No ensemble models found!")
