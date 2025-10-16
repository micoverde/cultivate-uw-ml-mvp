#!/usr/bin/env python3
"""
Download ML models from Azure Blob Storage at container startup.
Ensures models are available for the API without needing Git LFS in Docker.
"""

import os
import sys
from pathlib import Path


def download_models_from_blob():
    """Download models from Azure Blob Storage."""
    print("🚀 Cultivate ML API Startup")
    print("=" * 50)

    models_dir = Path("/app/models")
    models_dir.mkdir(exist_ok=True)

    # Check if REAL model files exist (filter out Git LFS pointer files)
    # LFS pointers are ~131 bytes, real models are >1MB
    # This is necessary because when cloning without Git LFS (local dev, Docker builds),
    # the models/ directory contains 131-byte pointer files. Without size filtering,
    # we'd skip downloading and the API would try to load pointers instead of real models.
    existing_models = [
        f for f in models_dir.glob("*.pkl")
        if f.stat().st_size > 1000
    ]

    if existing_models:
        print("✅ Models directory already populated, skipping download")
        print(f"   Found {len(existing_models)} model files")
        return True

    # Clean up any LFS pointers that might exist
    lfs_pointers = [
        f for f in models_dir.glob("*.pkl")
        if f.stat().st_size <= 1000
    ]
    if lfs_pointers:
        print(f"⚠️  Found {len(lfs_pointers)} Git LFS pointer files (removing)")
        for pointer in lfs_pointers:
            pointer.unlink()
            print(f"   Deleted: {pointer.name}")

    print("📥 Downloading ML models from Azure Blob Storage...")

    # Get connection string from environment
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    print(f"   Connection string set: {bool(connection_string)}")
    if not connection_string:
        print("⚠️  AZURE_STORAGE_CONNECTION_STRING not set")
        print("   Models will not be available in this environment")
        print("   Ensemble endpoint will use heuristic fallback")
        return False

    try:
        from azure.storage.blob import BlobServiceClient

        # Create blob client
        client = BlobServiceClient.from_connection_string(connection_string)
        container_client = client.get_container_client("ml-models")

        models_downloaded = 0

        # Download ensemble models
        print("   Downloading ensemble models...")
        for blob in container_client.list_blobs(name_starts_with="ensemble_"):
            if blob.name.endswith(".pkl"):
                model_path = models_dir / blob.name
                print(f"   📥 {blob.name}", end="", flush=True)
                blob_client = container_client.get_blob_client(blob.name)
                with open(model_path, "wb") as f:
                    f.write(blob_client.download_blob().readall())
                print(" ✅")
                models_downloaded += 1

        # Download classic models
        print("   Downloading classic models...")
        for blob in container_client.list_blobs(name_starts_with="classic_"):
            if blob.name.endswith(".pkl"):
                model_path = models_dir / blob.name
                print(f"   📥 {blob.name}", end="", flush=True)
                blob_client = container_client.get_blob_client(blob.name)
                with open(model_path, "wb") as f:
                    f.write(blob_client.download_blob().readall())
                print(" ✅")
                models_downloaded += 1

        # Verify models were downloaded
        if models_downloaded > 0:
            print(f"✅ Models downloaded successfully ({models_downloaded} files)")
            return True
        else:
            print("❌ Failed to download any models from blob storage")
            return False

    except ImportError as e:
        print(f"❌ azure-storage-blob package not installed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_available_models():
    """List available model files."""
    models_dir = Path("/app/models")
    print()
    print("📦 Available Models:")

    pkl_files = list(models_dir.glob("*.pkl"))
    if pkl_files:
        for model_file in sorted(pkl_files):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"   {model_file.name} ({size_mb:.2f} MB)")
    else:
        print("   (none)")


def main():
    """Main startup sequence."""
    try:
        # Download models
        success = download_models_from_blob()

        # List available models
        list_available_models()

        print()
        print("🎯 Starting ML API Server...")
        print("=" * 50)

        # If models failed to download and this is production, exit
        if not success:
            # In production (with connection string), we should fail fast
            if os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
                print("⚠️  Production environment with missing models - this may cause API failures")
                # Continue anyway - API has fallback heuristic for ensemble

        # Start the API
        os.execvp("python", ["python", "run_api.py"])

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
