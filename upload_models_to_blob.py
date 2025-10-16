#!/usr/bin/env python3
"""
Upload trained ML models to Azure Blob Storage.

Usage:
    python upload_models_to_blob.py                    # Upload all models from ./models
    python upload_models_to_blob.py --dir /path/to/models
    python upload_models_to_blob.py --pattern ensemble_*.pkl
    python upload_models_to_blob.py --dry-run          # Show what would be uploaded

Requires:
    AZURE_STORAGE_CONNECTION_STRING environment variable
"""

import os
import sys
import argparse
from pathlib import Path
from azure.storage.blob import BlobServiceClient


def upload_models(
    models_dir: Path,
    pattern: str = "*.pkl",
    dry_run: bool = False,
    overwrite: bool = True,
) -> tuple[int, int]:
    """
    Upload model files to Azure Blob Storage.

    Args:
        models_dir: Directory containing model files
        pattern: Glob pattern for model files to upload
        dry_run: If True, show what would be uploaded without uploading
        overwrite: If True, overwrite existing blobs

    Returns:
        Tuple of (uploaded_count, total_count)
    """
    # Validate models directory
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        sys.exit(1)

    if not models_dir.is_dir():
        print(f"❌ Not a directory: {models_dir}")
        sys.exit(1)

    # Get connection string
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("❌ AZURE_STORAGE_CONNECTION_STRING environment variable not set")
        print("   Set it with: export AZURE_STORAGE_CONNECTION_STRING='...'")
        sys.exit(1)

    # Find model files
    model_files = sorted(models_dir.glob(pattern))
    if not model_files:
        print(f"❌ No model files found matching '{pattern}' in {models_dir}")
        sys.exit(1)

    print("🚀 Upload Models to Azure Blob Storage")
    print("=" * 60)
    print(f"📁 Source directory: {models_dir}")
    print(f"📋 Pattern: {pattern}")
    print(f"📊 Found {len(model_files)} model file(s)")

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be uploaded\n")
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"   Would upload: {model_file.name} ({size_mb:.2f} MB)")
        return 0, len(model_files)

    print("\n📤 Uploading to Azure Blob Storage...\n")

    try:
        # Connect to Azure Blob Storage
        client = BlobServiceClient.from_connection_string(connection_string)
        container_client = client.get_container_client("ml-models")

        uploaded_count = 0

        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"   📥 {model_file.name} ({size_mb:.2f} MB)", end=" ", flush=True)

            try:
                with open(model_file, "rb") as f:
                    container_client.upload_blob(
                        model_file.name,
                        f,
                        overwrite=overwrite,
                    )
                print("✅")
                uploaded_count += 1
            except Exception as e:
                print(f"❌ Error: {e}")

        print("\n" + "=" * 60)
        print(f"✅ Uploaded {uploaded_count}/{len(model_files)} model files")
        return uploaded_count, len(model_files)

    except Exception as e:
        print(f"\n❌ Failed to connect to Azure Blob Storage: {e}")
        print("\nTroubleshooting:")
        print("  • Verify AZURE_STORAGE_CONNECTION_STRING is set correctly")
        print("  • Verify 'ml-models' container exists in the storage account")
        print("  • Check your network connection to Azure")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload trained ML models to Azure Blob Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all .pkl files from ./models
  python upload_models_to_blob.py

  # Upload from custom directory
  python upload_models_to_blob.py --dir /path/to/models

  # Upload only ensemble models
  python upload_models_to_blob.py --pattern "ensemble_*.pkl"

  # Preview what would be uploaded
  python upload_models_to_blob.py --dry-run

Environment:
  AZURE_STORAGE_CONNECTION_STRING - Required. Azure Blob Storage connection string
        """,
    )

    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("models"),
        help="Directory containing model files (default: ./models)",
    )

    parser.add_argument(
        "--pattern",
        default="*.pkl",
        help="Glob pattern for model files (default: *.pkl)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Don't overwrite existing blobs (default: overwrite)",
    )

    args = parser.parse_args()

    # Run upload
    uploaded, total = upload_models(
        args.dir,
        pattern=args.pattern,
        dry_run=args.dry_run,
        overwrite=not args.no_overwrite,
    )

    # Exit with appropriate code
    if uploaded == total:
        sys.exit(0)
    elif uploaded == 0:
        sys.exit(1)
    else:
        sys.exit(1)  # Partial success is still a failure


if __name__ == "__main__":
    main()
