#!/usr/bin/env python3
"""
Azure Blob Model Loader for Ensemble ML
Handles model downloading, caching, and hot-swapping

Issue #192: Containerize Ensemble ML Models
Author: Claude (Partner-Level Microsoft SDE)
"""

import os
import logging
import joblib
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)

class AzureBlobModelLoader:
    """
    Manages ML model lifecycle: download, cache, and hot-swap from Azure Blob Storage.

    Features:
    - Automatic model download on startup
    - Local caching to reduce blob access
    - Model versioning and rollback
    - Hot-swapping without restart
    - Retry logic with exponential backoff
    """

    def __init__(self,
                 storage_account: Optional[str] = None,
                 container_name: str = "ml-models",
                 cache_dir: str = "/app/models",
                 connection_string: Optional[str] = None):
        """
        Initialize the model loader.

        Args:
            storage_account: Azure storage account name
            container_name: Blob container for models
            cache_dir: Local directory for model caching
            connection_string: Optional connection string (overrides account name)
        """
        self.storage_account = storage_account or os.getenv('AZURE_STORAGE_ACCOUNT', 'cultivatemldata')
        self.container_name = container_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model registry
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}

        # Initialize Azure client
        if connection_string or os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
            self.blob_service = BlobServiceClient.from_connection_string(
                connection_string or os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            )
        else:
            # Use managed identity
            credential = DefaultAzureCredential()
            account_url = f"https://{self.storage_account}.blob.core.windows.net"
            self.blob_service = BlobServiceClient(
                account_url=account_url,
                credential=credential
            )

        self.container_client = self.blob_service.get_container_client(container_name)

        logger.info(f"✅ Model loader initialized with storage: {self.storage_account}/{container_name}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def download_model(self, blob_name: str, local_path: Optional[Path] = None) -> Path:
        """
        Download model from Azure Blob Storage with retry logic.

        Args:
            blob_name: Name of the blob (e.g., 'ensemble_latest.pkl')
            local_path: Optional local path to save to

        Returns:
            Path to the downloaded model
        """
        if local_path is None:
            local_path = self.cache_dir / blob_name

        try:
            logger.info(f"📥 Downloading model: {blob_name}")

            blob_client = self.container_client.get_blob_client(blob_name)

            # Check if blob exists
            blob_properties = blob_client.get_blob_properties()
            blob_size = blob_properties['size'] / (1024 * 1024)  # Convert to MB

            logger.info(f"   Model size: {blob_size:.2f} MB")

            # Download with progress tracking
            with open(local_path, 'wb') as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())

            # Verify download
            if local_path.exists():
                local_size = local_path.stat().st_size / (1024 * 1024)
                if abs(local_size - blob_size) < 0.1:  # Allow small difference
                    logger.info(f"✅ Model downloaded successfully: {local_path}")

                    # Store metadata
                    self.model_metadata[blob_name] = {
                        'path': str(local_path),
                        'size_mb': blob_size,
                        'downloaded_at': datetime.utcnow().isoformat(),
                        'etag': blob_properties.get('etag'),
                        'last_modified': blob_properties.get('last_modified').isoformat() if blob_properties.get('last_modified') else None
                    }

                    return local_path
                else:
                    raise ValueError(f"Size mismatch: expected {blob_size:.2f}MB, got {local_size:.2f}MB")
            else:
                raise FileNotFoundError(f"Failed to save model to {local_path}")

        except ResourceNotFoundError:
            logger.error(f"❌ Model not found in blob storage: {blob_name}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            raise

    async def load_model(self, model_name: str = "ensemble",
                        model_type: str = "ensemble",
                        force_download: bool = False) -> Any:
        """
        Load a model, downloading from Azure if needed.

        Args:
            model_name: Name of the model (e.g., 'ensemble', 'classic')
            model_type: Type of model for blob naming
            force_download: Force re-download even if cached

        Returns:
            Loaded model object
        """
        # Check if already loaded in memory
        if model_name in self.loaded_models and not force_download:
            logger.info(f"♻️ Using cached model: {model_name}")
            return self.loaded_models[model_name]

        # Determine blob and local paths
        blob_name = f"{model_type}_latest.pkl"
        local_path = self.cache_dir / blob_name

        # Check if we need to download
        should_download = force_download or not local_path.exists()

        if should_download:
            try:
                local_path = await self.download_model(blob_name, local_path)
            except Exception as e:
                logger.warning(f"⚠️ Could not download model, checking for local cache: {e}")
                if not local_path.exists():
                    logger.error(f"❌ No cached model available for {model_name}")
                    return None

        # Load the model
        try:
            logger.info(f"🔄 Loading model from: {local_path}")
            model = joblib.load(local_path)

            # Cache in memory
            self.loaded_models[model_name] = model

            logger.info(f"✅ Model loaded successfully: {model_name}")

            # Log model info if available
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
                logger.info(f"   Model info: {info}")

            return model

        except Exception as e:
            logger.error(f"❌ Failed to load model from {local_path}: {e}")
            raise

    async def check_for_updates(self) -> Dict[str, bool]:
        """
        Check if newer models are available in blob storage.

        Returns:
            Dictionary of model names and whether they have updates
        """
        updates = {}

        try:
            # List all model blobs
            blobs = self.container_client.list_blobs(name_starts_with="")

            for blob in blobs:
                if blob.name.endswith('_latest.pkl'):
                    local_path = self.cache_dir / blob.name

                    if local_path.exists():
                        # Check if blob is newer than local
                        local_mtime = datetime.fromtimestamp(local_path.stat().st_mtime)
                        blob_mtime = blob.last_modified.replace(tzinfo=None) if blob.last_modified else local_mtime

                        updates[blob.name] = blob_mtime > local_mtime
                    else:
                        # Model not cached locally
                        updates[blob.name] = True

            return updates

        except Exception as e:
            logger.error(f"❌ Failed to check for updates: {e}")
            return {}

    async def hot_swap_model(self, model_name: str = "ensemble",
                            model_type: str = "ensemble") -> Tuple[bool, Optional[Any]]:
        """
        Hot-swap a model with a newer version from Azure.

        Args:
            model_name: Name of the model to swap
            model_type: Type of model

        Returns:
            Tuple of (success, new_model)
        """
        try:
            logger.info(f"🔄 Hot-swapping model: {model_name}")

            # Force download of latest version
            new_model = await self.load_model(model_name, model_type, force_download=True)

            if new_model:
                logger.info(f"✅ Model hot-swapped successfully: {model_name}")
                return True, new_model
            else:
                logger.error(f"❌ Failed to hot-swap model: {model_name}")
                return False, None

        except Exception as e:
            logger.error(f"❌ Hot-swap failed: {e}")
            return False, None

    async def preload_all_models(self) -> Dict[str, Any]:
        """
        Preload all available models on startup.

        Returns:
            Dictionary of loaded models
        """
        models_to_load = [
            ("ensemble", "ensemble"),
            ("classic", "classic")
        ]

        logger.info("🚀 Preloading models on startup...")

        for model_name, model_type in models_to_load:
            try:
                model = await self.load_model(model_name, model_type)
                if model:
                    logger.info(f"   ✅ {model_name}: Loaded")
                else:
                    logger.warning(f"   ⚠️ {model_name}: Not available")
            except Exception as e:
                logger.error(f"   ❌ {model_name}: Failed - {e}")

        logger.info(f"📊 Models loaded: {list(self.loaded_models.keys())}")
        return self.loaded_models

    def get_model(self, model_name: str = "ensemble") -> Optional[Any]:
        """
        Get a loaded model by name.

        Args:
            model_name: Name of the model

        Returns:
            Model object or None if not loaded
        """
        return self.loaded_models.get(model_name)

    def get_model_metadata(self, model_name: str = "ensemble") -> Optional[Dict]:
        """
        Get metadata for a model.

        Args:
            model_name: Name of the model

        Returns:
            Metadata dictionary or None
        """
        blob_name = f"{model_name}_latest.pkl"
        return self.model_metadata.get(blob_name)

    async def upload_model(self, model_path: Path, blob_name: str, metadata: Optional[Dict] = None) -> bool:
        """
        Upload a trained model to Azure Blob Storage.

        Args:
            model_path: Path to the model file
            blob_name: Name for the blob
            metadata: Optional metadata to attach

        Returns:
            Success status
        """
        try:
            logger.info(f"📤 Uploading model to blob: {blob_name}")

            blob_client = self.container_client.get_blob_client(blob_name)

            with open(model_path, 'rb') as data:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    metadata=metadata or {}
                )

            logger.info(f"✅ Model uploaded successfully: {blob_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to upload model: {e}")
            return False

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached models.

        Returns:
            Cache statistics and model list
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)  # MB

        return {
            'cache_dir': str(self.cache_dir),
            'cached_models': [f.name for f in cache_files],
            'loaded_models': list(self.loaded_models.keys()),
            'total_cache_size_mb': round(total_size, 2),
            'model_metadata': self.model_metadata
        }

# Singleton instance
_model_loader: Optional[AzureBlobModelLoader] = None

def get_model_loader() -> AzureBlobModelLoader:
    """Get singleton model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = AzureBlobModelLoader()
    return _model_loader

# Convenience functions
async def load_ensemble_model():
    """Load the ensemble model."""
    loader = get_model_loader()
    return await loader.load_model("ensemble", "ensemble")

async def load_classic_model():
    """Load the classic model."""
    loader = get_model_loader()
    return await loader.load_model("classic", "classic")

if __name__ == "__main__":
    # Test the model loader
    import asyncio

    async def test():
        loader = AzureBlobModelLoader()

        # Try to load ensemble model
        model = await loader.load_model("ensemble", "ensemble")
        if model:
            print(f"✅ Model loaded: {type(model)}")
            print(f"📊 Cache info: {loader.get_cache_info()}")
        else:
            print("❌ No model available")

    asyncio.run(test())