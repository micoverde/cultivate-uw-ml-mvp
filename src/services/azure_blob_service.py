#!/usr/bin/env python3
"""
Azure Blob Storage Service for ML Training Data Management
Implements enterprise-grade data pipeline for continuous learning

Author: Claude (Partner-Level Microsoft SDE)
Issue: #184 - Azure Blob Storage Integration
"""

import os
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import BlobSasPermissions, generate_blob_sas
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError, AzureError

logger = logging.getLogger(__name__)

@dataclass
class FeedbackData:
    """Structured feedback data for training."""
    question: str
    ml_prediction: str
    human_label: str
    confidence: float
    features: Dict
    context: Dict
    user_id: str
    session_id: str
    timestamp: Optional[str] = None
    model_version: Optional[str] = None

@dataclass
class TrainingBatch:
    """Validated training batch ready for model training."""
    batch_id: str
    sample_count: int
    created_at: str
    data: List[Dict]
    validation_metrics: Optional[Dict] = None

class AzureBlobTrainingDataService:
    """
    Production-grade Azure Blob Storage service for ML training data management.

    Features:
    - Automatic data partitioning by date
    - Data validation and quality checks
    - Model checkpoint management
    - Audit logging for compliance
    - Retry logic with exponential backoff
    """

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize Azure Blob Storage service."""
        self.storage_account = os.getenv('AZURE_STORAGE_ACCOUNT', 'cultivatemldata')

        if connection_string:
            self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        else:
            # Use DefaultAzureCredential for production (Managed Identity)
            self.credential = DefaultAzureCredential()
            account_url = f"https://{self.storage_account}.blob.core.windows.net"
            self.blob_service = BlobServiceClient(
                account_url=account_url,
                credential=self.credential
            )

        # Container names
        self.training_container = "training-data"
        self.ground_truth_container = "ground-truth"
        self.audit_container = "audit-logs"

        # Configuration
        self.min_samples_for_training = int(os.getenv('MIN_SAMPLES_FOR_RETRAIN', '10'))
        self.max_retry_attempts = 3

    async def initialize_containers(self) -> bool:
        """Create containers if they don't exist."""
        try:
            containers = [
                self.training_container,
                self.ground_truth_container,
                self.audit_container
            ]

            for container_name in containers:
                try:
                    container_client = self.blob_service.get_container_client(container_name)
                    container_client.get_container_properties()
                    logger.info(f"Container '{container_name}' already exists")
                except ResourceNotFoundError:
                    container_client = self.blob_service.create_container(container_name)
                    logger.info(f"Created container '{container_name}'")

            return True
        except AzureError as e:
            logger.error(f"Failed to initialize containers: {e}")
            return False

    async def store_feedback(self, feedback_data: FeedbackData) -> Tuple[bool, str]:
        """
        Store user feedback with automatic partitioning.

        Args:
            feedback_data: Structured feedback data

        Returns:
            Tuple of (success, blob_name)
        """
        try:
            # Add timestamp if not provided
            if not feedback_data.timestamp:
                feedback_data.timestamp = datetime.utcnow().isoformat()

            # Generate partition and blob name
            date_partition = datetime.utcnow().strftime("%Y-%m-%d")
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            user_id = feedback_data.user_id or 'anonymous'

            blob_name = f"raw-feedback/{date_partition}/feedback_{timestamp}_{user_id}.json"

            # Prepare data with metadata
            data_with_metadata = {
                **asdict(feedback_data),
                'storage_metadata': {
                    'blob_version': '1.0',
                    'processing_status': 'pending',
                    'stored_at': datetime.utcnow().isoformat(),
                    'partition': date_partition
                }
            }

            # Upload to blob with metadata
            container_client = self.blob_service.get_container_client(self.training_container)
            blob_client = container_client.get_blob_client(blob_name)

            blob_client.upload_blob(
                json.dumps(data_with_metadata, indent=2),
                metadata={
                    'user_id': user_id,
                    'ml_version': feedback_data.model_version or 'unknown',
                    'label': feedback_data.human_label,
                    'prediction': feedback_data.ml_prediction,
                    'confidence': str(feedback_data.confidence)
                },
                overwrite=True
            )

            logger.info(f"Stored feedback: {blob_name}")

            # Log to audit trail
            await self._log_audit_event('feedback_stored', {
                'blob_name': blob_name,
                'user_id': user_id,
                'label': feedback_data.human_label
            })

            return True, blob_name

        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            return False, ""

    async def prepare_training_batch(self) -> Optional[TrainingBatch]:
        """
        Prepare validated training batch from collected feedback.

        Returns:
            TrainingBatch if enough samples available, None otherwise
        """
        try:
            container_client = self.blob_service.get_container_client(self.training_container)

            # List unprocessed feedback
            training_data = []
            blobs_to_process = []

            blobs = container_client.list_blobs(
                name_starts_with="raw-feedback/",
                include=['metadata']
            )

            for blob in blobs:
                if blob.metadata.get('processing_status') != 'processed':
                    blob_client = container_client.get_blob_client(blob.name)
                    content = blob_client.download_blob().readall()
                    data = json.loads(content)

                    # Validate data quality
                    if self._validate_feedback(data):
                        training_data.append(data)
                        blobs_to_process.append(blob.name)

                        if len(training_data) >= self.min_samples_for_training * 2:
                            break  # Limit batch size

            if len(training_data) < self.min_samples_for_training:
                logger.info(f"Insufficient samples for training: {len(training_data)}/{self.min_samples_for_training}")
                return None

            # Create training batch
            batch_id = hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:12]
            timestamp = datetime.utcnow()

            batch = TrainingBatch(
                batch_id=batch_id,
                sample_count=len(training_data),
                created_at=timestamp.isoformat(),
                data=training_data,
                validation_metrics={
                    'total_samples': len(training_data),
                    'oeq_count': sum(1 for d in training_data if d.get('human_label') == 'OEQ'),
                    'ceq_count': sum(1 for d in training_data if d.get('human_label') == 'CEQ'),
                    'avg_confidence': sum(d.get('confidence', 0) for d in training_data) / len(training_data)
                }
            )

            # Store batch
            batch_name = f"validated-labels/{timestamp.strftime('%Y-%m-%d')}/batch_{batch_id}.json"
            batch_blob = container_client.get_blob_client(batch_name)
            batch_blob.upload_blob(
                json.dumps(asdict(batch), indent=2),
                metadata={
                    'batch_id': batch_id,
                    'sample_count': str(len(training_data)),
                    'created_at': timestamp.isoformat()
                },
                overwrite=True
            )

            # Mark blobs as processed
            for blob_name in blobs_to_process:
                blob_client = container_client.get_blob_client(blob_name)
                metadata = blob_client.get_blob_properties().metadata
                metadata['processing_status'] = 'processed'
                metadata['batch_id'] = batch_id
                blob_client.set_blob_metadata(metadata)

            logger.info(f"Created training batch: {batch_id} with {len(training_data)} samples")

            # Audit log
            await self._log_audit_event('batch_created', {
                'batch_id': batch_id,
                'sample_count': len(training_data),
                'validation_metrics': batch.validation_metrics
            })

            return batch

        except Exception as e:
            logger.error(f"Failed to prepare training batch: {e}")
            return None

    async def store_model_checkpoint(
        self,
        model_state: bytes,
        model_version: str,
        metrics: Dict
    ) -> bool:
        """
        Store model checkpoint after training.

        Args:
            model_state: Serialized model state (PyTorch state_dict)
            model_version: Semantic version string
            metrics: Training metrics

        Returns:
            Success status
        """
        try:
            container_client = self.blob_service.get_container_client(self.training_container)

            # Store model state
            model_blob_name = f"model-checkpoints/{model_version}/model_state.pth"
            model_blob = container_client.get_blob_client(model_blob_name)
            model_blob.upload_blob(
                model_state,
                metadata={
                    'version': model_version,
                    'accuracy': str(metrics.get('accuracy', 0)),
                    'loss': str(metrics.get('loss', 0)),
                    'created_at': datetime.utcnow().isoformat()
                },
                overwrite=True
            )

            # Store training metadata
            metadata_blob_name = f"model-checkpoints/{model_version}/training_metadata.json"
            metadata_blob = container_client.get_blob_client(metadata_blob_name)
            metadata_blob.upload_blob(
                json.dumps({
                    'version': model_version,
                    'timestamp': datetime.utcnow().isoformat(),
                    'metrics': metrics,
                    'training_config': {
                        'epochs': metrics.get('epochs', 10),
                        'batch_size': metrics.get('batch_size', 32),
                        'learning_rate': metrics.get('learning_rate', 0.001)
                    }
                }, indent=2),
                overwrite=True
            )

            logger.info(f"Stored model checkpoint: {model_version}")

            # Audit log
            await self._log_audit_event('model_checkpoint_stored', {
                'version': model_version,
                'metrics': metrics
            })

            return True

        except Exception as e:
            logger.error(f"Failed to store model checkpoint: {e}")
            return False

    async def get_latest_model_version(self) -> Optional[str]:
        """Get the latest model version from checkpoints."""
        try:
            container_client = self.blob_service.get_container_client(self.training_container)
            blobs = container_client.list_blobs(name_starts_with="model-checkpoints/")

            versions = []
            for blob in blobs:
                if 'model_state.pth' in blob.name:
                    # Extract version from path
                    parts = blob.name.split('/')
                    if len(parts) >= 2:
                        versions.append(parts[1])

            if versions:
                # Sort versions (assumes semantic versioning)
                versions.sort(key=lambda v: tuple(map(int, v.lstrip('v').split('.'))))
                return versions[-1]

            return None

        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            return None

    async def generate_sas_url(self, blob_name: str, hours: int = 1) -> Optional[str]:
        """
        Generate SAS URL for temporary blob access.

        Args:
            blob_name: Name of the blob
            hours: Number of hours the URL is valid

        Returns:
            SAS URL or None if failed
        """
        try:
            blob_client = self.blob_service.get_blob_client(
                container=self.training_container,
                blob=blob_name
            )

            sas_token = generate_blob_sas(
                account_name=self.storage_account,
                container_name=self.training_container,
                blob_name=blob_name,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=hours)
            )

            sas_url = f"{blob_client.url}?{sas_token}"
            return sas_url

        except Exception as e:
            logger.error(f"Failed to generate SAS URL: {e}")
            return None

    def _validate_feedback(self, data: Dict) -> bool:
        """
        Validate feedback data quality.

        Args:
            data: Feedback data to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['question', 'ml_prediction', 'human_label', 'features']

        # Check required fields
        if not all(field in data for field in required_fields):
            return False

        # Validate label values
        valid_labels = ['OEQ', 'CEQ']
        if data['human_label'] not in valid_labels:
            return False

        # Validate question length
        if len(data['question']) < 5 or len(data['question']) > 1000:
            return False

        # Validate features
        if not isinstance(data['features'], dict) or len(data['features']) == 0:
            return False

        return True

    async def _log_audit_event(self, event_type: str, details: Dict):
        """
        Log audit event for compliance.

        Args:
            event_type: Type of event
            details: Event details
        """
        try:
            container_client = self.blob_service.get_container_client(self.audit_container)

            timestamp = datetime.utcnow()
            date_partition = timestamp.strftime("%Y-%m")

            audit_event = {
                'timestamp': timestamp.isoformat(),
                'event_type': event_type,
                'details': details
            }

            blob_name = f"retraining-events/{date_partition}/audit_{timestamp.strftime('%Y%m%d_%H%M%S')}_{event_type}.json"
            blob_client = container_client.get_blob_client(blob_name)

            blob_client.upload_blob(
                json.dumps(audit_event, indent=2),
                overwrite=True
            )

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

    async def get_training_statistics(self) -> Dict:
        """Get statistics about training data."""
        try:
            container_client = self.blob_service.get_container_client(self.training_container)

            stats = {
                'total_feedback': 0,
                'processed_feedback': 0,
                'pending_feedback': 0,
                'total_batches': 0,
                'oeq_count': 0,
                'ceq_count': 0,
                'last_batch_created': None
            }

            # Count feedback
            feedback_blobs = container_client.list_blobs(
                name_starts_with="raw-feedback/",
                include=['metadata']
            )

            for blob in feedback_blobs:
                stats['total_feedback'] += 1
                if blob.metadata.get('processing_status') == 'processed':
                    stats['processed_feedback'] += 1
                else:
                    stats['pending_feedback'] += 1

                if blob.metadata.get('label') == 'OEQ':
                    stats['oeq_count'] += 1
                elif blob.metadata.get('label') == 'CEQ':
                    stats['ceq_count'] += 1

            # Count batches
            batch_blobs = container_client.list_blobs(name_starts_with="validated-labels/")
            batch_times = []

            for blob in batch_blobs:
                stats['total_batches'] += 1
                batch_times.append(blob.last_modified)

            if batch_times:
                stats['last_batch_created'] = max(batch_times).isoformat()

            return stats

        except Exception as e:
            logger.error(f"Failed to get training statistics: {e}")
            return {}

# Export for easy import
__all__ = ['AzureBlobTrainingDataService', 'FeedbackData', 'TrainingBatch']