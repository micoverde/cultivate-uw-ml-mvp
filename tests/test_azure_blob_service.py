#!/usr/bin/env python3
"""
Unit tests for Azure Blob Storage Service
Tests the core functionality of training data management

Author: Claude (Partner-Level Microsoft SDE)
Issue: #184 - Azure Blob Storage Integration
"""

import os
import sys
import json
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.azure_blob_service import (
    AzureBlobTrainingDataService,
    FeedbackData,
    TrainingBatch
)


class TestAzureBlobService(unittest.TestCase):
    """Test suite for Azure Blob Storage service."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = None
        self.mock_blob_client = MagicMock()
        self.mock_container_client = MagicMock()

        # Sample feedback data
        self.sample_feedback = FeedbackData(
            question="Why is the sky blue?",
            ml_prediction="OEQ",
            human_label="CEQ",
            confidence=0.85,
            features={"length": 20, "has_why": True},
            context={"source": "demo1"},
            user_id="test_user",
            session_id="test_session",
            model_version="v1.0"
        )

    @patch('src.services.azure_blob_service.BlobServiceClient')
    def test_service_initialization(self, mock_blob_service):
        """Test service initialization with connection string."""
        connection_string = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test_key"
        service = AzureBlobTrainingDataService(connection_string)

        mock_blob_service.from_connection_string.assert_called_once_with(connection_string)
        self.assertEqual(service.training_container, "training-data")
        self.assertEqual(service.ground_truth_container, "ground-truth")
        self.assertEqual(service.audit_container, "audit-logs")

    @patch('src.services.azure_blob_service.DefaultAzureCredential')
    @patch('src.services.azure_blob_service.BlobServiceClient')
    def test_service_initialization_with_managed_identity(self, mock_blob_service, mock_credential):
        """Test service initialization with managed identity."""
        os.environ['AZURE_STORAGE_ACCOUNT'] = 'testaccount'
        service = AzureBlobTrainingDataService()

        mock_credential.assert_called_once()
        mock_blob_service.assert_called_once()

    @patch('src.services.azure_blob_service.BlobServiceClient')
    async def test_initialize_containers(self, mock_blob_service):
        """Test container initialization."""
        service = AzureBlobTrainingDataService("test_connection_string")
        service.blob_service = mock_blob_service

        mock_container = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container
        mock_container.get_container_properties.side_effect = [None, None, None]

        result = await service.initialize_containers()

        self.assertTrue(result)
        self.assertEqual(mock_blob_service.get_container_client.call_count, 3)

    @patch('src.services.azure_blob_service.BlobServiceClient')
    async def test_store_feedback_success(self, mock_blob_service):
        """Test successful feedback storage."""
        service = AzureBlobTrainingDataService("test_connection_string")

        mock_container = MagicMock()
        mock_blob = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob
        service.blob_service = mock_blob_service

        success, blob_name = await service.store_feedback(self.sample_feedback)

        self.assertTrue(success)
        self.assertIn("raw-feedback", blob_name)
        self.assertIn("test_user", blob_name)
        mock_blob.upload_blob.assert_called_once()

    def test_validate_feedback_valid(self):
        """Test feedback validation with valid data."""
        service = AzureBlobTrainingDataService("test_connection_string")

        valid_data = {
            'question': 'What is photosynthesis?',
            'ml_prediction': 'OEQ',
            'human_label': 'CEQ',
            'features': {'length': 23}
        }

        self.assertTrue(service._validate_feedback(valid_data))

    def test_validate_feedback_invalid_label(self):
        """Test feedback validation with invalid label."""
        service = AzureBlobTrainingDataService("test_connection_string")

        invalid_data = {
            'question': 'What is photosynthesis?',
            'ml_prediction': 'OEQ',
            'human_label': 'INVALID',
            'features': {'length': 23}
        }

        self.assertFalse(service._validate_feedback(invalid_data))

    def test_validate_feedback_missing_fields(self):
        """Test feedback validation with missing fields."""
        service = AzureBlobTrainingDataService("test_connection_string")

        invalid_data = {
            'question': 'What is photosynthesis?',
            'ml_prediction': 'OEQ'
            # Missing human_label and features
        }

        self.assertFalse(service._validate_feedback(invalid_data))

    def test_validate_feedback_short_question(self):
        """Test feedback validation with too short question."""
        service = AzureBlobTrainingDataService("test_connection_string")

        invalid_data = {
            'question': 'Hi?',
            'ml_prediction': 'OEQ',
            'human_label': 'CEQ',
            'features': {'length': 3}
        }

        self.assertFalse(service._validate_feedback(invalid_data))

    @patch('src.services.azure_blob_service.BlobServiceClient')
    async def test_prepare_training_batch_insufficient_samples(self, mock_blob_service):
        """Test training batch preparation with insufficient samples."""
        service = AzureBlobTrainingDataService("test_connection_string")
        service.blob_service = mock_blob_service

        mock_container = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container

        # Mock only 5 blobs (less than minimum required)
        mock_blobs = []
        for i in range(5):
            blob = MagicMock()
            blob.name = f"raw-feedback/2025-01-27/feedback_{i}.json"
            blob.metadata = {'processing_status': 'pending'}
            mock_blobs.append(blob)

        mock_container.list_blobs.return_value = mock_blobs

        # Mock blob download
        mock_blob_client = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_blob_client.download_blob.return_value.readall.return_value = json.dumps({
            'question': 'Test question?',
            'ml_prediction': 'OEQ',
            'human_label': 'CEQ',
            'features': {'length': 15}
        })

        result = await service.prepare_training_batch()

        self.assertIsNone(result)

    @patch('src.services.azure_blob_service.BlobServiceClient')
    async def test_prepare_training_batch_success(self, mock_blob_service):
        """Test successful training batch preparation."""
        service = AzureBlobTrainingDataService("test_connection_string")
        service.blob_service = mock_blob_service

        mock_container = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container

        # Mock 15 blobs (more than minimum required)
        mock_blobs = []
        for i in range(15):
            blob = MagicMock()
            blob.name = f"raw-feedback/2025-01-27/feedback_{i}.json"
            blob.metadata = {'processing_status': 'pending'}
            mock_blobs.append(blob)

        mock_container.list_blobs.return_value = mock_blobs

        # Mock blob operations
        mock_blob_client = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client

        # Create valid feedback data
        feedback_data = {
            'question': 'Test question?',
            'ml_prediction': 'OEQ',
            'human_label': 'CEQ',
            'features': {'length': 15},
            'confidence': 0.9
        }
        mock_blob_client.download_blob.return_value.readall.return_value = json.dumps(feedback_data)
        mock_blob_client.get_blob_properties.return_value.metadata = {'processing_status': 'pending'}

        result = await service.prepare_training_batch()

        self.assertIsNotNone(result)
        self.assertIsInstance(result, TrainingBatch)
        self.assertEqual(result.sample_count, 15)
        self.assertTrue(len(result.batch_id) > 0)

    @patch('src.services.azure_blob_service.BlobServiceClient')
    async def test_store_model_checkpoint(self, mock_blob_service):
        """Test model checkpoint storage."""
        service = AzureBlobTrainingDataService("test_connection_string")
        service.blob_service = mock_blob_service

        mock_container = MagicMock()
        mock_blob = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob

        model_state = b"mock_model_state_dict"
        model_version = "v2.0.1"
        metrics = {
            'accuracy': 0.95,
            'loss': 0.12,
            'epochs': 10
        }

        result = await service.store_model_checkpoint(model_state, model_version, metrics)

        self.assertTrue(result)
        # Should create 2 blobs: model state and metadata
        self.assertEqual(mock_container.get_blob_client.call_count, 2)

    @patch('src.services.azure_blob_service.BlobServiceClient')
    async def test_get_latest_model_version(self, mock_blob_service):
        """Test getting latest model version."""
        service = AzureBlobTrainingDataService("test_connection_string")
        service.blob_service = mock_blob_service

        mock_container = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container

        # Mock model checkpoint blobs
        mock_blobs = []
        versions = ['v1.0.0', 'v1.1.0', 'v2.0.0', 'v1.2.0']
        for version in versions:
            blob = MagicMock()
            blob.name = f"model-checkpoints/{version}/model_state.pth"
            mock_blobs.append(blob)

        mock_container.list_blobs.return_value = mock_blobs

        latest_version = await service.get_latest_model_version()

        self.assertEqual(latest_version, 'v2.0.0')

    @patch('src.services.azure_blob_service.generate_blob_sas')
    @patch('src.services.azure_blob_service.BlobServiceClient')
    async def test_generate_sas_url(self, mock_blob_service, mock_generate_sas):
        """Test SAS URL generation."""
        service = AzureBlobTrainingDataService("test_connection_string")
        service.blob_service = mock_blob_service

        mock_blob = MagicMock()
        mock_blob.url = "https://test.blob.core.windows.net/container/blob"
        mock_blob_service.get_blob_client.return_value = mock_blob

        mock_generate_sas.return_value = "sas_token_here"

        sas_url = await service.generate_sas_url("test_blob.json", hours=2)

        self.assertIsNotNone(sas_url)
        self.assertIn("sas_token_here", sas_url)

    @patch('src.services.azure_blob_service.BlobServiceClient')
    async def test_get_training_statistics(self, mock_blob_service):
        """Test getting training statistics."""
        service = AzureBlobTrainingDataService("test_connection_string")
        service.blob_service = mock_blob_service

        mock_container = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container

        # Mock feedback blobs
        feedback_blobs = []
        for i in range(20):
            blob = MagicMock()
            blob.metadata = {
                'processing_status': 'processed' if i < 15 else 'pending',
                'label': 'OEQ' if i % 2 == 0 else 'CEQ'
            }
            feedback_blobs.append(blob)

        # Mock batch blobs
        batch_blobs = []
        for i in range(3):
            blob = MagicMock()
            blob.last_modified = datetime.utcnow() - timedelta(days=i)
            batch_blobs.append(blob)

        mock_container.list_blobs.side_effect = [feedback_blobs, batch_blobs]

        stats = await service.get_training_statistics()

        self.assertEqual(stats['total_feedback'], 20)
        self.assertEqual(stats['processed_feedback'], 15)
        self.assertEqual(stats['pending_feedback'], 5)
        self.assertEqual(stats['oeq_count'], 10)
        self.assertEqual(stats['ceq_count'], 10)
        self.assertEqual(stats['total_batches'], 3)
        self.assertIsNotNone(stats['last_batch_created'])


class TestFeedbackData(unittest.TestCase):
    """Test suite for FeedbackData dataclass."""

    def test_feedback_data_creation(self):
        """Test FeedbackData creation with all fields."""
        feedback = FeedbackData(
            question="What causes rain?",
            ml_prediction="OEQ",
            human_label="OEQ",
            confidence=0.92,
            features={"length": 17, "has_what": True},
            context={"source": "demo2"},
            user_id="teacher123",
            session_id="session456",
            timestamp="2025-01-27T10:30:00",
            model_version="v1.2.3"
        )

        self.assertEqual(feedback.question, "What causes rain?")
        self.assertEqual(feedback.ml_prediction, "OEQ")
        self.assertEqual(feedback.human_label, "OEQ")
        self.assertEqual(feedback.confidence, 0.92)
        self.assertEqual(feedback.user_id, "teacher123")

    def test_feedback_data_optional_fields(self):
        """Test FeedbackData with optional fields."""
        feedback = FeedbackData(
            question="Is this a test?",
            ml_prediction="CEQ",
            human_label="CEQ",
            confidence=0.75,
            features={},
            context={},
            user_id="anonymous",
            session_id="test"
        )

        self.assertIsNone(feedback.timestamp)
        self.assertIsNone(feedback.model_version)


class TestTrainingBatch(unittest.TestCase):
    """Test suite for TrainingBatch dataclass."""

    def test_training_batch_creation(self):
        """Test TrainingBatch creation."""
        batch = TrainingBatch(
            batch_id="batch_123",
            sample_count=50,
            created_at="2025-01-27T10:30:00",
            data=[
                {"question": "Q1", "label": "OEQ"},
                {"question": "Q2", "label": "CEQ"}
            ],
            validation_metrics={
                "total_samples": 50,
                "oeq_count": 25,
                "ceq_count": 25
            }
        )

        self.assertEqual(batch.batch_id, "batch_123")
        self.assertEqual(batch.sample_count, 50)
        self.assertEqual(len(batch.data), 2)
        self.assertEqual(batch.validation_metrics["oeq_count"], 25)


def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == '__main__':
    unittest.main()