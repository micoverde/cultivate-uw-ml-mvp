#!/usr/bin/env python3
"""
Unit tests for Training Data Management API Endpoints
Tests feedback collection, model retraining, and status endpoints

Author: Claude (Partner-Level Microsoft SDE)
Issue: #184 - Azure Blob Storage Integration
"""

import os
import sys
import json
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException
from fastapi.testclient import TestClient

# Import models and endpoints
from src.api.endpoints.training_data_management import (
    FeedbackRequest,
    FeedbackResponse,
    RetrainRequest,
    RetrainResponse,
    TrainingStatusResponse,
    collect_feedback,
    retrain_model,
    get_training_status,
    get_training_statistics,
    export_training_data,
    training_jobs,
    run_training_job
)


class TestFeedbackEndpoints(unittest.TestCase):
    """Test suite for feedback collection endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_feedback = FeedbackRequest(
            question="Why do birds fly?",
            ml_prediction="OEQ",
            human_label="CEQ",
            confidence=0.88,
            features={"length": 17, "has_why": True},
            context={"source": "test"},
            user_id="test_user",
            session_id="test_session"
        )

    @patch('src.api.endpoints.training_data_management.get_blob_service')
    async def test_collect_feedback_success(self, mock_get_blob_service):
        """Test successful feedback collection."""
        mock_blob_service = AsyncMock()
        mock_blob_service.store_feedback.return_value = (True, "raw-feedback/2025-01-27/feedback_test.json")
        mock_blob_service.get_training_statistics.return_value = {
            'total_feedback': 100,
            'pending_feedback': 10
        }
        mock_get_blob_service.return_value = mock_blob_service

        mock_background_tasks = MagicMock()

        response = await collect_feedback(
            request=self.sample_feedback,
            background_tasks=mock_background_tasks,
            x_client_ip="192.168.1.1"
        )

        self.assertIsInstance(response, FeedbackResponse)
        self.assertTrue(response.success)
        self.assertEqual(response.message, "Feedback successfully stored")
        self.assertIn("feedback_test", response.feedback_id)

    @patch('src.api.endpoints.training_data_management.get_blob_service')
    async def test_collect_feedback_storage_failure(self, mock_get_blob_service):
        """Test feedback collection when storage fails."""
        mock_blob_service = AsyncMock()
        mock_blob_service.store_feedback.return_value = (False, "")
        mock_get_blob_service.return_value = mock_blob_service

        mock_background_tasks = MagicMock()

        with self.assertRaises(HTTPException) as context:
            await collect_feedback(
                request=self.sample_feedback,
                background_tasks=mock_background_tasks
            )

        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("Failed to store feedback", context.exception.detail)


class TestRetrainEndpoints(unittest.TestCase):
    """Test suite for model retraining endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.retrain_request = RetrainRequest(
            model_type="oeq_ceq_classifier",
            training_config={
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        )
        # Clear training jobs
        training_jobs.clear()

    @patch('src.api.endpoints.training_data_management.get_auth_service')
    @patch('src.api.endpoints.training_data_management.get_blob_service')
    async def test_retrain_model_success(self, mock_get_blob_service, mock_get_auth_service):
        """Test successful model retraining initiation."""
        # Mock auth service
        mock_auth = AsyncMock()
        mock_auth.validate_password.return_value = {
            'authenticated': True,
            'session_token': 'test_token'
        }
        mock_get_auth_service.return_value = mock_auth

        # Mock blob service
        mock_blob_service = AsyncMock()
        mock_batch = MagicMock()
        mock_batch.batch_id = "batch_123"
        mock_batch.sample_count = 50
        mock_blob_service.prepare_training_batch.return_value = mock_batch
        mock_get_blob_service.return_value = mock_blob_service

        mock_background_tasks = MagicMock()

        response = await retrain_model(
            request=self.retrain_request,
            background_tasks=mock_background_tasks,
            x_retrain_password="lev",
            x_client_ip="192.168.1.1"
        )

        self.assertIsInstance(response, RetrainResponse)
        self.assertEqual(response.status, "queued")
        self.assertIn("50 samples", response.message)
        self.assertEqual(len(training_jobs), 1)

    @patch('src.api.endpoints.training_data_management.get_auth_service')
    async def test_retrain_model_auth_failure(self, mock_get_auth_service):
        """Test retraining with authentication failure."""
        mock_auth = AsyncMock()
        mock_auth.validate_password.side_effect = HTTPException(
            status_code=401,
            detail="Invalid password"
        )
        mock_get_auth_service.return_value = mock_auth

        mock_background_tasks = MagicMock()

        with self.assertRaises(HTTPException) as context:
            await retrain_model(
                request=self.retrain_request,
                background_tasks=mock_background_tasks,
                x_retrain_password="wrong_password",
                x_client_ip="192.168.1.1"
            )

        self.assertEqual(context.exception.status_code, 401)

    @patch('src.api.endpoints.training_data_management.get_auth_service')
    @patch('src.api.endpoints.training_data_management.get_blob_service')
    async def test_retrain_model_insufficient_data(self, mock_get_blob_service, mock_get_auth_service):
        """Test retraining with insufficient training data."""
        # Mock auth service
        mock_auth = AsyncMock()
        mock_auth.validate_password.return_value = {
            'authenticated': True,
            'session_token': 'test_token'
        }
        mock_get_auth_service.return_value = mock_auth

        # Mock blob service with no training batch
        mock_blob_service = AsyncMock()
        mock_blob_service.prepare_training_batch.return_value = None
        mock_blob_service.get_training_statistics.return_value = {
            'pending_feedback': 5
        }
        mock_get_blob_service.return_value = mock_blob_service

        mock_background_tasks = MagicMock()

        with self.assertRaises(HTTPException) as context:
            await retrain_model(
                request=self.retrain_request,
                background_tasks=mock_background_tasks,
                x_retrain_password="lev",
                x_client_ip="192.168.1.1"
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("Insufficient training data", context.exception.detail)


class TestTrainingStatusEndpoints(unittest.TestCase):
    """Test suite for training status endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear and populate training jobs
        training_jobs.clear()
        training_jobs["test_job_123"] = {
            'status': 'running',
            'progress': 0.5,
            'current_epoch': 5,
            'config': {'epochs': 10},
            'start_time': datetime.utcnow().isoformat(),
            'metrics': {
                'accuracy': 0.90,
                'loss': 0.25
            }
        }

    async def test_get_training_status_existing_job(self):
        """Test getting status of existing training job."""
        response = await get_training_status("test_job_123")

        self.assertIsInstance(response, TrainingStatusResponse)
        self.assertEqual(response.job_id, "test_job_123")
        self.assertEqual(response.status, "running")
        self.assertEqual(response.progress, 0.5)
        self.assertEqual(response.current_epoch, 5)

    async def test_get_training_status_nonexistent_job(self):
        """Test getting status of non-existent job."""
        with self.assertRaises(HTTPException) as context:
            await get_training_status("nonexistent_job")

        self.assertEqual(context.exception.status_code, 404)
        self.assertIn("not found", context.exception.detail)


class TestStatisticsEndpoints(unittest.TestCase):
    """Test suite for statistics endpoints."""

    @patch('src.api.endpoints.training_data_management.get_blob_service')
    async def test_get_training_statistics(self, mock_get_blob_service):
        """Test getting training statistics."""
        mock_blob_service = AsyncMock()
        mock_blob_service.get_training_statistics.return_value = {
            'total_feedback': 150,
            'processed_feedback': 120,
            'pending_feedback': 30,
            'oeq_count': 75,
            'ceq_count': 75
        }
        mock_get_blob_service.return_value = mock_blob_service

        # Add some training jobs
        training_jobs['job1'] = {'status': 'running'}
        training_jobs['job2'] = {'status': 'completed'}

        response = await get_training_statistics()
        content = json.loads(response.body)

        self.assertTrue(content['success'])
        self.assertEqual(content['statistics']['total_feedback'], 150)
        self.assertEqual(content['statistics']['active_training_jobs'], 1)
        self.assertEqual(content['statistics']['total_training_jobs'], 2)


class TestExportEndpoints(unittest.TestCase):
    """Test suite for data export endpoints."""

    @patch('src.api.endpoints.training_data_management.get_blob_service')
    async def test_export_training_data_authorized(self, mock_get_blob_service):
        """Test data export with admin authorization."""
        mock_blob_service = AsyncMock()
        mock_blob_service.get_training_statistics.return_value = {
            'total_feedback': 100
        }
        mock_get_blob_service.return_value = mock_blob_service

        os.environ['ADMIN_TOKEN'] = 'admin_secret_token'

        response = await export_training_data(
            format="json",
            date_range="7d",
            x_admin_token="admin_secret_token"
        )
        content = json.loads(response.body)

        self.assertTrue(content['success'])
        self.assertEqual(content['format'], "json")

    async def test_export_training_data_unauthorized(self):
        """Test data export without proper authorization."""
        os.environ['ADMIN_TOKEN'] = 'admin_secret_token'

        with self.assertRaises(HTTPException) as context:
            await export_training_data(
                format="json",
                date_range="7d",
                x_admin_token="wrong_token"
            )

        self.assertEqual(context.exception.status_code, 403)
        self.assertIn("Admin authentication required", context.exception.detail)


class TestTrainingJob(unittest.TestCase):
    """Test suite for background training job."""

    @patch('src.api.endpoints.training_data_management.asyncio.sleep')
    async def test_run_training_job_success(self, mock_sleep):
        """Test successful training job execution."""
        mock_sleep.return_value = None  # Make it instant

        mock_batch = MagicMock()
        mock_batch.batch_id = "batch_123"
        mock_batch.sample_count = 50

        mock_blob_service = AsyncMock()
        mock_blob_service.store_model_checkpoint.return_value = True

        training_jobs.clear()

        await run_training_job(
            job_id="test_job",
            training_batch=mock_batch,
            config={'epochs': 2},
            blob_service=mock_blob_service
        )

        self.assertEqual(training_jobs["test_job"]["status"], "completed")
        self.assertIn("model_version", training_jobs["test_job"])
        self.assertEqual(mock_sleep.call_count, 2)  # Called once per epoch

    @patch('src.api.endpoints.training_data_management.asyncio.sleep')
    async def test_run_training_job_failure(self, mock_sleep):
        """Test training job failure handling."""
        mock_sleep.side_effect = Exception("Training failed")

        mock_batch = MagicMock()
        mock_blob_service = AsyncMock()

        training_jobs.clear()

        await run_training_job(
            job_id="failing_job",
            training_batch=mock_batch,
            config={'epochs': 10},
            blob_service=mock_blob_service
        )

        self.assertEqual(training_jobs["failing_job"]["status"], "failed")
        self.assertIn("error_message", training_jobs["failing_job"])


class TestRequestModels(unittest.TestCase):
    """Test suite for request/response models."""

    def test_feedback_request_validation(self):
        """Test FeedbackRequest model validation."""
        valid_request = FeedbackRequest(
            question="Test question?",
            ml_prediction="OEQ",
            human_label="CEQ",
            confidence=0.5,
            features={},
            context={}
        )

        self.assertEqual(valid_request.question, "Test question?")
        self.assertEqual(valid_request.user_id, "anonymous")  # Default value

    def test_feedback_request_invalid_label(self):
        """Test FeedbackRequest with invalid label."""
        with self.assertRaises(ValueError):
            FeedbackRequest(
                question="Test?",
                ml_prediction="INVALID",  # Should fail pattern validation
                human_label="CEQ",
                confidence=0.5,
                features={},
                context={}
            )

    def test_retrain_request_defaults(self):
        """Test RetrainRequest default values."""
        request = RetrainRequest()

        self.assertEqual(request.model_type, "oeq_ceq_classifier")
        self.assertEqual(request.data_source, "azure_blob")
        self.assertEqual(request.min_samples, 10)
        self.assertEqual(request.training_config["epochs"], 10)

    def test_training_status_response(self):
        """Test TrainingStatusResponse model."""
        response = TrainingStatusResponse(
            job_id="job_123",
            status="running",
            progress=0.75,
            current_epoch=7,
            total_epochs=10,
            metrics={"accuracy": 0.92}
        )

        self.assertEqual(response.job_id, "job_123")
        self.assertEqual(response.progress, 0.75)
        self.assertIsNone(response.error_message)


def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == '__main__':
    unittest.main()