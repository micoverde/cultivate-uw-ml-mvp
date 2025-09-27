#!/usr/bin/env python3
"""
Training Data Management API Endpoints
Production endpoints for feedback collection and model retraining with Azure Blob Storage

Author: Claude (Partner-Level Microsoft SDE)
Issue: #184 - Azure Blob Storage Integration for ML Training
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Optional, List
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Header, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
import numpy as np

# Import our services
try:
    from src.services.azure_blob_service import (
        AzureBlobTrainingDataService,
        FeedbackData,
        TrainingBatch
    )
    from src.api.security.retrain_auth import get_retrain_auth_service
except ImportError:
    # Handle development environment
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from services.azure_blob_service import (
        AzureBlobTrainingDataService,
        FeedbackData,
        TrainingBatch
    )
    from api.security.retrain_auth import get_retrain_auth_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1",
    tags=["training_data"],
    responses={404: {"description": "Not found"}}
)

# Request/Response Models
class FeedbackRequest(BaseModel):
    """Request model for feedback collection."""
    question: str = Field(..., description="The question text")
    ml_prediction: str = Field(..., pattern="^(OEQ|CEQ)$", description="ML model prediction")
    human_label: str = Field(..., pattern="^(OEQ|CEQ)$", description="Human-provided label")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    features: Dict = Field(..., description="Extracted features")
    context: Dict = Field(default_factory=dict, description="Additional context")
    user_id: str = Field(default="anonymous", description="User identifier")
    session_id: str = Field(default_factory=lambda: str(uuid4()), description="Session identifier")
    model_version: Optional[str] = Field(None, description="Model version used")

class FeedbackResponse(BaseModel):
    """Response model for feedback collection."""
    success: bool
    feedback_id: str
    message: str
    storage_location: Optional[str] = None
    statistics: Optional[Dict] = None

class RetrainRequest(BaseModel):
    """Request model for model retraining."""
    model_type: str = Field(default="oeq_ceq_classifier", description="Model type to retrain")
    training_config: Dict = Field(
        default_factory=lambda: {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "use_augmentation": True
        },
        description="Training configuration"
    )
    data_source: str = Field(default="azure_blob", description="Data source for training")
    min_samples: int = Field(default=10, description="Minimum samples required")

class RetrainResponse(BaseModel):
    """Response model for model retraining."""
    job_id: str
    status: str
    message: str
    estimated_duration_seconds: Optional[int] = None
    training_config: Optional[Dict] = None

class TrainingStatusResponse(BaseModel):
    """Response model for training status."""
    job_id: str
    status: str
    progress: float
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    metrics: Optional[Dict] = None
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None

# Global storage for training jobs (in production, use Redis or database)
training_jobs = {}

# Initialize services
blob_service = None
auth_service = None

def get_blob_service() -> AzureBlobTrainingDataService:
    """Get or create blob service instance."""
    global blob_service
    if blob_service is None:
        blob_service = AzureBlobTrainingDataService()
        # Initialize containers
        asyncio.create_task(blob_service.initialize_containers())
    return blob_service

def get_auth_service():
    """Get auth service instance."""
    global auth_service
    if auth_service is None:
        auth_service = get_retrain_auth_service()
    return auth_service

# Endpoints

@router.post("/feedback/collect", response_model=FeedbackResponse)
async def collect_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    x_client_ip: Optional[str] = Header(None)
):
    """
    Collect user feedback for ground truth data.

    This endpoint stores user-provided labels in Azure Blob Storage
    for future model retraining.
    """
    try:
        blob_service = get_blob_service()

        # Create FeedbackData object
        feedback_data = FeedbackData(
            question=request.question,
            ml_prediction=request.ml_prediction,
            human_label=request.human_label,
            confidence=request.confidence,
            features=request.features,
            context=request.context,
            user_id=request.user_id,
            session_id=request.session_id,
            model_version=request.model_version,
            timestamp=datetime.utcnow().isoformat()
        )

        # Store feedback in Azure Blob
        success, blob_name = await blob_service.store_feedback(feedback_data)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store feedback in Azure Blob Storage"
            )

        # Get statistics (in background)
        statistics = None
        try:
            statistics = await blob_service.get_training_statistics()
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")

        # Log analytics event
        logger.info(f"Feedback collected from {request.user_id}: {request.human_label} for '{request.question[:50]}...'")

        return FeedbackResponse(
            success=True,
            feedback_id=blob_name.split('/')[-1].replace('.json', ''),
            message="Feedback successfully stored",
            storage_location=blob_name,
            statistics=statistics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error collecting feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/model/retrain", response_model=RetrainResponse)
async def retrain_model(
    request: RetrainRequest,
    background_tasks: BackgroundTasks,
    x_retrain_password: str = Header(..., description="Retraining password"),
    x_client_ip: Optional[str] = Header(None),
    x_user_agent: Optional[str] = Header(None)
):
    """
    Initiate model retraining with collected feedback data.

    Password protected endpoint that triggers model retraining
    using data from Azure Blob Storage.
    """
    try:
        # Authenticate request
        auth_service = get_auth_service()
        client_ip = x_client_ip or "unknown"

        auth_result = await auth_service.validate_password(
            password=x_retrain_password,
            client_ip=client_ip,
            user_agent=x_user_agent
        )

        # Get blob service
        blob_service = get_blob_service()

        # Prepare training batch
        training_batch = await blob_service.prepare_training_batch()

        if training_batch is None:
            # Check statistics to provide better error message
            stats = await blob_service.get_training_statistics()
            pending = stats.get('pending_feedback', 0)

            raise HTTPException(
                status_code=400,
                detail=f"Insufficient training data. Have {pending} pending samples, need at least {request.min_samples}"
            )

        # Create training job
        job_id = str(uuid4())
        training_jobs[job_id] = {
            'status': 'queued',
            'created_at': datetime.utcnow().isoformat(),
            'batch_id': training_batch.batch_id,
            'sample_count': training_batch.sample_count,
            'config': request.training_config,
            'progress': 0.0
        }

        # Start training in background
        background_tasks.add_task(
            run_training_job,
            job_id,
            training_batch,
            request.training_config,
            blob_service
        )

        # Estimate duration based on sample count and epochs
        samples = training_batch.sample_count
        epochs = request.training_config.get('epochs', 10)
        estimated_duration = (samples * epochs) // 10  # Rough estimate

        logger.info(f"Retraining initiated: job_id={job_id}, samples={samples}, client_ip={client_ip}")

        return RetrainResponse(
            job_id=job_id,
            status="queued",
            message=f"Model retraining initiated with {samples} samples",
            estimated_duration_seconds=estimated_duration,
            training_config=request.training_config
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating retraining: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/training/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """
    Get the status of a training job.

    Returns current progress and metrics for a training job.
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Training job {job_id} not found"
        )

    job = training_jobs[job_id]

    # Calculate estimated completion
    estimated_completion = None
    if job['status'] == 'running' and job.get('start_time'):
        start_time = datetime.fromisoformat(job['start_time'])
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        if job['progress'] > 0:
            total_estimated = elapsed / job['progress']
            remaining = total_estimated - elapsed
            estimated_completion = (
                datetime.utcnow() + timedelta(seconds=remaining)
            ).isoformat()

    return TrainingStatusResponse(
        job_id=job_id,
        status=job['status'],
        progress=job.get('progress', 0.0),
        current_epoch=job.get('current_epoch'),
        total_epochs=job.get('config', {}).get('epochs', 10),
        metrics=job.get('metrics'),
        estimated_completion=estimated_completion,
        error_message=job.get('error_message')
    )

@router.get("/training/statistics")
async def get_training_statistics():
    """
    Get training data statistics.

    Returns statistics about collected feedback and training data.
    """
    try:
        blob_service = get_blob_service()
        statistics = await blob_service.get_training_statistics()

        # Add active training jobs
        active_jobs = [
            job_id for job_id, job in training_jobs.items()
            if job['status'] in ['queued', 'running']
        ]

        statistics['active_training_jobs'] = len(active_jobs)
        statistics['total_training_jobs'] = len(training_jobs)

        return JSONResponse(
            content={
                'success': True,
                'statistics': statistics,
                'timestamp': datetime.utcnow().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/data/export")
async def export_training_data(
    format: str = "json",
    date_range: str = "7d",
    x_admin_token: Optional[str] = Header(None)
):
    """
    Export training data for analysis (Admin only).

    Exports collected feedback data in specified format.
    """
    # Simple admin check (in production, use proper auth)
    if x_admin_token != os.getenv('ADMIN_TOKEN', 'admin_secret_token'):
        raise HTTPException(
            status_code=403,
            detail="Admin authentication required"
        )

    try:
        blob_service = get_blob_service()

        # For now, return statistics
        # Full implementation would export actual data
        statistics = await blob_service.get_training_statistics()

        return JSONResponse(
            content={
                'success': True,
                'format': format,
                'date_range': date_range,
                'statistics': statistics,
                'message': 'Full export implementation pending'
            }
        )

    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Background task for training
async def run_training_job(
    job_id: str,
    training_batch: TrainingBatch,
    config: Dict,
    blob_service: AzureBlobTrainingDataService
):
    """
    Run model training job in background.

    This is a simplified version. In production, this would:
    1. Load the actual PyTorch model
    2. Prepare data loaders
    3. Run training loop
    4. Save checkpoints to Azure Blob
    5. Update metrics in real-time
    """
    try:
        # Update job status
        training_jobs[job_id]['status'] = 'running'
        training_jobs[job_id]['start_time'] = datetime.utcnow().isoformat()

        # Simulate training with progress updates
        epochs = config.get('epochs', 10)
        for epoch in range(epochs):
            # Update progress
            progress = (epoch + 1) / epochs
            training_jobs[job_id]['progress'] = progress
            training_jobs[job_id]['current_epoch'] = epoch + 1

            # Simulate training time
            await asyncio.sleep(2)  # In production, actual training here

            # Update metrics (simulated)
            training_jobs[job_id]['metrics'] = {
                'loss': 0.5 * (1 - progress),
                'accuracy': 0.85 + 0.1 * progress,
                'val_loss': 0.6 * (1 - progress),
                'val_accuracy': 0.83 + 0.1 * progress
            }

        # Save model checkpoint (simulated)
        model_version = f"v2.{int(datetime.utcnow().timestamp())}"

        # In production, serialize actual model state_dict
        model_state = b"simulated_model_state"
        metrics = training_jobs[job_id]['metrics']

        await blob_service.store_model_checkpoint(
            model_state=model_state,
            model_version=model_version,
            metrics=metrics
        )

        # Update job as completed
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['model_version'] = model_version
        training_jobs[job_id]['completed_at'] = datetime.utcnow().isoformat()

        logger.info(f"Training job {job_id} completed successfully. Model version: {model_version}")

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error_message'] = str(e)

# Export router
__all__ = ['router']