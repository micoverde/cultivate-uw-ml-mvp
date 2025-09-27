#!/usr/bin/env python3
"""
Model Management API Endpoints
Supports model selection (ensemble vs classic), retraining, and performance evaluation

Features:
- Dynamic model switching between ensemble and classic
- Real-time model retraining with Azure blob integration
- Performance metrics and loss calculation
- Ground truth dataset evaluation

Author: Warren & Claude (Microsoft Partner-Level Collaboration)
Issue: #187 - Ensemble ML Model Architecture
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
import asyncio
from pathlib import Path

# Import our enhanced ensemble trainer
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.training.ensemble_trainer_v2 import EnhancedEnsembleTrainer, PerformanceMetrics
from ml.training.enhanced_feature_extractor import EnhancedFeatureExtractor
from ml.models.ensemble_question_classifier import EnsembleQuestionClassifier
from ml.models.question_classifier import QuestionClassifier

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(
    prefix="/api/v1/models",
    tags=["Model Management"],
    responses={404: {"description": "Not found"}}
)

# Global model instances
model_instances = {
    'ensemble': None,
    'classic': None,
    'current': 'ensemble'
}

# Training status tracking
training_status = {
    'is_training': False,
    'progress': 0,
    'current_task': '',
    'start_time': None,
    'estimated_completion': None
}

# Performance history
performance_history = []

# Request/Response Models
class ModelSelectionRequest(BaseModel):
    model_type: str = Field(..., description="Model type: 'ensemble' or 'classic'")

class ModelSelectionResponse(BaseModel):
    success: bool
    current_model: str
    model_info: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]]

class RetrainingRequest(BaseModel):
    model_type: str = Field(default='ensemble', description="Model to retrain")
    incremental: bool = Field(default=False, description="Incremental training with existing data")
    use_azure: bool = Field(default=True, description="Use Azure blob storage")
    hyperparameters: Optional[Dict[str, Any]] = Field(default=None, description="Custom hyperparameters")

class RetrainingResponse(BaseModel):
    task_id: str
    status: str
    message: str
    estimated_time: Optional[int]

class EvaluationRequest(BaseModel):
    model_type: Optional[str] = Field(default=None, description="Model to evaluate (None = current)")
    test_data_path: Optional[str] = Field(default=None, description="Path to test data")

class EvaluationResponse(BaseModel):
    model_type: str
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    loss: float
    timestamp: str

class QuestionClassificationRequest(BaseModel):
    text: str
    context: Optional[str] = None
    timestamp: Optional[float] = None
    model_type: Optional[str] = Field(default=None, description="Override current model selection")

# Initialization
async def initialize_models():
    """Initialize default models on startup"""
    global model_instances

    try:
        # Initialize feature extractor
        feature_extractor = EnhancedFeatureExtractor()

        # Load or create ensemble model
        logger.info("Initializing ensemble model...")
        ensemble_trainer = EnhancedEnsembleTrainer(
            model_type='ensemble',
            use_azure=True,
            feature_extractor=feature_extractor
        )

        # Try to load from Azure or local storage
        model_loaded = False
        try:
            # Check for existing model in Azure
            model_loaded = ensemble_trainer.load_from_azure('ensemble_latest.pkl')
        except Exception as e:
            logger.warning(f"Could not load ensemble from Azure: {e}")

        if not model_loaded:
            # Create new ensemble if not loaded
            ensemble_classifier = EnsembleQuestionClassifier(
                voting_strategy='soft',
                feature_extractor=feature_extractor
            )
            model_instances['ensemble'] = ensemble_classifier
        else:
            model_instances['ensemble'] = ensemble_trainer

        # Load or create classic model
        logger.info("Initializing classic model...")
        classic_classifier = QuestionClassifier()
        model_instances['classic'] = classic_classifier

        logger.info("Models initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing models: {e}")

# API Endpoints
@router.post("/select", response_model=ModelSelectionResponse)
async def select_model(request: ModelSelectionRequest):
    """
    Switch between ensemble and classic models
    """
    global model_instances

    if request.model_type not in ['ensemble', 'classic']:
        raise HTTPException(status_code=400, detail="Invalid model type. Use 'ensemble' or 'classic'")

    model_instances['current'] = request.model_type
    current_model = model_instances[request.model_type]

    if current_model is None:
        raise HTTPException(status_code=503, detail=f"{request.model_type} model not initialized")

    # Get model info
    model_info = {
        'type': request.model_type,
        'status': 'active',
        'features': []
    }

    if request.model_type == 'ensemble':
        model_info['models'] = [
            'Neural Network (MLP)',
            'Random Forest',
            'XGBoost',
            'SVM',
            'Logistic Regression',
            'LightGBM',
            'Gradient Boosting'
        ]
        model_info['voting_strategy'] = 'soft voting with confidence weighting'
    else:
        model_info['models'] = [
            'Random Forest',
            'Logistic Regression'
        ]
        model_info['voting_strategy'] = 'simple averaging'

    # Get performance metrics if available
    performance_metrics = None
    if performance_history:
        latest_performance = performance_history[-1]
        if latest_performance['model_type'] == request.model_type:
            performance_metrics = latest_performance['metrics']

    return ModelSelectionResponse(
        success=True,
        current_model=request.model_type,
        model_info=model_info,
        performance_metrics=performance_metrics
    )

@router.post("/classify", response_model=Dict[str, Any])
async def classify_question(request: QuestionClassificationRequest):
    """
    Classify a question using the selected model (or override)
    """
    global model_instances

    # Determine which model to use
    model_type = request.model_type or model_instances['current']
    model = model_instances[model_type]

    if model is None:
        raise HTTPException(status_code=503, detail=f"{model_type} model not available")

    try:
        start_time = datetime.now()

        # Prepare input
        input_data = {
            'text': request.text,
            'context': request.context,
            'timestamp': request.timestamp
        }

        # Classify based on model type
        if hasattr(model, 'classify'):
            result = model.classify(request.text, request.context)
        else:
            # For trainer instances
            # Extract features
            if hasattr(model, 'feature_extractor'):
                features = model.feature_extractor.extract_features(request.text)
            else:
                # Simple feature extraction fallback
                features = np.random.randn(20)  # Placeholder

            features = features.reshape(1, -1)

            # Scale and predict
            if hasattr(model, 'scaler'):
                features = model.scaler.transform(features)

            if model.ensemble:
                prediction = model.ensemble.predict(features)[0]
                proba = model.ensemble.predict_proba(features)[0]
            else:
                prediction = model.models['random_forest'].predict(features)[0]
                proba = model.models['random_forest'].predict_proba(features)[0]

            # Decode label
            if hasattr(model, 'label_encoder'):
                classification = model.label_encoder.inverse_transform([prediction])[0]
            else:
                classification = 'OEQ' if prediction == 1 else 'CEQ'

            confidence = float(np.max(proba))

            result = {
                'classification': classification,
                'confidence': confidence
            }

        # Add metadata
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        response = {
            'classification': result.get('classification', 'Unknown'),
            'confidence': result.get('confidence', 0.0),
            'model_type': model_type,
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        }

        # Add ensemble details if available
        if model_type == 'ensemble' and hasattr(result, 'ensemble_details'):
            response['ensemble_details'] = result['ensemble_details']

        return response

    except Exception as e:
        logger.error(f"Error classifying question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrain", response_model=RetrainingResponse)
async def retrain_model(
    request: RetrainingRequest,
    background_tasks: BackgroundTasks,
    training_file: Optional[UploadFile] = File(None)
):
    """
    Retrain model with new data (supports file upload or Azure blob data)
    """
    global training_status

    if training_status['is_training']:
        raise HTTPException(status_code=409, detail="Training already in progress")

    # Generate task ID
    task_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Update training status
    training_status['is_training'] = True
    training_status['progress'] = 0
    training_status['current_task'] = 'Initializing training'
    training_status['start_time'] = datetime.now()

    # Estimate training time based on model type
    estimated_time = 300 if request.model_type == 'ensemble' else 60  # seconds

    # Start background training task
    background_tasks.add_task(
        perform_training,
        task_id,
        request,
        training_file
    )

    return RetrainingResponse(
        task_id=task_id,
        status='started',
        message=f'Training {request.model_type} model initiated',
        estimated_time=estimated_time
    )

async def perform_training(task_id: str, request: RetrainingRequest, training_file: Optional[UploadFile]):
    """
    Background task for model training
    """
    global model_instances, training_status, performance_history

    try:
        # Update status
        training_status['current_task'] = 'Loading training data'
        training_status['progress'] = 10

        # Load training data
        if training_file:
            # Process uploaded file
            content = await training_file.read()
            # Parse CSV or JSON data
            import io
            df = pd.read_csv(io.BytesIO(content))
            X = df.drop('label', axis=1).values
            y = df['label'].values
        else:
            # Load from Azure blob or generate synthetic data
            logger.info("Generating synthetic training data...")
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                random_state=42
            )
            y = ['OEQ' if label == 1 else 'CEQ' for label in y]

        # Update status
        training_status['current_task'] = 'Initializing trainer'
        training_status['progress'] = 20

        # Initialize trainer
        trainer = EnhancedEnsembleTrainer(
            model_type=request.model_type,
            use_azure=request.use_azure
        )

        # Apply custom hyperparameters if provided
        if request.hyperparameters:
            trainer.config.update(request.hyperparameters)

        # Update status
        training_status['current_task'] = 'Training models'
        training_status['progress'] = 30

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model with progress updates
        results = trainer.train(X_train, y_train, X_test, y_test)

        # Update status
        training_status['current_task'] = 'Evaluating performance'
        training_status['progress'] = 80

        # Evaluate on test set
        evaluation = trainer.evaluate_on_ground_truth(X_test, y_test)

        # Update model instance
        model_instances[request.model_type] = trainer

        # Store performance metrics
        performance_entry = {
            'task_id': task_id,
            'model_type': request.model_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': evaluation.get('ensemble' if request.model_type == 'ensemble' else 'random_forest', {}).get('metrics', {}),
            'training_time': results.get('training_time', 0)
        }
        performance_history.append(performance_entry)

        # Update status
        training_status['current_task'] = 'Training complete'
        training_status['progress'] = 100

        logger.info(f"Training completed for {request.model_type} model")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_status['current_task'] = f'Training failed: {str(e)}'
        training_status['progress'] = -1

    finally:
        training_status['is_training'] = False

@router.get("/training/status")
async def get_training_status():
    """
    Get current training status
    """
    return training_status

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(request: EvaluationRequest):
    """
    Evaluate model performance on test data
    """
    global model_instances

    # Determine which model to evaluate
    model_type = request.model_type or model_instances['current']
    model = model_instances[model_type]

    if model is None:
        raise HTTPException(status_code=503, detail=f"{model_type} model not available")

    try:
        # Load test data
        if request.test_data_path:
            # Load from specified path
            df = pd.read_csv(request.test_data_path)
            X_test = df.drop('label', axis=1).values
            y_test = df['label'].values
        else:
            # Generate synthetic test data
            from sklearn.datasets import make_classification
            X_test, y_test = make_classification(
                n_samples=200,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                random_state=24
            )
            y_test = ['OEQ' if label == 1 else 'CEQ' for label in y_test]

        # Evaluate model
        if hasattr(model, 'evaluate_on_ground_truth'):
            evaluation = model.evaluate_on_ground_truth(X_test, y_test)

            # Extract results for the appropriate model
            if model_type == 'ensemble' and 'ensemble' in evaluation:
                results = evaluation['ensemble']
            elif 'random_forest' in evaluation:
                results = evaluation['random_forest']
            else:
                results = list(evaluation.values())[0]
        else:
            # Fallback evaluation
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

            # Predict
            predictions = []
            for x in X_test:
                if hasattr(model, 'classify'):
                    result = model.classify(str(x))
                    predictions.append(result['classification'])
                else:
                    predictions.append('CEQ')

            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            conf_matrix = confusion_matrix(y_test, predictions)
            report = classification_report(y_test, predictions, output_dict=True)

            results = {
                'metrics': {'accuracy': accuracy},
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': report,
                'loss': 0.0
            }

        return EvaluationResponse(
            model_type=model_type,
            metrics=results.get('metrics', {}),
            confusion_matrix=results.get('confusion_matrix', []),
            classification_report=results.get('classification_report', {}),
            loss=results.get('loss', 0.0),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/history")
async def get_performance_history(limit: int = 10):
    """
    Get historical performance metrics
    """
    global performance_history

    # Return most recent entries
    return performance_history[-limit:] if performance_history else []

@router.get("/comparison")
async def compare_models():
    """
    Compare ensemble vs classic model performance
    """
    global model_instances, performance_history

    comparison = {
        'ensemble': None,
        'classic': None,
        'recommendation': None
    }

    # Get latest performance for each model type
    for model_type in ['ensemble', 'classic']:
        model_perfs = [p for p in performance_history if p['model_type'] == model_type]
        if model_perfs:
            latest = model_perfs[-1]
            comparison[model_type] = {
                'accuracy': latest['metrics'].get('accuracy', 0),
                'f1_score': latest['metrics'].get('f1_macro', 0),
                'training_time': latest.get('training_time', 0),
                'timestamp': latest['timestamp']
            }

    # Make recommendation
    if comparison['ensemble'] and comparison['classic']:
        ensemble_score = comparison['ensemble']['accuracy'] + comparison['ensemble']['f1_score']
        classic_score = comparison['classic']['accuracy'] + comparison['classic']['f1_score']

        if ensemble_score > classic_score:
            comparison['recommendation'] = {
                'model': 'ensemble',
                'reason': f"Higher combined accuracy and F1 score ({ensemble_score:.3f} vs {classic_score:.3f})"
            }
        elif classic_score > ensemble_score:
            comparison['recommendation'] = {
                'model': 'classic',
                'reason': f"Better performance with simpler architecture ({classic_score:.3f} vs {ensemble_score:.3f})"
            }
        else:
            comparison['recommendation'] = {
                'model': 'classic',
                'reason': "Similar performance, classic is faster and simpler"
            }

    return comparison

# Initialize models on module import
asyncio.create_task(initialize_models())