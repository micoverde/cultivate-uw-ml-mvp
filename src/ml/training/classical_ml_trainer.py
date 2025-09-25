#!/usr/bin/env python3
"""
Classical ML Training Orchestrator for Educational Domain
Implements Issue #109 design specification for production-ready training pipeline.

Orchestrates complete training workflow from CSV annotations to deployed models,
optimized for Azure Container Apps deployment (Issue #79).

Author: Claude-4 (Partner-Level Microsoft SDE)
Issue: #109 - Classical ML Model Training on Expert Annotations
Context: Phase 1 of Issue #76 + Azure Infrastructure Issue #79
"""

import os
import json
import joblib
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

from feature_extractor import ExpertAnnotationFeatureExtractor
from model_factory import ClassicalMLModelFactory

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Structured training metrics for comprehensive evaluation."""
    model_name: str
    task_type: str  # classification or regression
    cv_scores: List[float]
    mean_cv_score: float
    std_cv_score: float
    train_score: float
    test_score: float
    feature_importance: Optional[List[float]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict] = None
    training_time: float = 0.0
    inference_time: float = 0.0

@dataclass
class TrainingReport:
    """Comprehensive training report for all models."""
    experiment_id: str
    timestamp: str
    dataset_info: Dict[str, Any]
    models_trained: List[str]
    overall_metrics: Dict[str, TrainingMetrics]
    hyperparameters: Dict[str, Dict[str, Any]]
    feature_extraction_time: float
    total_training_time: float
    model_sizes: Dict[str, int]
    success: bool
    error_message: Optional[str] = None

class ClassicalMLTrainer:
    """
    Production training pipeline with enterprise monitoring and validation.
    Implements Issue #76 Phase 1 specifications with Azure Container Apps readiness.

    Features:
    - Cross-validation with stratified sampling
    - Comprehensive metrics collection
    - Model persistence with metadata
    - Container-ready file organization
    - Production monitoring integration
    """

    def __init__(self,
                 csv_path: str,
                 output_dir: str = 'src/ml/trained_models',
                 experiment_name: Optional[str] = None):
        """
        Initialize training pipeline.

        Args:
            csv_path: Path to expert annotations CSV
            output_dir: Directory for saved models and reports
            experiment_name: Optional experiment identifier
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate experiment ID
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = experiment_name or f"classical_ml_{timestamp}"

        # Initialize components
        self.feature_extractor = ExpertAnnotationFeatureExtractor()
        self.model_factory = ClassicalMLModelFactory()
        self.training_report = None

        logger.info(f"🚀 Initialized ClassicalMLTrainer for experiment: {self.experiment_id}")

    def train_all_models(self,
                        use_grid_search: bool = False,
                        cv_folds: int = 5) -> TrainingReport:
        """
        Complete training pipeline with cross-validation and persistence.

        Args:
            use_grid_search: Whether to use hyperparameter optimization
            cv_folds: Number of cross-validation folds

        Returns:
            Comprehensive training report with metrics and model paths
        """
        start_time = time.time()
        logger.info(f"🚀 Starting Classical ML training pipeline")

        try:
            # 1. Feature extraction
            logger.info("📊 Step 1/4: Feature extraction")
            feature_start = time.time()
            X, y_multi = self.feature_extractor.extract_features(str(self.csv_path))
            feature_time = time.time() - feature_start

            logger.info(f"✅ Extracted features: {X.shape[0]} samples × {X.shape[1]} features")

            # Dataset information
            dataset_info = {
                'total_samples': X.shape[0],
                'feature_dimensions': X.shape[1],
                'question_type_distribution': np.bincount(y_multi['question_type']).tolist(),
                'wait_time_distribution': np.bincount(y_multi['wait_time']).tolist(),
                'class_score_stats': {
                    'mean': float(np.mean(y_multi['class_scores'])),
                    'std': float(np.std(y_multi['class_scores'])),
                    'min': float(np.min(y_multi['class_scores'])),
                    'max': float(np.max(y_multi['class_scores']))
                }
            }

            # 2. Model training
            logger.info("🤖 Step 2/4: Model training and validation")

            if use_grid_search:
                optimized_models = self.model_factory.create_optimized_models_with_gridsearch(
                    X, y_multi, cv_folds
                )
                models = {name: info['model'] for name, info in optimized_models.items()}
                hyperparameters = {name: info['best_params'] for name, info in optimized_models.items()}
            else:
                models = {
                    'question_classifier': self.model_factory.create_question_classifier(),
                    'wait_time_detector': self.model_factory.create_wait_time_detector(),
                    'class_scorer': self.model_factory.create_class_scorer()
                }
                hyperparameters = self._extract_default_hyperparameters(models)

            # 3. Comprehensive evaluation
            logger.info("📈 Step 3/4: Model evaluation and validation")
            overall_metrics = {}

            # Train and evaluate each model
            overall_metrics['question_classifier'] = self._evaluate_classifier(
                models['question_classifier'], X, y_multi['question_type'],
                'question_classifier', cv_folds
            )

            overall_metrics['wait_time_detector'] = self._evaluate_classifier(
                models['wait_time_detector'], X, y_multi['wait_time'],
                'wait_time_detector', cv_folds
            )

            overall_metrics['class_scorer'] = self._evaluate_regressor(
                models['class_scorer'], X, y_multi['class_scores'],
                'class_scorer', cv_folds
            )

            # 4. Model persistence and metadata
            logger.info("💾 Step 4/4: Model persistence and reporting")
            model_sizes = self._save_models_with_metadata(models, X, y_multi)

            # Generate comprehensive report
            total_time = time.time() - start_time

            self.training_report = TrainingReport(
                experiment_id=self.experiment_id,
                timestamp=datetime.utcnow().isoformat(),
                dataset_info=dataset_info,
                models_trained=list(models.keys()),
                overall_metrics=overall_metrics,
                hyperparameters=hyperparameters,
                feature_extraction_time=feature_time,
                total_training_time=total_time,
                model_sizes=model_sizes,
                success=True
            )

            # Save training report
            self._save_training_report()

            logger.info(f"✅ Classical ML training completed successfully in {total_time:.2f}s")
            self._log_summary_metrics()

            return self.training_report

        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")

            self.training_report = TrainingReport(
                experiment_id=self.experiment_id,
                timestamp=datetime.utcnow().isoformat(),
                dataset_info={},
                models_trained=[],
                overall_metrics={},
                hyperparameters={},
                feature_extraction_time=0.0,
                total_training_time=time.time() - start_time,
                model_sizes={},
                success=False,
                error_message=str(e)
            )

            raise e

    def _evaluate_classifier(self, model, X: np.ndarray, y: np.ndarray,
                           model_name: str, cv_folds: int) -> TrainingMetrics:
        """Comprehensive evaluation for classification models."""
        start_time = time.time()

        # Cross-validation with stratified sampling
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

        # Train on full dataset for final model
        model.fit(X, y)
        train_predictions = model.predict(X)
        train_score = accuracy_score(y, train_predictions)

        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_).flatten().tolist()

        # Classification metrics
        conf_matrix = confusion_matrix(y, train_predictions).tolist()
        class_report = classification_report(y, train_predictions, output_dict=True)

        # Inference timing
        inference_start = time.time()
        _ = model.predict(X[:10])  # Sample inference
        inference_time = (time.time() - inference_start) / 10 * 1000  # ms per sample

        training_time = time.time() - start_time

        return TrainingMetrics(
            model_name=model_name,
            task_type='classification',
            cv_scores=cv_scores.tolist(),
            mean_cv_score=float(cv_scores.mean()),
            std_cv_score=float(cv_scores.std()),
            train_score=float(train_score),
            test_score=float(cv_scores.mean()),  # Use CV as test score
            feature_importance=feature_importance,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            training_time=training_time,
            inference_time=inference_time
        )

    def _evaluate_regressor(self, model, X: np.ndarray, y: np.ndarray,
                          model_name: str, cv_folds: int) -> TrainingMetrics:
        """Comprehensive evaluation for regression models."""
        start_time = time.time()

        # Cross-validation for regression
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)

        # Train on full dataset
        model.fit(X, y)
        train_predictions = model.predict(X)

        # Regression metrics
        train_r2 = r2_score(y, train_predictions)
        train_mse = mean_squared_error(y, train_predictions)
        train_mae = mean_absolute_error(y, train_predictions)

        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_.tolist()

        # Inference timing
        inference_start = time.time()
        _ = model.predict(X[:10])
        inference_time = (time.time() - inference_start) / 10 * 1000  # ms per sample

        training_time = time.time() - start_time

        # Create regression-specific report
        regression_report = {
            'r2_score': float(train_r2),
            'mse': float(train_mse),
            'mae': float(train_mae),
            'rmse': float(np.sqrt(train_mse))
        }

        return TrainingMetrics(
            model_name=model_name,
            task_type='regression',
            cv_scores=cv_scores.tolist(),
            mean_cv_score=float(cv_scores.mean()),
            std_cv_score=float(cv_scores.std()),
            train_score=float(train_r2),
            test_score=float(cv_scores.mean()),
            feature_importance=feature_importance,
            classification_report=regression_report,
            training_time=training_time,
            inference_time=inference_time
        )

    def _save_models_with_metadata(self, models: Dict, X: np.ndarray, y_multi: Dict) -> Dict[str, int]:
        """Save models with comprehensive metadata for production deployment."""
        model_sizes = {}

        for model_name, model in models.items():
            # Model file path
            model_path = self.output_dir / f'{model_name}.pkl'

            # Create model bundle with metadata
            model_bundle = {
                'model': model,
                'feature_extractor': self.feature_extractor,
                'metadata': {
                    'experiment_id': self.experiment_id,
                    'model_name': model_name,
                    'model_type': type(model).__name__,
                    'feature_dimensions': X.shape[1],
                    'training_samples': X.shape[0],
                    'created_at': datetime.utcnow().isoformat(),
                    'sklearn_version': joblib.__version__,
                    'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                    'container_ready': True,  # Azure Container Apps compatibility
                    'azure_deployment': True  # Issue #79 infrastructure
                }
            }

            # Save with compression for container efficiency
            joblib.dump(model_bundle, model_path, compress=3)
            model_sizes[model_name] = model_path.stat().st_size

            logger.info(f"💾 Saved {model_name} model: {model_path} ({model_sizes[model_name]} bytes)")

        # Save feature extractor separately for inference
        extractor_path = self.output_dir / 'feature_extractor.pkl'
        joblib.dump(self.feature_extractor, extractor_path, compress=3)

        return model_sizes

    def _save_training_report(self):
        """Save comprehensive training report as JSON."""
        report_path = self.output_dir / f'training_report_{self.experiment_id}.json'

        # Convert to JSON-serializable format
        report_dict = asdict(self.training_report)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"📊 Saved training report: {report_path}")

    def _extract_default_hyperparameters(self, models: Dict) -> Dict[str, Dict[str, Any]]:
        """Extract hyperparameters from trained models."""
        hyperparameters = {}

        for model_name, model in models.items():
            params = model.get_params()
            # Filter to important hyperparameters
            important_params = {
                key: value for key, value in params.items()
                if not key.endswith('_') and value is not None
            }
            hyperparameters[model_name] = important_params

        return hyperparameters

    def _log_summary_metrics(self):
        """Log summary metrics for monitoring."""
        if not self.training_report or not self.training_report.success:
            return

        logger.info("📊 TRAINING SUMMARY METRICS:")
        logger.info(f"   Experiment ID: {self.training_report.experiment_id}")
        logger.info(f"   Total Training Time: {self.training_report.total_training_time:.2f}s")
        logger.info(f"   Dataset: {self.training_report.dataset_info['total_samples']} samples")

        for model_name, metrics in self.training_report.overall_metrics.items():
            if metrics.task_type == 'classification':
                logger.info(f"   {model_name}: {metrics.mean_cv_score:.3f} ± {metrics.std_cv_score:.3f} accuracy")
            else:
                logger.info(f"   {model_name}: {metrics.mean_cv_score:.3f} ± {metrics.std_cv_score:.3f} R²")
            logger.info(f"      Inference: {metrics.inference_time:.2f}ms per sample")

def main():
    """Main training script for containerized deployment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configuration for container deployment
    csv_path = os.getenv('TRAINING_CSV_PATH', '/home/warrenjo/src/tmp2/secure data/VideosAskingQuestions CSV.csv')
    output_dir = os.getenv('MODEL_OUTPUT_DIR', 'src/ml/trained_models')
    use_grid_search = os.getenv('USE_GRID_SEARCH', 'false').lower() == 'true'
    cv_folds = int(os.getenv('CV_FOLDS', '5'))

    if not os.path.exists(csv_path):
        logger.error(f"❌ Training data not found: {csv_path}")
        return False

    try:
        # Initialize and run training
        trainer = ClassicalMLTrainer(csv_path=csv_path, output_dir=output_dir)
        report = trainer.train_all_models(use_grid_search=use_grid_search, cv_folds=cv_folds)

        if report.success:
            logger.info("✅ Training completed successfully!")

            # Check if models meet target thresholds (from Issue #76)
            question_acc = report.overall_metrics['question_classifier'].mean_cv_score
            wait_time_acc = report.overall_metrics['wait_time_detector'].mean_cv_score
            class_r2 = report.overall_metrics['class_scorer'].mean_cv_score

            logger.info("🎯 TARGET PERFORMANCE CHECK:")
            logger.info(f"   Question Classification: {question_acc:.3f} ≥ 0.85: {'✅' if question_acc >= 0.85 else '❌'}")
            logger.info(f"   Wait Time Detection: {wait_time_acc:.3f} ≥ 0.80: {'✅' if wait_time_acc >= 0.80 else '❌'}")
            logger.info(f"   CLASS Scoring: {class_r2:.3f} ≥ 0.75: {'✅' if class_r2 >= 0.75 else '❌'}")

            return True
        else:
            logger.error("❌ Training failed!")
            return False

    except Exception as e:
        logger.error(f"❌ Training error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)