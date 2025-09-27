#!/usr/bin/env python3
"""
Enhanced Ensemble Trainer V2 with Azure Blob Integration and Performance Metrics
Implements Issue #187: Advanced Multi-Model Fusion for Educational Coaching System

Features:
- Azure Blob Storage integration for training data and model persistence
- Real-time retraining capabilities
- Comprehensive loss calculation and performance metrics
- Support for both ensemble and classic model switching
- Ground truth dataset evaluation

Author: Warren & Claude (Microsoft Partner-Level Collaboration)
Issue: #187 - Ensemble ML Model Architecture
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import time
from io import BytesIO
import os

# Azure Storage imports
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    log_loss, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

# XGBoost for advanced gradient boosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# LightGBM for faster training
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)

class AzureBlobManager:
    """Manages Azure Blob Storage operations for models and datasets"""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize Azure Blob Storage connection"""
        self.connection_string = connection_string or os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not self.connection_string:
            logger.warning("Azure Storage connection string not found. Using local storage.")
            self.blob_service_client = None
        else:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            self.container_name = "ml-models"
            self._ensure_container_exists()

    def _ensure_container_exists(self):
        """Ensure the ML models container exists"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            if not container_client.exists():
                self.blob_service_client.create_container(self.container_name)
                logger.info(f"Created container: {self.container_name}")
        except Exception as e:
            logger.error(f"Error ensuring container exists: {e}")

    def upload_model(self, model_data: bytes, model_name: str, metadata: Dict[str, Any]):
        """Upload trained model to Azure Blob Storage"""
        if not self.blob_service_client:
            return self._save_local(model_data, model_name, metadata)

        try:
            blob_name = f"models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            # Upload model with metadata
            blob_client.upload_blob(
                model_data,
                overwrite=True,
                metadata=metadata
            )

            logger.info(f"Model uploaded to Azure Blob: {blob_name}")
            return blob_name

        except Exception as e:
            logger.error(f"Error uploading model to Azure: {e}")
            return self._save_local(model_data, model_name, metadata)

    def download_model(self, blob_name: str) -> Tuple[bytes, Dict[str, Any]]:
        """Download model from Azure Blob Storage"""
        if not self.blob_service_client:
            return self._load_local(blob_name)

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            # Download model and metadata
            blob_data = blob_client.download_blob()
            model_data = blob_data.readall()
            metadata = blob_client.get_blob_properties().metadata

            return model_data, metadata

        except ResourceNotFoundError:
            logger.error(f"Model not found in Azure: {blob_name}")
            return self._load_local(blob_name)

    def _save_local(self, model_data: bytes, model_name: str, metadata: Dict[str, Any]):
        """Fallback to local storage"""
        local_path = Path(f"./models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, 'wb') as f:
            f.write(model_data)

        # Save metadata
        with open(local_path.with_suffix('.meta.json'), 'w') as f:
            json.dump(metadata, f)

        logger.info(f"Model saved locally: {local_path}")
        return str(local_path)

    def _load_local(self, path: str) -> Tuple[bytes, Dict[str, Any]]:
        """Load from local storage"""
        local_path = Path(path)

        with open(local_path, 'rb') as f:
            model_data = f.read()

        # Load metadata
        meta_path = local_path.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return model_data, metadata

class PerformanceMetrics:
    """Comprehensive performance metrics calculator"""

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
        }

        # Add probability-based metrics if available
        if y_proba is not None:
            try:
                metrics['log_loss'] = log_loss(y_true, y_proba)
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate probability metrics: {e}")

        # Per-class metrics
        unique_classes = np.unique(y_true)
        for class_label in unique_classes:
            class_mask = y_true == class_label
            class_pred_mask = y_pred == class_label

            tp = np.sum((class_mask) & (class_pred_mask))
            fp = np.sum((~class_mask) & (class_pred_mask))
            fn = np.sum((class_mask) & (~class_pred_mask))
            tn = np.sum((~class_mask) & (~class_pred_mask))

            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) \
                if (class_precision + class_recall) > 0 else 0

            metrics[f'class_{class_label}_precision'] = class_precision
            metrics[f'class_{class_label}_recall'] = class_recall
            metrics[f'class_{class_label}_f1'] = class_f1

        return metrics

    @staticmethod
    def calculate_confusion_matrix(y_true, y_pred) -> np.ndarray:
        """Calculate confusion matrix"""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def calculate_loss(y_true, y_proba) -> float:
        """Calculate cross-entropy loss"""
        return log_loss(y_true, y_proba)

class EnhancedEnsembleTrainer:
    """
    Enhanced ensemble trainer with Azure integration and comprehensive metrics
    Supports both classic and advanced ensemble models
    """

    def __init__(self,
                 model_type: str = 'ensemble',
                 use_azure: bool = True,
                 feature_extractor = None):
        """
        Initialize enhanced trainer

        Args:
            model_type: 'ensemble', 'classic', or 'hybrid'
            use_azure: Whether to use Azure Blob Storage
            feature_extractor: Feature extraction pipeline
        """
        self.model_type = model_type
        self.use_azure = use_azure
        self.feature_extractor = feature_extractor
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Azure blob manager
        self.blob_manager = AzureBlobManager() if use_azure else None

        # Model storage
        self.models = {}
        self.ensemble = None
        self.performance_history = []

        # Training configuration
        self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration"""
        return {
            'ensemble': {
                'voting': 'soft',
                'weights': None,  # Will be optimized
                'cv_folds': 5,
                'calibration': True
            },
            'models': {
                'neural_network': {
                    'hidden_layer_sizes': (256, 128, 64),
                    'activation': 'relu',
                    'solver': 'adam',
                    'learning_rate_init': 0.001,
                    'max_iter': 1000,
                    'early_stopping': True,
                    'validation_fraction': 0.15
                },
                'random_forest': {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'n_jobs': -1
                },
                'xgboost': {
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'multi:softprob'
                },
                'svm': {
                    'kernel': 'rbf',
                    'C': 10.0,
                    'gamma': 'scale',
                    'probability': True
                },
                'logistic': {
                    'solver': 'saga',
                    'penalty': 'elasticnet',
                    'l1_ratio': 0.5,
                    'max_iter': 1000,
                    'multi_class': 'multinomial'
                }
            }
        }

    def create_models(self) -> Dict[str, Any]:
        """Create all model instances based on configuration"""
        models = {}

        if self.model_type in ['ensemble', 'hybrid']:
            # Neural Network
            models['neural_network'] = MLPClassifier(
                **self.config['models']['neural_network'],
                random_state=42
            )

            # Random Forest
            models['random_forest'] = RandomForestClassifier(
                **self.config['models']['random_forest'],
                random_state=42
            )

            # XGBoost if available
            if XGBOOST_AVAILABLE:
                models['xgboost'] = xgb.XGBClassifier(
                    **self.config['models']['xgboost'],
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )

            # SVM
            models['svm'] = SVC(
                **self.config['models']['svm'],
                random_state=42
            )

            # Logistic Regression
            models['logistic'] = LogisticRegression(
                **self.config['models']['logistic'],
                random_state=42
            )

            # LightGBM if available
            if LIGHTGBM_AVAILABLE:
                models['lightgbm'] = lgb.LGBMClassifier(
                    n_estimators=200,
                    num_leaves=31,
                    learning_rate=0.05,
                    feature_fraction=0.9,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    random_state=42,
                    verbose=-1
                )

        elif self.model_type == 'classic':
            # Classic models only
            models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
            models['logistic'] = LogisticRegression(
                solver='liblinear',
                random_state=42
            )

        return models

    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        """
        Train ensemble or classic models with comprehensive metrics

        Returns:
            Dictionary with trained models, metrics, and metadata
        """
        start_time = time.time()

        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # Create models
        self.models = self.create_models()

        # Train individual models
        model_performances = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            # Train with cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train_encoded,
                cv=StratifiedKFold(n_splits=self.config['ensemble']['cv_folds']),
                scoring='accuracy'
            )

            # Final training
            model.fit(X_train_scaled, y_train_encoded)

            # Calibrate if requested
            if self.config['ensemble']['calibration']:
                model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
                model.fit(X_train_scaled, y_train_encoded)

            self.models[name] = model

            # Evaluate on validation set if provided
            if X_val is not None:
                y_pred = model.predict(X_val_scaled)
                y_proba = model.predict_proba(X_val_scaled) if hasattr(model, 'predict_proba') else None

                metrics = PerformanceMetrics.calculate_metrics(y_val_encoded, y_pred, y_proba)
                model_performances[name] = {
                    'cv_accuracy': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'val_metrics': metrics
                }

                logger.info(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                logger.info(f"{name} - Val Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")

        # Create ensemble if requested
        if self.model_type in ['ensemble', 'hybrid']:
            # Optimize ensemble weights based on validation performance
            if model_performances:
                weights = self._optimize_ensemble_weights(model_performances)
            else:
                weights = None

            # Create voting classifier
            estimators = [(name, model) for name, model in self.models.items()]
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting=self.config['ensemble']['voting'],
                weights=weights
            )

            # Train ensemble
            self.ensemble.fit(X_train_scaled, y_train_encoded)

            # Evaluate ensemble
            if X_val is not None:
                y_pred_ensemble = self.ensemble.predict(X_val_scaled)
                y_proba_ensemble = self.ensemble.predict_proba(X_val_scaled)

                ensemble_metrics = PerformanceMetrics.calculate_metrics(
                    y_val_encoded, y_pred_ensemble, y_proba_ensemble
                )

                model_performances['ensemble'] = {
                    'weights': weights,
                    'val_metrics': ensemble_metrics
                }

                logger.info(f"Ensemble - Accuracy: {ensemble_metrics['accuracy']:.4f}, "
                          f"F1: {ensemble_metrics['f1_macro']:.4f}")

        # Calculate training time
        training_time = time.time() - start_time

        # Prepare results
        results = {
            'models': self.models,
            'ensemble': self.ensemble,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'performances': model_performances,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }

        # Save to Azure if configured
        if self.blob_manager:
            self._save_to_azure(results)

        # Update performance history
        self.performance_history.append({
            'timestamp': results['timestamp'],
            'performances': model_performances,
            'training_time': training_time
        })

        return results

    def _optimize_ensemble_weights(self, performances: Dict[str, Any]) -> List[float]:
        """Optimize ensemble weights based on validation performance"""
        weights = []
        total_score = 0

        for name in self.models.keys():
            if name in performances and 'val_metrics' in performances[name]:
                # Use F1 macro score for weighting
                score = performances[name]['val_metrics']['f1_macro']
                weights.append(score)
                total_score += score
            else:
                weights.append(1.0)
                total_score += 1.0

        # Normalize weights
        if total_score > 0:
            weights = [w / total_score for w in weights]

        return weights

    def retrain_on_new_data(self, X_new, y_new, incremental: bool = False) -> Dict[str, Any]:
        """
        Retrain models on new data

        Args:
            X_new: New training features
            y_new: New training labels
            incremental: Whether to combine with existing training data

        Returns:
            Training results with updated metrics
        """
        logger.info(f"Retraining models with {len(X_new)} new samples...")

        if incremental and hasattr(self, 'X_train_history'):
            # Combine with historical data
            X_combined = np.vstack([self.X_train_history, X_new])
            y_combined = np.hstack([self.y_train_history, y_new])
        else:
            X_combined = X_new
            y_combined = y_new

        # Store for future incremental training
        self.X_train_history = X_combined
        self.y_train_history = y_combined

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )

        # Retrain
        results = self.train(X_train, y_train, X_val, y_val)

        return results

    def evaluate_on_ground_truth(self, X_test, y_test) -> Dict[str, Any]:
        """
        Evaluate models on ground truth test set

        Returns:
            Comprehensive evaluation metrics
        """
        # Encode and scale
        y_test_encoded = self.label_encoder.transform(y_test)
        X_test_scaled = self.scaler.transform(X_test)

        evaluation_results = {}

        # Evaluate individual models
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None

            metrics = PerformanceMetrics.calculate_metrics(y_test_encoded, y_pred, y_proba)
            conf_matrix = PerformanceMetrics.calculate_confusion_matrix(y_test_encoded, y_pred)

            evaluation_results[name] = {
                'metrics': metrics,
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': classification_report(
                    y_test_encoded, y_pred,
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                )
            }

            if y_proba is not None:
                evaluation_results[name]['loss'] = PerformanceMetrics.calculate_loss(
                    y_test_encoded, y_proba
                )

        # Evaluate ensemble if available
        if self.ensemble:
            y_pred_ensemble = self.ensemble.predict(X_test_scaled)
            y_proba_ensemble = self.ensemble.predict_proba(X_test_scaled)

            ensemble_metrics = PerformanceMetrics.calculate_metrics(
                y_test_encoded, y_pred_ensemble, y_proba_ensemble
            )
            ensemble_conf_matrix = PerformanceMetrics.calculate_confusion_matrix(
                y_test_encoded, y_pred_ensemble
            )

            evaluation_results['ensemble'] = {
                'metrics': ensemble_metrics,
                'confusion_matrix': ensemble_conf_matrix.tolist(),
                'loss': PerformanceMetrics.calculate_loss(y_test_encoded, y_proba_ensemble),
                'classification_report': classification_report(
                    y_test_encoded, y_pred_ensemble,
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                )
            }

        return evaluation_results

    def _save_to_azure(self, results: Dict[str, Any]):
        """Save trained models to Azure Blob Storage"""
        try:
            # Serialize models
            model_data = BytesIO()
            joblib.dump(results, model_data)
            model_data.seek(0)

            # Prepare metadata
            metadata = {
                'model_type': self.model_type,
                'timestamp': results['timestamp'],
                'training_time': str(results['training_time']),
                'num_models': str(len(self.models))
            }

            # Add performance metrics to metadata
            if 'ensemble' in results['performances']:
                metadata['ensemble_accuracy'] = str(
                    results['performances']['ensemble']['val_metrics']['accuracy']
                )
                metadata['ensemble_f1'] = str(
                    results['performances']['ensemble']['val_metrics']['f1_macro']
                )

            # Upload to Azure
            blob_name = self.blob_manager.upload_model(
                model_data.getvalue(),
                f"{self.model_type}_ensemble",
                metadata
            )

            logger.info(f"Models saved to Azure: {blob_name}")

        except Exception as e:
            logger.error(f"Error saving models to Azure: {e}")

    def load_from_azure(self, blob_name: str):
        """Load trained models from Azure Blob Storage"""
        if not self.blob_manager:
            logger.error("Azure Blob Manager not configured")
            return False

        try:
            model_data, metadata = self.blob_manager.download_model(blob_name)

            # Deserialize models
            model_buffer = BytesIO(model_data)
            results = joblib.load(model_buffer)

            # Restore models
            self.models = results.get('models', {})
            self.ensemble = results.get('ensemble')
            self.scaler = results.get('scaler')
            self.label_encoder = results.get('label_encoder')
            self.config = results.get('config', self.config)

            logger.info(f"Models loaded from Azure: {blob_name}")
            logger.info(f"Metadata: {metadata}")

            return True

        except Exception as e:
            logger.error(f"Error loading models from Azure: {e}")
            return False

    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all model performances"""
        if not self.performance_history:
            return pd.DataFrame()

        # Collect all metrics
        comparison_data = []
        latest = self.performance_history[-1]

        for model_name, perf in latest['performances'].items():
            if 'val_metrics' in perf:
                row = {'model': model_name}
                row.update(perf['val_metrics'])
                comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by F1 macro score
        if 'f1_macro' in df.columns:
            df = df.sort_values('f1_macro', ascending=False)

        return df

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Example: Train ensemble with Azure integration
    trainer = EnhancedEnsembleTrainer(
        model_type='ensemble',
        use_azure=True
    )

    # Generate sample data (replace with real data)
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    # Convert to string labels for educational context
    y = ['OEQ' if label == 1 else 'CEQ' for label in y]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    results = trainer.train(X_train, y_train, X_test, y_test)

    # Evaluate on ground truth
    evaluation = trainer.evaluate_on_ground_truth(X_test, y_test)

    # Display results
    print("\n=== Model Performance Comparison ===")
    print(trainer.get_model_comparison())

    print("\n=== Ensemble Performance ===")
    if 'ensemble' in evaluation:
        print(f"Accuracy: {evaluation['ensemble']['metrics']['accuracy']:.4f}")
        print(f"F1 Score: {evaluation['ensemble']['metrics']['f1_macro']:.4f}")
        print(f"Loss: {evaluation['ensemble']['loss']:.4f}")