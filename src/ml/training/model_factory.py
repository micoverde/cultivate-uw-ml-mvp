#!/usr/bin/env python3
"""
Classical ML Model Factory for Educational Domain
Implements Issue #109 design specification with optimized scikit-learn
configurations for 112-sample dataset.

Provides hyperparameters tuned for limited training data with cross-validation
to prevent overfitting while maintaining high accuracy.

Author: Claude-4 (Partner-Level Microsoft SDE)
Issue: #109 - Classical ML Model Training on Expert Annotations
Context: Phase 1 of Issue #76 Comprehensive ML Architecture
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ClassicalMLModelFactory:
    """
    Optimized scikit-learn configurations for educational domain.
    Hyperparameters tuned for 112-sample dataset with cross-validation.

    Design Principles:
    - Conservative parameters to prevent overfitting
    - Balanced class weights for imbalanced data
    - Ensemble methods for robustness
    - Fast inference for production deployment
    """

    @staticmethod
    def create_question_classifier() -> RandomForestClassifier:
        """
        RandomForest for OEQ/CEQ/Rhetorical classification.
        Optimized for small dataset with high interpretability.

        Configuration rationale:
        - n_estimators=100: Sufficient trees for stability without overfitting
        - max_depth=8: Conservative depth for 112 samples
        - min_samples_split=5: Prevent splitting on small groups
        - class_weight='balanced': Handle OEQ(39) vs CEQ(73) imbalance
        """
        return RandomForestClassifier(
            n_estimators=100,           # Sufficient trees for stability
            max_depth=8,                # Prevent overfitting with limited data
            min_samples_split=5,        # Conservative splitting
            min_samples_leaf=3,         # Minimum leaf size for generalization
            max_features='sqrt',        # Feature randomness for robustness
            bootstrap=True,             # Bootstrap sampling
            oob_score=True,             # Out-of-bag validation
            random_state=42,            # Reproducibility
            n_jobs=-1,                  # Parallel processing
            class_weight='balanced',    # Handle class imbalance (39 OEQ vs 73 CEQ)
            criterion='gini',           # Gini impurity for classification
            max_samples=0.8             # Use 80% of samples per tree
        )

    @staticmethod
    def create_wait_time_detector() -> SVC:
        """
        SVM with RBF kernel for wait time behavior detection.
        Effective for non-linear patterns in prosodic features.

        Configuration rationale:
        - C=1.0: Balanced regularization for small dataset
        - kernel='rbf': Non-linear decision boundary for prosodic patterns
        - gamma='scale': Automatic gamma scaling for 79 features
        - probability=True: Enable confidence scores for inference
        """
        return SVC(
            C=1.0,                      # Regularization parameter (balanced)
            kernel='rbf',               # Non-linear decision boundary
            gamma='scale',              # Automatic gamma scaling for 79 features
            probability=True,           # Enable probability estimates
            cache_size=200,             # Memory optimization
            class_weight='balanced',    # Handle imbalanced classes
            random_state=42,            # Reproducibility
            decision_function_shape='ovr',  # One-vs-rest for multi-class
            tol=1e-3,                   # Tolerance for stopping criterion
            max_iter=1000               # Maximum iterations
        )

    @staticmethod
    def create_class_scorer() -> GradientBoostingRegressor:
        """
        Gradient Boosting for CLASS framework scoring.
        Handles continuous scores (1-5) with ensemble robustness.

        Configuration rationale:
        - n_estimators=200: Moderate boosting rounds for small dataset
        - learning_rate=0.1: Conservative rate to prevent overfitting
        - max_depth=6: Tree complexity appropriate for 112 samples
        - subsample=0.8: Stochastic gradient boosting for regularization
        """
        return GradientBoostingRegressor(
            n_estimators=200,           # Moderate boosting rounds
            learning_rate=0.1,          # Conservative learning rate
            max_depth=6,                # Tree complexity for small dataset
            min_samples_split=5,        # Conservative splitting criteria
            min_samples_leaf=3,         # Leaf size constraints
            subsample=0.8,              # Stochastic gradient boosting
            random_state=42,            # Reproducibility
            alpha=0.9,                  # Huber loss robustness (90th percentile)
            loss='huber',               # Robust to outliers in CLASS scores
            max_features='sqrt',        # Feature randomness
            validation_fraction=0.1,    # Early stopping validation
            n_iter_no_change=10,        # Early stopping patience
            tol=1e-4                    # Tolerance for early stopping
        )

    @staticmethod
    def create_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
        """
        Hyperparameter grids for GridSearchCV optimization.
        Focused search spaces for small dataset.

        Returns:
            Dictionary of parameter grids for each model type
        """
        return {
            'question_classifier': {
                'n_estimators': [50, 100, 150],
                'max_depth': [6, 8, 10],
                'min_samples_split': [3, 5, 7],
                'max_features': ['sqrt', 'log2']
            },
            'wait_time_detector': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            },
            'class_scorer': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8]
            }
        }

    @staticmethod
    def create_optimized_models_with_gridsearch(X, y_multi, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Create models with hyperparameter optimization using GridSearchCV.

        Args:
            X: Feature matrix (112, 79)
            y_multi: Multi-task labels dictionary
            cv_folds: Cross-validation folds

        Returns:
            Dictionary of optimized models with best parameters
        """
        logger.info("🔍 Starting hyperparameter optimization with GridSearchCV")

        grids = ClassicalMLModelFactory.create_hyperparameter_grids()
        optimized_models = {}

        # Question Classifier optimization
        logger.info("📊 Optimizing question classifier...")
        base_rf = RandomForestClassifier(
            bootstrap=True, oob_score=True, random_state=42,
            n_jobs=-1, class_weight='balanced'
        )

        grid_search_rf = GridSearchCV(
            base_rf, grids['question_classifier'],
            cv=cv_folds, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search_rf.fit(X, y_multi['question_type'])

        optimized_models['question_classifier'] = {
            'model': grid_search_rf.best_estimator_,
            'best_params': grid_search_rf.best_params_,
            'best_score': grid_search_rf.best_score_,
            'cv_results': grid_search_rf.cv_results_
        }

        # Wait Time Detector optimization
        logger.info("📊 Optimizing wait time detector...")
        base_svm = SVC(
            probability=True, random_state=42,
            class_weight='balanced', cache_size=200
        )

        grid_search_svm = GridSearchCV(
            base_svm, grids['wait_time_detector'],
            cv=cv_folds, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search_svm.fit(X, y_multi['wait_time'])

        optimized_models['wait_time_detector'] = {
            'model': grid_search_svm.best_estimator_,
            'best_params': grid_search_svm.best_params_,
            'best_score': grid_search_svm.best_score_,
            'cv_results': grid_search_svm.cv_results_
        }

        # CLASS Scorer optimization
        logger.info("📊 Optimizing CLASS scorer...")
        base_gbr = GradientBoostingRegressor(
            random_state=42, alpha=0.9, loss='huber',
            validation_fraction=0.1, n_iter_no_change=10
        )

        grid_search_gbr = GridSearchCV(
            base_gbr, grids['class_scorer'],
            cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
        )
        grid_search_gbr.fit(X, y_multi['class_scores'])

        optimized_models['class_scorer'] = {
            'model': grid_search_gbr.best_estimator_,
            'best_params': grid_search_gbr.best_params_,
            'best_score': grid_search_gbr.best_score_,
            'cv_results': grid_search_gbr.cv_results_
        }

        logger.info("✅ Hyperparameter optimization complete")
        return optimized_models

    @staticmethod
    def get_model_info() -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive model information and rationale.

        Returns:
            Dictionary with model specifications and design rationale
        """
        return {
            'question_classifier': {
                'algorithm': 'RandomForest',
                'task': 'Multi-class classification (OEQ/CEQ/Rhetorical)',
                'target_accuracy': 0.85,
                'key_features': ['TF-IDF', 'keyword_patterns', 'question_structure'],
                'rationale': 'Robust to overfitting, provides feature importance, handles class imbalance',
                'hyperparameters': {
                    'n_estimators': '100 trees for stability',
                    'max_depth': '8 levels to prevent overfitting',
                    'class_weight': 'balanced for OEQ/CEQ imbalance'
                }
            },
            'wait_time_detector': {
                'algorithm': 'SVM with RBF kernel',
                'task': 'Multi-class classification (appropriate/insufficient/none)',
                'target_accuracy': 0.80,
                'key_features': ['pause_patterns', 'temporal_features', 'prosodic_indicators'],
                'rationale': 'Effective for non-linear patterns in temporal data',
                'hyperparameters': {
                    'C': '1.0 for balanced regularization',
                    'kernel': 'RBF for non-linear decision boundaries',
                    'probability': 'True for confidence scores'
                }
            },
            'class_scorer': {
                'algorithm': 'Gradient Boosting Regression',
                'task': 'Regression (CLASS framework scores 1-5)',
                'target_correlation': 0.75,
                'key_features': ['interaction_quality', 'response_patterns', 'pedagogical_indicators'],
                'rationale': 'Handles continuous scores, robust to outliers with Huber loss',
                'hyperparameters': {
                    'n_estimators': '200 boosting rounds',
                    'learning_rate': '0.1 conservative rate',
                    'loss': 'huber for outlier robustness'
                }
            }
        }

if __name__ == "__main__":
    # Test model factory
    logging.basicConfig(level=logging.INFO)

    factory = ClassicalMLModelFactory()

    # Create default models
    question_clf = factory.create_question_classifier()
    wait_time_clf = factory.create_wait_time_detector()
    class_scorer = factory.create_class_scorer()

    print("✅ Model Factory Test Results:")
    print(f"Question Classifier: {type(question_clf).__name__} with {question_clf.n_estimators} estimators")
    print(f"Wait Time Detector: {type(wait_time_clf).__name__} with {wait_time_clf.kernel} kernel")
    print(f"CLASS Scorer: {type(class_scorer).__name__} with {class_scorer.n_estimators} estimators")

    # Display model info
    model_info = factory.get_model_info()
    print("\n📊 Model Information:")
    for model_name, info in model_info.items():
        print(f"\n{model_name}:")
        print(f"  Algorithm: {info['algorithm']}")
        print(f"  Task: {info['task']}")
        print(f"  Rationale: {info['rationale']}")