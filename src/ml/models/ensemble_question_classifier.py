#!/usr/bin/env python3
"""
Ensemble Question Classifier for Superior OEQ/CEQ Classification
Implements Issues #118, #120: Multi-Model Voting for Educational Analysis

Warren - This is the missing ensemble architecture for robust classification!

Combines three complementary models:
- Neural Network (MLPClassifier): Deep semantic pattern recognition
- Random Forest: Feature importance and interaction analysis
- Logistic Regression: Interpretable linear boundaries

Author: Claude (Partner-Level Microsoft SDE)
Issues: #118 (Synthetic Data), #120 (Gradient Descent Tuning)
"""

import numpy as np
import joblib
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
from abc import ABC, abstractmethod

# Scikit-learn imports for ensemble components
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available for ensemble classification")

logger = logging.getLogger(__name__)

class BaseVotingStrategy(ABC):
    """Base class for different voting strategies"""

    @abstractmethod
    def vote(self, predictions: List[np.ndarray], probabilities: List[np.ndarray]) -> Tuple[int, float]:
        """
        Combine predictions from multiple models

        Args:
            predictions: List of class predictions from each model
            probabilities: List of probability arrays from each model

        Returns:
            (final_prediction, confidence_score)
        """
        pass

class HardVotingStrategy(BaseVotingStrategy):
    """Simple majority voting (2 out of 3 models agree)"""

    def vote(self, predictions: List[np.ndarray], probabilities: List[np.ndarray]) -> Tuple[int, float]:
        """Hard voting with confidence from agreement level"""
        votes = np.array(predictions)

        # Count votes for each class
        unique, counts = np.unique(votes, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        agreement_ratio = np.max(counts) / len(votes)

        # Confidence based on agreement level
        confidence = 0.5 + (agreement_ratio - 0.5) * 0.8  # Scale 0.5-1.0 to 0.5-0.9

        return int(majority_class), float(confidence)

class SoftVotingStrategy(BaseVotingStrategy):
    """Probability averaging with optional model weighting"""

    def __init__(self, model_weights: Optional[List[float]] = None):
        self.model_weights = model_weights or [1.0, 1.0, 1.0]  # Equal weights by default
        # Normalize weights
        total_weight = sum(self.model_weights)
        self.model_weights = [w / total_weight for w in self.model_weights]

    def vote(self, predictions: List[np.ndarray], probabilities: List[np.ndarray]) -> Tuple[int, float]:
        """Weighted probability averaging"""
        if not probabilities or len(probabilities) == 0:
            # Fallback to hard voting
            return HardVotingStrategy().vote(predictions, probabilities)

        # Weight and average probabilities
        weighted_probs = np.zeros_like(probabilities[0])
        for i, (probs, weight) in enumerate(zip(probabilities, self.model_weights)):
            weighted_probs += probs * weight

        final_prediction = np.argmax(weighted_probs)
        confidence = float(np.max(weighted_probs))

        return int(final_prediction), confidence

class ConfidenceWeightedStrategy(BaseVotingStrategy):
    """Dynamic weighting based on individual model confidence"""

    def vote(self, predictions: List[np.ndarray], probabilities: List[np.ndarray]) -> Tuple[int, float]:
        """Weight models by their confidence in current prediction"""
        if not probabilities or len(probabilities) == 0:
            return HardVotingStrategy().vote(predictions, probabilities)

        # Calculate confidence for each model (max probability)
        confidences = [np.max(probs) for probs in probabilities]
        total_confidence = sum(confidences)

        if total_confidence == 0:
            # Equal weighting fallback
            return SoftVotingStrategy().vote(predictions, probabilities)

        # Weight by confidence
        weights = [conf / total_confidence for conf in confidences]

        # Use weighted averaging
        weighted_strategy = SoftVotingStrategy(model_weights=weights)
        return weighted_strategy.vote(predictions, probabilities)

class EnsembleQuestionClassifier:
    """
    Ensemble classifier combining Neural Network, Random Forest, and Logistic Regression
    for superior OEQ/CEQ classification performance.
    """

    def __init__(self,
                 voting_strategy: str = 'soft',
                 model_weights: Optional[List[float]] = None,
                 feature_extractor = None):
        """
        Initialize ensemble classifier

        Args:
            voting_strategy: 'hard', 'soft', or 'confidence_weighted'
            model_weights: Optional weights for models [NN, RF, LR]
            feature_extractor: Feature extraction instance
        """
        self.voting_strategy_name = voting_strategy
        self.model_weights = model_weights or [0.4, 0.35, 0.25]  # NN, RF, LR
        self.feature_extractor = feature_extractor
        self.scaler = StandardScaler()

        # Initialize models
        self.models = {}
        self.ensemble = None
        self.trained = False

        # Setup voting strategy
        self._setup_voting_strategy()

        # Initialize individual models
        if SKLEARN_AVAILABLE:
            self._initialize_models()
        else:
            logger.warning("Scikit-learn not available, ensemble will use fallback")

    def _setup_voting_strategy(self):
        """Setup the voting strategy based on configuration"""
        if self.voting_strategy_name == 'hard':
            self.voting_strategy = HardVotingStrategy()
        elif self.voting_strategy_name == 'soft':
            self.voting_strategy = SoftVotingStrategy(self.model_weights)
        elif self.voting_strategy_name == 'confidence_weighted':
            self.voting_strategy = ConfidenceWeightedStrategy()
        else:
            logger.warning(f"Unknown voting strategy: {self.voting_strategy_name}, using soft")
            self.voting_strategy = SoftVotingStrategy(self.model_weights)

    def _initialize_models(self):
        """Initialize the three base classifiers"""

        # Neural Network: Deep semantic pattern recognition
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            dropout=0.3,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )

        # Random Forest: Feature importance and interactions
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )

        # Logistic Regression: Interpretable linear boundaries
        self.models['logistic_regression'] = LogisticRegression(
            C=1.0,
            solver='liblinear',
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )

        logger.info("✅ Initialized ensemble with 3 base classifiers")
        logger.info("   - Neural Network: MLPClassifier(128, 64 hidden)")
        logger.info("   - Random Forest: 100 trees, depth=10")
        logger.info("   - Logistic Regression: L2 regularized")

    def train(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Train the ensemble on expert annotation data

        Args:
            X: Feature matrix (N x features)
            y: Labels (OEQ=0, CEQ=1, Rhetorical=2)
            validation_data: Optional (X_val, y_val) for validation

        Returns:
            Training results and performance metrics
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Scikit-learn required for ensemble training")

        start_time = time.time()

        # Scale features for neural network
        X_scaled = self.scaler.fit_transform(X)

        # Train individual models
        model_scores = {}

        for name, model in self.models.items():
            logger.info(f"🎯 Training {name}...")

            # Use scaled features for neural network, original for tree-based
            X_train = X_scaled if name == 'neural_network' else X

            # Train model
            model.fit(X_train, y)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y, cv=5, scoring='accuracy')
            model_scores[name] = {
                'mean_cv_accuracy': float(np.mean(cv_scores)),
                'std_cv_accuracy': float(np.std(cv_scores)),
                'train_accuracy': float(accuracy_score(y, model.predict(X_train)))
            }

            logger.info(f"   ✅ {name}: {model_scores[name]['mean_cv_accuracy']:.3f} ± {model_scores[name]['std_cv_accuracy']:.3f}")

        # Test ensemble performance
        ensemble_predictions = []
        ensemble_confidences = []

        for i in range(len(X)):
            # Get predictions from all models
            predictions = []
            probabilities = []

            for name, model in self.models.items():
                X_sample = X_scaled[i:i+1] if name == 'neural_network' else X[i:i+1]
                pred = model.predict(X_sample)[0]

                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_sample)[0]
                    probabilities.append(proba)
                else:
                    # Create one-hot encoding for models without probabilities
                    proba = np.zeros(len(np.unique(y)))
                    proba[pred] = 1.0
                    probabilities.append(proba)

                predictions.append(pred)

            # Apply voting strategy
            final_pred, confidence = self.voting_strategy.vote(predictions, probabilities)
            ensemble_predictions.append(final_pred)
            ensemble_confidences.append(confidence)

        # Calculate ensemble performance
        ensemble_accuracy = accuracy_score(y, ensemble_predictions)
        mean_confidence = np.mean(ensemble_confidences)

        training_time = time.time() - start_time
        self.trained = True

        results = {
            'ensemble_accuracy': float(ensemble_accuracy),
            'mean_confidence': float(mean_confidence),
            'individual_models': model_scores,
            'training_time_seconds': training_time,
            'voting_strategy': self.voting_strategy_name,
            'model_weights': self.model_weights,
            'samples_trained': len(X),
            'features_count': X.shape[1]
        }

        logger.info(f"🎉 Ensemble Training Complete!")
        logger.info(f"   📊 Ensemble Accuracy: {ensemble_accuracy:.3f}")
        logger.info(f"   🎯 Mean Confidence: {mean_confidence:.3f}")
        logger.info(f"   ⏱️  Training Time: {training_time:.1f}s")

        return results

    def predict(self, question_text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Classify a question using ensemble voting

        Args:
            question_text: Question to classify
            metadata: Optional metadata for enhanced features

        Returns:
            Classification results with ensemble consensus
        """
        if not self.trained:
            raise RuntimeError("Ensemble must be trained before prediction")

        start_time = time.time()

        try:
            # Extract features
            if self.feature_extractor:
                features = self.feature_extractor.extract_features(question_text, metadata)
                features = features.reshape(1, -1)
            else:
                # Fallback to basic features if no extractor
                features = self._extract_basic_features(question_text)

            # Get predictions from all models
            predictions = []
            probabilities = []
            individual_results = {}

            for name, model in self.models.items():
                # Use scaled features for neural network
                X_sample = self.scaler.transform(features) if name == 'neural_network' else features

                pred = model.predict(X_sample)[0]
                predictions.append(pred)

                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_sample)[0]
                    probabilities.append(proba)

                    individual_results[name] = {
                        'prediction': int(pred),
                        'confidence': float(np.max(proba)),
                        'probabilities': {
                            'OEQ': float(proba[0]),
                            'CEQ': float(proba[1]) if len(proba) > 1 else 0.0,
                            'Rhetorical': float(proba[2]) if len(proba) > 2 else 0.0
                        }
                    }
                else:
                    individual_results[name] = {
                        'prediction': int(pred),
                        'confidence': 0.8,  # Default confidence
                        'probabilities': {}
                    }

            # Apply ensemble voting
            final_prediction, ensemble_confidence = self.voting_strategy.vote(predictions, probabilities)

            # Map prediction to label
            question_types = ['OEQ', 'CEQ', 'Rhetorical']
            predicted_type = question_types[final_prediction] if final_prediction < len(question_types) else 'Unknown'

            inference_time = (time.time() - start_time) * 1000  # ms

            return {
                'version': 'ensemble_v1.0_issues_118_120',
                'model_type': 'EnsembleClassifier',
                'voting_strategy': self.voting_strategy_name,
                'analysis_method': 'ensemble_voting',
                'question_type': predicted_type,
                'confidence': ensemble_confidence,
                'individual_models': individual_results,
                'ensemble_consensus': {
                    'all_predictions': [question_types[p] for p in predictions],
                    'agreement_level': len(set(predictions)) == 1,  # All models agree
                    'voting_weights': self.model_weights
                },
                'performance': {
                    'inference_time_ms': round(inference_time, 2),
                    'models_count': len(self.models),
                    'features_used': features.shape[1] if hasattr(features, 'shape') else 0
                },
                'metadata': {
                    'timestamp': time.time(),
                    'question_length': len(question_text),
                    'has_metadata': metadata is not None
                }
            }

        except Exception as e:
            logger.error(f"❌ Ensemble prediction failed: {e}")
            return self._create_fallback_response(question_text, str(e))

    def _extract_basic_features(self, text: str) -> np.ndarray:
        """Fallback basic feature extraction if no feature extractor available"""
        features = [
            len(text),                                    # Length
            text.count(' ') + 1,                         # Word count
            text.count('?'),                              # Question marks
            float(text.lower().startswith(('what', 'how', 'why'))),  # WH-question
            float('think' in text.lower() or 'feel' in text.lower()),  # Opinion seeking
            float(text.lower().startswith(('is', 'are', 'do', 'can'))),  # Yes/no start
            text.count(','),                              # Complexity
            float(len(text.split()) > 7),                 # Long question
            float('or' in text.lower()),                  # Options
            float('how many' in text.lower() or 'how much' in text.lower())  # Counting
        ]
        return np.array(features).reshape(1, -1)

    def _create_fallback_response(self, question_text: str, error_message: str) -> Dict[str, Any]:
        """Create fallback response when ensemble fails"""
        return {
            'version': 'ensemble_fallback_v1.0',
            'model_type': 'EnsembleFallback',
            'analysis_method': 'fallback',
            'question_type': 'Unknown',
            'confidence': 0.5,
            'individual_models': {},
            'ensemble_consensus': {
                'error': error_message,
                'fallback_used': True
            },
            'performance': {
                'inference_time_ms': 0,
                'models_count': 0,
                'error': error_message
            },
            'metadata': {
                'timestamp': time.time(),
                'question_length': len(question_text),
                'fallback_reason': error_message
            }
        }

    def save_ensemble(self, model_path: str) -> bool:
        """
        Save the trained ensemble to disk

        Args:
            model_path: Path to save the ensemble bundle

        Returns:
            Success status
        """
        if not self.trained:
            logger.error("Cannot save untrained ensemble")
            return False

        try:
            ensemble_bundle = {
                'models': self.models,
                'scaler': self.scaler,
                'voting_strategy_name': self.voting_strategy_name,
                'model_weights': self.model_weights,
                'feature_extractor': self.feature_extractor,
                'trained': self.trained,
                'version': 'ensemble_v1.0',
                'sklearn_available': SKLEARN_AVAILABLE
            }

            joblib.dump(ensemble_bundle, model_path)
            logger.info(f"✅ Ensemble saved to {model_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save ensemble: {e}")
            return False

    @classmethod
    def load_ensemble(cls, model_path: str) -> 'EnsembleQuestionClassifier':
        """
        Load a trained ensemble from disk

        Args:
            model_path: Path to the saved ensemble bundle

        Returns:
            Loaded ensemble classifier
        """
        try:
            ensemble_bundle = joblib.load(model_path)

            # Create instance
            ensemble = cls(
                voting_strategy=ensemble_bundle['voting_strategy_name'],
                model_weights=ensemble_bundle['model_weights'],
                feature_extractor=ensemble_bundle.get('feature_extractor')
            )

            # Restore state
            ensemble.models = ensemble_bundle['models']
            ensemble.scaler = ensemble_bundle['scaler']
            ensemble.trained = ensemble_bundle['trained']

            logger.info(f"✅ Ensemble loaded from {model_path}")
            logger.info(f"   📊 Models: {list(ensemble.models.keys())}")
            logger.info(f"   🗳️  Voting: {ensemble.voting_strategy_name}")

            return ensemble

        except Exception as e:
            logger.error(f"❌ Failed to load ensemble: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the ensemble"""
        return {
            'ensemble_type': 'QuestionClassifier',
            'models': list(self.models.keys()) if self.models else [],
            'voting_strategy': self.voting_strategy_name,
            'model_weights': self.model_weights,
            'trained': self.trained,
            'sklearn_available': SKLEARN_AVAILABLE,
            'feature_extractor_available': self.feature_extractor is not None
        }