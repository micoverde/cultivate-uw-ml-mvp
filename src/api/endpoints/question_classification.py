"""
Question Classification API Endpoint - Real ML Classification
Warren's Requirement: NO simulations - only REAL ML API calls

Provides real-time OEQ/CEQ classification using ensemble ML models.
Returns JSON responses with detailed classification results for console debugging.

Author: Claude (Partner-Level Microsoft SDE)
Feature: Real-time question classification for Warren's authentic teaching demo
"""

import uuid
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.logger import get_logger
from ...ml.models.ensemble_question_classifier import EnsembleQuestionClassifier
from ...ml.training.enhanced_feature_extractor import EnhancedFeatureExtractor

logger = get_logger(__name__)

# Configure router
router = APIRouter(prefix="/classify", tags=["question-classification"])

# Global ML models (initialized once)
ensemble_classifier = None
feature_extractor = None

def initialize_ml_models():
    """Initialize ensemble classifier and feature extractor"""
    global ensemble_classifier, feature_extractor

    if ensemble_classifier is None:
        logger.info("🧠 Initializing ensemble question classifier...")
        try:
            ensemble_classifier = EnsembleQuestionClassifier()
            feature_extractor = EnhancedFeatureExtractor()
            logger.info("✅ Ensemble classifier loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load ML models: {e}")
            raise HTTPException(status_code=500, detail=f"ML model initialization failed: {e}")

# Request Models
class QuestionRequest(BaseModel):
    """Single question classification request"""
    text: str = Field(..., min_length=1, description="Question text to classify")
    context: Optional[str] = Field(None, description="Optional context around the question")
    timestamp: Optional[float] = Field(None, description="Timestamp in video/audio")

class BatchQuestionRequest(BaseModel):
    """Batch question classification request"""
    questions: List[QuestionRequest] = Field(..., min_items=1, max_items=100, description="List of questions to classify")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking")

# Response Models
class QuestionClassificationResult(BaseModel):
    """Individual question classification result"""
    question_text: str
    classification: str = Field(..., pattern="^(OEQ|CEQ)$", description="Open-Ended or Closed-Ended Question")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    features_extracted: int = Field(..., description="Number of features used in classification")
    ensemble_details: Dict[str, Any] = Field(..., description="Individual model predictions")
    processing_time_ms: float = Field(..., description="Time taken for classification")
    timestamp: Optional[float] = None

class BatchClassificationResponse(BaseModel):
    """Batch classification response"""
    session_id: str
    total_questions: int
    processed_questions: int
    processing_time_ms: float
    results: List[QuestionClassificationResult]
    summary: Dict[str, Any]

@router.post("/question", response_model=QuestionClassificationResult)
async def classify_single_question(request: QuestionRequest) -> QuestionClassificationResult:
    """
    Classify a single question as OEQ or CEQ using ensemble ML models.

    Returns real-time classification with detailed debugging information.
    """
    start_time = time.time()

    # Initialize models if needed
    initialize_ml_models()

    try:
        logger.info(f"🔍 Classifying question: '{request.text[:50]}...'")

        # Extract features using real ML pipeline
        features = feature_extractor.extract_features(request.text)
        logger.info(f"📊 Extracted {len(features)} features")

        # Get ensemble prediction
        prediction = ensemble_classifier.predict([features])[0]
        confidence = ensemble_classifier.predict_proba([features])[0]

        # Get individual model predictions for debugging
        ensemble_details = {
            "neural_network": {
                "prediction": ensemble_classifier.models['neural_network'].predict([features])[0],
                "confidence": float(ensemble_classifier.models['neural_network'].predict_proba([features])[0].max())
            },
            "random_forest": {
                "prediction": ensemble_classifier.models['random_forest'].predict([features])[0],
                "confidence": float(ensemble_classifier.models['random_forest'].predict_proba([features])[0].max())
            },
            "logistic_regression": {
                "prediction": ensemble_classifier.models['logistic_regression'].predict([features])[0],
                "confidence": float(ensemble_classifier.models['logistic_regression'].predict_proba([features])[0].max())
            },
            "voting_strategy": ensemble_classifier.voting_strategy,
            "model_weights": ensemble_classifier.model_weights
        }

        processing_time = (time.time() - start_time) * 1000

        result = QuestionClassificationResult(
            question_text=request.text,
            classification=prediction,
            confidence=float(confidence.max()),
            features_extracted=len(features),
            ensemble_details=ensemble_details,
            processing_time_ms=processing_time,
            timestamp=request.timestamp
        )

        logger.info(f"✅ Classification complete: {prediction} ({result.confidence:.3f})")
        return result

    except Exception as e:
        logger.error(f"❌ Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

@router.post("/questions/batch", response_model=BatchClassificationResponse)
async def classify_questions_batch(request: BatchQuestionRequest) -> BatchClassificationResponse:
    """
    Classify multiple questions in batch for Warren's 95-question demo.

    Optimized for processing Warren's authentic teaching video questions.
    """
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    # Initialize models if needed
    initialize_ml_models()

    try:
        logger.info(f"🚀 Batch processing {len(request.questions)} questions (session: {session_id})")

        results = []
        oeq_count = 0
        ceq_count = 0

        for i, question_req in enumerate(request.questions):
            question_start = time.time()

            # Extract features
            features = feature_extractor.extract_features(question_req.text)

            # Get ensemble prediction
            prediction = ensemble_classifier.predict([features])[0]
            confidence = ensemble_classifier.predict_proba([features])[0]

            # Get ensemble details for debugging
            ensemble_details = {
                "neural_network": {
                    "prediction": ensemble_classifier.models['neural_network'].predict([features])[0],
                    "confidence": float(ensemble_classifier.models['neural_network'].predict_proba([features])[0].max())
                },
                "random_forest": {
                    "prediction": ensemble_classifier.models['random_forest'].predict([features])[0],
                    "confidence": float(ensemble_classifier.models['random_forest'].predict_proba([features])[0].max())
                },
                "logistic_regression": {
                    "prediction": ensemble_classifier.models['logistic_regression'].predict([features])[0],
                    "confidence": float(ensemble_classifier.models['logistic_regression'].predict_proba([features])[0].max())
                }
            }

            question_time = (time.time() - question_start) * 1000

            result = QuestionClassificationResult(
                question_text=question_req.text,
                classification=prediction,
                confidence=float(confidence.max()),
                features_extracted=len(features),
                ensemble_details=ensemble_details,
                processing_time_ms=question_time,
                timestamp=question_req.timestamp
            )

            results.append(result)

            if prediction == "OEQ":
                oeq_count += 1
            else:
                ceq_count += 1

            if (i + 1) % 10 == 0:
                logger.info(f"📈 Processed {i + 1}/{len(request.questions)} questions")

        total_time = (time.time() - start_time) * 1000
        avg_confidence = sum(r.confidence for r in results) / len(results)

        summary = {
            "total_questions": len(results),
            "oeq_count": oeq_count,
            "ceq_count": ceq_count,
            "oeq_percentage": (oeq_count / len(results)) * 100,
            "ceq_percentage": (ceq_count / len(results)) * 100,
            "average_confidence": avg_confidence,
            "average_processing_time_ms": total_time / len(results),
            "ensemble_architecture": "Neural Network + Random Forest + Logistic Regression",
            "features_per_question": results[0].features_extracted if results else 0
        }

        logger.info(f"✅ Batch complete: {oeq_count} OEQ, {ceq_count} CEQ, avg confidence: {avg_confidence:.3f}")

        return BatchClassificationResponse(
            session_id=session_id,
            total_questions=len(request.questions),
            processed_questions=len(results),
            processing_time_ms=total_time,
            results=results,
            summary=summary
        )

    except Exception as e:
        logger.error(f"❌ Batch classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {e}")

@router.get("/health")
async def classification_health():
    """Health check for question classification service"""
    try:
        initialize_ml_models()
        return {
            "status": "healthy",
            "service": "Question Classification API",
            "ensemble_models": ["neural_network", "random_forest", "logistic_regression"],
            "timestamp": datetime.utcnow().isoformat(),
            "ready_for_warren_demo": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }