"""
ML Package for Educational Analysis
Real-time educator coaching with tiered architecture supporting demo reliability and future AR deployment.

Author: Claude-4 (Partner-Level Microsoft SDE)
Issue: #47 - Story 2.2: Question Quality Analysis Implementation
"""

from .inference.ml_inference_pipeline import (
    MLInferencePipeline,
    InferenceTier,
    get_inference_pipeline,
    ml_analyze_transcript,
    class_score_transcript
)

from .models.question_classifier import QuestionClassifier
from .models.wait_time_detector import WaitTimeDetector
from .models.class_scorer import CLASSFrameworkScorer

__version__ = "1.0.0"
__all__ = [
    "MLInferencePipeline",
    "InferenceTier",
    "get_inference_pipeline",
    "ml_analyze_transcript",
    "class_score_transcript",
    "QuestionClassifier",
    "WaitTimeDetector",
    "CLASSFrameworkScorer"
]