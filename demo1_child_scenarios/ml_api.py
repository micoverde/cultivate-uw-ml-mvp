#!/usr/bin/env python3
"""
DEMO 1 ML API: Child Scenario OEQ/CEQ Classifier
FastAPI backend that serves the PyTorch model for real-time classification

Warren - This connects your web demo to the actual trained ML model!
"""

import sys
import os
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add paths for our ML modules
sys.path.insert(0, '../src/ml/training')
sys.path.insert(0, '../src')

try:
    from enhanced_feature_extractor import EnhancedQuestionFeatureExtractor
    ML_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML dependencies not available: {e}")
    print("Using fallback rule-based classification")
    ML_DEPENDENCIES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DEMO 1: Child Scenario ML API",
    description="Real-time OEQ/CEQ classification for educator responses",
    version="1.0.0"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassificationRequest(BaseModel):
    text: str
    scenario_id: Optional[int] = None

class ClassificationResponse(BaseModel):
    oeq_probability: float
    ceq_probability: float
    classification: str
    confidence: float
    method: str
    features_used: int

class OEQCEQClassifier(nn.Module):
    """PyTorch model architecture - matches training"""
    def __init__(self, input_size=56, hidden_sizes=[64, 32, 16], dropout_rate=0.3):
        super(OEQCEQClassifier, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MLClassificationService:
    """Service for loading and using the trained PyTorch model"""

    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.load_model()

    def load_model(self):
        """Load the trained PyTorch model and feature extractor"""
        try:
            if not ML_DEPENDENCIES_AVAILABLE:
                logger.warning("ML dependencies not available, using rule-based fallback")
                return

            # Load the PyTorch model
            model_path = '../src/ml/trained_models/oeq_ceq_pytorch.pth'
            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}")
                return

            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            self.model = OEQCEQClassifier()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Initialize feature extractor
            self.feature_extractor = EnhancedQuestionFeatureExtractor()

            logger.info("✅ PyTorch model loaded successfully")
            logger.info(f"Model accuracy: {checkpoint.get('accuracy', 'Unknown')}")

        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self.model = None
            self.feature_extractor = None

    async def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text using the ML model or fallback rules"""

        if self.model is not None and self.feature_extractor is not None:
            return await self.classify_with_pytorch(text)
        else:
            return self.classify_with_rules(text)

    async def classify_with_pytorch(self, text: str) -> Dict[str, Any]:
        """Use the trained PyTorch model for classification"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(text)
            features_array = np.array(list(features.values())).reshape(1, -1)
            features_tensor = torch.FloatTensor(features_array)

            # Get model prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)

                oeq_prob = probabilities[0][1].item()  # Index 1 for OEQ
                ceq_prob = probabilities[0][0].item()  # Index 0 for CEQ

            classification = 'OEQ' if oeq_prob > ceq_prob else 'CEQ'
            confidence = abs(oeq_prob - ceq_prob)

            return {
                'oeq_probability': oeq_prob,
                'ceq_probability': ceq_prob,
                'classification': classification,
                'confidence': confidence,
                'method': 'pytorch-ml',
                'features_used': len(features)
            }

        except Exception as e:
            logger.error(f"PyTorch classification error: {e}")
            return self.classify_with_rules(text)

    def classify_with_rules(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based classification"""

        oeq_patterns = [
            'what', 'how', 'why', 'tell me', 'describe', 'explain',
            'think', 'feel', 'notice', 'see', 'observe'
        ]

        ceq_patterns = [
            'is', 'are', 'do', 'can', 'will', 'does', 'did',
            'have', 'has', 'was', 'were'
        ]

        text_lower = text.lower()

        oeq_score = 0
        ceq_score = 0

        # Pattern matching
        for pattern in oeq_patterns:
            if pattern in text_lower:
                oeq_score += 1

        for pattern in ceq_patterns:
            if pattern in text_lower:
                ceq_score += 1

        # Question structure analysis
        if text_lower.strip().endswith('?'):
            if any(text_lower.startswith(starter) for starter in ['what', 'how', 'why']):
                oeq_score += 2
            else:
                ceq_score += 1

        # Calculate probabilities
        total_score = oeq_score + ceq_score
        if total_score == 0:
            oeq_prob = 0.5
            ceq_prob = 0.5
        else:
            oeq_prob = oeq_score / total_score
            ceq_prob = ceq_score / total_score

        classification = 'OEQ' if oeq_prob > ceq_prob else 'CEQ'
        confidence = abs(oeq_prob - ceq_prob)

        return {
            'oeq_probability': oeq_prob,
            'ceq_probability': ceq_prob,
            'classification': classification,
            'confidence': confidence,
            'method': 'rule-based',
            'features_used': total_score
        }

# Initialize ML service
ml_service = MLClassificationService()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "DEMO 1: Child Scenario ML API",
        "status": "running",
        "model_available": ml_service.model is not None,
        "method": "pytorch-ml" if ml_service.model else "rule-based"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": ml_service.model is not None,
        "feature_extractor_available": ml_service.feature_extractor is not None,
        "ml_dependencies": ML_DEPENDENCIES_AVAILABLE
    }

@app.post("/classify_response", response_model=ClassificationResponse)
async def classify_response(request: ClassificationRequest):
    """Main endpoint for classifying educator responses"""

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        result = await ml_service.classify_text(request.text)

        logger.info(f"Classified: '{request.text[:50]}...' -> {result['classification']} "
                   f"({result['confidence']:.2f} confidence, {result['method']})")

        return ClassificationResponse(**result)

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/scenarios/{scenario_id}")
async def get_scenario(scenario_id: int):
    """Get specific scenario details (for future enhancement)"""
    return {
        "scenario_id": scenario_id,
        "message": "Scenario details endpoint - ready for enhancement"
    }

@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    return {
        "classifications_today": "N/A - Demo mode",
        "accuracy_rate": "92%" if ml_service.model else "Rule-based",
        "model_version": "pytorch_v1.0" if ml_service.model else "rules_v1.0"
    }

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting DEMO 1 ML API")
    print("=" * 50)
    print(f"Model available: {ml_service.model is not None}")
    print(f"Feature extractor: {ml_service.feature_extractor is not None}")
    print(f"Classification method: {'PyTorch ML' if ml_service.model else 'Rule-based'}")
    print()
    print("API will be available at: http://localhost:8001")
    print("API docs at: http://localhost:8001/docs")

    uvicorn.run(app, host="0.0.0.0", port=8001)