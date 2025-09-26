#!/usr/bin/env python3
"""
DEMO 1 REAL ML API: Actual PyTorch Neural Network OEQ/CEQ Classifier
FastAPI backend with the REAL trained PyTorch model + DEBUG mode

Warren - This is the ACTUAL neural network model with 56 features!
"""

import sys
import os
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add paths for our ML modules
sys.path.insert(0, '../src/ml/training')
sys.path.insert(0, '../src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DEMO 1: REAL PyTorch ML API",
    description="Actual neural network OEQ/CEQ classification with DEBUG mode",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassificationRequest(BaseModel):
    text: str
    scenario_id: int = None
    debug_mode: bool = True  # Enable DEBUG by default

class ClassificationResponse(BaseModel):
    oeq_probability: float
    ceq_probability: float
    classification: str
    confidence: float
    method: str
    features_used: int

class OEQCEQClassifier(nn.Module):
    """Exact PyTorch model architecture from training"""
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

class FeatureExtractor:
    """Feature extraction matching the training pipeline"""

    def __init__(self):
        self.oeq_keywords = [
            'what', 'how', 'why', 'describe', 'explain', 'tell me',
            'think', 'feel', 'notice', 'see', 'observe', 'imagine',
            'wonder', 'curious', 'explore', 'discover'
        ]

        self.ceq_keywords = [
            'is', 'are', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'have', 'has', 'had'
        ]

    def extract_features(self, text: str) -> np.ndarray:
        """Extract 56 features exactly as in training"""
        text_lower = text.lower().strip()
        words = text_lower.split()

        features = []

        # 1-16: OEQ keyword counts (16 features)
        for keyword in self.oeq_keywords:
            count = text_lower.count(keyword)
            features.append(count)

        # 17-29: CEQ keyword counts (13 features)
        for keyword in self.ceq_keywords:
            count = text_lower.count(keyword)
            features.append(count)

        # 30: Word count
        features.append(len(words))

        # 31: Character count
        features.append(len(text))

        # 32: Average word length
        avg_len = np.mean([len(word) for word in words]) if words else 0
        features.append(avg_len)

        # 33: Question mark count
        features.append(text.count('?'))

        # 34: Exclamation mark count
        features.append(text.count('!'))

        # 35: Comma count
        features.append(text.count(','))

        # 36: Starts with question word (0 or 1)
        question_starters = ['what', 'how', 'why', 'when', 'where', 'who']
        starts_with_q = 1 if any(text_lower.startswith(q) for q in question_starters) else 0
        features.append(starts_with_q)

        # 37: Contains "you" (0 or 1)
        features.append(1 if 'you' in text_lower else 0)

        # 38: Contains "think" (0 or 1)
        features.append(1 if 'think' in text_lower else 0)

        # 39: Contains "feel" (0 or 1)
        features.append(1 if 'feel' in text_lower else 0)

        # 40-45: Sentence structure features (6 features)
        features.append(1 if text_lower.startswith('what') else 0)
        features.append(1 if text_lower.startswith('how') else 0)
        features.append(1 if text_lower.startswith('why') else 0)
        features.append(1 if text_lower.startswith('is') else 0)
        features.append(1 if text_lower.startswith('are') else 0)
        features.append(1 if text_lower.startswith('do') else 0)

        # 46-50: Advanced linguistic features (5 features)
        features.append(text_lower.count('do you'))
        features.append(text_lower.count('can you'))
        features.append(text_lower.count('what do you'))
        features.append(text_lower.count('how do you'))
        features.append(text_lower.count('tell me'))

        # 51-56: Context features (6 features)
        features.append(1 if 'because' in text_lower else 0)
        features.append(1 if 'maybe' in text_lower else 0)
        features.append(1 if 'might' in text_lower else 0)
        features.append(1 if any(word in text_lower for word in ['good', 'bad', 'happy', 'sad']) else 0)
        features.append(1 if any(word in text_lower for word in ['big', 'small', 'tall', 'short']) else 0)
        features.append(len([word for word in words if len(word) > 6]))  # Complex words

        return np.array(features, dtype=np.float32)

class RealMLService:
    """Real PyTorch ML model service"""

    def __init__(self):
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.load_model()

    def load_model(self):
        """Load the actual trained PyTorch model"""
        try:
            model_path = '../src/ml/trained_models/oeq_ceq_pytorch.pth'

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Initialize model
            self.model = OEQCEQClassifier()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            logger.info("🧠 REAL PyTorch model loaded successfully!")
            logger.info(f"📊 Model accuracy: {checkpoint.get('accuracy', 'Unknown')}")
            logger.info(f"📈 Training epoch: {checkpoint.get('epoch', 'Unknown')}")

        except Exception as e:
            logger.error(f"❌ Error loading PyTorch model: {e}")
            raise e

    def classify_text(self, text: str, debug_mode: bool = True) -> Dict[str, Any]:
        """Classify using REAL neural network with DEBUG output"""

        if debug_mode:
            logger.info(f"🧠 REAL NEURAL NETWORK Classification: '{text[:50]}...'")

        # Extract 56 features
        features = self.feature_extractor.extract_features(text)

        if debug_mode:
            logger.info(f"📊 Extracted {len(features)} features (neural network input)")
            logger.info(f"🔢 Feature vector sample: {features[:10].tolist()}")
            logger.info(f"🎯 Key features: word_count={features[29]}, has_question_mark={features[33]}, starts_with_what={features[40]}")

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)

        # Neural network inference
        with torch.no_grad():
            raw_outputs = self.model(features_tensor)
            probabilities = torch.softmax(raw_outputs, dim=1)

            if debug_mode:
                logger.info(f"🎯 Raw neural network outputs: {raw_outputs[0].numpy()}")
                logger.info(f"📈 Softmax probabilities: {probabilities[0].numpy()}")

        # Extract probabilities (CEQ=index 0, OEQ=index 1)
        ceq_prob = probabilities[0][0].item()
        oeq_prob = probabilities[0][1].item()

        classification = 'OEQ' if oeq_prob > ceq_prob else 'CEQ'
        confidence = abs(oeq_prob - ceq_prob)

        if debug_mode:
            logger.info(f"✅ NEURAL NETWORK RESULT: {classification} (confidence: {confidence:.3f})")
            logger.info(f"🔍 OEQ: {oeq_prob:.3f}, CEQ: {ceq_prob:.3f}")

        return {
            'oeq_probability': float(oeq_prob),
            'ceq_probability': float(ceq_prob),
            'classification': classification,
            'confidence': float(confidence),
            'method': 'pytorch-neural-network',
            'features_used': len(features),
            'debug_info': {
                'feature_vector': features.tolist(),
                'raw_outputs': raw_outputs[0].numpy().tolist(),
                'softmax_probs': probabilities[0].numpy().tolist(),
                'feature_breakdown': {
                    'oeq_keywords': features[:16].sum(),
                    'ceq_keywords': features[16:29].sum(),
                    'word_count': features[29],
                    'char_count': features[30],
                    'question_marks': features[33],
                    'starts_with_question': features[36]
                }
            }
        }

# Initialize the REAL ML service
try:
    ml_service = RealMLService()
    logger.info("🚀 REAL ML API initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize REAL ML service: {e}")
    ml_service = None

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "DEMO 1: REAL PyTorch ML API",
        "status": "running",
        "model": "pytorch-neural-network",
        "features": 56,
        "architecture": "4-layer neural network [56→64→32→16→2]",
        "real_ml": True
    }

@app.post("/classify_response", response_model=ClassificationResponse)
async def classify_response(request: ClassificationRequest):
    """Classify using REAL neural network"""

    if ml_service is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    try:
        logger.info(f"🎯 REAL ML Classification request for scenario {request.scenario_id}")

        result = ml_service.classify_text(request.text, request.debug_mode)

        return ClassificationResponse(**result)

    except Exception as e:
        logger.error(f"❌ Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": ml_service is not None,
        "model_type": "pytorch-neural-network",
        "features": 56,
        "debug_mode": True
    }

@app.get("/model_info")
async def model_info():
    """Get detailed model information"""
    if ml_service is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "architecture": "4-layer feedforward neural network",
        "input_features": 56,
        "hidden_layers": [64, 32, 16],
        "output_classes": 2,
        "activation": "ReLU + BatchNorm + Dropout",
        "training_framework": "PyTorch",
        "model_file": "oeq_ceq_pytorch.pth"
    }

if __name__ == "__main__":
    import uvicorn

    print("🧠 Starting DEMO 1 REAL PyTorch ML API")
    print("=" * 50)
    print("Model: Actual 4-layer neural network")
    print("Features: 56 linguistic features")
    print("Architecture: [56→64→32→16→2]")
    print("DEBUG mode: ENABLED")
    print("Framework: PyTorch")
    print()
    print("API will be available at: http://localhost:8001")
    print("API docs at: http://localhost:8001/docs")

    uvicorn.run(app, host="0.0.0.0", port=8001)