#!/usr/bin/env python3
"""
DEMO 1 ML API: Simple Rule-Based OEQ/CEQ Classifier
FastAPI backend with enhanced rule-based classification and debug logging

Warren - This gives you a working API while PyTorch installs!
"""

import logging
import re
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from human_feedback_storage import HumanFeedbackStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DEMO 1: Child Scenario ML API",
    description="Enhanced rule-based OEQ/CEQ classification for educator responses",
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
    scenario_id: int = None

class ClassificationResponse(BaseModel):
    oeq_probability: float
    ceq_probability: float
    classification: str
    confidence: float
    method: str
    features_used: int

class EnhancedRuleBased:
    """Enhanced rule-based classifier with detailed analysis"""

    def __init__(self):
        # Comprehensive OEQ patterns
        self.oeq_patterns = [
            r'\bwhat\s+do\s+you\s+(think|see|notice|feel|observe)',
            r'\bhow\s+(does|do|did|would|could)',
            r'\bwhy\s+(do|did|would|might)',
            r'\btell\s+me\s+(about|what)',
            r'\bdescribe\s+',
            r'\bexplain\s+',
            r'\bwhat\s+happens?\s+(when|if)',
            r'\bwhat\s+would\s+happen',
            r'\bhow\s+could\s+we',
            r'\bwhat\s+if\b',
            r'\bwhat\s+makes?\s+you',
            r'\bhow\s+did\s+that\s+make\s+you\s+feel',
            r'\bwhat\s+are\s+you\s+thinking',
            r'\bcan\s+you\s+tell\s+me\s+more',
            r'\bwhat\s+else\s+do\s+you\s+notice',
            r'\bwhat\s+reminds?\s+you',
            r'\bhow\s+is\s+this\s+similar',
            r'\bwhat\s+patterns?\s+do\s+you\s+see'
        ]

        # Strong CEQ patterns
        self.ceq_patterns = [
            r'\bis\s+(this|that|it)\s+',
            r'\bare\s+(you|they|we)\s+',
            r'\bdo\s+you\s+(like|want|have|see)\s+',
            r'\bcan\s+you\s+(see|find|count)\s+',
            r'\bdid\s+you\s+(have|make|do|eat)\s+',
            r'\bwill\s+you\s+',
            r'\bhave\s+you\s+(ever|been|seen|done)',
            r'\bshould\s+we\s+',
            r'\bwould\s+you\s+like\s+to\s+',
            r'^(yes|no)\s+or\s+(yes|no)',
            r'\bwhich\s+one\s+',
            r'\bis\s+your\s+favorite',
            r'\bwhere\s+is\s+the\s+',
            r'\bwhen\s+did\s+you\s+',
            r'\bhow\s+many\s+',
            r'\bhow\s+old\s+',
            r'\bwhat\s+color\s+',
            r'\bwhat\s+time\s+'
        ]

        # Question starters that lean OEQ
        self.oeq_starters = [
            'what do you think', 'how does', 'why do you',
            'tell me about', 'describe', 'explain',
            'what happens when', 'how could we', 'what if'
        ]

        # Question starters that lean CEQ
        self.ceq_starters = [
            'is this', 'are you', 'do you like', 'can you see',
            'did you', 'will you', 'have you', 'should we'
        ]

    def classify_text(self, text: str) -> Dict[str, Any]:
        """Enhanced rule-based classification with detailed analysis"""

        logger.info(f"🔍 Analyzing text: '{text[:50]}...'")

        text_lower = text.lower().strip()

        # Initialize scoring
        oeq_score = 0.0
        ceq_score = 0.0
        features_found = []

        # Pattern matching with weights
        for pattern in self.oeq_patterns:
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                weight = 2.0 if 'think' in pattern or 'feel' in pattern else 1.5
                oeq_score += matches * weight
                features_found.append(f"OEQ pattern: {pattern[:20]}...")
                logger.info(f"  ✅ OEQ pattern match: {pattern[:30]}... (weight: {weight})")

        for pattern in self.ceq_patterns:
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                weight = 1.8 if 'is' in pattern or 'are' in pattern else 1.4
                ceq_score += matches * weight
                features_found.append(f"CEQ pattern: {pattern[:20]}...")
                logger.info(f"  ❌ CEQ pattern match: {pattern[:30]}... (weight: {weight})")

        # Starter phrase analysis
        for starter in self.oeq_starters:
            if text_lower.startswith(starter):
                oeq_score += 2.5
                features_found.append(f"OEQ starter: {starter}")
                logger.info(f"  🎯 Strong OEQ starter: {starter}")
                break

        for starter in self.ceq_starters:
            if text_lower.startswith(starter):
                ceq_score += 2.5
                features_found.append(f"CEQ starter: {starter}")
                logger.info(f"  🎯 Strong CEQ starter: {starter}")
                break

        # Question mark analysis
        if '?' in text:
            features_found.append("Question mark present")
        else:
            # Statements might be commands/prompts (often OEQ)
            oeq_score += 0.5
            features_found.append("Statement form (slight OEQ bias)")

        # Word count analysis
        word_count = len(text.split())
        if word_count > 6:
            oeq_score += 0.3  # Longer questions often more open-ended
            features_found.append("Long question (OEQ bias)")
        elif word_count <= 3:
            ceq_score += 0.3  # Very short questions often yes/no
            features_found.append("Short question (CEQ bias)")

        # Calculate probabilities
        total_score = oeq_score + ceq_score
        if total_score == 0:
            # Default for ambiguous cases
            oeq_prob = 0.55
            ceq_prob = 0.45
            confidence = 0.1
        else:
            oeq_prob = oeq_score / total_score
            ceq_prob = ceq_score / total_score
            confidence = abs(oeq_prob - ceq_prob)

        # Ensure probabilities sum to 1
        total_prob = oeq_prob + ceq_prob
        oeq_prob = oeq_prob / total_prob
        ceq_prob = ceq_prob / total_prob

        classification = 'OEQ' if oeq_prob > ceq_prob else 'CEQ'

        logger.info(f"📊 Final scores: OEQ={oeq_score:.2f}, CEQ={ceq_score:.2f}")
        logger.info(f"✅ Classification: {classification} (confidence: {confidence:.3f})")
        logger.info(f"🔍 Features found: {len(features_found)}")

        return {
            'oeq_probability': float(oeq_prob),
            'ceq_probability': float(ceq_prob),
            'classification': classification,
            'confidence': float(confidence),
            'method': 'enhanced-rule-based',
            'features_used': len(features_found),
            'debug_info': {
                'raw_scores': {'oeq': oeq_score, 'ceq': ceq_score},
                'features_found': features_found[:5],  # Limit for readability
                'word_count': word_count,
                'has_question_mark': '?' in text
            }
        }

# Initialize classifier and feedback storage
classifier = EnhancedRuleBased()
feedback_storage = HumanFeedbackStorage()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "DEMO 1: Child Scenario ML API",
        "status": "running",
        "model": "enhanced-rule-based",
        "version": "1.0.0"
    }

@app.post("/classify_response", response_model=ClassificationResponse)
async def classify_response(request: ClassificationRequest):
    """Classify educator response as OEQ or CEQ"""

    try:
        logger.info(f"🎯 Classification request for scenario {request.scenario_id}")

        result = classifier.classify_text(request.text)

        return ClassificationResponse(**result)

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "classifier": "enhanced-rule-based",
        "features": ["pattern_matching", "starter_analysis", "length_analysis", "debug_logging"]
    }

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "oeq_patterns": len(classifier.oeq_patterns),
        "ceq_patterns": len(classifier.ceq_patterns),
        "oeq_starters": len(classifier.oeq_starters),
        "ceq_starters": len(classifier.ceq_starters)
    }

# Human Feedback API Endpoints
class FeedbackRequest(BaseModel):
    scenario_id: int
    user_response: str
    ml_prediction: str
    ml_confidence: float
    human_label: str
    session_id: str = None
    additional_notes: str = None

@app.post("/submit_feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit human feedback for ML model training"""
    try:
        feedback_id = feedback_storage.store_feedback(
            scenario_id=request.scenario_id,
            user_response=request.user_response,
            ml_prediction=request.ml_prediction,
            ml_confidence=request.ml_confidence,
            human_label=request.human_label,
            session_id=request.session_id,
            additional_notes=request.additional_notes
        )

        logger.info(f"📝 Feedback stored with ID: {feedback_id}")

        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Feedback stored successfully"
        }
    except Exception as e:
        logger.error(f"Feedback storage error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {str(e)}")

@app.get("/feedback_summary")
async def get_feedback_summary():
    """Get summary of all human feedback"""
    try:
        summary = feedback_storage.get_feedback_summary()
        return summary
    except Exception as e:
        logger.error(f"Feedback summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback summary: {str(e)}")

@app.get("/training_data")
async def get_training_data():
    """Get all feedback data for model retraining"""
    try:
        training_data = feedback_storage.get_training_data()
        return {
            "data": training_data,
            "total_records": len(training_data)
        }
    except Exception as e:
        logger.error(f"Training data error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training data: {str(e)}")

@app.post("/export_training_data")
async def export_training_data(format: str = "json"):
    """Export training data to file"""
    try:
        filename = feedback_storage.export_for_retraining(format=format)
        return {
            "success": True,
            "filename": filename,
            "message": f"Training data exported to {filename}"
        }
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export training data: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    print("🎭 Starting DEMO 1 Enhanced Rule-Based ML API")
    print("=" * 50)
    print("Classification method: Enhanced rule-based")
    print("Debug logging: Enabled")
    print("Pattern matching: Advanced")
    print("Confidence scoring: Yes")
    print()
    print("API will be available at: http://localhost:8001")
    print("API docs at: http://localhost:8001/docs")

    uvicorn.run(app, host="0.0.0.0", port=8001)