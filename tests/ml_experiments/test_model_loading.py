#!/usr/bin/env python3
"""Test that ML models load correctly"""

import sys
import os
import asyncio

# Add training directory to path for feature_extractor
sys.path.insert(0, 'src/ml/training')
sys.path.insert(0, 'src')

from ml.models.question_classifier import ClassicalQuestionClassifier

async def test_model():
    # Test loading with our trained model
    classifier = ClassicalQuestionClassifier(model_path='src/ml/trained_models/question_classifier.pkl')
    
    # Test prediction
    test_text = "Why do you think the circle has no sides?"
    result = await classifier.analyze(test_text)
    
    print(f"✅ Model loaded successfully!")
    print(f"Test text: '{test_text}'")
    print(f"Result: {result}")
    print(f"Using: {'Trained ML Model' if hasattr(classifier, 'model') and classifier.model else 'Fallback heuristics'}")
    
    # Test a few more examples
    test_cases = [
        "What shape is this?",
        "Can you show me how to do it?",  
        "Excellent work!",
        "How many sides does a triangle have?"
    ]
    
    print("\nAdditional test cases:")
    for text in test_cases:
        result = await classifier.analyze(text)
        print(f"  '{text}' -> {result.get('classification', result)}")

asyncio.run(test_model())
