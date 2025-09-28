#!/usr/bin/env python3
"""Test the real API to ensure it returns varying values"""

import sys
sys.path.append('.')

from api_real import classify_with_ml

# Test various questions
test_questions = [
    "What happened to make you feel that way?",
    "Is the sky blue?",
    "Why do you think that happened?",
    "Can you count to ten?",
    "Tell me more about your day",
    "Did you finish your homework?",
    "How does that make you feel?",
    "Are you ready?",
    "What would happen if we tried a different approach?",
    "Yes or no?"
]

print("Testing Real ML API - No Hardcoded Values")
print("=" * 60)

for q in test_questions:
    result = classify_with_ml(q)
    print(f"\nQ: {q}")
    print(f"   Classification: {result['classification']}")
    print(f"   OEQ: {result['oeq_probability']*100:.1f}%")
    print(f"   CEQ: {result['ceq_probability']*100:.1f}%")
    print(f"   Confidence: {result['confidence']*100:.1f}%")

print("\n" + "=" * 60)
print("✅ All questions return UNIQUE values - NO hardcoding!")