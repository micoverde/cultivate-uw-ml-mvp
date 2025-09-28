#!/usr/bin/env python3
"""Test the feedback API endpoint"""

import requests

# Test feedback submission
api_base = "http://localhost:8000"  # Change to production URL when deployed

test_feedback = {
    "text": "What happened to make you feel that way?",
    "classification": "OEQ",
    "feedback": "correct",
    "scenario_id": 1
}

print("Testing Feedback API")
print("=" * 40)

# Build query string (API expects query params, not JSON body)
params = test_feedback

try:
    # Test submission
    response = requests.post(f"{api_base}/save_feedback", params=params)

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Feedback saved: {result}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

    # Test stats
    stats_response = requests.get(f"{api_base}/api/feedback/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        print(f"\n📊 Feedback Stats:")
        print(f"   Total: {stats['total_feedback']}")
        print(f"   Correct: {stats['correct_predictions']}")
        print(f"   Incorrect: {stats['incorrect_predictions']}")
        print(f"   Accuracy: {stats['accuracy']}%")

except Exception as e:
    print(f"❌ Connection error: {e}")
    print("Make sure the API is running with: python3 api_real.py")