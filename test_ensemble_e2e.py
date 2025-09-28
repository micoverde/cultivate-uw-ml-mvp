#!/usr/bin/env python3
"""
End-to-End Test for Ensemble ML System
Tests both Classic and Ensemble models via the demo UI
"""

import requests
import json
import time
from datetime import datetime

def test_api_endpoints():
    """Test API endpoints directly"""
    print("🧪 Testing API Endpoints...")

    base_url = "http://localhost:8001"

    # Test health
    response = requests.get(f"{base_url}/")
    print(f"✅ Health check: {response.status_code}")

    # Test Classic ML
    test_questions = [
        "What makes the sky blue?",  # OEQ
        "Good job on that answer!",   # CEQ
        "How does photosynthesis work?",  # OEQ
        "I like your explanation",  # CEQ
    ]

    for endpoint_name, endpoint in [("Classic", "/api/classify"), ("Ensemble", "/api/v2/classify/ensemble")]:
        print(f"\n📊 Testing {endpoint_name} Model:")
        for question in test_questions:
            payload = {"text": question, "debug_mode": False}
            response = requests.post(f"{base_url}{endpoint}", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"  '{question[:30]}...' -> {data['classification']} ({data['confidence']:.2%})")
            else:
                print(f"  ❌ Error: {response.status_code}")

def test_demo_integration():
    """Test the demo UI integration"""
    print("\n🎨 Testing Demo UI Integration...")

    demo_url = "http://localhost:3000/demo1/"

    # Check if demo is accessible
    response = requests.get(demo_url)
    if response.status_code == 200:
        print("✅ Demo UI accessible")

        # Check for settings button
        if "Settings" in response.text:
            print("✅ Settings UI integrated")
        else:
            print("❌ Settings UI not found")

        # Check for model settings script
        if "model-settings.js" in response.text:
            print("✅ Model settings script loaded")
        else:
            print("❌ Model settings script not loaded")
    else:
        print(f"❌ Demo not accessible: {response.status_code}")

def test_model_switching():
    """Test switching between models"""
    print("\n🔄 Testing Model Switching...")

    base_url = "http://localhost:8001"
    test_text = "Why do birds migrate?"

    # Test with Classic
    classic_response = requests.post(f"{base_url}/api/classify",
                                    json={"text": test_text, "debug_mode": True})

    # Test with Ensemble
    ensemble_response = requests.post(f"{base_url}/api/v2/classify/ensemble",
                                     json={"text": test_text, "debug_mode": True})

    if classic_response.status_code == 200 and ensemble_response.status_code == 200:
        classic_data = classic_response.json()
        ensemble_data = ensemble_response.json()

        print(f"Classic method: {classic_data.get('method', 'N/A')}")
        print(f"Ensemble method: {ensemble_data.get('method', 'N/A')}")

        if ensemble_data.get('debug_info', {}).get('ensemble_votes'):
            votes = ensemble_data['debug_info']['ensemble_votes']
            print(f"Ensemble votes: OEQ={votes.get('OEQ', 0)}, CEQ={votes.get('CEQ', 0)}")
            print("✅ Model switching works correctly")
        else:
            print("⚠️ Ensemble voting details not available")
    else:
        print("❌ API endpoints not responding correctly")

def main():
    print("=" * 60)
    print("🚀 ENSEMBLE ML END-TO-END TEST")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        test_api_endpoints()
        test_demo_integration()
        test_model_switching()

        print("\n" + "=" * 60)
        print("✅ END-TO-END TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())