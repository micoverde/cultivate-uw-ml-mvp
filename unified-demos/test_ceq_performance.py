#!/usr/bin/env python3
"""
Test CEQ performance issue with "Did the tower fall" question
Warren reported ensemble model has very bad CEQ performance
"""

import requests
import json
import time

def test_classification(text, model_type, environment="local"):
    """Test classification with specified model and environment"""

    # Set up endpoints based on environment and model
    if environment == "local":
        base_url = "http://localhost:8001"
        if model_type == "classic":
            endpoint = f"{base_url}/classify_response"
        else:  # ensemble
            endpoint = f"{base_url}/api/v2/classify/ensemble"
    else:  # azure
        base_url = "https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io"
        if model_type == "classic":
            endpoint = f"{base_url}/api/classify"
        else:  # ensemble
            endpoint = f"{base_url}/api/v2/classify/ensemble"

    print(f"\n{'='*60}")
    print(f"Testing: {model_type.upper()} model on {environment.upper()}")
    print(f"Endpoint: {endpoint}")
    print(f"Text: '{text}'")
    print(f"{'='*60}")

    try:
        # Make request
        start_time = time.time()

        # Different request formats for different endpoints
        if "classify_response" in endpoint:
            # Legacy local classic endpoint
            response = requests.post(endpoint, json={
                "text": text,
                "scenario_id": 1,
                "debug_mode": True
            })
        elif "/api/classify" in endpoint and environment == "azure" and model_type == "classic":
            # Azure classic uses query parameter
            response = requests.post(f"{endpoint}?text={text}")
        else:
            # Standard JSON body for ensemble and other endpoints
            response = requests.post(endpoint, json={
                "text": text,
                "debug": True
            })

        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

        if response.status_code == 200:
            data = response.json()

            # Extract classification result
            classification = data.get('classification', data.get('predicted_class', 'Unknown'))
            confidence = data.get('confidence', data.get('confidence_score', 0))

            print(f"✅ Success!")
            print(f"📊 Classification: {classification}")
            print(f"💯 Confidence: {confidence:.2%}")
            print(f"⏱️  Response time: {elapsed_time:.0f}ms")

            # If ensemble, show voting details
            if model_type == "ensemble" and 'debug_info' in data:
                debug_info = data['debug_info']
                if 'ensemble_votes' in debug_info:
                    votes = debug_info['ensemble_votes']
                    print(f"🗳️  Ensemble votes: OEQ={votes.get('OEQ', 0)}, CEQ={votes.get('CEQ', 0)}")
                if 'individual_predictions' in debug_info:
                    print(f"\n📋 Individual model predictions:")
                    for model, pred in debug_info['individual_predictions'].items():
                        print(f"   - {model}: {pred}")

            # Check for feature extraction if available
            if 'features' in data:
                print(f"\n🔍 Features extracted: {len(data['features'])} features")

            return classification, confidence, data

        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None, None, None

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return None, None, None


def main():
    """Test CEQ performance issue"""

    test_questions = [
        "Did the tower fall?",  # Warren's specific test case
        "What happened to the tower?",  # More open-ended CEQ
        "Can you build a tower?",  # Should be OEQ
        "Why did the tower fall down?",  # CEQ with reasoning
        "How tall was your tower?",  # CEQ about measurement
    ]

    print("\n" + "="*60)
    print("🔬 TESTING CEQ PERFORMANCE ISSUE")
    print("="*60)

    # Test in local environment first (if available)
    print("\n📍 TESTING LOCAL ENVIRONMENT")

    for question in test_questions:
        print(f"\n{'─'*60}")
        print(f"Question: '{question}'")

        # Test classic model
        classic_result, classic_conf, _ = test_classification(question, "classic", "local")

        # Test ensemble model
        ensemble_result, ensemble_conf, ensemble_data = test_classification(question, "ensemble", "local")

        # Compare results
        if classic_result and ensemble_result:
            print(f"\n📊 COMPARISON:")
            print(f"   Classic:  {classic_result} ({classic_conf:.1%})")
            print(f"   Ensemble: {ensemble_result} ({ensemble_conf:.1%})")

            if classic_result != ensemble_result:
                print(f"   ⚠️  MODELS DISAGREE!")

            # Special analysis for "Did the tower fall?"
            if question == "Did the tower fall?":
                print(f"\n🎯 CRITICAL TEST CASE ANALYSIS:")
                print(f"   Expected: CEQ (Closed-Ended Question)")
                print(f"   Classic got:  {classic_result}")
                print(f"   Ensemble got: {ensemble_result}")

                if ensemble_result != "CEQ":
                    print(f"   ❌ ENSEMBLE FAILED TO IDENTIFY CEQ!")
                    if ensemble_data and 'debug_info' in ensemble_data:
                        print(f"   📊 Voting pattern suggests model bias")

    print("\n" + "="*60)
    print("💡 ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()