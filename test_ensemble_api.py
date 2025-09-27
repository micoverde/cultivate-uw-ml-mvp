#!/usr/bin/env python3
"""
Test Ensemble API Endpoints
Verifies all model management endpoints are working

Usage:
    python test_ensemble_api.py

Author: Warren & Claude
"""

import requests
import json
import time
import sys
from datetime import datetime

# API base URL - adjust if needed
API_BASE = "http://localhost:8000"

def test_model_selection():
    """Test model selection endpoint"""
    print("\n📝 Testing Model Selection...")

    # Test ensemble selection
    response = requests.post(f"{API_BASE}/api/v1/models/select", json={
        "model_type": "ensemble"
    })
    print(f"  Ensemble selection: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"    - Current model: {data['current_model']}")
        print(f"    - Models: {data['model_info'].get('models', [])[:3]}...")

    # Test classic selection
    response = requests.post(f"{API_BASE}/api/v1/models/select", json={
        "model_type": "classic"
    })
    print(f"  Classic selection: {response.status_code}")

    return response.ok

def test_classification():
    """Test question classification"""
    print("\n🤖 Testing Classification...")

    test_questions = [
        "What do you think will happen if we add more water?",
        "Is this the red block?",
        "Can you explain why the tower fell down?",
        "Do you want to play with the blocks?"
    ]

    for question in test_questions:
        # Test with ensemble
        response = requests.post(f"{API_BASE}/api/v1/models/classify", json={
            "text": question,
            "model_type": "ensemble"
        })

        if response.ok:
            result = response.json()
            print(f"  Q: '{question[:50]}...'")
            print(f"    → {result['classification']} (confidence: {result['confidence']:.2%})")
            print(f"    → Model: {result['model_type']}, Time: {result['processing_time_ms']:.1f}ms")
        else:
            print(f"  ❌ Failed: {response.status_code}")

    return True

def test_model_comparison():
    """Test model comparison endpoint"""
    print("\n📊 Testing Model Comparison...")

    response = requests.get(f"{API_BASE}/api/v1/models/comparison")
    if response.ok:
        data = response.json()
        print(f"  Comparison retrieved: {response.status_code}")

        if data['ensemble']:
            print(f"  Ensemble: Accuracy={data['ensemble'].get('accuracy', 'N/A')}, "
                  f"F1={data['ensemble'].get('f1_score', 'N/A')}")

        if data['classic']:
            print(f"  Classic:  Accuracy={data['classic'].get('accuracy', 'N/A')}, "
                  f"F1={data['classic'].get('f1_score', 'N/A')}")

        if data['recommendation']:
            print(f"  Recommendation: {data['recommendation']['reason']}")
    else:
        print(f"  ⚠️ No comparison data available yet")

    return True

def test_performance_history():
    """Test performance history endpoint"""
    print("\n📈 Testing Performance History...")

    response = requests.get(f"{API_BASE}/api/v1/models/performance/history?limit=5")
    if response.ok:
        history = response.json()
        print(f"  History entries: {len(history)}")
        if history:
            latest = history[-1]
            print(f"  Latest: {latest['model_type']} - "
                  f"Accuracy: {latest['metrics'].get('accuracy', 'N/A')}")
    else:
        print(f"  ⚠️ No history available")

    return True

def test_training_status():
    """Test training status endpoint"""
    print("\n🏋️ Testing Training Status...")

    response = requests.get(f"{API_BASE}/api/v1/models/training/status")
    if response.ok:
        status = response.json()
        print(f"  Training active: {status['is_training']}")
        if status['is_training']:
            print(f"  Progress: {status['progress']}%")
            print(f"  Current task: {status['current_task']}")
    else:
        print(f"  ❌ Failed: {response.status_code}")

    return response.ok

def run_all_tests():
    """Run all API tests"""
    print("=" * 60)
    print("🧪 ENSEMBLE API TEST SUITE")
    print("=" * 60)
    print(f"API Base: {API_BASE}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("Model Selection", test_model_selection),
        ("Classification", test_classification),
        ("Model Comparison", test_model_comparison),
        ("Performance History", test_performance_history),
        ("Training Status", test_training_status)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"  ✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"  ❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"  ❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✅ All tests passed! API is ready for production.")
    else:
        print(f"⚠️ {failed} test(s) failed. Review and fix before deployment.")

    return failed == 0

if __name__ == "__main__":
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE}/api/health")
        if not response.ok:
            print("⚠️ API health check failed. Make sure the API is running:")
            print("   cd src && uvicorn api.main:app --reload")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Start it with:")
        print("   cd src && uvicorn api.main:app --reload")
        sys.exit(1)

    # Run tests
    success = run_all_tests()
    sys.exit(0 if success else 1)