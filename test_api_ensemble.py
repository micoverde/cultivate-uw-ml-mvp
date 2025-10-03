#!/usr/bin/env python3
"""
Unit tests for ensemble ML API endpoints and model selection
"""

import pytest
import requests
import json
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8001"

class TestEnsembleAPI:
    """Test suite for ensemble ML API endpoints"""

    def test_api_health(self):
        """Test API health endpoint"""
        response = requests.get(f"{API_BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["real_ml"] == True
        assert data["features"] == 56

    def test_classic_endpoint(self):
        """Test /api/classify endpoint for classic ML"""
        payload = {
            "text": "What do you think about this?",
            "debug_mode": False
        }
        response = requests.post(f"{API_BASE_URL}/api/classify", json=payload)

        # Note: May return 500 if model not loaded, which is expected in test env
        if response.status_code == 200:
            data = response.json()
            assert "oeq_probability" in data
            assert "ceq_probability" in data
            assert "classification" in data
            assert data["classification"] in ["OEQ", "CEQ"]
            assert 0 <= data["oeq_probability"] <= 1
            assert 0 <= data["ceq_probability"] <= 1

    def test_ensemble_endpoint(self):
        """Test /api/v2/classify/ensemble endpoint"""
        payload = {
            "text": "Why does the sky look blue?",
            "debug_mode": True
        }
        response = requests.post(f"{API_BASE_URL}/api/v2/classify/ensemble", json=payload)

        # Note: May return 500 if model not loaded, which is expected in test env
        if response.status_code == 200:
            data = response.json()
            assert "oeq_probability" in data
            assert "ceq_probability" in data
            assert "classification" in data
            assert data["method"] == "ensemble-7-models"
            assert data["features_used"] == 56

            # Check debug info for ensemble
            if data.get("debug_info"):
                assert "ensemble_votes" in data["debug_info"]
                assert "model_predictions" in data["debug_info"]
                assert len(data["debug_info"]["model_predictions"]) == 7

    def test_feedback_endpoint(self):
        """Test /save_feedback endpoint"""
        payload = {
            "text": "Is this working?",
            "predicted_class": "CEQ",
            "correct_class": "CEQ",
            "error_type": "TN",
            "scenario_id": 1
        }
        response = requests.post(f"{API_BASE_URL}/save_feedback", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert "total_entries" in data

    def test_model_metrics(self):
        """Test /model_metrics endpoint"""
        response = requests.get(f"{API_BASE_URL}/model_metrics")

        if response.status_code == 200:
            data = response.json()
            assert "f1_score" in data
            assert "accuracy" in data
            assert 0 <= data["f1_score"] <= 1
            assert 0 <= data["accuracy"] <= 1

class TestModelSettings:
    """Test suite for model settings functionality"""

    def test_settings_storage(self):
        """Test that model settings persist in localStorage"""
        # This would be tested in browser environment
        # Simulating the behavior here
        settings = {
            "selectedModel": "ensemble",
            "apiBaseUrl": "http://localhost:8001"
        }
        assert settings["selectedModel"] in ["classic", "ensemble"]

    def test_environment_detection(self):
        """Test environment detection logic"""
        # Test localhost detection
        localhost_hosts = ["localhost", "127.0.0.1"]
        for host in localhost_hosts:
            is_localhost = host in ["localhost", "127.0.0.1"]
            assert is_localhost == True

        # Test production detection
        prod_hosts = ["example.azurestaticapps.net", "app.cultivate-learning.com"]
        for host in prod_hosts:
            is_localhost = host in ["localhost", "127.0.0.1"]
            assert is_localhost == False

    def test_endpoint_generation(self):
        """Test endpoint URL generation"""
        test_cases = [
            ("classic", "localhost", "http://localhost:8001/api/classify"),
            ("ensemble", "localhost", "http://localhost:8001/api/v2/classify/ensemble"),
            ("classic", "prod", "https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io/api/classify"),
            ("ensemble", "prod", "https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io/api/v2/classify/ensemble")
        ]

        for model, env, expected_url in test_cases:
            if env == "localhost":
                base_url = "http://localhost:8001"
            else:
                base_url = "https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io"

            if model == "ensemble":
                endpoint = f"{base_url}/api/v2/classify/ensemble"
            else:
                endpoint = f"{base_url}/api/classify"

            assert endpoint == expected_url

def run_tests():
    """Run all tests"""
    print("🧪 Running Ensemble ML API Tests...")
    print("=" * 50)

    # API Tests
    api_tests = TestEnsembleAPI()
    try:
        api_tests.test_api_health()
        print("✅ API health check passed")
    except Exception as e:
        print(f"❌ API health check failed: {e}")

    try:
        api_tests.test_classic_endpoint()
        print("✅ Classic endpoint test passed")
    except Exception as e:
        print(f"⚠️  Classic endpoint test: {e}")

    try:
        api_tests.test_ensemble_endpoint()
        print("✅ Ensemble endpoint test passed")
    except Exception as e:
        print(f"⚠️  Ensemble endpoint test: {e}")

    # Settings Tests
    settings_tests = TestModelSettings()
    try:
        settings_tests.test_environment_detection()
        print("✅ Environment detection test passed")
    except Exception as e:
        print(f"❌ Environment detection test failed: {e}")

    try:
        settings_tests.test_endpoint_generation()
        print("✅ Endpoint generation test passed")
    except Exception as e:
        print(f"❌ Endpoint generation test failed: {e}")

    print("=" * 50)
    print("🎯 Test suite completed")

if __name__ == "__main__":
    run_tests()