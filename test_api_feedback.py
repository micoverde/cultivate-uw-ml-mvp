#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Feedback API
Warren's requirement: TDD - Test, verify, think before deploy
"""

import pytest
import json
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the API
from api_real import app, FeedbackRequest, FeedbackV1Request

# Create test client
client = TestClient(app)

class TestClassificationAPI:
    """Test ML classification endpoints"""

    def test_classify_oeq_question(self):
        """Test OEQ classification"""
        response = client.post("/api/classify?text=What%20happened%20to%20make%20you%20feel%20that%20way%3F")
        assert response.status_code == 200
        data = response.json()
        assert "classification" in data
        assert "confidence" in data
        assert data["classification"] in ["OEQ", "CEQ"]
        # Should classify as OEQ based on "what" and "feel"
        assert data["classification"] == "OEQ"

    def test_classify_ceq_question(self):
        """Test CEQ classification"""
        response = client.post("/api/classify?text=Is%20the%20sky%20blue%3F")
        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "CEQ"

    def test_classify_returns_unique_values(self):
        """Test that different questions get different confidence values"""
        q1 = client.post("/api/classify?text=Why%20did%20that%20happen%3F").json()
        q2 = client.post("/api/classify?text=Can%20you%20count%20to%20ten%3F").json()
        q3 = client.post("/api/classify?text=Tell%20me%20about%20your%20day").json()

        # Each should have different confidence values (not hardcoded 0.85)
        confidences = [q1["confidence"], q2["confidence"], q3["confidence"]]
        assert len(set(confidences)) > 1  # At least 2 different values
        assert not all(c == 0.85 for c in confidences)  # Not all 0.85

class TestFeedbackEndpoints:
    """Test feedback submission endpoints"""

    def test_save_feedback_json_body(self):
        """Test /save_feedback with JSON body (Demo 1 format)"""
        feedback_data = {
            "text": "What makes you happy?",
            "classification": "OEQ",
            "feedback": "correct",
            "scenario_id": 1
        }

        response = client.post(
            "/save_feedback",
            json=feedback_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "feedback_id" in data
        assert "total_feedback_collected" in data

    def test_save_feedback_query_params(self):
        """Test /save_feedback with query parameters"""
        response = client.post(
            "/save_feedback?text=Test%20question&classification=OEQ&feedback=correct"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

    def test_save_feedback_v1_json_body(self):
        """Test /api/v1/feedback/save with JSON body (Demo 2 format)"""
        feedback_data = {
            "text": "Who wants to play?",
            "predicted_class": "CEQ",
            "correct_class": "CEQ",
            "confidence": 0.75,
            "error_type": "TP"
        }

        response = client.post(
            "/api/v1/feedback/save",
            json=feedback_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "feedback_id" in data

    def test_save_feedback_v1_incorrect_classification(self):
        """Test feedback when classification is incorrect"""
        feedback_data = {
            "text": "What do you think about that?",
            "predicted_class": "CEQ",
            "correct_class": "OEQ",
            "confidence": 0.65,
            "error_type": "FP"
        }

        response = client.post(
            "/api/v1/feedback/save",
            json=feedback_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

    def test_save_feedback_missing_fields(self):
        """Test feedback with missing required fields"""
        incomplete_data = {
            "text": "Test question"
            # Missing classification
        }

        response = client.post(
            "/save_feedback",
            json=incomplete_data
        )

        # Should still work with partial data
        assert response.status_code == 200

    def test_feedback_with_predicted_class(self):
        """Test feedback using predicted_class field (alternate format)"""
        feedback_data = {
            "text": "How does that work?",
            "predicted_class": "OEQ",
            "correct_class": "OEQ"
        }

        response = client.post(
            "/save_feedback",
            json=feedback_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

class TestFeedbackStats:
    """Test feedback statistics endpoint"""

    def test_get_feedback_stats(self):
        """Test /api/feedback/stats endpoint"""
        response = client.get("/api/feedback/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_feedback" in data
        assert "correct_predictions" in data
        assert "incorrect_predictions" in data
        assert "accuracy" in data
        assert data["total_feedback"] >= 0
        assert 0 <= data["accuracy"] <= 100

class TestAPIHealth:
    """Test API health and status endpoints"""

    def test_health_endpoint(self):
        """Test /health endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["real_ml"] == True  # Must be real ML, not mock

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

class TestIntegrationScenarios:
    """Test complete user scenarios"""

    def test_demo1_complete_workflow(self):
        """Test Demo 1 complete workflow: classify + feedback"""
        # Step 1: Classify a question
        text = "What makes you feel proud?"
        classify_response = client.post(f"/api/classify?text={text}")
        assert classify_response.status_code == 200

        classification = classify_response.json()
        assert classification["classification"] == "OEQ"  # Should be OEQ

        # Step 2: Submit feedback that it was correct
        feedback_data = {
            "text": text,
            "classification": classification["classification"],
            "feedback": "correct",
            "scenario_id": 1
        }

        feedback_response = client.post("/save_feedback", json=feedback_data)
        assert feedback_response.status_code == 200
        assert feedback_response.json()["success"] == True

    def test_demo2_complete_workflow(self):
        """Test Demo 2 complete workflow with error correction"""
        # Step 1: Simulate a misclassification
        text = "Tell me more about your feelings"

        # Step 2: Submit corrective feedback
        feedback_data = {
            "text": text,
            "predicted_class": "CEQ",  # Wrong prediction
            "correct_class": "OEQ",     # Correct answer
            "confidence": 0.6,
            "error_type": "FN"
        }

        response = client.post("/api/v1/feedback/save", json=feedback_data)
        assert response.status_code == 200
        assert response.json()["success"] == True

if __name__ == "__main__":
    # Run tests
    print("🧪 Running Comprehensive Feedback API Tests")
    print("=" * 60)

    # Test classification
    print("\n📊 Testing Classification Endpoints...")
    test_cls = TestClassificationAPI()
    test_cls.test_classify_oeq_question()
    print("✅ OEQ classification works")
    test_cls.test_classify_ceq_question()
    print("✅ CEQ classification works")
    test_cls.test_classify_returns_unique_values()
    print("✅ Returns unique values (not hardcoded)")

    # Test feedback
    print("\n💾 Testing Feedback Endpoints...")
    test_fb = TestFeedbackEndpoints()
    test_fb.test_save_feedback_json_body()
    print("✅ JSON body feedback works")
    test_fb.test_save_feedback_query_params()
    print("✅ Query param feedback works")
    test_fb.test_save_feedback_v1_json_body()
    print("✅ Demo 2 v1 feedback works")
    test_fb.test_save_feedback_v1_incorrect_classification()
    print("✅ Incorrect classification feedback works")
    test_fb.test_feedback_with_predicted_class()
    print("✅ Alternate field names work")

    # Test stats
    print("\n📈 Testing Statistics...")
    test_stats = TestFeedbackStats()
    test_stats.test_get_feedback_stats()
    print("✅ Feedback stats work")

    # Test health
    print("\n🏥 Testing Health Endpoints...")
    test_health = TestAPIHealth()
    test_health.test_health_endpoint()
    print("✅ Health endpoint works")
    test_health.test_root_endpoint()
    print("✅ Root endpoint works")

    # Test integration
    print("\n🔄 Testing Complete Workflows...")
    test_int = TestIntegrationScenarios()
    test_int.test_demo1_complete_workflow()
    print("✅ Demo 1 workflow complete")
    test_int.test_demo2_complete_workflow()
    print("✅ Demo 2 workflow complete")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Ready for deployment!")
    print("=" * 60)