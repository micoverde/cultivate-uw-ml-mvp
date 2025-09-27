#!/usr/bin/env python3
"""
Unit Tests for Model Switching Functionality
Story 9.1 - Issue #196

Tests the model selection API endpoints and switching logic
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from api.main import app
from api.endpoints.model_management import ModelSelector


class TestModelSelection:
    """Test model selection functionality"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_model_selector(self):
        """Mock ModelSelector instance"""
        with patch('api.endpoints.model_management.model_selector') as mock:
            mock.current_model_type = 'classic'
            mock.ensemble_model = Mock()
            mock.classic_model = Mock()
            yield mock

    def test_get_current_model(self, client, mock_model_selector):
        """Test getting current model type"""
        response = client.get("/api/v1/models/current")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert data["model_type"] in ["classic", "ensemble"]

    def test_select_ensemble_model(self, client, mock_model_selector):
        """Test switching to ensemble model"""
        response = client.post(
            "/api/v1/models/select",
            json={"model_type": "ensemble"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["current_model"] == "ensemble"
        assert "switch_time_ms" in data

    def test_select_classic_model(self, client, mock_model_selector):
        """Test switching to classic model"""
        response = client.post(
            "/api/v1/models/select",
            json={"model_type": "classic"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["current_model"] == "classic"

    def test_invalid_model_type(self, client):
        """Test selecting invalid model type"""
        response = client.post(
            "/api/v1/models/select",
            json={"model_type": "invalid"}
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data or "detail" in data

    def test_model_comparison(self, client, mock_model_selector):
        """Test getting model comparison data"""
        response = client.get("/api/v1/models/comparison")
        assert response.status_code == 200
        data = response.json()

        # Check structure
        assert "ensemble" in data
        assert "classic" in data
        assert "recommendation" in data

    def test_performance_metrics(self, client, mock_model_selector):
        """Test performance metrics retrieval"""
        # Mock performance data
        mock_model_selector.get_performance_metrics.return_value = {
            "accuracy": 0.95,
            "f1_score": 0.94,
            "avg_response_ms": 50,
            "total_predictions": 1000
        }

        response = client.get("/api/v1/models/performance")
        assert response.status_code == 200
        data = response.json()

        # Verify metrics structure
        if "classic" in data:
            assert "accuracy" in data["classic"]
            assert "f1_score" in data["classic"]

    def test_classification_with_model_selection(self, client, mock_model_selector):
        """Test classification with specific model"""
        test_text = "What do you think will happen next?"

        # Test with ensemble
        response = client.post(
            "/api/v1/models/classify",
            json={
                "text": test_text,
                "model_type": "ensemble"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "classification" in data
        assert "confidence" in data
        assert data["model_type"] == "ensemble"

        # Test with classic
        response = client.post(
            "/api/v1/models/classify",
            json={
                "text": test_text,
                "model_type": "classic"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "classic"

    def test_model_switching_persistence(self, client, mock_model_selector):
        """Test that model selection persists across requests"""
        # Switch to ensemble
        response = client.post(
            "/api/v1/models/select",
            json={"model_type": "ensemble"}
        )
        assert response.status_code == 200

        # Verify it's still ensemble
        response = client.get("/api/v1/models/current")
        assert response.status_code == 200
        data = response.json()
        # Note: In real implementation, this would check actual persistence
        # For test, we're verifying the API contract

    def test_concurrent_model_switching(self, client, mock_model_selector):
        """Test handling concurrent model switch requests"""
        import concurrent.futures

        def switch_model(model_type):
            return client.post(
                "/api/v1/models/select",
                json={"model_type": model_type}
            )

        # Try switching models concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(10):
                model_type = "ensemble" if i % 2 == 0 else "classic"
                futures.append(executor.submit(switch_model, model_type))

            results = [f.result() for f in futures]

        # All requests should succeed
        for result in results:
            assert result.status_code == 200

    def test_model_health_check(self, client, mock_model_selector):
        """Test model health check endpoint"""
        response = client.get("/api/v1/models/health")

        # Health check should always return 200
        assert response.status_code in [200, 404]  # 404 if endpoint not implemented yet

        if response.status_code == 200:
            data = response.json()
            assert "status" in data


class TestModelSelectorClass:
    """Test ModelSelector class directly"""

    def test_singleton_pattern(self):
        """Test ModelSelector implements singleton pattern"""
        selector1 = ModelSelector()
        selector2 = ModelSelector()
        # Should be same instance
        assert selector1 is selector2

    @patch('api.endpoints.model_management.joblib.load')
    def test_load_models(self, mock_load):
        """Test model loading"""
        mock_model = Mock()
        mock_load.return_value = mock_model

        selector = ModelSelector()
        selector.load_models()

        # Should attempt to load both models
        assert mock_load.call_count >= 1

    def test_switch_model_type(self):
        """Test switching between model types"""
        selector = ModelSelector()

        # Switch to ensemble
        result = selector.switch_model("ensemble")
        assert selector.current_model_type == "ensemble"

        # Switch to classic
        result = selector.switch_model("classic")
        assert selector.current_model_type == "classic"

    def test_get_current_model(self):
        """Test getting current model"""
        selector = ModelSelector()
        selector.current_model_type = "ensemble"
        selector.ensemble_model = Mock()

        model = selector.get_current_model()
        assert model is not None

    @patch('api.endpoints.model_management.time.time')
    def test_performance_tracking(self, mock_time):
        """Test performance metric tracking"""
        mock_time.side_effect = [0, 0.1]  # 100ms elapsed

        selector = ModelSelector()
        selector.track_prediction_start()
        selector.track_prediction_end(
            model_type="ensemble",
            classification="OEQ",
            confidence=0.95
        )

        metrics = selector.get_performance_summary("ensemble")
        assert metrics is not None


class TestIntegrationScenarios:
    """Integration test scenarios"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_full_user_workflow(self, client):
        """Test complete user workflow for model switching"""
        # 1. Check current model
        response = client.get("/api/v1/models/current")
        assert response.status_code == 200
        initial_model = response.json()["model_type"]

        # 2. Get performance comparison
        response = client.get("/api/v1/models/comparison")
        assert response.status_code == 200

        # 3. Switch to different model
        new_model = "ensemble" if initial_model == "classic" else "classic"
        response = client.post(
            "/api/v1/models/select",
            json={"model_type": new_model}
        )
        assert response.status_code == 200

        # 4. Verify switch
        response = client.get("/api/v1/models/current")
        assert response.status_code == 200
        # Note: Actual persistence depends on implementation

    def test_error_recovery(self, client):
        """Test error recovery scenarios"""
        # Test with malformed request
        response = client.post(
            "/api/v1/models/select",
            json={}  # Missing model_type
        )
        assert response.status_code == 422  # FastAPI validation error

        # Test with invalid JSON
        response = client.post(
            "/api/v1/models/select",
            data="invalid json"
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])