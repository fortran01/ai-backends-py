"""
High-level integration tests for AI backends Flask application.

Following the coding guidelines:
- Explicit type annotations for all functions
- Focus on integration and end-to-end testing
- Test complete API workflows and user journeys
"""

import pytest
import requests
from typing import Dict, Any, Optional, List
import json
import time


class TestAPIIntegration:
    """Integration tests for Flask API endpoints."""

    BASE_URL: str = "http://localhost:5001"

    def test_health_endpoint(self) -> None:
        """Test health check endpoint availability."""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "model_info" in data

    def test_classify_basic_endpoint(self) -> None:
        """Test basic classification endpoint with valid iris data."""
        payload: Dict[str, float] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }

        response = requests.post(
            f"{self.BASE_URL}/api/v1/classify",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert data["predicted_class"] in ["setosa", "versicolor", "virginica"]
        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["confidence"] <= 1.0

    def test_classify_detailed_endpoint(self) -> None:
        """Test detailed classification with additional metadata."""
        payload: Dict[str, float] = {
            "sepal_length": 6.5,
            "sepal_width": 3.0,
            "petal_length": 5.2,
            "petal_width": 2.0
        }

        response = requests.post(
            f"{self.BASE_URL}/api/v1/classify-detailed",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        assert "predicted_class" in data
        assert "raw_probabilities" in data
        assert "feature_importances" in data
        assert "model_info" in data
        assert len(data["raw_probabilities"]) == 3  # Three iris classes

    def test_mlflow_registry_classification(self) -> None:
        """Test MLflow model registry classification endpoint."""
        payload: Dict[str, Any] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }

        response = requests.post(
            f"{self.BASE_URL}/api/v1/classify-registry?model_format=sklearn&alias=staging",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "model_registry_info" in data
        assert "model_version" in data["model_registry_info"]

    def test_drift_report_endpoint(self) -> None:
        """Test drift monitoring report endpoint."""
        response = requests.get(f"{self.BASE_URL}/api/v1/drift-report?limit=50")

        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        assert "drift_analysis" in data
        assert "recommendations" in data
        assert "data_summary" in data
        assert "drift_detected" in data["drift_analysis"]
        assert "retrain_model" in data["recommendations"]
        assert "feature_drift_scores" in data["drift_analysis"]
        assert isinstance(data["drift_analysis"]["drift_detected"], bool)
        assert isinstance(data["recommendations"]["retrain_model"], bool)
        assert "analyzed_samples" in data["data_summary"]

    def test_chat_endpoint_basic(self) -> None:
        """Test basic chat endpoint functionality."""
        payload: Dict[str, str] = {
            "prompt": "Hello, how are you?",
            "session_id": "test123"
        }

        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        assert "response" in data
        assert "session_id" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_generate_endpoint_basic(self) -> None:
        """Test basic text generation endpoint."""
        payload: Dict[str, str] = {
            "prompt": "Write a short greeting"
        }

        response = requests.post(
            f"{self.BASE_URL}/api/v1/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        # Generate endpoint returns plain text, not JSON
        generated_text: str = response.text
        assert isinstance(generated_text, str)
        assert len(generated_text.strip()) > 0

    def test_invalid_classification_data(self) -> None:
        """Test classification endpoint with invalid data."""
        payload: Dict[str, str] = {
            "invalid_field": "invalid_value"
        }

        response = requests.post(
            f"{self.BASE_URL}/api/v1/classify",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 400
        data: Dict[str, Any] = response.json()
        assert "error" in data

    def test_missing_required_fields(self) -> None:
        """Test classification endpoint with missing required fields."""
        payload: Dict[str, float] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5
            # Missing petal_length and petal_width
        }

        response = requests.post(
            f"{self.BASE_URL}/api/v1/classify",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 400
        data: Dict[str, Any] = response.json()
        assert "error" in data


class TestPerformanceAndReliability:
    """Performance and reliability tests."""

    BASE_URL: str = "http://localhost:5001"

    def test_multiple_classification_requests(self) -> None:
        """Test handling multiple classification requests."""
        payload: Dict[str, float] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }

        responses: List[requests.Response] = []
        for _ in range(5):
            response = requests.post(
                f"{self.BASE_URL}/api/v1/classify",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            responses.append(response)

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data: Dict[str, Any] = response.json()
            assert "predicted_class" in data
            assert "confidence" in data

    def test_classification_response_time(self) -> None:
        """Test classification endpoint response time."""
        payload: Dict[str, float] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }

        start_time: float = time.time()
        response = requests.post(
            f"{self.BASE_URL}/api/v1/classify",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        end_time: float = time.time()

        response_time: float = end_time - start_time

        assert response.status_code == 200
        assert response_time < 2.0  # Should respond within 2 seconds


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """Setup test environment - ensure Flask app is running."""
    # Wait for Flask app to be available
    max_retries: int = 30
    for _ in range(max_retries):
        try:
            response = requests.get("http://localhost:5001/health", timeout=5)
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
            continue
    else:
        pytest.fail("Flask application is not running on localhost:5001")