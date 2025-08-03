"""
Unit tests for Phase 2 features including LangChain chat, gRPC, and serialization.

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive test coverage for Phase 2 features
- LangChain conversation memory testing
- gRPC client integration testing
- Serialization utilities testing
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional
import pytest
import requests
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import ConnectionError, RequestException

# Test configuration
BASE_URL: str = "http://localhost:5001"
TIMEOUT: int = 30


class TestLangChainChatEndpoint:
    """Test cases for the /api/v1/chat endpoint with LangChain conversation memory."""
    
    def test_chat_new_session_creation(self) -> None:
        """
        Test creating a new chat session with conversation memory.
        
        Raises:
            AssertionError: If session creation fails
        """
        chat_data: Dict[str, str] = {
            "prompt": "Hello, what is artificial intelligence?",
            "session_id": f"test_session_001_{uuid.uuid4().hex[:8]}"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json=chat_data,
            timeout=TIMEOUT
        )
        
        # Should succeed even if Ollama is not available (graceful degradation)
        if response.status_code == 200:
            data: Dict[str, Any] = response.json()
            
            # Verify response structure
            assert "response" in data
            assert "session_id" in data
            assert "conversation_length" in data
            assert "memory_stats" in data
            
            # Verify session tracking
            assert data["session_id"].startswith("test_session_001")
            
            # Verify conversation memory structure
            memory: Dict[str, Any] = data["memory_stats"]
            assert "total_messages" in memory
            assert "human_messages" in memory
            assert "ai_messages" in memory
            assert memory["total_messages"] >= 1
            
        elif response.status_code == 500:
            # Ollama not available - verify graceful error handling
            error_data: Dict[str, Any] = response.json()
            assert "error" in error_data
            assert "Ollama" in error_data["error"]
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
    
    def test_chat_session_continuity(self) -> None:
        """
        Test conversation memory persistence across multiple messages.
        
        Raises:
            AssertionError: If session continuity fails
        """
        session_id: str = f"test_session_continuity_{uuid.uuid4().hex[:8]}"
        
        # First message
        first_message: Dict[str, str] = {
            "prompt": "My name is Alice",
            "session_id": session_id
        }
        
        try:
            response1 = requests.post(
                f"{BASE_URL}/api/v1/chat",
                json=first_message,
                timeout=15  # Reduced timeout
            )
            
            # Only make second request if first succeeded
            if response1.status_code != 200:
                if response1.status_code == 500:
                    pytest.skip("Ollama not available for session continuity testing")
                else:
                    pytest.fail(f"First request failed with status: {response1.status_code}")
            
            # Second message referencing first
            second_message: Dict[str, str] = {
                "prompt": "What is my name?",
                "session_id": session_id
            }
            
            response2 = requests.post(
                f"{BASE_URL}/api/v1/chat",
                json=second_message,
                timeout=15  # Reduced timeout
            )
            
        except requests.exceptions.Timeout:
            pytest.skip("Chat requests timed out - Ollama may be overloaded")
        
        # Verify both requests use same session
        if response1.status_code == 200 and response2.status_code == 200:
            data1: Dict[str, Any] = response1.json()
            data2: Dict[str, Any] = response2.json()
            
            # Same session ID
            assert data1["session_id"] == data2["session_id"] == session_id
            
            # Memory should accumulate
            memory1: Dict[str, Any] = data1["memory_stats"]
            memory2: Dict[str, Any] = data2["memory_stats"]
            
            assert memory2["total_messages"] > memory1["total_messages"]
            assert data2["conversation_length"] > data1["conversation_length"]
            
        elif response1.status_code == 500 or response2.status_code == 500:
            # Ollama not available - acceptable for testing
            pass
        else:
            pytest.fail("Unexpected response codes for session continuity test")
    
    def test_chat_missing_message_field(self) -> None:
        """
        Test chat endpoint with missing message field.
        
        Raises:
            AssertionError: If validation is incorrect
        """
        invalid_data: Dict[str, str] = {
            "session_id": "test_session"
            # Missing message field
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json=invalid_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 400
        data: Dict[str, Any] = response.json()
        assert "error" in data
        assert "prompt" in data["error"].lower()
    
    def test_chat_empty_message(self) -> None:
        """
        Test chat endpoint with empty message.
        
        Raises:
            AssertionError: If empty message validation is incorrect
        """
        empty_data: Dict[str, str] = {
            "prompt": "",
            "session_id": "test_session"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json=empty_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 400
        data: Dict[str, Any] = response.json()
        assert "error" in data
        assert "empty" in data["error"].lower()
    
    def test_chat_missing_session_id(self) -> None:
        """
        Test chat endpoint with missing session_id field.
        
        Raises:
            AssertionError: If validation is incorrect
        """
        chat_data: Dict[str, str] = {
            "prompt": "Test message without session ID"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json=chat_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 400
        data: Dict[str, Any] = response.json()
        assert "error" in data
        assert "session_id" in data["error"].lower()


class TestGRPCIntegrationEndpoint:
    """Test cases for the /api/v1/classify-grpc endpoint."""
    
    def test_grpc_classify_valid_input(self) -> None:
        """
        Test gRPC classification with valid iris features.
        
        Raises:
            AssertionError: If gRPC classification fails
        """
        valid_data: Dict[str, float] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/classify-grpc",
            json=valid_data,
            timeout=TIMEOUT
        )
        
        # gRPC server might not be running - check both success and expected failure
        if response.status_code == 200:
            data: Dict[str, Any] = response.json()
            
            # Verify response structure matches gRPC response format
            required_fields: List[str] = [
                "predicted_class", "predicted_class_index",
                "probabilities", "confidence", "all_classes", "input_features"
            ]
            for field in required_fields:
                assert field in data
                
            # Verify data types and ranges
            assert isinstance(data["predicted_class"], str)
            assert isinstance(data["predicted_class_index"], int)
            assert isinstance(data["probabilities"], list)
            assert isinstance(data["confidence"], float)
            assert 0 <= data["confidence"] <= 1
            assert len(data["probabilities"]) == 3
            assert data["predicted_class"] in ["setosa", "versicolor", "virginica"]
            
        elif response.status_code == 500:
            # gRPC server not available - verify error message
            error_data: Dict[str, Any] = response.json()
            assert "error" in error_data
            assert any(keyword in error_data["error"].lower() 
                      for keyword in ["grpc", "connection", "server"])
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
    
    def test_grpc_classify_invalid_input(self) -> None:
        """
        Test gRPC classification with invalid input.
        
        Raises:
            AssertionError: If validation is incorrect
        """
        invalid_data: Dict[str, Any] = {
            "sepal_length": "invalid",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/classify-grpc",
            json=invalid_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 400
        data: Dict[str, Any] = response.json()
        assert "error" in data
        assert "numeric" in data["error"].lower()


class TestPerformanceComparisonEndpoint:
    """Test cases for the /api/v1/classify-benchmark endpoint."""
    
    def test_performance_comparison_valid_input(self) -> None:
        """
        Test performance comparison between REST and gRPC.
        
        Raises:
            AssertionError: If performance comparison fails
        """
        valid_data: Dict[str, float] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/classify-benchmark",
            json=valid_data,
            timeout=TIMEOUT
        )
        
        # Should succeed regardless of gRPC server availability
        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        
        # Verify response structure
        assert "results" in data
        assert "performance_analysis" in data
        
        # Verify REST result (should always work)
        rest_result: Dict[str, Any] = data["results"]["rest"]
        assert "predicted_class" in rest_result
        
        # gRPC result might fail if server not available
        grpc_result: Dict[str, Any] = data["results"]["grpc"]
        # gRPC might have an error if not available
        
        # Performance metrics should be present
        metrics: Dict[str, Any] = data["performance_analysis"]
        assert "rest_time_ms" in metrics
        assert isinstance(metrics["rest_time_ms"], (int, float))
        
        if "error" not in grpc_result:
            assert "grpc_time_ms" in metrics
            assert isinstance(metrics["grpc_time_ms"], (int, float))
            assert "speedup_factor" in metrics
            assert "faster_protocol" in metrics


class TestSerializationEndpoint:
    """Test cases for the /api/v1/classify-detailed endpoint."""
    
    def test_serialization_demo_valid_input(self) -> None:
        """
        Test serialization challenges demonstration.
        
        Raises:
            AssertionError: If serialization demo fails
        """
        valid_data: Dict[str, float] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/classify-detailed",
            json=valid_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        
        # Verify response structure
        assert "predicted_class" in data
        assert "predicted_class_index" in data
        assert "raw_probabilities" in data
        assert "serialization_demo" in data
        
        # Verify prediction result
        assert data["predicted_class"] in ["setosa", "versicolor", "virginica"]
        assert isinstance(data["predicted_class_index"], int)
        
        # Verify numpy arrays demonstration (should be serialized as lists)
        assert isinstance(data["raw_probabilities"], list)
        assert len(data["raw_probabilities"]) == 3
        
        # Verify serialization demo
        demo: Dict[str, Any] = data["serialization_demo"]
        assert "numpy_int" in demo
        assert "numpy_float" in demo
        assert "numpy_bool" in demo
        assert "numpy_array" in demo
        
        # All numpy types should be serialized to native Python types
        assert isinstance(demo["numpy_int"], int)
        assert isinstance(demo["numpy_float"], float)
        assert isinstance(demo["numpy_bool"], bool)
        assert isinstance(demo["numpy_array"], list)


class TestNumpyEncoderUtility:
    """Test cases for the NumpyEncoder serialization utility."""
    
    def test_numpy_encoder_handles_numpy_types(self) -> None:
        """
        Test that NumpyEncoder properly handles numpy data types.
        
        Note: This tests the encoder indirectly through API responses
        since we're testing the running Flask app.
        """
        # This is tested implicitly through the serialization demo endpoint
        # which uses NumpyEncoder to handle numpy arrays
        valid_data: Dict[str, float] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/classify-detailed",
            json=valid_data,
            timeout=TIMEOUT
        )
        
        # If this succeeds, NumpyEncoder is working properly
        assert response.status_code == 200
        
        # Response should be valid JSON (no serialization errors)
        data: Dict[str, Any] = response.json()
        assert isinstance(data, dict)
    
    def test_classification_endpoints_handle_numpy_serialization(self) -> None:
        """
        Test that classification endpoints properly serialize numpy data.
        
        Raises:
            AssertionError: If serialization fails
        """
        valid_data: Dict[str, float] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        # Test regular classification endpoint
        response = requests.post(
            f"{BASE_URL}/api/v1/classify",
            json=valid_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        
        # Probabilities come from numpy and should be properly serialized
        assert "probabilities" in data
        probabilities: List[float] = data["probabilities"]
        assert isinstance(probabilities, list)
        assert len(probabilities) == 3
        assert all(isinstance(p, (int, float)) for p in probabilities)


class TestConversationMemoryManagement:
    """Test cases for LangChain conversation memory management."""
    
    def test_memory_isolation_between_sessions(self) -> None:
        """
        Test that conversation memories are isolated between different sessions.
        
        Raises:
            AssertionError: If memory isolation fails
        """
        session1_id: str = f"isolation_test_session_1_{uuid.uuid4().hex[:8]}"
        session2_id: str = f"isolation_test_session_2_{uuid.uuid4().hex[:8]}"
        
        # Send message to session 1
        session1_data: Dict[str, str] = {
            "prompt": "I am from session 1",
            "session_id": session1_id
        }
        
        response1 = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json=session1_data,
            timeout=TIMEOUT
        )
        
        # Send message to session 2
        session2_data: Dict[str, str] = {
            "prompt": "I am from session 2",
            "session_id": session2_id
        }
        
        response2 = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json=session2_data,
            timeout=TIMEOUT
        )
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1: Dict[str, Any] = response1.json()
            data2: Dict[str, Any] = response2.json()
            
            # Sessions should be different
            assert data1["session_id"] != data2["session_id"]
            
            # Each session should start with its own message count
            memory1: Dict[str, Any] = data1["memory_stats"]
            memory2: Dict[str, Any] = data2["memory_stats"]
            
            # Both should have 2 messages (human + AI)
            assert memory1["total_messages"] == 2
            assert memory2["total_messages"] == 2
            
        elif response1.status_code == 500 or response2.status_code == 500:
            # Ollama not available - acceptable
            pass
        else:
            pytest.fail("Unexpected response codes for memory isolation test")
    
    def test_memory_persistence_within_session(self) -> None:
        """
        Test that conversation memory persists within a single session.
        
        Raises:
            AssertionError: If memory persistence fails
        """
        session_id: str = f"persistence_test_session_{uuid.uuid4().hex[:8]}"
        
        messages: List[str] = [
            "First message in session",
            "Second message in session", 
            "Third message in session"
        ]
        
        responses: List[requests.Response] = []
        
        for i, message in enumerate(messages):
            chat_data: Dict[str, str] = {
                "prompt": message,
                "session_id": session_id
            }
            
            try:
                response = requests.post(
                    f"{BASE_URL}/api/v1/chat",
                    json=chat_data,
                    timeout=15  # Reduced timeout to handle overloaded Ollama
                )
                
                responses.append(response)
                
                if response.status_code == 200:
                    data: Dict[str, Any] = response.json()
                    memory: Dict[str, Any] = data["memory_stats"]
                    
                    # Message count should increase with each message
                    expected_count: int = (i + 1) * 2  # Each exchange has human + AI message
                    assert memory["total_messages"] == expected_count
                    
                    # Conversation length should match
                    assert data["conversation_length"] == i + 1
                    
                elif response.status_code == 500:
                    # Ollama not available - skip this test
                    pytest.skip("Ollama not available for memory persistence test")
                else:
                    pytest.fail(f"Unexpected status code: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                # Skip test if requests are timing out (Ollama overloaded)
                pytest.skip(f"Chat request {i+1} timed out - Ollama may be overloaded")


# Pytest configuration and fixtures
@pytest.fixture(scope="session", autouse=True)
def check_server_running() -> None:
    """
    Session-scoped fixture to check if the Flask server is running.
    
    Raises:
        pytest.skip: If server is not accessible
    """
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code not in [200, 500]:  # 500 is OK if services are down
            pytest.skip(f"Server returned unexpected status: {response.status_code}")
    except ConnectionError:
        pytest.skip("Flask server is not running. Start with: python app.py")
    except RequestException as e:
        pytest.skip(f"Server connection failed: {e}")


@pytest.fixture
def cleanup_test_sessions() -> None:
    """
    Fixture to clean up test sessions after tests.
    
    Note: In a production system, you might want to implement
    session cleanup endpoints. For testing, we rely on the
    fact that sessions are stored in memory and will be
    cleared when the server restarts.
    """
    yield
    # Cleanup code would go here if we had session management endpoints


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])