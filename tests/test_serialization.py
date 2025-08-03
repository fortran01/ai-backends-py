"""
Unit tests for serialization utilities and numpy data handling.

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive test coverage for serialization challenges
- NumpyEncoder testing and validation
- JSON serialization edge cases
"""

import json
from typing import Dict, Any, List, Union, Optional
from unittest.mock import Mock, patch
import pytest
import requests
import numpy as np

# Test configuration
BASE_URL: str = "http://localhost:5001"
TIMEOUT: int = 30


class TestNumpyEncoderUnit:
    """Unit tests for the NumpyEncoder utility class."""
    
    def test_numpy_encoder_import(self) -> None:
        """
        Test that NumpyEncoder can be imported from the app module.
        
        Raises:
            AssertionError: If NumpyEncoder cannot be imported
        """
        try:
            import app
            
            # Verify NumpyEncoder exists
            assert hasattr(app, 'NumpyEncoder')
            
            # Verify it's a class
            assert isinstance(app.NumpyEncoder, type)
            
            # Verify it inherits from JSONEncoder
            import json
            assert issubclass(app.NumpyEncoder, json.JSONEncoder)
            
        except ImportError as e:
            pytest.fail(f"Could not import app module: {e}")
    
    def test_numpy_encoder_numpy_array_serialization(self) -> None:
        """
        Test NumpyEncoder handles numpy arrays correctly.
        
        Raises:
            AssertionError: If numpy array serialization fails
        """
        try:
            import app
            
            # Test different numpy array types
            test_arrays: List[np.ndarray] = [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([1.1, 2.2, 3.3], dtype=np.float32),
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
                np.array([[1, 2], [3, 4]], dtype=np.int64),
                np.array([True, False, True], dtype=np.bool_)
            ]
            
            encoder = app.NumpyEncoder()
            
            for test_array in test_arrays:
                # Test direct encoding
                encoded = encoder.default(test_array)
                
                # Should return a list
                assert isinstance(encoded, list)
                
                # Should preserve array structure
                expected_list = test_array.tolist()
                assert encoded == expected_list
                
                # Test full JSON serialization
                json_str: str = json.dumps(test_array, cls=app.NumpyEncoder)
                parsed_back: List[Union[int, float, bool]] = json.loads(json_str)
                
                assert isinstance(parsed_back, list)
                
        except ImportError as e:
            pytest.fail(f"Could not import required modules: {e}")
    
    def test_numpy_encoder_numpy_scalar_serialization(self) -> None:
        """
        Test NumpyEncoder handles numpy scalar types correctly.
        
        Raises:
            AssertionError: If numpy scalar serialization fails
        """
        try:
            import app
            
            # Test different numpy scalar types
            test_scalars: List[np.generic] = [
                np.int32(42),
                np.int64(123),
                np.float32(3.14),
                np.float64(2.718),
                np.bool_(True),
                np.bool_(False)
            ]
            
            encoder = app.NumpyEncoder()
            
            for test_scalar in test_scalars:
                # Test direct encoding
                encoded = encoder.default(test_scalar)
                
                # Should return a Python native type
                assert isinstance(encoded, (int, float, bool))
                
                # Test full JSON serialization
                json_str: str = json.dumps(test_scalar, cls=app.NumpyEncoder)
                parsed_back: Union[int, float, bool] = json.loads(json_str)
                
                assert type(parsed_back) in [int, float, bool]
                
        except ImportError as e:
            pytest.fail(f"Could not import required modules: {e}")
    
    def test_numpy_encoder_non_numpy_passthrough(self) -> None:
        """
        Test NumpyEncoder passes non-numpy objects to default JSON encoder.
        
        Raises:
            AssertionError: If non-numpy passthrough fails
        """
        try:
            import app
            
            encoder = app.NumpyEncoder()
            
            # Test with Python objects that should raise TypeError
            # Note: The NumpyEncoder has a fallback for objects with __dict__
            # Only objects without __dict__ and not handled by numpy cases should raise TypeError
            non_serializable_objects: List[Any] = [
                set([1, 2, 3]),  # Set object (not JSON serializable)
                complex(1, 2),   # Complex number (not JSON serializable)
                frozenset([1, 2])  # Frozenset (not JSON serializable)
            ]
            
            for obj in non_serializable_objects:
                with pytest.raises(TypeError):
                    encoder.default(obj)
                    
        except ImportError as e:
            pytest.fail(f"Could not import required modules: {e}")
    
    def test_numpy_encoder_complex_structure_serialization(self) -> None:
        """
        Test NumpyEncoder handles complex nested structures with numpy data.
        
        Raises:
            AssertionError: If complex structure serialization fails
        """
        try:
            import app
            
            # Create complex structure with numpy data
            complex_data: Dict[str, Any] = {
                "regular_list": [1, 2, 3],
                "numpy_array": np.array([4, 5, 6], dtype=np.float32),
                "nested": {
                    "probabilities": np.array([0.1, 0.7, 0.2], dtype=np.float64),
                    "prediction": np.int32(1),
                    "confidence": np.float32(0.75)
                },
                "mixed_list": [
                    1, 
                    np.float32(2.5), 
                    np.array([3, 4]), 
                    "string"
                ]
            }
            
            # Test JSON serialization
            json_str: str = json.dumps(complex_data, cls=app.NumpyEncoder)
            parsed_back: Dict[str, Any] = json.loads(json_str)
            
            # Verify structure is preserved
            assert "regular_list" in parsed_back
            assert "numpy_array" in parsed_back
            assert "nested" in parsed_back
            assert "mixed_list" in parsed_back
            
            # Verify numpy data is converted to native types
            assert isinstance(parsed_back["numpy_array"], list)
            assert isinstance(parsed_back["nested"]["probabilities"], list)
            assert isinstance(parsed_back["nested"]["prediction"], int)
            assert isinstance(parsed_back["nested"]["confidence"], float)
            
        except ImportError as e:
            pytest.fail(f"Could not import required modules: {e}")


class TestSerializationDemoEndpoint:
    """Integration tests for the serialization demo endpoint."""
    
    def test_serialization_demo_endpoint_structure(self) -> None:
        """
        Test serialization demo endpoint returns proper structure.
        
        Raises:
            AssertionError: If endpoint structure is incorrect
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
        
        # Verify top-level structure
        required_fields: List[str] = [
            "predicted_class",
            "predicted_class_index",
            "raw_probabilities",
            "serialization_demo"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Verify prediction result
        assert data["predicted_class"] in ["setosa", "versicolor", "virginica"]
        assert isinstance(data["predicted_class_index"], int)
        assert 0 <= data["predicted_class_index"] <= 2
        
        # Verify numpy arrays are properly serialized as lists
        assert isinstance(data["raw_probabilities"], list)
        assert len(data["raw_probabilities"]) == 3
        
        # Verify serialization demo section
        demo: Dict[str, Any] = data["serialization_demo"]
        demo_fields: List[str] = [
            "numpy_int", "numpy_float", "numpy_bool", "numpy_array"
        ]
        for field in demo_fields:
            assert field in demo, f"Missing demo field: {field}"
        
        # All numpy types should be serialized to native Python types
        assert isinstance(demo["numpy_int"], int)
        assert isinstance(demo["numpy_float"], float)
        assert isinstance(demo["numpy_bool"], bool)
        assert isinstance(demo["numpy_array"], list)
    
    def test_serialization_demo_numpy_data_consistency(self) -> None:
        """
        Test that serialized numpy data maintains consistency with original.
        
        Raises:
            AssertionError: If numpy data consistency fails
        """
        test_cases: List[Dict[str, float]] = [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 7.0,
                "sepal_width": 3.2,
                "petal_length": 4.7,
                "petal_width": 1.4
            },
            {
                "sepal_length": 6.3,
                "sepal_width": 3.3,
                "petal_length": 6.0,
                "petal_width": 2.5
            }
        ]
        
        for test_case in test_cases:
            response = requests.post(
                f"{BASE_URL}/api/v1/classify-detailed",
                json=test_case,
                timeout=TIMEOUT
            )
            
            assert response.status_code == 200
            data: Dict[str, Any] = response.json()
            
            # Verify input features if present
            if "input_features" in data:
                input_features = data["input_features"]
                # Handle both flat and nested array formats
                if isinstance(input_features[0], list):
                    # Nested format: [[val1, val2, val3, val4]]
                    input_array: List[float] = input_features[0]
                else:
                    # Flat format: [val1, val2, val3, val4]
                    input_array: List[float] = input_features
                    
                expected_input: List[float] = [
                    test_case["sepal_length"],
                    test_case["sepal_width"],
                    test_case["petal_length"],
                    test_case["petal_width"]
                ]
                
                assert len(input_array) == len(expected_input)
                for i, (actual, expected) in enumerate(zip(input_array, expected_input)):
                    assert abs(actual - expected) < 0.001, f"Input mismatch at index {i}"
            
            # Verify probabilities sum to approximately 1.0
            probabilities: List[float] = data["raw_probabilities"]
            prob_sum: float = sum(probabilities)
            assert abs(prob_sum - 1.0) < 0.01, f"Probabilities don't sum to 1.0: {prob_sum}"
            
            # Verify all probabilities are in valid range
            for prob in probabilities:
                assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"
    
    def test_serialization_demo_error_handling(self) -> None:
        """
        Test serialization demo endpoint error handling.
        
        Raises:
            AssertionError: If error handling is incorrect
        """
        # Test with invalid input data
        invalid_test_cases: List[Dict[str, Any]] = [
            {
                "sepal_length": "invalid",
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4
                # Missing petal_width
            },
            {}  # Empty data
        ]
        
        for invalid_data in invalid_test_cases:
            response = requests.post(
                f"{BASE_URL}/api/v1/classify-detailed",
                json=invalid_data,
                timeout=TIMEOUT
            )
            
            # Should return 400 for invalid input
            assert response.status_code == 400
            error_data: Dict[str, Any] = response.json()
            assert "error" in error_data


class TestNumpySerializationInClassificationEndpoints:
    """Test numpy serialization in regular classification endpoints."""
    
    def test_classify_endpoint_numpy_serialization(self) -> None:
        """
        Test that regular classify endpoint properly serializes numpy data.
        
        Raises:
            AssertionError: If numpy serialization fails in classify endpoint
        """
        valid_data: Dict[str, float] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/classify",
            json=valid_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        
        # Verify probabilities are properly serialized (come from numpy)
        probabilities: List[float] = data["probabilities"]
        assert isinstance(probabilities, list)
        assert len(probabilities) == 3
        
        # All should be proper float values (not numpy float types)
        for prob in probabilities:
            assert isinstance(prob, (int, float))
            assert not isinstance(prob, np.floating)
            assert 0.0 <= prob <= 1.0
        
        # Verify confidence is properly serialized
        confidence: float = data["confidence"]
        assert isinstance(confidence, (int, float))
        assert not isinstance(confidence, np.floating)
        assert 0.0 <= confidence <= 1.0
        
        # Verify predicted_class_index is properly serialized
        class_index: int = data["predicted_class_index"]
        assert isinstance(class_index, int)
        assert not isinstance(class_index, np.integer)
        assert 0 <= class_index <= 2
    
    def test_grpc_classify_endpoint_numpy_serialization(self) -> None:
        """
        Test that gRPC classify endpoint properly serializes numpy data.
        
        Raises:
            AssertionError: If numpy serialization fails in gRPC endpoint
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
        
        if response.status_code == 200:
            data: Dict[str, Any] = response.json()
            
            # Verify probabilities are properly serialized
            probabilities: List[float] = data["probabilities"]
            assert isinstance(probabilities, list)
            assert len(probabilities) == 3
            
            for prob in probabilities:
                assert isinstance(prob, (int, float))
                assert 0.0 <= prob <= 1.0
            
            # Verify confidence is properly serialized
            confidence: float = data["confidence"]
            assert isinstance(confidence, (int, float))
            assert 0.0 <= confidence <= 1.0
            
            # Verify class index is properly serialized
            class_index: int = data["predicted_class_index"]
            assert isinstance(class_index, int)
            assert 0 <= class_index <= 2
            
        elif response.status_code == 500:
            # gRPC server not available - acceptable
            error_data: Dict[str, Any] = response.json()
            assert "error" in error_data
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
    
    def test_performance_comparison_numpy_serialization(self) -> None:
        """
        Test numpy serialization in performance comparison endpoint.
        
        Raises:
            AssertionError: If numpy serialization fails in performance endpoint
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
        
        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        
        # Verify REST result has properly serialized numpy data
        rest_result: Dict[str, Any] = data["results"]["rest"]
        if "probabilities" in rest_result:
            probabilities: List[float] = rest_result["probabilities"]
            assert isinstance(probabilities, list)
            for prob in probabilities:
                assert isinstance(prob, (int, float))
        
        # Verify gRPC result has properly serialized numpy data (if available)
        grpc_result: Dict[str, Any] = data["results"]["grpc"]
        if "error" not in grpc_result and "probabilities" in grpc_result:
            probabilities = grpc_result["probabilities"]
            assert isinstance(probabilities, list)
            for prob in probabilities:
                assert isinstance(prob, (int, float))


class TestJSONSerializationEdgeCases:
    """Test edge cases and error conditions in JSON serialization."""
    
    def test_response_is_valid_json(self) -> None:
        """
        Test that all API responses return valid JSON.
        
        Raises:
            AssertionError: If JSON is invalid
        """
        endpoints_to_test: List[Dict[str, Any]] = [
            {
                "url": f"{BASE_URL}/api/v1/classify",
                "method": "POST",
                "data": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            },
            {
                "url": f"{BASE_URL}/api/v1/classify-detailed",
                "method": "POST",
                "data": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            },
            {
                "url": f"{BASE_URL}/health",
                "method": "GET",
                "data": None
            }
        ]
        
        for endpoint in endpoints_to_test:
            if endpoint["method"] == "GET":
                response = requests.get(endpoint["url"], timeout=TIMEOUT)
            else:
                response = requests.post(
                    endpoint["url"],
                    json=endpoint["data"],
                    timeout=TIMEOUT
                )
            
            # Response should have valid JSON
            try:
                data: Dict[str, Any] = response.json()
                assert isinstance(data, dict)
                
                # JSON should be serializable again (round-trip test)
                json_str: str = json.dumps(data)
                reparsed: Dict[str, Any] = json.loads(json_str)
                assert isinstance(reparsed, dict)
                
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON from {endpoint['url']}: {e}")


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
        if response.status_code not in [200, 500]:
            pytest.skip(f"Server returned unexpected status: {response.status_code}")
    except Exception:
        pytest.skip("Flask server is not running. Start with: python app.py")


@pytest.fixture
def numpy_imports() -> None:
    """
    Fixture to check if NumPy is available.
    
    Raises:
        pytest.skip: If NumPy is not available
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy not available for serialization testing")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])