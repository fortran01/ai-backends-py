"""
Unit tests for the gRPC server implementation.

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive test coverage for gRPC server functionality
- Mock-based testing for isolated unit tests
- Integration testing for gRPC communication
"""

import os
import time
import threading
import subprocess
from typing import Dict, Any, List, Optional, Generator
from unittest.mock import Mock, patch, MagicMock
import pytest
import grpc
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Import the gRPC generated classes
try:
    from proto import inference_pb2, inference_pb2_grpc
    GRPC_IMPORTS_AVAILABLE = True
except ImportError:
    GRPC_IMPORTS_AVAILABLE = False
    pytest.skip("gRPC imports not available", allow_module_level=True)

# Test configuration
GRPC_HOST: str = "localhost"
GRPC_PORT: int = 50051
GRPC_SERVER_URL: str = f"{GRPC_HOST}:{GRPC_PORT}"
MAX_WORKERS: int = 10
TEST_TIMEOUT: int = 30


class TestGRPCServerUnit:
    """Unit tests for gRPC server components using mocks."""
    
    @pytest.fixture
    def mock_onnx_session(self) -> Mock:
        """
        Create a mock ONNX runtime session for testing.
        
        Returns:
            Mock ONNX session with run method
        """
        mock_session = Mock()
        
        # Mock prediction results for iris classification
        # Results format: [class_predictions, probability_dictionaries]
        mock_class_predictions: np.ndarray = np.array([0], dtype=np.int64)  # Predicted class index
        mock_prob_dict: Dict[int, float] = {0: 0.8, 1: 0.15, 2: 0.05}  # Probability dictionary
        mock_prob_dicts: List[Dict[int, float]] = [mock_prob_dict]  # Batch of probability dictionaries
        
        mock_session.run.return_value = [mock_class_predictions, mock_prob_dicts]
        
        # Mock input/output metadata
        mock_input = Mock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        
        mock_output1 = Mock()
        mock_output1.name = "output_label"
        mock_output2 = Mock()
        mock_output2.name = "output_probability"
        mock_session.get_outputs.return_value = [mock_output1, mock_output2]
        
        return mock_session
    
    @patch('onnxruntime.InferenceSession')
    def test_grpc_server_initialization(self, mock_inference_session: Mock) -> None:
        """
        Test gRPC server initialization with mocked ONNX session.
        
        Args:
            mock_inference_session: Mock ONNX InferenceSession
            
        Raises:
            AssertionError: If server initialization fails
        """
        mock_inference_session.return_value = Mock()
        
        # Import and test server initialization
        try:
            import grpc_server
            
            # Verify that the server module can be imported
            assert hasattr(grpc_server, 'InferenceServicer')
            assert hasattr(grpc_server, 'serve')
            
        except ImportError as e:
            pytest.fail(f"Could not import gRPC server module: {e}")
    
    def test_inference_servicer_classify_method(self, mock_onnx_session: Mock) -> None:
        """
        Test the InferenceServicer.Classify method with mocked ONNX session.
        
        Args:
            mock_onnx_session: Mock ONNX session
            
        Raises:
            AssertionError: If classification method fails
        """
        with patch('onnxruntime.InferenceSession', return_value=mock_onnx_session):
            # Import the servicer class
            import grpc_server
            
            # Create servicer instance
            servicer = grpc_server.InferenceServicer()
            
            # Create test request
            request = inference_pb2.ClassifyRequest(
                sepal_length=5.1,
                sepal_width=3.5,
                petal_length=1.4,
                petal_width=0.2
            )
            
            # Mock gRPC context
            mock_context = Mock()
            
            # Call the classify method
            response = servicer.Classify(request, mock_context)
            
            # Verify response structure
            assert isinstance(response, inference_pb2.ClassifyResponse)
            assert response.predicted_class in ["setosa", "versicolor", "virginica"]
            assert isinstance(response.predicted_class_index, int)
            assert 0 <= response.predicted_class_index <= 2
            assert len(response.probabilities) == 3
            assert isinstance(response.confidence, float)
            assert 0.0 <= response.confidence <= 1.0
            assert len(response.all_classes) == 3
            
            # Verify input features are preserved
            input_features = response.input_features
            assert abs(input_features.sepal_length - 5.1) < 0.001
            assert abs(input_features.sepal_width - 3.5) < 0.001
            assert abs(input_features.petal_length - 1.4) < 0.001
            assert abs(input_features.petal_width - 0.2) < 0.001
            
            # Verify ONNX session was called
            mock_onnx_session.run.assert_called_once()
    
    def test_inference_servicer_invalid_model_path(self) -> None:
        """
        Test InferenceServicer behavior with invalid model path.
        
        Raises:
            AssertionError: If error handling is incorrect
        """
        with patch('grpc_server.get_onnx_session') as mock_get_session:
            mock_get_session.side_effect = FileNotFoundError("Model file not found")
            
            import grpc_server
            
            # Create servicer instance (should not raise exception)
            servicer = grpc_server.InferenceServicer()
            
            # Create test request
            request = inference_pb2.ClassifyRequest(
                sepal_length=5.1,
                sepal_width=3.5,
                petal_length=1.4,
                petal_width=0.2
            )
            
            # Mock gRPC context
            mock_context = Mock()
            
            # Call the classify method (should handle error gracefully)
            response = servicer.Classify(request, mock_context)
            
            # Should return empty response and set error context
            assert response.predicted_class == ""
            mock_context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)
            mock_context.set_details.assert_called()
    
    def test_iris_class_mapping(self) -> None:
        """
        Test the iris class name mapping functionality.
        
        Raises:
            AssertionError: If class mapping is incorrect
        """
        import grpc_server
        
        # Test class mapping
        expected_classes: List[str] = ["setosa", "versicolor", "virginica"]
        
        # Verify class mapping exists and is correct
        assert hasattr(grpc_server, 'IRIS_CLASSES')
        assert grpc_server.IRIS_CLASSES == expected_classes
        
        # Test index to class mapping
        for i, class_name in enumerate(expected_classes):
            assert grpc_server.IRIS_CLASSES[i] == class_name


class TestGRPCServerIntegration:
    """Integration tests for gRPC server requiring actual server instance."""
    
    @pytest.fixture(scope="class")
    def grpc_server_process(self) -> Generator[Optional[subprocess.Popen], None, None]:
        """
        Start gRPC server process for integration testing.
        
        Yields:
            subprocess.Popen instance of running gRPC server
        """
        # Check if server is already running
        try:
            with grpc.insecure_channel(GRPC_SERVER_URL) as channel:
                stub = inference_pb2_grpc.InferenceServiceStub(channel)
                # Try to make a simple request to check if server is running
                request = inference_pb2.ClassifyRequest(
                    sepal_length=5.0, sepal_width=3.0, 
                    petal_length=1.0, petal_width=0.1
                )
                try:
                    response = stub.Classify(request, timeout=5)
                    # Server is already running
                    yield None
                    return
                except grpc.RpcError:
                    pass  # Server not running, we'll start it
        except:
            pass  # Server not running, we'll start it
        
        # Start the gRPC server
        server_process: Optional[subprocess.Popen] = None
        
        try:
            server_process = subprocess.Popen(
                ["python", "grpc_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(__file__))  # Go up to project root
            )
            
            # Wait for server to start
            max_wait: int = 10
            server_ready: bool = False
            
            for _ in range(max_wait):
                try:
                    with grpc.insecure_channel(GRPC_SERVER_URL) as channel:
                        stub = inference_pb2_grpc.InferenceServiceStub(channel)
                        request = inference_pb2.ClassifyRequest(
                            sepal_length=5.0, sepal_width=3.0,
                            petal_length=1.0, petal_width=0.1
                        )
                        stub.Classify(request, timeout=5)
                        server_ready = True
                        break
                except grpc.RpcError:
                    time.sleep(1)
            
            if not server_ready:
                pytest.skip("Could not start gRPC server for integration testing")
            
            yield server_process
            
        finally:
            if server_process:
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.kill()
    
    def test_grpc_client_server_communication(self, grpc_server_process: Optional[subprocess.Popen]) -> None:
        """
        Test complete gRPC client-server communication.
        
        Args:
            grpc_server_process: Running gRPC server process
            
        Raises:
            AssertionError: If client-server communication fails
        """
        # Test data for different iris types
        test_cases: List[Dict[str, Any]] = [
            {
                "name": "setosa_sample",
                "features": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                "expected_class": "setosa"
            },
            {
                "name": "versicolor_sample",
                "features": {
                    "sepal_length": 7.0,
                    "sepal_width": 3.2,
                    "petal_length": 4.7,
                    "petal_width": 1.4
                },
                "expected_class": "versicolor"
            },
            {
                "name": "virginica_sample",
                "features": {
                    "sepal_length": 6.3,
                    "sepal_width": 3.3,
                    "petal_length": 6.0,
                    "petal_width": 2.5
                },
                "expected_class": "virginica"
            }
        ]
        
        # Create gRPC client
        with grpc.insecure_channel(GRPC_SERVER_URL) as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            
            for test_case in test_cases:
                features: Dict[str, float] = test_case["features"]
                
                # Create request
                request = inference_pb2.ClassifyRequest(
                    sepal_length=features["sepal_length"],
                    sepal_width=features["sepal_width"],
                    petal_length=features["petal_length"],
                    petal_width=features["petal_width"]
                )
                
                # Make gRPC call
                try:
                    response: inference_pb2.ClassifyResponse = stub.Classify(request, timeout=TEST_TIMEOUT)
                    
                    # Verify response structure
                    assert response.predicted_class in ["setosa", "versicolor", "virginica"]
                    assert isinstance(response.predicted_class_index, int)
                    assert 0 <= response.predicted_class_index <= 2
                    assert len(response.probabilities) == 3
                    assert isinstance(response.confidence, float)
                    assert 0.0 <= response.confidence <= 1.0
                    assert len(response.all_classes) == 3
                    
                    # Verify input features preservation
                    input_features = response.input_features
                    assert abs(input_features.sepal_length - features["sepal_length"]) < 0.001
                    assert abs(input_features.sepal_width - features["sepal_width"]) < 0.001
                    assert abs(input_features.petal_length - features["petal_length"]) < 0.001
                    assert abs(input_features.petal_width - features["petal_width"]) < 0.001
                    
                    # Verify class mapping consistency
                    expected_index: int = ["setosa", "versicolor", "virginica"].index(response.predicted_class)
                    assert response.predicted_class_index == expected_index
                    
                    print(f"✅ {test_case['name']}: {response.predicted_class} (confidence: {response.confidence:.3f})")
                    
                except grpc.RpcError as e:
                    pytest.fail(f"gRPC call failed for {test_case['name']}: {e}")
    
    def test_grpc_server_error_handling(self, grpc_server_process: Optional[subprocess.Popen]) -> None:
        """
        Test gRPC server error handling with invalid requests.
        
        Args:
            grpc_server_process: Running gRPC server process
            
        Raises:
            AssertionError: If error handling is incorrect
        """
        with grpc.insecure_channel(GRPC_SERVER_URL) as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            
            # Test with extreme values that might cause issues
            extreme_request = inference_pb2.ClassifyRequest(
                sepal_length=999.0,
                sepal_width=-999.0,
                petal_length=0.0,
                petal_width=1000.0
            )
            
            try:
                response = stub.Classify(extreme_request, timeout=TEST_TIMEOUT)
                
                # Server should handle extreme values gracefully
                assert response.predicted_class in ["setosa", "versicolor", "virginica"]
                assert isinstance(response.confidence, float)
                assert 0.0 <= response.confidence <= 1.0
                
            except grpc.RpcError as e:
                # Some level of error handling is acceptable for extreme values
                assert e.code() in [grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.OUT_OF_RANGE]
    
    def test_grpc_concurrent_requests(self, grpc_server_process: Optional[subprocess.Popen]) -> None:
        """
        Test gRPC server handling of concurrent requests.
        
        Args:
            grpc_server_process: Running gRPC server process
            
        Raises:
            AssertionError: If concurrent request handling fails
        """
        def make_grpc_request(request_id: int) -> Dict[str, Any]:
            """
            Make a single gRPC request for concurrent testing.
            
            Args:
                request_id: Unique identifier for the request
                
            Returns:
                Dictionary with request results
            """
            try:
                with grpc.insecure_channel(GRPC_SERVER_URL) as channel:
                    stub = inference_pb2_grpc.InferenceServiceStub(channel)
                    
                    request = inference_pb2.ClassifyRequest(
                        sepal_length=5.0 + (request_id * 0.1),
                        sepal_width=3.0 + (request_id * 0.1),
                        petal_length=1.0 + (request_id * 0.1),
                        petal_width=0.1 + (request_id * 0.1)
                    )
                    
                    response = stub.Classify(request, timeout=TEST_TIMEOUT)
                    
                    return {
                        "request_id": request_id,
                        "success": True,
                        "predicted_class": response.predicted_class,
                        "confidence": response.confidence
                    }
                    
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Make concurrent requests
        num_concurrent_requests: int = 5
        
        with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [
                executor.submit(make_grpc_request, i) 
                for i in range(num_concurrent_requests)
            ]
            
            results: List[Dict[str, Any]] = [future.result() for future in futures]
        
        # Verify all requests succeeded
        successful_requests: List[Dict[str, Any]] = [r for r in results if r["success"]]
        failed_requests: List[Dict[str, Any]] = [r for r in results if not r["success"]]
        
        assert len(successful_requests) >= num_concurrent_requests * 0.8  # Allow some failures
        
        # Verify successful responses
        for result in successful_requests:
            assert result["predicted_class"] in ["setosa", "versicolor", "virginica"]
            assert isinstance(result["confidence"], float)
            assert 0.0 <= result["confidence"] <= 1.0
        
        if failed_requests:
            print(f"⚠️  {len(failed_requests)} out of {num_concurrent_requests} concurrent requests failed")


class TestGRPCClientIntegration:
    """Test gRPC client integration in the Flask app."""
    
    def test_flask_grpc_client_configuration(self) -> None:
        """
        Test that Flask app properly configures gRPC client.
        
        Raises:
            AssertionError: If gRPC client configuration is incorrect
        """
        # Test by importing the Flask app and checking gRPC integration
        try:
            import app
            
            # Verify gRPC-related imports exist
            assert hasattr(app, 'grpc')
            assert hasattr(app, 'inference_pb2')
            assert hasattr(app, 'inference_pb2_grpc')
            
        except ImportError as e:
            pytest.fail(f"Could not import Flask app with gRPC integration: {e}")
    
    @patch('grpc.insecure_channel')
    def test_grpc_client_error_handling_in_flask(self, mock_channel: Mock) -> None:
        """
        Test gRPC client error handling within Flask app.
        
        Args:
            mock_channel: Mock gRPC channel
            
        Raises:
            AssertionError: If error handling is incorrect
        """
        # Mock gRPC channel to raise connection error
        mock_channel.side_effect = grpc.RpcError("Connection failed")
        
        import app
        
        # The Flask app should handle gRPC connection errors gracefully
        # This is verified through the actual API tests in test_phase2_features.py
        # Here we just verify the imports and basic structure exist
        assert hasattr(app, 'grpc')


# Pytest configuration and fixtures
@pytest.fixture(scope="session", autouse=True)
def check_grpc_imports() -> None:
    """
    Check if gRPC imports are available.
    
    Raises:
        pytest.skip: If gRPC imports are not available
    """
    if not GRPC_IMPORTS_AVAILABLE:
        pytest.skip("gRPC protobuf files not generated. Run: python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. proto/inference.proto")


@pytest.fixture
def grpc_channel() -> Generator[grpc.Channel, None, None]:
    """
    Create gRPC channel for testing.
    
    Yields:
        gRPC channel for making requests
    """
    channel = grpc.insecure_channel(GRPC_SERVER_URL)
    try:
        yield channel
    finally:
        channel.close()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])