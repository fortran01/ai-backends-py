"""
Integration tests for gRPC client-server communication.

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive integration test coverage
- Real gRPC server testing
- Performance and reliability testing
"""

import json
import time
import threading
import statistics
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
import grpc
import requests

# Test configuration
FLASK_BASE_URL: str = "http://localhost:5001"
GRPC_HOST: str = "localhost"
GRPC_PORT: int = 50051
GRPC_SERVER_URL: str = f"{GRPC_HOST}:{GRPC_PORT}"
TEST_TIMEOUT: int = 30


class TestGRPCFlaskIntegration:
    """Integration tests between Flask app and gRPC server."""
    
    def test_flask_grpc_classify_endpoint_integration(self) -> None:
        """
        Test full integration of Flask app calling gRPC server for classification.
        
        Raises:
            AssertionError: If integration fails
        """
        test_data: Dict[str, float] = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        # Make request to Flask app's gRPC endpoint
        response = requests.post(
            f"{FLASK_BASE_URL}/api/v1/classify-grpc",
            json=test_data,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            # gRPC server is available and working
            data: Dict[str, Any] = response.json()
            
            # Verify response structure matches gRPC format
            required_fields: List[str] = [
                "predicted_class", "predicted_class_index",
                "probabilities", "confidence", "all_classes", "input_features"
            ]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            # Verify data types and ranges
            assert isinstance(data["predicted_class"], str)
            assert data["predicted_class"] in ["setosa", "versicolor", "virginica"]
            assert isinstance(data["predicted_class_index"], int)
            assert 0 <= data["predicted_class_index"] <= 2
            assert isinstance(data["probabilities"], list)
            assert len(data["probabilities"]) == 3
            assert isinstance(data["confidence"], float)
            assert 0.0 <= data["confidence"] <= 1.0
            
            # Verify input features are preserved
            input_features: Dict[str, float] = data["input_features"]
            for key, value in test_data.items():
                assert abs(input_features[key] - value) < 0.001
                
        elif response.status_code == 500:
            # gRPC server not available - verify error handling
            error_data: Dict[str, Any] = response.json()
            assert "error" in error_data
            assert any(keyword in error_data["error"].lower() 
                      for keyword in ["grpc", "connection", "server"])
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
    
    def test_performance_comparison_integration(self) -> None:
        """
        Test performance comparison between REST and gRPC through Flask app.
        
        Raises:
            AssertionError: If performance comparison fails
        """
        test_data: Dict[str, float] = {
            "sepal_length": 6.2,
            "sepal_width": 2.9,
            "petal_length": 4.3,
            "petal_width": 1.3
        }
        
        # Make request to performance comparison endpoint
        response = requests.post(
            f"{FLASK_BASE_URL}/api/v1/classify-benchmark",
            json=test_data,
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        data: Dict[str, Any] = response.json()
        
        # Verify response structure
        assert "results" in data
        assert "performance_analysis" in data
        
        # Verify REST result (should always work)
        rest_result: Dict[str, Any] = data["results"]["rest"]
        assert "predicted_class" in rest_result
        assert rest_result["predicted_class"] in ["setosa", "versicolor", "virginica"]
        
        # Verify gRPC result structure
        grpc_result: Dict[str, Any] = data["results"]["grpc"]
        # gRPC might have an error if not available
        
        # Verify performance metrics
        metrics: Dict[str, Any] = data["performance_analysis"]
        # New architecture uses http_time_ms instead of rest_time_ms
        assert "http_time_ms" in metrics
        assert isinstance(metrics["http_time_ms"], (int, float))
        assert metrics["http_time_ms"] > 0
        
        if "error" not in grpc_result:
            # gRPC server is available
            assert "predicted_class" in grpc_result
            assert grpc_result["predicted_class"] in ["setosa", "versicolor", "virginica"]
            assert "grpc_time_ms" in metrics
            assert isinstance(metrics["grpc_time_ms"], (int, float))
            assert metrics["grpc_time_ms"] > 0
            assert "faster_protocol" in metrics
            
            # Both should predict the same class for the same input
            assert rest_result["predicted_class"] == grpc_result["predicted_class"]
            
        else:
            # gRPC server not available - that's acceptable for testing
            pass
    
    def test_error_propagation_from_grpc_to_flask(self) -> None:
        """
        Test that gRPC errors are properly propagated through Flask app.
        
        Raises:
            AssertionError: If error propagation fails
        """
        # Test with invalid data that might cause gRPC errors
        invalid_data: Dict[str, Any] = {
            "sepal_length": "invalid_string",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(
            f"{FLASK_BASE_URL}/api/v1/classify-grpc",
            json=invalid_data,
            timeout=TEST_TIMEOUT
        )
        
        # Should return 400 for invalid input data
        assert response.status_code == 400
        error_data: Dict[str, Any] = response.json()
        assert "error" in error_data
        assert "numeric" in error_data["error"].lower()


class TestGRPCPerformanceCharacteristics:
    """Performance and reliability tests for gRPC integration."""
    
    def test_grpc_vs_rest_performance_comparison(self) -> None:
        """
        Compare performance characteristics of gRPC vs REST.
        
        Raises:
            AssertionError: If performance comparison fails
        """
        test_data: Dict[str, float] = {
            "sepal_length": 5.8,
            "sepal_width": 2.7,
            "petal_length": 5.1,
            "petal_width": 1.9
        }
        
        num_requests: int = 10
        rest_times: List[float] = []
        grpc_times: List[float] = []
        
        # Test REST endpoint performance
        for _ in range(num_requests):
            start_time: float = time.time()
            
            response = requests.post(
                f"{FLASK_BASE_URL}/api/v1/classify",
                json=test_data,
                timeout=TEST_TIMEOUT
            )
            
            end_time: float = time.time()
            
            if response.status_code == 200:
                rest_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Test gRPC endpoint performance (through Flask)
        for _ in range(num_requests):
            start_time = time.time()
            
            response = requests.post(
                f"{FLASK_BASE_URL}/api/v1/classify-grpc",
                json=test_data,
                timeout=TEST_TIMEOUT
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                grpc_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Analyze results
        if rest_times and grpc_times:
            rest_avg: float = statistics.mean(rest_times)
            grpc_avg: float = statistics.mean(grpc_times)
            
            print(f"📊 Performance Analysis:")
            print(f"   REST average: {rest_avg:.2f}ms")
            print(f"   gRPC average: {grpc_avg:.2f}ms")
            print(f"   Difference: {abs(rest_avg - grpc_avg):.2f}ms")
            
            # Both should complete within reasonable time
            assert rest_avg < 5000  # Less than 5 seconds
            assert grpc_avg < 5000  # Less than 5 seconds
            
        elif rest_times and not grpc_times:
            print("⚠️  gRPC server not available for performance testing")
            rest_avg = statistics.mean(rest_times)
            assert rest_avg < 5000  # REST should still work
            
        else:
            pytest.fail("No successful requests for performance testing")
    
    def test_concurrent_grpc_requests_through_flask(self) -> None:
        """
        Test handling of concurrent gRPC requests through Flask app.
        
        Raises:
            AssertionError: If concurrent request handling fails
        """
        def make_flask_grpc_request(request_id: int) -> Dict[str, Any]:
            """
            Make a Flask gRPC request for concurrent testing.
            
            Args:
                request_id: Unique identifier for the request
                
            Returns:
                Dictionary with request results
            """
            test_data: Dict[str, float] = {
                "sepal_length": 5.0 + (request_id * 0.1),
                "sepal_width": 3.0 + (request_id * 0.1),
                "petal_length": 1.0 + (request_id * 0.1),
                "petal_width": 0.1 + (request_id * 0.1)
            }
            
            try:
                start_time: float = time.time()
                
                response = requests.post(
                    f"{FLASK_BASE_URL}/api/v1/classify-grpc",
                    json=test_data,
                    timeout=TEST_TIMEOUT
                )
                
                end_time: float = time.time()
                
                if response.status_code == 200:
                    data: Dict[str, Any] = response.json()
                    return {
                        "request_id": request_id,
                        "success": True,
                        "predicted_class": data["predicted_class"],
                        "confidence": data["confidence"],
                        "response_time_ms": (end_time - start_time) * 1000
                    }
                else:
                    return {
                        "request_id": request_id,
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "response_time_ms": (end_time - start_time) * 1000
                    }
                    
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "response_time_ms": None
                }
        
        # Make concurrent requests
        num_concurrent_requests: int = 5
        
        with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [
                executor.submit(make_flask_grpc_request, i)
                for i in range(num_concurrent_requests)
            ]
            
            results: List[Dict[str, Any]] = []
            for future in as_completed(futures, timeout=TEST_TIMEOUT * 2):
                results.append(future.result())
        
        # Analyze results
        successful_requests: List[Dict[str, Any]] = [r for r in results if r["success"]]
        failed_requests: List[Dict[str, Any]] = [r for r in results if not r["success"]]
        
        print(f"📈 Concurrent Request Results:")
        print(f"   Successful: {len(successful_requests)}/{num_concurrent_requests}")
        print(f"   Failed: {len(failed_requests)}/{num_concurrent_requests}")
        
        if successful_requests:
            response_times: List[float] = [
                r["response_time_ms"] for r in successful_requests 
                if r["response_time_ms"] is not None
            ]
            
            if response_times:
                avg_time: float = statistics.mean(response_times)
                max_time: float = max(response_times)
                min_time: float = min(response_times)
                
                print(f"   Avg response time: {avg_time:.2f}ms")
                print(f"   Min response time: {min_time:.2f}ms") 
                print(f"   Max response time: {max_time:.2f}ms")
                
                # Verify reasonable performance under load
                assert avg_time < 10000  # Less than 10 seconds average
                assert max_time < 30000  # Less than 30 seconds max
        
        # Should have reasonable success rate
        success_rate: float = len(successful_requests) / num_concurrent_requests
        
        if success_rate < 0.6:  # Less than 60% success
            if any("grpc" in str(r.get("error", "")).lower() for r in failed_requests):
                pytest.skip("gRPC server not available for concurrent testing")
            else:
                pytest.fail(f"Low success rate for concurrent requests: {success_rate:.1%}")
    
    def test_grpc_connection_resilience(self) -> None:
        """
        Test gRPC connection resilience and error recovery.
        
        Raises:
            AssertionError: If connection resilience fails
        """
        test_data: Dict[str, float] = {
            "sepal_length": 5.4,
            "sepal_width": 3.9,
            "petal_length": 1.7,
            "petal_width": 0.4
        }
        
        # Make multiple requests to test connection stability
        num_requests: int = 5
        successful_requests: int = 0
        connection_errors: int = 0
        other_errors: int = 0
        
        for i in range(num_requests):
            try:
                response = requests.post(
                    f"{FLASK_BASE_URL}/api/v1/classify-grpc",
                    json=test_data,
                    timeout=TEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    successful_requests += 1
                elif response.status_code == 500:
                    error_data: Dict[str, Any] = response.json()
                    if any(keyword in error_data.get("error", "").lower() 
                           for keyword in ["grpc", "connection", "server"]):
                        connection_errors += 1
                    else:
                        other_errors += 1
                else:
                    other_errors += 1
                    
            except Exception:
                other_errors += 1
            
            # Small delay between requests
            time.sleep(0.1)
        
        print(f"🔄 Connection Resilience Test:")
        print(f"   Successful: {successful_requests}/{num_requests}")
        print(f"   Connection errors: {connection_errors}/{num_requests}")
        print(f"   Other errors: {other_errors}/{num_requests}")
        
        # At least some requests should succeed if gRPC server is available
        # Or all should fail consistently if server is not available
        total_errors: int = connection_errors + other_errors
        
        if successful_requests > 0:
            # Server is available, should have good success rate
            success_rate: float = successful_requests / num_requests
            assert success_rate >= 0.8, f"Poor success rate with available server: {success_rate:.1%}"
        else:
            # Server might not be available
            assert total_errors == num_requests, "Inconsistent error pattern"


class TestGRPCDataConsistency:
    """Tests for data consistency between REST and gRPC endpoints."""
    
    def test_rest_grpc_prediction_consistency(self) -> None:
        """
        Test that REST and gRPC endpoints produce consistent predictions.
        
        Raises:
            AssertionError: If predictions are inconsistent
        """
        # Test with multiple iris samples
        test_samples: List[Dict[str, Any]] = [
            {
                "name": "clear_setosa",
                "data": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            },
            {
                "name": "clear_versicolor",
                "data": {
                    "sepal_length": 7.0,
                    "sepal_width": 3.2,
                    "petal_length": 4.7,
                    "petal_width": 1.4
                }
            },
            {
                "name": "clear_virginica",
                "data": {
                    "sepal_length": 6.3,
                    "sepal_width": 3.3,
                    "petal_length": 6.0,
                    "petal_width": 2.5
                }
            }
        ]
        
        consistent_predictions: int = 0
        total_comparisons: int = 0
        
        for sample in test_samples:
            test_data: Dict[str, float] = sample["data"]
            
            # Get REST prediction
            rest_response = requests.post(
                f"{FLASK_BASE_URL}/api/v1/classify",
                json=test_data,
                timeout=TEST_TIMEOUT
            )
            
            # Get gRPC prediction
            grpc_response = requests.post(
                f"{FLASK_BASE_URL}/api/v1/classify-grpc",
                json=test_data,
                timeout=TEST_TIMEOUT
            )
            
            if rest_response.status_code == 200 and grpc_response.status_code == 200:
                total_comparisons += 1
                
                rest_data: Dict[str, Any] = rest_response.json()
                grpc_data: Dict[str, Any] = grpc_response.json()
                
                rest_prediction: str = rest_data["predicted_class"]
                grpc_prediction: str = grpc_data["predicted_class"]
                
                if rest_prediction == grpc_prediction:
                    consistent_predictions += 1
                    print(f"✅ {sample['name']}: {rest_prediction} (consistent)")
                else:
                    print(f"❌ {sample['name']}: REST={rest_prediction}, gRPC={grpc_prediction} (inconsistent)")
                
                # Confidence should be similar (within reasonable range)
                rest_confidence: float = rest_data["confidence"]
                grpc_confidence: float = grpc_data["confidence"]
                confidence_diff: float = abs(rest_confidence - grpc_confidence)
                
                # Allow some small differences due to floating point precision
                assert confidence_diff < 0.01, f"Large confidence difference for {sample['name']}: {confidence_diff}"
        
        if total_comparisons > 0:
            consistency_rate: float = consistent_predictions / total_comparisons
            print(f"🎯 Prediction Consistency: {consistency_rate:.1%} ({consistent_predictions}/{total_comparisons})")
            
            # Should have high consistency rate
            assert consistency_rate >= 0.9, f"Poor prediction consistency: {consistency_rate:.1%}"
        else:
            pytest.skip("No successful comparisons possible (gRPC server may not be available)")


# Pytest configuration and fixtures
@pytest.fixture(scope="session", autouse=True)
def check_services_available() -> None:
    """
    Check if required services are available for integration testing.
    
    Raises:
        pytest.skip: If services are not available
    """
    # Check Flask app
    try:
        response = requests.get(f"{FLASK_BASE_URL}/health", timeout=5)
        if response.status_code not in [200, 500]:
            pytest.skip("Flask server not available for integration testing")
    except Exception:
        pytest.skip("Flask server not available for integration testing")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])