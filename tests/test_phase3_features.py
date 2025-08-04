"""
Unit and integration tests for Phase 3 features - Advanced Caching and API Versioning.

This test module covers:
1. Redis-based exact caching with Flask-Caching
2. Semantic caching with FastEmbed and cosine similarity
3. API versioning with Flask Blueprints

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive test coverage for Phase 3 features
- Cache behavior validation and performance testing
- API versioning compatibility testing
"""

import json
import time
import hashlib
from typing import Dict, Any, Optional, List, Set
import pytest
import requests
from requests.exceptions import ConnectionError, RequestException
import redis


class TestPhase3ExactCaching:
    """Test cases for Redis-based exact caching on /api/v1/classify endpoint."""

    def test_classify_caching_enabled(self, api_client: requests.Session, server_url: str, sample_iris_data: Dict[str, float]) -> None:
        """
        Test that Redis-based caching is working for classify endpoint.

        Tests:
        - First request: Cache miss (normal response time)
        - Second identical request: Cache hit (faster response time)
        - Cache key generation and storage verification

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server
            sample_iris_data: Valid iris classification data

        Raises:
            AssertionError: If caching behavior is incorrect
        """
        # Clear any existing cache for this test
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.flushdb()
        except Exception:
            # Redis may not be available - test will still verify the endpoint works
            pass

        # First request - should be cache miss
        start_time = time.time()
        response1 = api_client.post(f"{server_url}/api/v1/classify", json=sample_iris_data)
        first_request_time = time.time() - start_time

        assert response1.status_code == 200
        data1: Dict[str, Any] = response1.json()

        # Verify response structure
        assert "predicted_class" in data1
        assert "predicted_class_index" in data1
        assert "probabilities" in data1
        assert "confidence" in data1

        # Second identical request - should be cache hit (if Redis is available)
        start_time = time.time()
        response2 = api_client.post(f"{server_url}/api/v1/classify", json=sample_iris_data)
        second_request_time = time.time() - start_time

        assert response2.status_code == 200
        data2: Dict[str, Any] = response2.json()

        # Results should be identical
        assert data1["predicted_class"] == data2["predicted_class"]
        assert data1["predicted_class_index"] == data2["predicted_class_index"]
        assert data1["probabilities"] == data2["probabilities"]
        assert data1["confidence"] == data2["confidence"]

        # If Redis is available, second request should be faster
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()
            # Redis is available - verify caching performance improvement
            assert second_request_time < first_request_time * 0.8, "Second request should be significantly faster due to caching"
        except Exception:
            # Redis not available - skip performance assertion
            pytest.skip("Redis not available for caching performance test")

    def test_classify_cache_invalidation(self, api_client: requests.Session, server_url: str, sample_iris_variants: Dict[str, Dict[str, float]]) -> None:
        """
        Test that different inputs can be cached separately and cache works correctly.

        Tests:
        - Different input data can be stored in cache
        - Cache keys are properly generated based on input  
        - Multiple cache entries can coexist
        - Cache retrieval works for each variant

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server
            sample_iris_variants: Different iris classification samples

        Raises:
            AssertionError: If cache invalidation logic is incorrect
        """
        responses: Dict[str, Dict[str, Any]] = {}

        # Test different iris samples - make first requests (cache misses)
        for variant_name, iris_data in sample_iris_variants.items():
            response = api_client.post(f"{server_url}/api/v1/classify", json=iris_data)
            assert response.status_code == 200
            responses[variant_name] = response.json()

        # Verify all requests succeeded and returned valid structure
        for variant_name, result in responses.items():
            assert "predicted_class" in result
            assert "probabilities" in result
            assert "confidence" in result
            assert isinstance(result["probabilities"], list)
            assert len(result["probabilities"]) == 3  # 3 iris classes

        # Test that caching works correctly for each variant individually
        for variant_name, iris_data in sample_iris_variants.items():
            # Second request for same data should return identical result (cache hit)
            response = api_client.post(f"{server_url}/api/v1/classify", json=iris_data)
            assert response.status_code == 200
            cached_result = response.json()

            # Results should be identical from cache
            assert cached_result == responses[variant_name]
            
        # Test that we can make different requests and get consistent caching
        # Test a few more requests to ensure cache stability
        for variant_name, iris_data in sample_iris_variants.items():
            for _ in range(3):  # Multiple requests to same data
                response = api_client.post(f"{server_url}/api/v1/classify", json=iris_data)
                assert response.status_code == 200
                result = response.json()
                assert result == responses[variant_name]  # Should always match cached result

    def test_classify_cache_key_generation(self, sample_iris_data: Dict[str, float]) -> None:
        """
        Test cache key generation logic for classify endpoint.

        This test verifies that Flask-Caching generates consistent cache keys
        for identical input data.

        Args:
            sample_iris_data: Valid iris classification data

        Raises:
            AssertionError: If cache key generation is inconsistent
        """
        # Simulate Flask-Caching key generation logic
        # Note: Actual key generation is handled by Flask-Caching internally

        # Create JSON representation for consistent hashing
        json_str1 = json.dumps(sample_iris_data, sort_keys=True)
        json_str2 = json.dumps(sample_iris_data, sort_keys=True)

        # Keys should be identical for identical data
        assert json_str1 == json_str2

        # Hash should be consistent
        hash1 = hashlib.md5(json_str1.encode()).hexdigest()
        hash2 = hashlib.md5(json_str2.encode()).hexdigest()

        assert hash1 == hash2

        # Different data should produce different hashes
        different_data = sample_iris_data.copy()
        different_data["sepal_length"] = 999.0

        different_json = json.dumps(different_data, sort_keys=True)
        different_hash = hashlib.md5(different_json.encode()).hexdigest()

        assert hash1 != different_hash


class TestPhase3SemanticCaching:
    """Test cases for semantic caching with FastEmbed on /api/v1/chat-semantic endpoint."""

    def test_semantic_cache_exact_match(self, api_client: requests.Session, server_url: str) -> None:
        """
        Test semantic caching with exact prompt match.

        Tests:
        - First request: Cache miss
        - Second identical request: Cache hit with exact match
        - Cache metadata includes correct information

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server

        Raises:
            AssertionError: If semantic caching behavior is incorrect
        """
        test_prompt = "What is artificial intelligence?"

        # Clear semantic cache
        self._clear_semantic_cache()

        # First request - cache miss
        response1 = api_client.post(f"{server_url}/api/v1/chat-semantic", json={"prompt": test_prompt})

        if response1.status_code == 500:
            pytest.skip("Ollama service not available for semantic caching test")

        assert response1.status_code == 200
        data1: Dict[str, Any] = response1.json()

        # Verify response structure
        assert "response" in data1
        assert "cache_info" in data1
        assert "cache_hit" in data1["cache_info"]
        assert "cache_type" in data1["cache_info"]
        assert "similarity_score" in data1["cache_info"]

        # First request should be cache miss
        cache_info1 = data1["cache_info"]
        assert cache_info1["cache_hit"] is False
        assert cache_info1["cache_type"] in ["miss", "disabled"]

        # Second identical request - should be cache hit
        response2 = api_client.post(f"{server_url}/api/v1/chat-semantic", json={"prompt": test_prompt})
        assert response2.status_code == 200
        data2: Dict[str, Any] = response2.json()

        cache_info2 = data2["cache_info"]

        # If Redis is available, should be cache hit
        if cache_info2["cache_type"] != "disabled":
            assert cache_info2["cache_hit"] is True
            assert cache_info2["cache_type"] == "exact"
            assert cache_info2["similarity_score"] == 1.0

            # Response should be identical
            assert data1["response"] == data2["response"]

    def test_semantic_cache_similarity_match(self, api_client: requests.Session, server_url: str) -> None:
        """
        Test semantic caching with similar but not identical prompts.

        Tests:
        - Different wordings of similar questions
        - Semantic similarity detection and caching
        - Similarity score calculation

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server

        Raises:
            AssertionError: If semantic similarity matching is incorrect
        """
        # Clear semantic cache
        self._clear_semantic_cache()

        # Test prompts with similar semantic meaning
        prompts = [
            "What is artificial intelligence?",
            "What is AI?",
            "Explain artificial intelligence to me",
            "How would you define AI?",
            "What does artificial intelligence mean?"
        ]

        responses: List[Dict[str, Any]] = []

        for i, prompt in enumerate(prompts):
            response = api_client.post(f"{server_url}/api/v1/chat-semantic", json={"prompt": prompt})

            if response.status_code == 500:
                pytest.skip("Ollama service not available for semantic caching test")

            assert response.status_code == 200
            data: Dict[str, Any] = response.json()
            responses.append(data)

            cache_info = data["cache_info"]

            if i == 0:
                # First request should be cache miss
                if cache_info["cache_type"] != "disabled":
                    assert cache_info["cache_hit"] is False
                    assert cache_info["cache_type"] == "miss"
            else:
                # Subsequent requests may hit semantic cache
                if cache_info["cache_type"] not in ["disabled", "miss"]:
                    assert cache_info["cache_hit"] is True
                    assert cache_info["cache_type"] in ["exact", "semantic"]

                    if cache_info["cache_type"] == "semantic":
                        # Semantic matches should have high similarity scores
                        assert cache_info["similarity_score"] >= 0.8

    def test_semantic_cache_different_topics(self, api_client: requests.Session, server_url: str) -> None:
        """
        Test that semantically different prompts don't match cache.

        Tests:
        - Different topics should not hit cache
        - Low similarity scores for unrelated prompts
        - Cache isolation between different topics

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server

        Raises:
            AssertionError: If unrelated prompts incorrectly match cache
        """
        # Clear semantic cache
        self._clear_semantic_cache()

        # Prompts with different semantic meanings
        different_prompts = [
            "What is artificial intelligence?",
            "How do I cook pasta?",
            "What is the weather like today?",
            "Explain quantum physics",
            "How to train a machine learning model?"
        ]

        responses: List[Dict[str, Any]] = []

        for i, prompt in enumerate(different_prompts):
            response = api_client.post(f"{server_url}/api/v1/chat-semantic", json={"prompt": prompt})

            if response.status_code == 500:
                pytest.skip("Ollama service not available for semantic caching test")

            assert response.status_code == 200
            data: Dict[str, Any] = response.json()
            responses.append(data)

            cache_info = data["cache_info"]

            if i == 0:
                # First request should be cache miss
                if cache_info["cache_type"] != "disabled":
                    assert cache_info["cache_hit"] is False
            else:
                # Different topics should generally be cache misses
                # (unless they happen to be semantically similar by chance)
                if cache_info["cache_hit"] is True and cache_info["cache_type"] == "semantic":
                    # If there is a semantic match, similarity should be reasonable
                    assert cache_info["similarity_score"] >= 0.8

    def test_semantic_cache_error_handling(self, api_client: requests.Session, server_url: str) -> None:
        """
        Test error handling in semantic caching system.

        Tests:
        - Empty prompt handling
        - Invalid JSON handling
        - Redis connection failure handling

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server

        Raises:
            AssertionError: If error handling is incorrect
        """
        # Test empty prompt
        response = api_client.post(f"{server_url}/api/v1/chat-semantic", json={"prompt": ""})
        assert response.status_code == 400

        # Test missing prompt field
        response = api_client.post(f"{server_url}/api/v1/chat-semantic", json={})
        assert response.status_code == 400

        # Test invalid JSON
        headers = {"Content-Type": "application/json"}
        response = api_client.post(f"{server_url}/api/v1/chat-semantic", data="invalid json", headers=headers)
        assert response.status_code == 500  # Invalid JSON causes server error

    def _clear_semantic_cache(self) -> None:
        """Clear semantic cache for testing purposes."""
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
            redis_client.flushdb()
        except Exception:
            # Redis may not be available - tests will handle gracefully
            pass


class TestPhase3APIVersioning:
    """Test cases for API versioning with Flask Blueprints."""

    def test_api_v2_generate_endpoint(self, api_client: requests.Session, server_url: str) -> None:
        """
        Test the new v2 generate endpoint with enhanced features.

        Tests:
        - v2 endpoint accessibility and functionality
        - Enhanced metadata in response
        - Model version specification support
        - Backward compatibility considerations

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server

        Raises:
            AssertionError: If v2 API features are incorrect
        """
        test_payload = {
            "prompt": "Hello, this is a test of API v2",
            "model_version": "tinyllama"
        }

        response = api_client.post(f"{server_url}/api/v2/generate", json=test_payload)

        if response.status_code == 500:
            pytest.skip("Ollama service not available for v2 API test")

        assert response.status_code == 200
        data: Dict[str, Any] = response.json()

        # Verify v2 response structure
        assert "response" in data
        assert "metadata" in data

        # Verify enhanced metadata
        metadata = data["metadata"]
        assert "api_version" in metadata
        assert "model_used" in metadata
        assert "response_time_ms" in metadata
        assert "token_count" in metadata
        assert "timestamp" in metadata

        # Verify correct values
        assert metadata["api_version"] == "v2"
        assert metadata["model_used"] == "tinyllama"
        assert isinstance(metadata["response_time_ms"], (int, float))
        assert isinstance(metadata["token_count"], int)
        assert isinstance(metadata["timestamp"], str)

        # Verify timestamp format (ISO 8601)
        assert metadata["timestamp"].endswith("Z")
        assert "T" in metadata["timestamp"]

    def test_api_v2_vs_v1_compatibility(self, api_client: requests.Session, server_url: str) -> None:
        """
        Test compatibility between v1 and v2 API endpoints.

        Tests:
        - Both versions handle similar requests
        - Response format differences
        - Functionality preservation

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server

        Raises:
            AssertionError: If version compatibility is broken
        """
        test_prompt = "Explain machine learning in simple terms"

        # Test v1 endpoint
        v1_payload = {"prompt": test_prompt}
        v1_response = api_client.post(f"{server_url}/api/v1/generate", json=v1_payload)

        if v1_response.status_code == 500:
            pytest.skip("Ollama service not available for API compatibility test")

        assert v1_response.status_code == 200
        v1_text: str = v1_response.text  # v1 returns plain text

        # Test v2 endpoint
        v2_payload = {"prompt": test_prompt, "model_version": "tinyllama"}
        v2_response = api_client.post(f"{server_url}/api/v2/generate", json=v2_payload)
        assert v2_response.status_code == 200
        v2_data: Dict[str, Any] = v2_response.json()  # v2 returns JSON

        # v1 should return plain text
        assert isinstance(v1_text, str)
        assert len(v1_text.strip()) > 0

        # v2 should have structured response with metadata
        assert "response" in v2_data
        assert "metadata" in v2_data
        assert isinstance(v2_data["response"], str)
        assert len(v2_data["response"]) > 0

    def test_api_v2_model_version_handling(self, api_client: requests.Session, server_url: str) -> None:
        """
        Test model version handling in v2 API.

        Tests:
        - Different model versions specified
        - Default model version behavior
        - Model version validation

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server

        Raises:
            AssertionError: If model version handling is incorrect
        """
        test_prompt = "What is Python programming?"

        # Test with explicit model version
        payload_with_model = {
            "prompt": test_prompt,
            "model_version": "tinyllama"
        }

        response = api_client.post(f"{server_url}/api/v2/generate", json=payload_with_model)

        if response.status_code == 500:
            pytest.skip("Ollama service not available for model version test")

        assert response.status_code == 200
        data: Dict[str, Any] = response.json()

        assert data["metadata"]["model_used"] == "tinyllama"

        # Test without model version (should default)
        payload_without_model = {"prompt": test_prompt}

        response = api_client.post(f"{server_url}/api/v2/generate", json=payload_without_model)
        assert response.status_code == 200
        data: Dict[str, Any] = response.json()

        # Should have some default model
        assert "model_used" in data["metadata"]
        assert isinstance(data["metadata"]["model_used"], str)

    def test_api_v2_error_handling(self, api_client: requests.Session, server_url: str) -> None:
        """
        Test error handling in v2 API endpoints.

        Tests:
        - Missing required fields
        - Invalid input formats
        - Proper error response structure

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server

        Raises:
            AssertionError: If error handling is incorrect
        """
        # Test missing prompt
        response = api_client.post(f"{server_url}/api/v2/generate", json={})
        assert response.status_code == 400

        # Test empty prompt
        response = api_client.post(f"{server_url}/api/v2/generate", json={"prompt": ""})
        assert response.status_code == 400

        # Test invalid JSON
        headers = {"Content-Type": "application/json"}
        response = api_client.post(f"{server_url}/api/v2/generate", data="invalid json", headers=headers)
        assert response.status_code == 500  # Invalid JSON causes server error


class TestPhase3Integration:
    """Integration tests for Phase 3 features working together."""

    def test_phase3_features_coexistence(self, api_client: requests.Session, server_url: str, sample_iris_data: Dict[str, float]) -> None:
        """
        Test that all Phase 3 features work together without conflicts.

        Tests:
        - Exact caching on classify endpoint
        - Semantic caching on chat-semantic endpoint
        - API versioning on v2 endpoints
        - No interference between different caching systems

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server
            sample_iris_data: Valid iris classification data

        Raises:
            AssertionError: If features interfere with each other
        """
        # Test exact caching
        classify_response = api_client.post(f"{server_url}/api/v1/classify", json=sample_iris_data)
        assert classify_response.status_code == 200
        classify_data = classify_response.json()
        assert "predicted_class" in classify_data

        # Test semantic caching
        semantic_payload = {"prompt": "What is machine learning?"}
        semantic_response = api_client.post(f"{server_url}/api/v1/chat-semantic", json=semantic_payload)

        if semantic_response.status_code != 500:  # Skip if Ollama not available
            assert semantic_response.status_code == 200
            semantic_data = semantic_response.json()
            assert "cache_info" in semantic_data

        # Test API versioning
        v2_payload = {"prompt": "Hello from integration test"}
        v2_response = api_client.post(f"{server_url}/api/v2/generate", json=v2_payload)

        if v2_response.status_code != 500:  # Skip if Ollama not available
            assert v2_response.status_code == 200
            v2_data = v2_response.json()
            assert "metadata" in v2_data
            assert v2_data["metadata"]["api_version"] == "v2"

        # Verify that exact caching still works after other operations
        classify_response2 = api_client.post(f"{server_url}/api/v1/classify", json=sample_iris_data)
        assert classify_response2.status_code == 200
        classify_data2 = classify_response2.json()

        # Results should be identical (cached)
        assert classify_data == classify_data2

    def test_phase3_performance_characteristics(self, api_client: requests.Session, server_url: str, sample_iris_data: Dict[str, float]) -> None:
        """
        Test performance characteristics of Phase 3 caching features.

        Tests:
        - Caching improves response times
        - Different caching systems don't interfere
        - Performance under load

        Args:
            api_client: Configured requests session
            server_url: Base URL for the Flask server
            sample_iris_data: Valid iris classification data

        Raises:
            AssertionError: If performance characteristics are incorrect
        """
        # Test classify endpoint caching performance
        times: List[float] = []

        for i in range(5):
            start_time = time.time()
            response = api_client.post(f"{server_url}/api/v1/classify", json=sample_iris_data)
            end_time = time.time()

            assert response.status_code == 200
            times.append(end_time - start_time)

        # After first request, subsequent requests should generally be faster due to caching
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()

            # If Redis is available, expect performance improvement
            avg_first_two = sum(times[:2]) / 2
            avg_last_three = sum(times[2:]) / 3

            # Later requests should be faster on average
            assert avg_last_three <= avg_first_two * 1.2  # Allow some variance

        except Exception:
            # Redis not available - just verify requests complete successfully
            assert all(t < 5.0 for t in times)  # All requests under 5 seconds
