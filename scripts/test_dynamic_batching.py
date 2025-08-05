#!/usr/bin/env python3
"""
Dynamic Batching Performance Demonstration for Triton Inference Server.

This script demonstrates the performance benefits of Triton's dynamic batching feature
by comparing individual requests vs concurrent requests that get automatically batched.

Following the coding guidelines: explicit type annotations and comprehensive documentation.
"""

import asyncio
import time
import logging
import statistics
import json
import sys
from typing import List, Dict, Any, Tuple
import concurrent.futures
import requests
import numpy as np
from sklearn.datasets import load_iris

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TritonBatchingDemo:
    """Demonstration class for Triton dynamic batching performance."""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        """
        Initialize the batching demo.
        
        Args:
            base_url: Base URL of the Flask application
        """
        self.base_url = base_url
        self.triton_endpoint = f"{base_url}/api/v1/classify-triton"
        self.onnx_endpoint = f"{base_url}/api/v1/classify"
        
        # Generate test data from Iris dataset
        iris = load_iris()
        self.test_samples = self._generate_test_samples(iris)
    
    def _generate_test_samples(self, iris) -> List[Dict[str, float]]:
        """
        Generate test samples from Iris dataset.
        
        Args:
            iris: Loaded Iris dataset
            
        Returns:
            List of test sample dictionaries
        """
        samples = []
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        # Use a subset of the actual Iris data for realistic testing
        for i in range(min(50, len(iris.data))):
            sample = {}
            for j, feature_name in enumerate(feature_names):
                sample[feature_name] = float(iris.data[i][j])
            samples.append(sample)
        
        return samples
    
    def make_single_request(self, endpoint: str, sample: Dict[str, float]) -> Tuple[float, bool]:
        """
        Make a single inference request.
        
        Args:
            endpoint: API endpoint URL
            sample: Input sample data
            
        Returns:
            Tuple of (response_time_ms, success_flag)
        """
        try:
            start_time = time.time()
            response = requests.post(
                endpoint,
                json=sample,
                headers={'Content-Type': 'application/json'},
                timeout=30.0
            )
            response_time = (time.time() - start_time) * 1000
            
            return response_time, response.status_code == 200
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return 0.0, False
    
    def make_concurrent_requests(
        self, 
        endpoint: str, 
        samples: List[Dict[str, float]], 
        max_workers: int = 10
    ) -> List[Tuple[float, bool]]:
        """
        Make concurrent inference requests.
        
        Args:
            endpoint: API endpoint URL
            samples: List of input samples
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of (response_time_ms, success_flag) tuples
        """
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests concurrently
            future_to_sample = {
                executor.submit(self.make_single_request, endpoint, sample): sample 
                for sample in samples
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_sample):
                try:
                    response_time, success = future.result()
                    results.append((response_time, success))
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    results.append((0.0, False))
        
        return results
    
    def run_sequential_test(self, endpoint: str, num_requests: int = 20) -> Dict[str, Any]:
        """
        Run sequential (individual) requests test.
        
        Args:
            endpoint: API endpoint URL
            num_requests: Number of requests to make
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running sequential test with {num_requests} requests...")
        
        start_time = time.time()
        results = []
        
        for i in range(num_requests):
            sample = self.test_samples[i % len(self.test_samples)]
            response_time, success = self.make_single_request(endpoint, sample)
            results.append((response_time, success))
            
            if not success:
                logger.warning(f"Request {i+1} failed")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        response_times = [rt for rt, success in results if success]
        success_count = sum(1 for _, success in results if success)
        
        return {
            "test_type": "sequential",
            "total_requests": num_requests,
            "successful_requests": success_count,
            "success_rate": success_count / num_requests,
            "total_time_seconds": total_time,
            "average_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "median_response_time_ms": statistics.median(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "throughput_requests_per_second": success_count / total_time if total_time > 0 else 0
        }
    
    def run_concurrent_test(self, endpoint: str, num_requests: int = 20, max_workers: int = 10) -> Dict[str, Any]:
        """
        Run concurrent requests test (enables dynamic batching).
        
        Args:
            endpoint: API endpoint URL
            num_requests: Number of requests to make
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running concurrent test with {num_requests} requests, {max_workers} workers...")
        
        # Prepare samples
        samples = [self.test_samples[i % len(self.test_samples)] for i in range(num_requests)]
        
        start_time = time.time()
        results = self.make_concurrent_requests(endpoint, samples, max_workers)
        total_time = time.time() - start_time
        
        # Calculate statistics
        response_times = [rt for rt, success in results if success]
        success_count = sum(1 for _, success in results if success)
        
        return {
            "test_type": "concurrent",
            "total_requests": num_requests,
            "successful_requests": success_count,
            "success_rate": success_count / num_requests,
            "total_time_seconds": total_time,
            "max_workers": max_workers,
            "average_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "median_response_time_ms": statistics.median(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "throughput_requests_per_second": success_count / total_time if total_time > 0 else 0
        }
    
    def compare_serving_infrastructures(self, num_requests: int = 20) -> Dict[str, Any]:
        """
        Compare performance between different serving infrastructures.
        
        Args:
            num_requests: Number of requests per test
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("=== Serving Infrastructure Comparison ===")
        
        results = {}
        
        # Test configurations
        test_configs = [
            {
                "name": "Direct ONNX (Sequential)",
                "endpoint": self.onnx_endpoint,
                "test_type": "sequential"
            },
            {
                "name": "Direct ONNX (Concurrent)",
                "endpoint": self.onnx_endpoint,
                "test_type": "concurrent"
            },
            {
                "name": "Triton (Sequential)",
                "endpoint": self.triton_endpoint,
                "test_type": "sequential"
            },
            {
                "name": "Triton (Concurrent - Dynamic Batching)",
                "endpoint": self.triton_endpoint,
                "test_type": "concurrent"
            }
        ]
        
        for config in test_configs:
            logger.info(f"\nTesting: {config['name']}")
            
            try:
                if config["test_type"] == "sequential":
                    result = self.run_sequential_test(config["endpoint"], num_requests)
                else:
                    result = self.run_concurrent_test(config["endpoint"], num_requests)
                
                result["configuration"] = config["name"]
                results[config["name"]] = result
                
                logger.info(f"✅ {config['name']}: {result['throughput_requests_per_second']:.2f} req/s")
                
            except Exception as e:
                logger.error(f"❌ {config['name']} failed: {e}")
                results[config["name"]] = {
                    "configuration": config["name"],
                    "error": str(e),
                    "throughput_requests_per_second": 0
                }
        
        return results
    
    def print_performance_analysis(self, results: Dict[str, Any]) -> None:
        """
        Print detailed performance analysis.
        
        Args:
            results: Results from comparison tests
        """
        print("\n" + "="*80)
        print("🚀 TRITON DYNAMIC BATCHING PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Extract key metrics
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not successful_results:
            print("❌ No successful tests to analyze")
            return
        
        # Sort by throughput
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]["throughput_requests_per_second"], reverse=True)
        
        print("\n📊 THROUGHPUT COMPARISON (Requests/Second)")
        print("-" * 50)
        for name, result in sorted_results:
            throughput = result["throughput_requests_per_second"]
            avg_latency = result["average_response_time_ms"]
            success_rate = result["success_rate"] * 100
            print(f"{name:<35} {throughput:>8.2f} req/s  |  {avg_latency:>6.1f}ms avg  |  {success_rate:>5.1f}% success")
        
        # Performance improvement analysis
        if len(sorted_results) >= 2:
            best_name, best_result = sorted_results[0]
            baseline_name, baseline_result = sorted_results[-1]
            
            improvement = (best_result["throughput_requests_per_second"] / baseline_result["throughput_requests_per_second"]) - 1
            
            print(f"\n🎯 PERFORMANCE IMPROVEMENT")
            print("-" * 50)
            print(f"Best: {best_name}")
            print(f"Baseline: {baseline_name}")
            print(f"Improvement: {improvement*100:.1f}% faster throughput")
        
        # Dynamic batching analysis
        triton_concurrent = results.get("Triton (Concurrent - Dynamic Batching)")
        triton_sequential = results.get("Triton (Sequential)")
        
        if triton_concurrent and triton_sequential and "error" not in triton_concurrent and "error" not in triton_sequential:
            batching_improvement = (triton_concurrent["throughput_requests_per_second"] / 
                                  triton_sequential["throughput_requests_per_second"]) - 1
            
            print(f"\n🔄 DYNAMIC BATCHING BENEFIT")
            print("-" * 50)
            print(f"Sequential Triton:  {triton_sequential['throughput_requests_per_second']:.2f} req/s")
            print(f"Concurrent Triton:  {triton_concurrent['throughput_requests_per_second']:.2f} req/s")
            print(f"Batching Benefit:   {batching_improvement*100:.1f}% throughput improvement")
        
        print("\n💡 KEY INSIGHTS:")
        print("- Triton's dynamic batching automatically groups concurrent requests")
        print("- This reduces per-request overhead and improves GPU/CPU utilization") 
        print("- Best for high-throughput scenarios with concurrent traffic")
        print("- Individual requests show similar latency across all serving methods")

def main() -> None:
    """Main function to run the dynamic batching demonstration."""
    
    logger.info("=== Triton Dynamic Batching Performance Demo ===")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Triton Dynamic Batching Performance Demo")
    parser.add_argument("--base-url", default="http://localhost:5001", help="Base URL of Flask app")
    parser.add_argument("--num-requests", type=int, default=20, help="Number of requests per test")
    parser.add_argument("--max-workers", type=int, default=8, help="Max concurrent workers")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = TritonBatchingDemo(args.base_url)
    
    logger.info(f"Testing with {args.num_requests} requests per test")
    logger.info(f"Using {args.max_workers} concurrent workers for batching tests")
    
    # Check if Flask app is running
    try:
        response = requests.get(f"{args.base_url}/health", timeout=5)
        if response.status_code != 200:
            logger.error("Flask application not responding properly")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Cannot connect to Flask app at {args.base_url}: {e}")
        logger.info("Please start the Flask app with: python app.py")
        sys.exit(1)
    
    # Run the comparison
    try:
        results = demo.compare_serving_infrastructures(args.num_requests)
        
        # Print analysis
        demo.print_performance_analysis(results)
        
        # Save results to file
        output_file = "triton_batching_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n📁 Results saved to: {output_file}")
        
        # Print setup instructions if Triton tests failed
        triton_failed = any("Triton" in k and "error" in v for k, v in results.items())
        if triton_failed:
            print("\n⚠️  TRITON SETUP REQUIRED")
            print("To test Triton Inference Server:")
            print("1. Install Triton: pip install tritonclient[http]")
            print("2. Download Triton server from NVIDIA NGC")
            print("3. Start Triton server:")
            print(f"   tritonserver --model-repository={demo.base_url.replace('http://localhost:5001', '')}/triton_model_repository")
            print("4. Re-run this script")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()