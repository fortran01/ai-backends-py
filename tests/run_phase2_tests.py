#!/usr/bin/env python3
"""
Test runner script for Phase 2 features of the AI backends project.

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive test execution with proper reporting
- Graceful handling of service dependencies
- Clear status reporting and error handling
"""

import os
import sys
import subprocess
import time
from typing import Dict, List, Any, Optional, Tuple
import requests
from requests.exceptions import ConnectionError, RequestException


# Configuration
FLASK_SERVER_URL: str = "http://localhost:5001"
GRPC_SERVER_URL: str = "localhost:50051"
OLLAMA_SERVER_URL: str = "http://localhost:11434"
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(__file__))


def check_service_availability(service_name: str, url: str, timeout: int = 5) -> bool:
    """
    Check if a service is available and responding.
    
    Args:
        service_name: Name of the service for logging
        url: URL to check
        timeout: Timeout in seconds
        
    Returns:
        True if service is available, False otherwise
    """
    try:
        if service_name == "gRPC":
            # For gRPC, we'll try to import and test connection
            import grpc
            from proto import inference_pb2, inference_pb2_grpc
            
            with grpc.insecure_channel(url) as channel:
                stub = inference_pb2_grpc.InferenceServiceStub(channel)
                request = inference_pb2.ClassifyRequest(
                    sepal_length=5.0, sepal_width=3.0,
                    petal_length=1.0, petal_width=0.1
                )
                stub.Classify(request, timeout=timeout)
            return True
        else:
            # For HTTP services
            if service_name == "Ollama":
                response = requests.get(f"{url}/api/tags", timeout=timeout)
            else:
                response = requests.get(f"{url}/health", timeout=timeout)
            return response.status_code in [200, 500]  # 500 is OK if dependencies are down
            
    except Exception:
        return False


def print_service_status() -> Dict[str, bool]:
    """
    Check and print the status of all required services.
    
    Returns:
        Dictionary mapping service names to availability status
    """
    services: Dict[str, Tuple[str, str]] = {
        "Flask": (FLASK_SERVER_URL, "Flask API Server"),
        "gRPC": (GRPC_SERVER_URL, "gRPC Inference Server"),
        "Ollama": (OLLAMA_SERVER_URL, "Ollama LLM Service")
    }
    
    print("🔍 Checking Service Availability:")
    print("=" * 50)
    
    status: Dict[str, bool] = {}
    
    for service_name, (url, description) in services.items():
        available: bool = check_service_availability(service_name, url)
        status[service_name] = available
        
        if available:
            print(f"✅ {description}: Available at {url}")
        else:
            print(f"❌ {description}: Not available at {url}")
    
    print()
    return status


def run_test_suite(test_module: str, test_description: str, markers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run a specific test suite and return results.
    
    Args:
        test_module: Python module name for the tests
        test_description: Human-readable description
        markers: Optional pytest markers to filter tests
        
    Returns:
        Dictionary with test results
    """
    print(f"🧪 Running {test_description}...")
    
    # Build pytest command
    cmd: List[str] = ["python", "-m", "pytest", f"tests/{test_module}", "-v", "--tb=short"]
    
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    try:
        # Run the test suite
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "description": test_description
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Test suite timed out after 5 minutes",
            "description": test_description
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "description": test_description
        }


def run_all_phase2_tests(service_status: Dict[str, bool]) -> Dict[str, Dict[str, Any]]:
    """
    Run all Phase 2 test suites based on service availability.
    
    Args:
        service_status: Dictionary of service availability
        
    Returns:
        Dictionary mapping test suite names to results
    """
    test_suites: List[Dict[str, Any]] = [
        {
            "module": "test_phase2_features.py",
            "description": "Phase 2 Flask API Features",
            "markers": None,
            "required_services": ["Flask"]
        },
        {
            "module": "test_serialization.py", 
            "description": "Serialization Utilities",
            "markers": None,
            "required_services": ["Flask"]
        },
        {
            "module": "test_langchain_memory.py",
            "description": "LangChain Memory Management",
            "markers": None,
            "required_services": ["Flask"]
        },
        {
            "module": "test_grpc_server.py",
            "description": "gRPC Server Unit Tests",
            "markers": ["unit"],
            "required_services": []
        },
        {
            "module": "test_grpc_integration.py",
            "description": "gRPC Integration Tests",
            "markers": None,
            "required_services": ["Flask"]
        },
        {
            "module": "test_e2e_playwright.py::TestPhase2APIWorkflows",
            "description": "Phase 2 E2E Playwright Tests",
            "markers": ["e2e"],
            "required_services": ["Flask"]
        }
    ]
    
    results: Dict[str, Dict[str, Any]] = {}
    
    for suite in test_suites:
        # Check if required services are available
        can_run: bool = all(
            service_status.get(service, False) 
            for service in suite["required_services"]
        )
        
        if not can_run and suite["required_services"]:
            missing_services: List[str] = [
                service for service in suite["required_services"]
                if not service_status.get(service, False)
            ]
            
            results[suite["description"]] = {
                "success": False,
                "returncode": -2,
                "stdout": "",
                "stderr": f"Skipped: Missing required services: {', '.join(missing_services)}",
                "description": suite["description"],
                "skipped": True
            }
            print(f"⏭️  Skipping {suite['description']} (missing services: {', '.join(missing_services)})")
            continue
        
        # Run the test suite
        result: Dict[str, Any] = run_test_suite(
            suite["module"],
            suite["description"],
            suite["markers"]
        )
        
        result["skipped"] = False
        results[suite["description"]] = result
        
        # Print immediate result
        if result["success"]:
            print(f"✅ {suite['description']}: PASSED")
        else:
            print(f"❌ {suite['description']}: FAILED")
            if result["stderr"]:
                print(f"   Error: {result['stderr'][:200]}...")
    
    return results


def print_test_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a comprehensive summary of test results.
    
    Args:
        results: Dictionary of test results
    """
    print("\n" + "=" * 60)
    print("📊 PHASE 2 TEST SUMMARY")
    print("=" * 60)
    
    total_suites: int = len(results)
    passed_suites: int = sum(1 for r in results.values() if r["success"])
    failed_suites: int = sum(1 for r in results.values() if not r["success"] and not r.get("skipped", False))
    skipped_suites: int = sum(1 for r in results.values() if r.get("skipped", False))
    
    print(f"Total Test Suites: {total_suites}")
    print(f"✅ Passed: {passed_suites}")
    print(f"❌ Failed: {failed_suites}")
    print(f"⏭️  Skipped: {skipped_suites}")
    print()
    
    # Detailed results
    for suite_name, result in results.items():
        status_icon: str = "✅" if result["success"] else ("⏭️" if result.get("skipped", False) else "❌")
        print(f"{status_icon} {suite_name}")
        
        if not result["success"] and not result.get("skipped", False):
            if result["stderr"]:
                print(f"   Error: {result['stderr']}")
            elif "FAILED" in result["stdout"]:
                # Extract failed test information
                lines: List[str] = result["stdout"].split('\n')
                failed_lines: List[str] = [line for line in lines if "FAILED" in line]
                for line in failed_lines[:3]:  # Show first 3 failures
                    print(f"   {line.strip()}")
                if len(failed_lines) > 3:
                    print(f"   ... and {len(failed_lines) - 3} more failures")
    
    print()
    
    # Recommendations
    if failed_suites > 0:
        print("🔧 RECOMMENDATIONS:")
        print("- Check that the Flask server is running: python app.py")
        if any("grpc" in r.get("stderr", "").lower() for r in results.values()):
            print("- Start gRPC server for full testing: python grpc_server.py")
        if any("ollama" in r.get("stderr", "").lower() for r in results.values()):
            print("- Install and start Ollama for LLM testing: https://ollama.ai/")
        print("- Review error details above for specific issues")
        print()
    
    # Overall status
    if failed_suites == 0:
        print("🎉 ALL TESTS PASSED! Phase 2 implementation is working correctly.")
    elif passed_suites > 0:
        print(f"⚠️  PARTIAL SUCCESS: {passed_suites}/{total_suites} test suites passed.")
    else:
        print("🚨 ALL TESTS FAILED: Please check your setup and configuration.")


def main() -> int:
    """
    Main function to run all Phase 2 tests.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("🚀 AI Backends Phase 2 Test Runner")
    print("=" * 40)
    print()
    
    # Change to project directory
    os.chdir(PROJECT_ROOT)
    
    # Check service availability
    service_status: Dict[str, bool] = print_service_status()
    
    # Check if we can run any tests
    if not service_status.get("Flask", False):
        print("❌ Cannot run tests: Flask server is not available")
        print("   Please start the Flask server with: python app.py")
        return 1
    
    # Run all test suites
    print("🏃 Starting Test Execution...")
    print("=" * 30)
    
    results: Dict[str, Dict[str, Any]] = run_all_phase2_tests(service_status)
    
    # Print summary
    print_test_summary(results)
    
    # Determine exit code
    failed_count: int = sum(1 for r in results.values() if not r["success"] and not r.get("skipped", False))
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())