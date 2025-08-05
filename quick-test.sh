#!/bin/bash

# AI Backend Quick Test Script
# Automates high-level tests from README.md curl examples
# Usage: ./quick-test.sh [options]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FLASK_URL="http://localhost:5001"
GRPC_SERVER_URL="http://localhost:50051"
HTTP_SERVER_URL="http://localhost:5002"
TF_SERVING_URL="http://localhost:8501"
TRITON_URL="http://localhost:8000"
MLFLOW_URL="http://localhost:5004"

# Default options
RUN_ALL=true
RUN_PHASE1=false
RUN_PHASE2=false
RUN_PHASE3=false
RUN_PHASE4=false
RUN_PHASE5=false
VERBOSE=false
SKIP_OPTIONAL=false

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO") echo -e "${BLUE}[INFO]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
        "WARNING") echo -e "${YELLOW}[WARNING]${NC} $message" ;;
    esac
}

# Function to check if a service is running
check_service() {
    local url=$1
    local service_name=$2
    if curl -s --connect-timeout 3 "$url" > /dev/null 2>&1; then
        print_status "SUCCESS" "$service_name is running"
        return 0
    else
        print_status "WARNING" "$service_name is not running"
        return 1
    fi
}

# Function to run a test
run_test() {
    local test_name=$1
    local curl_command=$2
    local optional=${3:-false}
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    print_status "INFO" "Running test: $test_name"
    
    if [ "$VERBOSE" = true ]; then
        echo "Command: $curl_command"
    fi
    
    if eval "$curl_command" > temp_response.json 2>&1; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        print_status "SUCCESS" "$test_name - PASSED"
        
        if [ "$VERBOSE" = true ]; then
            echo "Response:"
            cat temp_response.json | jq . 2>/dev/null || cat temp_response.json
            echo
        fi
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        if [ "$optional" = true ] && [ "$SKIP_OPTIONAL" = true ]; then
            print_status "WARNING" "$test_name - SKIPPED (optional service unavailable)"
        else
            print_status "ERROR" "$test_name - FAILED"
            if [ "$VERBOSE" = true ]; then
                echo "Error output:"
                cat temp_response.json 2>/dev/null || echo "No response captured"
                echo
            fi
        fi
    fi
    
    rm -f temp_response.json
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --phase1          Run Phase 1 tests only"
    echo "  --phase2          Run Phase 2 tests only"
    echo "  --phase3          Run Phase 3 tests only"
    echo "  --phase4          Run Phase 4 tests only"
    echo "  --phase5          Run Phase 5 tests only"
    echo "  --verbose, -v     Verbose output (show requests/responses)"
    echo "  --skip-optional   Skip optional tests if services unavailable"
    echo "  --help, -h        Show this help message"
    echo
    echo "Examples:"
    echo "  $0                Run all tests"
    echo "  $0 --phase1 -v    Run Phase 1 tests with verbose output"
    echo "  $0 --skip-optional Run tests, skip optional services"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase1)
            RUN_ALL=false
            RUN_PHASE1=true
            shift
            ;;
        --phase2)
            RUN_ALL=false
            RUN_PHASE2=true
            shift
            ;;
        --phase3)
            RUN_ALL=false
            RUN_PHASE3=true
            shift
            ;;
        --phase4)
            RUN_ALL=false
            RUN_PHASE4=true
            shift
            ;;
        --phase5)
            RUN_ALL=false
            RUN_PHASE5=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --skip-optional)
            SKIP_OPTIONAL=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_status "INFO" "Starting AI Backend Quick Tests"
echo

# Check required dependencies
print_status "INFO" "Checking dependencies..."
command -v curl >/dev/null 2>&1 || { print_status "ERROR" "curl is required but not installed"; exit 1; }
command -v jq >/dev/null 2>&1 || print_status "WARNING" "jq not installed - JSON responses won't be pretty-printed"

# Check service availability
print_status "INFO" "Checking service availability..."
FLASK_AVAILABLE=$(check_service "$FLASK_URL/health" "Flask API"; echo $?)
GRPC_AVAILABLE=$(check_service "$GRPC_SERVER_URL" "gRPC Server"; echo $?)  
HTTP_AVAILABLE=$(check_service "$HTTP_SERVER_URL" "HTTP Server"; echo $?)
TF_SERVING_AVAILABLE=$(check_service "$TF_SERVING_URL/v1/models/iris" "TensorFlow Serving"; echo $?)
TRITON_AVAILABLE=$(check_service "$TRITON_URL/v2/health/ready" "Triton Inference Server"; echo $?)
MLFLOW_AVAILABLE=$(check_service "$MLFLOW_URL" "MLflow Server"; echo $?)

echo

# Phase 1 Tests
if [ "$RUN_ALL" = true ] || [ "$RUN_PHASE1" = true ]; then
    print_status "INFO" "=== PHASE 1 TESTS ==="
    
    # Health check
    run_test "Health Check" \
        "curl -s $FLASK_URL/health"
    
    # Basic LLM generation
    run_test "LLM Generation (Basic)" \
        "curl -s -X POST $FLASK_URL/api/v1/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"What is AI?\"}'"
    
    # Secure LLM generation
    run_test "LLM Generation (Secure)" \
        "curl -s -X POST $FLASK_URL/api/v1/generate-secure -H 'Content-Type: application/json' -d '{\"prompt\": \"Explain machine learning\"}'"
    
    # Iris classification
    run_test "Iris Classification (ONNX)" \
        "curl -s -X POST $FLASK_URL/api/v1/classify -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'"
    
    echo
fi

# Phase 2 Tests
if [ "$RUN_ALL" = true ] || [ "$RUN_PHASE2" = true ]; then
    print_status "INFO" "=== PHASE 2 TESTS ==="
    
    # Stateful chat
    run_test "Stateful Chat (MCP)" \
        "curl -s -X POST $FLASK_URL/api/v1/chat -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello, what can you help me with?\", \"session_id\": \"test-session-1\"}'"
    
    # Detailed classification
    run_test "Detailed Classification" \
        "curl -s -X POST $FLASK_URL/api/v1/classify-detailed -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'"
    
    # HTTP classification (optional)
    if [ "$HTTP_AVAILABLE" -eq 0 ] || [ "$SKIP_OPTIONAL" = false ]; then
        run_test "HTTP Classification" \
            "curl -s -X POST $FLASK_URL/api/v1/classify-http -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'" \
            true
    fi
    
    # gRPC classification (optional)
    if [ "$GRPC_AVAILABLE" -eq 0 ] || [ "$SKIP_OPTIONAL" = false ]; then
        run_test "gRPC Classification" \
            "curl -s -X POST $FLASK_URL/api/v1/classify-grpc -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'" \
            true
    fi
    
    # Performance benchmark (optional)
    if [ "$GRPC_AVAILABLE" -eq 0 ] && [ "$HTTP_AVAILABLE" -eq 0 ] || [ "$SKIP_OPTIONAL" = false ]; then
        run_test "Performance Benchmark (HTTP vs gRPC)" \
            "curl -s -X POST $FLASK_URL/api/v1/classify-benchmark -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2, \"iterations\": 5}'" \
            true
    fi
    
    echo
fi

# Phase 3 Tests
if [ "$RUN_ALL" = true ] || [ "$RUN_PHASE3" = true ]; then
    print_status "INFO" "=== PHASE 3 TESTS ==="
    
    # Semantic caching chat
    run_test "Semantic Caching Chat" \
        "curl -s -X POST $FLASK_URL/api/v1/chat-semantic -H 'Content-Type: application/json' -d '{\"prompt\": \"What is AI?\"}'"
    
    # API versioning (v2)
    run_test "API Versioning (v2)" \
        "curl -s -X POST $FLASK_URL/api/v2/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello, this is API v2 test\", \"model_version\": \"tinyllama\"}'"
    
    # Test semantic similarity (second call should be cached)
    run_test "Semantic Similarity Test" \
        "curl -s -X POST $FLASK_URL/api/v1/chat-semantic -H 'Content-Type: application/json' -d '{\"prompt\": \"What is artificial intelligence?\"}'"
    
    echo
fi

# Phase 4 Tests
if [ "$RUN_ALL" = true ] || [ "$RUN_PHASE4" = true ]; then
    print_status "INFO" "=== PHASE 4 TESTS ==="
    
    # Drift monitoring report
    run_test "Drift Monitoring Report" \
        "curl -s '$FLASK_URL/api/v1/drift-report?limit=50'"
    
    # Drift simulation
    run_test "Drift Simulation" \
        "curl -s -X POST $FLASK_URL/api/v1/classify-shifted -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'"
    
    # Model registry classification (modern approach with aliases)
    if [ "$MLFLOW_AVAILABLE" -eq 0 ] || [ "$SKIP_OPTIONAL" = false ]; then
        run_test "Model Registry Classification (Alias)" \
            "curl -s -X POST '$FLASK_URL/api/v1/classify-registry?model_format=sklearn&alias=production' -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'" \
            true
    fi
    
    echo
fi

# Phase 5 Tests
if [ "$RUN_ALL" = true ] || [ "$RUN_PHASE5" = true ]; then
    print_status "INFO" "=== PHASE 5 TESTS ==="
    
    # TensorFlow Serving classification (optional)
    if [ "$TF_SERVING_AVAILABLE" -eq 0 ] || [ "$SKIP_OPTIONAL" = false ]; then
        run_test "TensorFlow Serving Classification" \
            "curl -s -X POST $FLASK_URL/api/v1/classify-tf-serving -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'" \
            true
    fi
    
    # Triton Inference Server classification (optional)
    if [ "$TRITON_AVAILABLE" -eq 0 ] || [ "$SKIP_OPTIONAL" = false ]; then
        run_test "Triton Inference Server Classification" \
            "curl -s -X POST $FLASK_URL/api/v1/classify-triton -H 'Content-Type: application/json' -d '{\"sepal_length\": 6.4, \"sepal_width\": 3.2, \"petal_length\": 4.5, \"petal_width\": 1.5}'" \
            true
    fi
    
    # Serving infrastructure performance comparison
    run_test "Serving Infrastructure Comparison" \
        "curl -s -X POST $FLASK_URL/api/v1/serving-comparison -H 'Content-Type: application/json' -d '{\"sepal_length\": 6.3, \"sepal_width\": 3.3, \"petal_length\": 6.0, \"petal_width\": 2.5, \"iterations\": 5}'"
    
    echo
fi

# Summary
print_status "INFO" "=== TEST SUMMARY ==="
echo "Total tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo

if [ $FAILED_TESTS -eq 0 ]; then
    print_status "SUCCESS" "All tests passed! 🎉"
    exit 0
else
    print_status "ERROR" "$FAILED_TESTS test(s) failed"
    echo
    print_status "INFO" "Troubleshooting tips:"
    echo "1. Ensure Flask app is running: python app.py"
    echo "2. Check Ollama service: brew services start ollama && ollama pull tinyllama"
    echo "3. For Phase 2: Start gRPC server (python grpc_server.py) and HTTP server (python http_server.py)"
    echo "4. For Phase 3: Start Redis: brew services start redis"
    echo "5. For Phase 4: Start MLflow: mlflow server --host 0.0.0.0 --port 5004 --serve-artifacts"
    echo "6. For Phase 5: Start TensorFlow Serving and/or Triton Inference Server"
    echo "7. Use --skip-optional to skip tests for unavailable services"
    echo "8. Use --verbose to see detailed request/response information"
    exit 1
fi