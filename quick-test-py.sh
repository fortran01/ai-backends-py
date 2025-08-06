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
RUN_PHASE6=false
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
    local port_only=$3
    
    if [ "$port_only" = "true" ]; then
        # For gRPC servers, just check if the port is open
        local port=$(echo "$url" | grep -o '[0-9]\+$')
        if nc -z localhost "$port" 2>/dev/null; then
            print_status "SUCCESS" "$service_name is running"
            return 0
        else
            print_status "WARNING" "$service_name is not running"
            return 1
        fi
    else
        # For HTTP services, use curl
        if curl -s --connect-timeout 3 "$url" > /dev/null 2>&1; then
            print_status "SUCCESS" "$service_name is running"
            return 0
        else
            print_status "WARNING" "$service_name is not running"
            return 1
        fi
    fi
}

# Function to validate JSON response
validate_response() {
    local response_file=$1
    local test_name=$2
    local validation_errors=""
    
    # Check if response should be JSON (most endpoints except basic generation)
    if [[ "$test_name" != *"Generation (Basic)"* ]]; then
        if ! jq empty "$response_file" 2>/dev/null; then
            validation_errors+="\n  - Invalid JSON format"
        fi
    fi
    
    # Check for common error patterns
    if grep -q '"error"' "$response_file" 2>/dev/null; then
        local error_msg=$(jq -r '.error // .detail // .message // "Unknown error"' "$response_file" 2>/dev/null)
        validation_errors+="\n  - API Error: $error_msg"
    fi
    
    # Check response size (detect truncated responses)
    local response_size=$(wc -c < "$response_file" 2>/dev/null || echo "0")
    if [ "$response_size" -lt 10 ]; then
        validation_errors+="\n  - Response too small ($response_size bytes)"
    fi
    
    # Specific validations based on test type
    case "$test_name" in
        *"Classification"*|*"Registry"*)
            # Validate classification responses
            if jq -e '.predicted_class' "$response_file" >/dev/null 2>&1; then
                local confidence=$(jq -r '.confidence // .probabilities[0] // 0' "$response_file" 2>/dev/null)
                if (( $(echo "$confidence < 0.1" | bc -l 2>/dev/null || echo "0") )); then
                    validation_errors+="\n  - Low confidence score: $confidence"
                fi
                if (( $(echo "$confidence > 1.0" | bc -l 2>/dev/null || echo "0") )); then
                    validation_errors+="\n  - Invalid confidence score: $confidence"
                fi
            else
                validation_errors+="\n  - Missing predicted_class field"
            fi
            ;;
        *"Performance"*|*"Benchmark"*)
            # Validate performance responses
            if ! jq -e '.performance_analysis // .results' "$response_file" >/dev/null 2>&1; then
                validation_errors+="\n  - Missing performance data"
            fi
            ;;
        *"Health"*)
            # Validate health check responses
            if ! jq -e '.status' "$response_file" >/dev/null 2>&1; then
                validation_errors+="\n  - Missing status field"
            fi
            ;;
        *"Chat"*|*"Generate"*)
            # Validate chat/generation responses
            if [[ "$test_name" == *"Generation (Basic)"* ]]; then
                # Basic generation returns plain text
                local response_text=$(cat "$response_file" 2>/dev/null)
                if [ ${#response_text} -lt 10 ]; then
                    validation_errors+="\n  - Response text too short"
                fi
            else
                # Other generation/chat endpoints return JSON
                if ! jq -e '.response' "$response_file" >/dev/null 2>&1; then
                    validation_errors+="\n  - Missing response field"
                else
                    local response_text=$(jq -r '.response' "$response_file" 2>/dev/null)
                    if [ ${#response_text} -lt 10 ]; then
                        validation_errors+="\n  - Response text too short"
                    fi
                fi
            fi
            ;;
        *"Drift"*)
            # Validate drift monitoring responses
            if ! jq -e '.drift_analysis // .drift_detected' "$response_file" >/dev/null 2>&1; then
                validation_errors+="\n  - Missing drift analysis data"
            fi
            ;;
    esac
    
    echo "$validation_errors"
}

# Function to check performance thresholds
check_performance() {
    local response_file=$1
    local test_name=$2
    local warnings=""
    
    # Check response time thresholds
    local response_time=$(jq -r '.response_time_ms // .processing_time_ms // .metadata.response_time_ms // 0' "$response_file" 2>/dev/null)
    if [ "$response_time" != "0" ] && [ "$response_time" != "null" ]; then
        if (( $(echo "$response_time > 10000" | bc -l 2>/dev/null || echo "0") )); then
            warnings+="\n  - Slow response time: ${response_time}ms"
        fi
    fi
    
    # Check for cache performance
    if jq -e '.cache_info' "$response_file" >/dev/null 2>&1; then
        local cache_time=$(jq -r '.cache_info.response_time_ms // 0' "$response_file" 2>/dev/null)
        local cache_hit=$(jq -r '.cache_info.cache_hit' "$response_file" 2>/dev/null)
        if [ "$cache_hit" = "true" ] && (( $(echo "$cache_time > 1000" | bc -l 2>/dev/null || echo "0") )); then
            warnings+="\n  - Slow cache response: ${cache_time}ms"
        fi
    fi
    
    echo "$warnings"
}

# Function to run a test with enhanced validation
run_test() {
    local test_name=$1
    local curl_command=$2
    local optional=${3:-false}
    local expected_keys=${4:-""}
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    print_status "INFO" "Running test: $test_name"
    
    if [ "$VERBOSE" = true ]; then
        echo "Command: $curl_command"
    fi
    
    # Execute the curl command
    local curl_exit_code=0
    local start_time=$(date +%s%N)
    eval "$curl_command" > temp_response.json 2>temp_error.log || curl_exit_code=$?
    local end_time=$(date +%s%N)
    local request_time_ms=$(( (end_time - start_time) / 1000000 ))
    
    if [ $curl_exit_code -eq 0 ] && [ -s temp_response.json ]; then
        # Validate response quality
        local validation_errors=$(validate_response temp_response.json "$test_name")
        local performance_warnings=$(check_performance temp_response.json "$test_name")
        
        if [ -z "$validation_errors" ]; then
            PASSED_TESTS=$((PASSED_TESTS + 1))
            print_status "SUCCESS" "$test_name - PASSED (${request_time_ms}ms)"
            
            # Show performance warnings if any
            if [ -n "$performance_warnings" ]; then
                print_status "WARNING" "Performance concerns:$performance_warnings"
            fi
            
            if [ "$VERBOSE" = true ]; then
                echo "Response:"
                cat temp_response.json | jq . 2>/dev/null || cat temp_response.json
                echo
            fi
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
            print_status "ERROR" "$test_name - VALIDATION FAILED"
            print_status "ERROR" "Validation issues:$validation_errors"
            
            if [ "$VERBOSE" = true ]; then
                echo "Response received:"
                cat temp_response.json | jq . 2>/dev/null || cat temp_response.json
                echo
            fi
        fi
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        if [ "$optional" = true ] && [ "$SKIP_OPTIONAL" = true ]; then
            print_status "WARNING" "$test_name - SKIPPED (optional service unavailable)"
        else
            print_status "ERROR" "$test_name - FAILED (${request_time_ms}ms)"
            if [ "$VERBOSE" = true ]; then
                echo "Error output:"
                cat temp_error.log 2>/dev/null || echo "No error details captured"
                if [ -s temp_response.json ]; then
                    echo "Response received:"
                    cat temp_response.json
                fi
                echo
            fi
        fi
    fi
    
    rm -f temp_response.json temp_error.log
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
    echo "  --phase6          Run Phase 6 tests only"
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
        --phase6)
            RUN_ALL=false
            RUN_PHASE6=true
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
command -v nc >/dev/null 2>&1 || { print_status "ERROR" "nc (netcat) is required but not installed"; exit 1; }
command -v jq >/dev/null 2>&1 || { print_status "ERROR" "jq is required for response validation but not installed. Install with: brew install jq"; exit 1; }
command -v bc >/dev/null 2>&1 || { print_status "WARNING" "bc not installed - some numeric comparisons may not work"; }

# Check service availability
print_status "INFO" "Checking service availability..."
check_service "$FLASK_URL/health" "Flask API"
FLASK_AVAILABLE=$?
check_service "$GRPC_SERVER_URL" "gRPC Server" true
GRPC_AVAILABLE=$?
check_service "$HTTP_SERVER_URL" "HTTP Server"
HTTP_AVAILABLE=$?
check_service "$TF_SERVING_URL/v1/models/iris" "TensorFlow Serving"
TF_SERVING_AVAILABLE=$?
check_service "$TRITON_URL/v2/health/ready" "Triton Inference Server"
TRITON_AVAILABLE=$?
check_service "$MLFLOW_URL" "MLflow Server"
MLFLOW_AVAILABLE=$?

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
    if [[ "$HTTP_AVAILABLE" -eq 0 ]] || [[ "$SKIP_OPTIONAL" = false ]]; then
        run_test "HTTP Classification" \
            "curl -s -X POST $FLASK_URL/api/v1/classify-http -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'" \
            true
    fi
    
    # gRPC classification (optional)
    if [[ "$GRPC_AVAILABLE" -eq 0 ]] || [[ "$SKIP_OPTIONAL" = false ]]; then
        run_test "gRPC Classification" \
            "curl -s -X POST $FLASK_URL/api/v1/classify-grpc -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'" \
            true
    fi
    
    # Performance benchmark (optional)
    if ([[ "$GRPC_AVAILABLE" -eq 0 ]] && [[ "$HTTP_AVAILABLE" -eq 0 ]]) || [[ "$SKIP_OPTIONAL" = false ]]; then
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
    if [[ "$MLFLOW_AVAILABLE" -eq 0 ]] || [[ "$SKIP_OPTIONAL" = false ]]; then
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
    if [[ "$TF_SERVING_AVAILABLE" -eq 0 ]] || [[ "$SKIP_OPTIONAL" = false ]]; then
        run_test "TensorFlow Serving Classification" \
            "curl -s -X POST $FLASK_URL/api/v1/classify-tf-serving -H 'Content-Type: application/json' -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'" \
            true
    fi
    
    # Triton Inference Server classification (optional)
    if [[ "$TRITON_AVAILABLE" -eq 0 ]] || [[ "$SKIP_OPTIONAL" = false ]]; then
        run_test "Triton Inference Server Classification" \
            "curl -s -X POST $FLASK_URL/api/v1/classify-triton -H 'Content-Type: application/json' -d '{\"sepal_length\": 6.4, \"sepal_width\": 3.2, \"petal_length\": 4.5, \"petal_width\": 1.5}'" \
            true
    fi
    
    # Serving infrastructure performance comparison
    run_test "Serving Infrastructure Comparison" \
        "curl -s -X POST $FLASK_URL/api/v1/serving-comparison -H 'Content-Type: application/json' -d '{\"sepal_length\": 6.3, \"sepal_width\": 3.3, \"petal_length\": 6.0, \"petal_width\": 2.5, \"iterations\": 5}'"
    
    echo
fi

# Phase 6 Tests
if [ "$RUN_ALL" = true ] || [ "$RUN_PHASE6" = true ]; then
    print_status "INFO" "=== PHASE 6 TESTS ==="
    
    # RAG Chat with ML query
    run_test "RAG Chat (ML Query)" \
        "curl -s -X POST $FLASK_URL/api/v1/rag-chat -H 'Content-Type: application/json' -d '{\"query\": \"What is machine learning and how does it work?\"}'"
    
    # RAG Chat with supervised learning query
    run_test "RAG Chat (Supervised Learning)" \
        "curl -s -X POST $FLASK_URL/api/v1/rag-chat -H 'Content-Type: application/json' -d '{\"query\": \"Explain the difference between supervised and unsupervised learning\"}'"
    
    # RAG Chat with neural networks query
    run_test "RAG Chat (Neural Networks)" \
        "curl -s -X POST $FLASK_URL/api/v1/rag-chat -H 'Content-Type: application/json' -d '{\"query\": \"What are neural networks and how do they work?\"}'"
    
    # RAG Domain Filtering Test (should be filtered out)
    run_test "RAG Domain Filtering (Kangaroo Query)" \
        "curl -s -X POST $FLASK_URL/api/v1/rag-chat -H 'Content-Type: application/json' -d '{\"query\": \"What is a kangaroo?\"}'"
    
    # RAG with source options
    run_test "RAG with Source Configuration" \
        "curl -s -X POST $FLASK_URL/api/v1/rag-chat -H 'Content-Type: application/json' -d '{\"query\": \"What is model drift and how do you monitor it?\", \"max_sources\": 3, \"include_sources\": true}'"
    
    echo
fi

# Enhanced Summary with Quality Metrics
print_status "INFO" "=== COMPREHENSIVE TEST SUMMARY ==="
echo "┌─────────────────────────────────────────────┐"
echo "│                TEST RESULTS                 │"
echo "└─────────────────────────────────────────────┘"
echo "Total tests executed: $TOTAL_TESTS"
echo "✅ Passed: $PASSED_TESTS"
echo "❌ Failed: $FAILED_TESTS"

if [ $TOTAL_TESTS -gt 0 ]; then
    success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "📊 Success rate: ${success_rate}%"
fi

# Calculate service availability summary
service_summary=""
if [[ $FLASK_AVAILABLE -eq 0 ]]; then service_summary+="Flask(✅) "; else service_summary+="Flask(❌) "; fi
if [[ $GRPC_AVAILABLE -eq 0 ]]; then service_summary+="gRPC(✅) "; else service_summary+="gRPC(❌) "; fi
if [[ $HTTP_AVAILABLE -eq 0 ]]; then service_summary+="HTTP(✅) "; else service_summary+="HTTP(❌) "; fi
if [[ $TF_SERVING_AVAILABLE -eq 0 ]]; then service_summary+="TF-Serving(✅) "; else service_summary+="TF-Serving(❌) "; fi
if [[ $TRITON_AVAILABLE -eq 0 ]]; then service_summary+="Triton(✅) "; else service_summary+="Triton(❌) "; fi
if [[ $MLFLOW_AVAILABLE -eq 0 ]]; then service_summary+="MLflow(✅)"; else service_summary+="MLflow(❌)"; fi

echo "🏥 Service Status: $service_summary"

# Generate recommendations based on results
echo
echo "┌─────────────────────────────────────────────┐"
echo "│            RECOMMENDATIONS                  │"
echo "└─────────────────────────────────────────────┘"

if [ $FAILED_TESTS -eq 0 ]; then
    print_status "SUCCESS" "🎉 All tests passed! Your AI backend is healthy"
    echo "✨ Quality indicators:"
    echo "   • All API endpoints responding correctly"
    echo "   • Model predictions within expected confidence ranges"
    echo "   • Response formats validated successfully"
    echo "   • Performance metrics within acceptable thresholds"
    
    if [ $success_rate -eq 100 ]; then
        echo "🏆 Perfect score! Consider this setup production-ready."
    fi
    exit 0
else
    print_status "ERROR" "🚨 $FAILED_TESTS test(s) failed - System needs attention"
    echo
    
    # Service-specific troubleshooting
    if [[ $FLASK_AVAILABLE -ne 0 ]]; then
        print_status "ERROR" "🔧 Flask API Issues:"
        echo "   → Start Flask: python app.py"
        echo "   → Check port 5001 availability"
        echo "   → Verify dependencies: pip install -r requirements.txt"
    fi
    
    if [[ $GRPC_AVAILABLE -ne 0 ]] && [[ $RUN_PHASE2 == true || $RUN_ALL == true ]]; then
        print_status "WARNING" "⚠️  gRPC Server Offline:"
        echo "   → Start gRPC server: python grpc_server.py"
        echo "   → Check port 50051 availability"
    fi
    
    if [[ $HTTP_AVAILABLE -ne 0 ]] && [[ $RUN_PHASE2 == true || $RUN_ALL == true ]]; then
        print_status "WARNING" "⚠️  HTTP Server Offline:"
        echo "   → Start HTTP server: python http_server.py"
        echo "   → Check port 5002 availability"
    fi
    
    if [[ $TF_SERVING_AVAILABLE -ne 0 ]] && [[ $RUN_PHASE5 == true || $RUN_ALL == true ]]; then
        print_status "WARNING" "⚠️  TensorFlow Serving Offline:"
        echo "   → Start TF Serving with iris model"
        echo "   → Check port 8501 availability"
    fi
    
    if [[ $TRITON_AVAILABLE -ne 0 ]] && [[ $RUN_PHASE5 == true || $RUN_ALL == true ]]; then
        print_status "WARNING" "⚠️  Triton Inference Server Offline:"
        echo "   → Start Triton with iris model"
        echo "   → Check port 8000 availability"
    fi
    
    if [[ $MLFLOW_AVAILABLE -ne 0 ]] && [[ $RUN_PHASE4 == true || $RUN_ALL == true ]]; then
        print_status "WARNING" "⚠️  MLflow Server Offline:"
        echo "   → Start MLflow: mlflow server --host 0.0.0.0 --port 5004"
        echo "   → Set up model registry"
    fi
    
    echo
    print_status "INFO" "💡 General Troubleshooting:"
    echo "1. 🔍 Use --verbose for detailed request/response logs"
    echo "2. 🎯 Use --phase[1-6] to test specific components"
    echo "3. ⏭️  Use --skip-optional to bypass unavailable services"
    echo "4. 🔄 Check service logs for specific error details"
    echo "5. 🧪 Validate your models are properly loaded"
    echo "6. 🏥 Run health checks on individual services first"
    
    echo
    echo "📋 Quick Service Start Commands:"
    echo "   Flask:     python app.py"
    echo "   Ollama:    brew services start ollama && ollama pull tinyllama"
    echo "   Redis:     brew services start redis"
    echo "   MLflow:    mlflow server --host 0.0.0.0 --port 5004"
    
    exit 1
fi