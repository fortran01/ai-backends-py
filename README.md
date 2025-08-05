# AI Back-End Demo: Python/Flask Project - Phase 4

[![CI/CD Pipeline](https://github.com/your-username/ai-backends-py/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/your-username/ai-backends-py/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-blue.svg)](https://docs.pytest.org/)
[![End-to-end tested with Playwright](https://img.shields.io/badge/e2e%20tested%20with-playwright-green.svg)](https://playwright.dev/)
[![MLflow](https://img.shields.io/badge/mlflow-2.9%2B%20%7C%203.x-blue.svg)](https://mlflow.org/)
[![Evidently AI](https://img.shields.io/badge/evidently-0.4%2B-orange.svg)](https://evidentlyai.com/)

This project demonstrates **Phase 5** of the AI Back-End architecture course, implementing dedicated model serving infrastructure that provides production-grade optimizations beyond direct model loading in applications. This phase showcases the evolution from prototype to production-grade serving infrastructure.

## 🎯 Phase 5 Objectives

**Goal:** Showcase dedicated model serving frameworks that provide production-grade optimizations beyond direct model loading in the application, demonstrating the performance characteristics and trade-offs between different serving architectures.

## 🚀 Features Implemented

### ✅ Phase 1 Completed Tasks

1. **Environment Setup** - Python virtual environment with required dependencies
2. **Model Training & Security Demo** - Iris dataset training with pickle/ONNX comparison  
3. **Basic Flask API** - Production-ready Flask application with proper structure
4. **Stateless LLM Generation** - `/api/v1/generate` endpoint using Ollama API
5. **Security Demonstration** - `/api/v1/generate-secure` with prompt injection prevention
6. **ONNX Model Inference** - `/api/v1/classify` endpoint for iris classification
7. **Batch Processing** - Offline inference script for bulk prompt processing

### ✅ Phase 2 Completed Tasks

8. **Stateful Chat Endpoint** - `/api/v1/chat` with conversation memory using LangChain
9. **gRPC Protocol Buffer Service** - High-performance binary communication protocol
10. **gRPC Server Implementation** - Standalone gRPC server for iris classification
11. **gRPC Client Integration** - `/api/v1/classify-grpc` endpoint calling gRPC service
12. **Performance Comparison** - `/api/v1/classify-benchmark` REST vs gRPC timing analysis
13. **Serialization Challenges** - `/api/v1/classify-detailed` with custom JSON encoders
14. **Model Context Protocol (MCP)** - Comprehensive conversation memory and orchestration

### ✅ Phase 3 Completed Tasks

15. **Advanced Exact Caching** - Redis-based caching with Flask-Caching for `/api/v1/classify` endpoint
16. **Semantic Caching Implementation** - FastEmbed-powered semantic similarity caching using `all-MiniLM-L6-v2` model
17. **Semantic Cache Service** - `/api/v1/chat-semantic` endpoint demonstrating intelligent cache hits for similar prompts
18. **Cache Analytics** - Comprehensive cache hit/miss statistics, similarity scores, and performance metrics
19. **API Versioning** - Flask Blueprint-based `/api/v2/generate` endpoint with enhanced metadata and features
20. **Redis Integration** - Production-ready Redis configuration for both exact and semantic caching

### ✅ Phase 4 Completed Tasks

21. **Production Logging System** - Automated request/response logging to CSV files for `/api/v1/classify` endpoint
22. **Reference Dataset Creation** - Iris training dataset stored as reference baseline for drift comparison
23. **Evidently AI Integration** - Production-grade drift monitoring with statistical tests and visualizations
24. **Drift Detection Endpoint** - `/api/v1/drift-report` with comprehensive data and prediction drift analysis
25. **Drift Simulation** - `/api/v1/classify-shifted` endpoint demonstrating systematic bias injection for testing
26. **MLflow Model Registry** - Centralized model lifecycle management with version control and metadata
27. **Model Registration** - Enhanced training script with automated MLflow registration for both sklearn and ONNX models
28. **Registry-Based Inference** - `/api/v1/classify-registry` endpoint loading models programmatically from MLflow
29. **Model Management CLI** - `scripts/manage_models.py` for model lifecycle operations (staging, promotion, comparison)
30. **Production Monitoring** - Complete MLOps pipeline with drift detection and model registry integration

### ✅ Phase 5 Completed Tasks

31. **TensorFlow SavedModel Creation** - Convert ONNX model to TensorFlow SavedModel format for TensorFlow Serving
32. **TensorFlow Serving Integration** - `/api/v1/classify-tf-serving` endpoint calling TensorFlow Serving REST API
33. **Triton Model Repository** - NVIDIA Triton Inference Server model repository with ONNX model and configuration
34. **Triton Integration** - `/api/v1/classify-triton` endpoint with advanced Triton client features
35. **Dynamic Batching Configuration** - Triton config with automatic request batching for improved throughput
36. **Performance Comparison Endpoint** - `/api/v1/serving-comparison` comparing all serving infrastructures
37. **Dynamic Batching Demo Script** - `scripts/test_dynamic_batching.py` demonstrating batching performance benefits
38. **Serving Architecture Analysis** - Comprehensive comparison of direct loading vs dedicated serving

### 🔒 Security Features

- **Prompt Injection Detection** - Pattern-based detection of malicious prompts
- **Input Sanitization** - Safe prompt processing with length limits
- **ONNX Model Format** - Secure model format that prevents code execution
- **Pickle Security Demo** - Educational demonstration of pickle vulnerabilities

### 📊 Model Formats Demonstrated

- **Pickle Format** (`.pkl`) - Traditional Python serialization with security risks
- **ONNX Format** (`.onnx`) - Cross-framework, secure model format for production
- **Security Comparison** - Live demonstration of pickle arbitrary code execution

## 🛠️ Installation & Setup

### Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
3. **TinyLlama model** pulled via Ollama
4. **Redis** installed and running (for Phase 3 caching)
5. **MLflow** server running (for Phase 4 model registry)
6. **TensorFlow Serving** installed (for Phase 5 dedicated serving)
7. **Triton Inference Server** installed (for Phase 5 high-performance serving)

### Setup Instructions

```bash
# 1. Clone/navigate to the project directory
cd ai-backends-py

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the Iris model (creates ONNX and pickle files)
python scripts/train_iris_model.py

# 5. Ensure Ollama is running and TinyLlama is available
brew services start ollama  # macOS
ollama pull tinyllama

# 6. (Phase 3) Start Redis for caching
brew services start redis  # macOS

# 7. (Phase 4) Start MLflow server with artifact serving in a separate terminal
mlflow server --host 0.0.0.0 --port 5004 --serve-artifacts

# 8. Start the Flask application
python app.py

# 9. (Phase 2) Start the HTTP inference server in a separate terminal
python http_server.py

# 10. (Phase 2) Start the gRPC server in another separate terminal  
python grpc_server.py

# 11. (Phase 5) Create TensorFlow SavedModel for TensorFlow Serving
python scripts/keras_export_savedmodel.py

# 12. (Phase 5) Install and Start TensorFlow Serving

## Install TensorFlow Model Server on Mac:

### Option 1: Using Docker (Recommended)

#### For Mac M1/M2 (ARM64) - RECOMMENDED:
```bash
# Use native Python TensorFlow Serving alternative (works perfectly on ARM64)
python tf_serving_local.py

# Note: Docker TensorFlow Serving has compatibility issues on Apple Silicon
# The Python alternative provides the same REST API and runs natively
```

#### For Intel Mac (x86_64):
```bash
# Pull standard TensorFlow Serving Docker image
docker pull tensorflow/serving

# Run TensorFlow Serving with your model
docker run -t --rm -p 8501:8501 \
    -v "$(pwd)/models/iris_tensorflow_savedmodel:/models/iris" \
    -e MODEL_NAME=iris \
    tensorflow/serving
```

#### Alternative: Emulation (slower but compatible)
```bash
# Force x86_64 emulation on ARM64 Mac (slower)
docker run --platform linux/amd64 -t --rm -p 8501:8501 \
    -v "$(pwd)/models/iris_tensorflow_savedmodel:/models/iris" \
    -e MODEL_NAME=iris \
    tensorflow/serving
```

### Option 2: Build from Source (Advanced)
```bash
# Install Bazel build tool
brew install bazel

# Clone TensorFlow Serving repository
git clone https://github.com/tensorflow/serving
cd serving

# Build TensorFlow Model Server
bazel build //tensorflow_serving/model_servers:tensorflow_model_server

# Run the server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
    --rest_api_port=8501 \
    --model_name=iris \
    --model_base_path=$(pwd)/../models/iris_tensorflow_savedmodel
```

### Option 3: Python TensorFlow Serving Alternative (Mac M1 Native)

The `tf_serving_local.py` script has been created and provides a native ARM64 solution that mimics TensorFlow Serving's REST API:

```bash
# Start the native Python TensorFlow Serving alternative
python tf_serving_local.py
```

This script provides:
- Full TensorFlow Serving REST API compatibility
- Native ARM64 performance (no emulation)
- Same endpoints: `/v1/models/iris:predict`, `/v1/models/iris`, `/v1/models/iris/metadata`
- Proper error handling and logging

### Option 4: Direct Command (if binary available)
```bash
tensorflow_model_server --rest_api_port=8501 --model_name=iris --model_base_path=$(pwd)/models/iris_tensorflow_savedmodel
```

# 13. (Phase 5) Install and Start Triton Inference Server

## Install Triton Inference Server on Mac:

### Option 1: Using Docker (Recommended)

#### For Mac M1/M2 (ARM64):
```bash
# Pull Triton Server container for ARM64
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# Run Triton Server
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v "$(pwd)/triton_model_repository:/models" \
    nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver \
    --model-repository=/models \
    --http-port=8000

# Note: GPU acceleration not available on Mac, CPU inference only
```

#### For Intel Mac (x86_64):
```bash
# Pull standard Triton Server container
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# Run Triton Server
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v "$(pwd)/triton_model_repository:/models" \
    nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver \
    --model-repository=/models \
    --http-port=8000
```

### Option 2: Native Binary (Not Available for Mac)

Unfortunately, NVIDIA does not provide native Triton Inference Server binaries for macOS. The `tritonserver` command requires Linux environment.

### Option 3: Build from Source (Advanced, Linux VM Required)

For development on Mac, consider using a Linux VM or cloud instance:

```bash
# On Linux VM/Cloud Instance:
# Install dependencies
apt-get update && apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get update && apt-get install -y python3.8 python3.8-dev python3-pip

# Clone and build Triton
git clone https://github.com/triton-inference-server/server.git
cd server
python3 build.py --build-dir=/tmp/citritonbuild --cmake-build-type=Release --enable-logging --enable-stats --enable-tracing --enable-metrics --enable-gpu-metrics=false --backend=onnxruntime

# Run the server
/tmp/citritonbuild/install/bin/tritonserver --model-repository=$(pwd)/triton_model_repository --http-port=8000
```

### Option 4: Use Development Setup

For Mac users who want to test the Triton integration without Docker:

1. **Skip Triton Server**: The Flask app gracefully handles Triton unavailability
2. **Test with other serving methods**: Direct ONNX and TensorFlow Serving work natively
3. **Use Cloud/Remote Triton**: Connect to a remote Triton instance

```bash
# Test if Triton is available (will show graceful fallback)
curl -X POST http://localhost:5001/api/v1/classify-triton \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Alternative: Test serving comparison (works without Triton)
curl -X POST http://localhost:5001/api/v1/serving-comparison \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

### Troubleshooting Triton Installation

If you encounter the error `bash: tritonserver: command not found`, this is expected on Mac as there's no native binary. Use Docker instead:

```bash
# Verify Docker is installed
docker --version

# Pull and run Triton via Docker
docker pull nvcr.io/nvidia/tritonserver:24.01-py3
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v "$(pwd)/triton_model_repository:/models" \
    nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver \
    --model-repository=/models --http-port=8000
```

### Common Triton Configuration Issues

#### Issue 1: `max_queue_size` Deprecated Error
```
Error parsing text-format inference.ModelConfig: Message type "inference.ModelDynamicBatching" has no field named "max_queue_size".
```

**Solution**: In Triton 24.01+, `max_queue_size` has been moved to `default_queue_policy`:

```protobuf
# Before (deprecated)
dynamic_batching {
  max_queue_size: 100
}

# After (Triton 24.01+)
dynamic_batching {
  default_queue_policy {
    max_queue_size: 100
  }
}
```

#### Issue 2: ONNX Model Compatibility
```
Unsupported ONNX Type 'ONNX_TYPE_SEQUENCE' for I/O 'output_probability', expected 'ONNX_TYPE_TENSOR'.
```

**Solution**: Use the improved ONNX model that outputs tensors instead of sequences:
```bash
# Replace the model with the tensor-compatible version
cp models/iris_classifier_improved.onnx triton_model_repository/iris_onnx/1/model.onnx
```

#### Issue 3: Model Configuration Auto-Completion
For complex models, let Triton auto-detect the configuration:

```protobuf
name: "iris_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 32

# Dynamic batching configuration
dynamic_batching {
  max_queue_delay_microseconds: 100
  preferred_batch_size: [ 4, 8 ]
  default_queue_policy {
    max_queue_size: 100
  }
}
```

### Verify Triton Installation

Once running, verify the server status:

```bash
# Check server health
curl http://localhost:8000/v2/health/ready

# List available models (correct endpoint)
curl -X POST http://localhost:8000/v2/repository/index

# Get model configuration
curl http://localhost:8000/v2/models/iris_onnx/config

# Get model metadata  
curl http://localhost:8000/v2/models/iris_onnx
```

Expected outputs:

**Health check**: Returns empty response with HTTP 200 status if healthy.

**Model list**:
```json
[{"name":"iris_onnx","version":"1","state":"READY"}]
```

**Model metadata**:
```json
{
  "name": "iris_onnx",
  "versions": ["1"],
  "platform": "onnxruntime_onnx",
  "inputs": [...],
  "outputs": [...]
}
```

**Test inference**:
```bash
# Test Triton inference via Flask app
curl -X POST http://localhost:5001/api/v1/classify-triton \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## 📡 API Endpoints

### Phase 1 Endpoints

#### Health Check
```bash
GET /health
# Returns service status and dependency health
```

#### LLM Text Generation (Basic)
```bash
POST /api/v1/generate
Content-Type: application/json

{
  "prompt": "What is artificial intelligence?"
}
```

#### LLM Text Generation (Secure)
```bash
POST /api/v1/generate-secure
Content-Type: application/json

{
  "prompt": "Explain machine learning"
}

# Returns response with security analysis
{
  "response": "Machine learning is...",
  "security_analysis": {
    "injection_detected": false,
    "detected_patterns": [],
    "sanitized": false,
    "original_length": 25,
    "sanitized_length": 25,
    "blocked": false
  }
}
```

#### Iris Classification (ONNX)
```bash
POST /api/v1/classify  
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

# Returns prediction with probabilities
{
  "predicted_class": "setosa",
  "predicted_class_index": 0,
  "probabilities": [0.95, 0.03, 0.02],
  "confidence": 0.95,
  "all_classes": ["setosa", "versicolor", "virginica"]
}
```

### Phase 2 Endpoints

#### Stateful Chat with Memory (MCP)
```bash
POST /api/v1/chat
Content-Type: application/json

{
  "prompt": "Hello, what can you help me with?",
  "session_id": "user123-session"
}

# Returns chat response with conversation stats
{
  "response": "Hello! I can help you with...",
  "session_id": "user123-session",
  "conversation_length": 1,
  "memory_stats": {
    "total_messages": 2,
    "human_messages": 1,
    "ai_messages": 1
  }
}
```

#### Detailed Classification with Serialization Demo
```bash
POST /api/v1/classify-detailed
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

# Returns detailed prediction with NumPy arrays (custom JSON encoder)
{
  "predicted_class": "setosa",
  "raw_probabilities": [0.95, 0.03, 0.02],
  "feature_importances": [0.1, 0.2, 0.3, 0.4],
  "model_info": {
    "type": "RandomForestClassifier",
    "n_estimators": 100
  },
  "serialization_demo": {
    "numpy_int": 42,
    "numpy_float": 3.14159,
    "numpy_array": [1, 2, 3, 4, 5]
  }
}
```

#### Network-Based HTTP Classification
```bash
POST /api/v1/classify-http
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

# Returns HTTP server classification via network call
{
  "predicted_class": "setosa",
  "predicted_class_index": 0,
  "probabilities": [0.95, 0.03, 0.02],
  "confidence": 0.95,
  "protocol": "HTTP",
  "processing_time_ms": 23.45
}
```

#### High-Performance gRPC Classification
```bash
POST /api/v1/classify-grpc
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

# Returns fast binary protocol classification
{
  "predicted_class": "setosa",
  "predicted_class_index": 0,
  "probabilities": [0.95, 0.03, 0.02],
  "confidence": 0.95,
  "protocol": "gRPC",
  "processing_time_ms": 12.34
}
```

#### Performance Comparison (HTTP vs gRPC)
```bash
POST /api/v1/classify-benchmark
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2,
  "iterations": 10
}

# Returns comprehensive performance analysis
{
  "results": {
    "rest": {
      "predicted_class": "setosa",
      "protocol": "HTTP",
      "avg_processing_time_ms": 45.67
    },
    "grpc": {
      "predicted_class": "setosa", 
      "protocol": "gRPC",
      "avg_processing_time_ms": 23.45
    }
  },
  "performance_analysis": {
    "http_time_ms": 45.67,
    "grpc_time_ms": 23.45,
    "speedup_factor": 1.95,
    "faster_protocol": "gRPC",
    "time_difference_ms": 22.22
  }
}
```

### Phase 3 Endpoints

#### Semantic Caching Chat
```bash
POST /api/v1/chat-semantic
Content-Type: application/json

{
  "prompt": "What is AI?"
}

# Returns response with cache information
{
  "response": "AI (Artificial Intelligence) is...",
  "cache_info": {
    "cache_hit": true,
    "cache_type": "semantic",
    "similarity_score": 0.92,
    "response_time_ms": 336.77
  }
}

# Test semantic similarity with:
# - "What is AI?"
# - "What is artificial intelligence?"
# - "Explain artificial intelligence to me"
```

#### API Versioning (v2)
```bash
POST /api/v2/generate
Content-Type: application/json

{
  "prompt": "Hello, this is a test of API v2",
  "model_version": "tinyllama"
}

# Returns enhanced response with metadata
{
  "response": "Generated text...",
  "metadata": {
    "api_version": "v2",
    "model_used": "tinyllama",
    "response_time_ms": 150.23,
    "token_count": 45,
    "timestamp": "2025-08-04T00:58:23.697079Z"
  }
}
```

#### Cached Classification (Exact Caching)
```bash
POST /api/v1/classify
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

# First call: Normal response time
# Subsequent identical calls: Cached response (much faster)
{
  "predicted_class": "setosa",
  "predicted_class_index": 0,
  "probabilities": [1.0, 0.0, 0.0],
  "confidence": 1.0
}
```

### Phase 4 Endpoints

#### Drift Monitoring Report
```bash
GET /api/v1/drift-report?limit=50
# Generate comprehensive drift analysis comparing recent production data vs reference

{
  "drift_analysis": {
    "drift_detected": true,
    "overall_drift_score": 0.6234,
    "feature_drift_scores": {
      "sepal_length": {
        "drift_detected": true,
        "drift_score": 0.8456,
        "statistical_test": "kolmogorov_smirnov"
      }
    },
    "analysis_timestamp": "2025-08-04T00:58:23.697079Z"
  },
  "data_summary": {
    "reference_samples": 120,
    "analyzed_samples": 50,
    "total_production_samples": 150
  },
  "recommendations": {
    "retrain_model": true,
    "investigate_features": ["sepal_length", "petal_width"],
    "monitoring_status": "MEDIUM_DRIFT"
  }
}
```

#### Drift Simulation
```bash
POST /api/v1/classify-shifted
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

# Returns both original and shifted predictions with drift analysis
{
  "original_prediction": {
    "predicted_class": "setosa",
    "confidence": 0.95
  },
  "shifted_prediction": {
    "predicted_class": "versicolor",
    "confidence": 0.78
  },
  "drift_simulation": {
    "applied_shifts": {
      "sepal_length": "+1.5 units",
      "petal_width": "*1.3 multiplier"
    },
    "prediction_changed": true,
    "confidence_change": -0.17
  }
}
```

#### Model Registry Classification
```bash
# Modern approach using aliases (MLflow 3.x recommended)
POST /api/v1/classify-registry?model_format=sklearn&alias=production
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

# Returns prediction with model registry metadata including alias
{
  "predicted_class": "setosa",
  "predicted_class_index": 0,
  "probabilities": [0.8, 0.1, 0.1],
  "confidence": 0.8,
  "model_registry_info": {
    "model_name": "iris-classifier-sklearn",
    "model_format": "sklearn",
    "model_version": "1",
    "model_stage": "None",
    "model_alias": "production",
    "model_uri": "models:/iris-classifier-sklearn@production",
    "mlflow_tracking_uri": "http://localhost:5004"
  }
}

# Legacy approach using stages (still supported but deprecated)
POST /api/v1/classify-registry?model_format=sklearn&stage=Production
```

### Phase 5 Endpoints

#### TensorFlow Serving Classification

```bash
# TensorFlow Serving Classification
curl -X POST http://localhost:5001/api/v1/classify-tf-serving \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'

# Response includes serving infrastructure details
{
  "predicted_class": "setosa",
  "predicted_class_index": 0,
  "probabilities": [0.98, 0.01, 0.01],
  "confidence": 0.98,
  "serving_infrastructure": "tensorflow_serving",
  "model_version": 1,
  "response_time_ms": 25.3
}
```

#### Triton Inference Server Classification

```bash
# Triton Inference Server Classification
curl -X POST http://localhost:5001/api/v1/classify-triton \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 6.4,
    "sepal_width": 3.2,
    "petal_length": 4.5,
    "petal_width": 1.5
  }'

# Response includes Triton-specific optimizations
{
  "predicted_class": "versicolor",
  "predicted_class_index": 1,
  "probabilities": [0.02, 0.95, 0.03],
  "confidence": 0.95,
  "serving_infrastructure": "triton_inference_server",
  "model_version": "1",
  "response_time_ms": 18.7,
  "batch_size": 1
}
```

#### Serving Infrastructure Performance Comparison

```bash
# Serving Infrastructure Performance Comparison
curl -X POST http://localhost:5001/api/v1/serving-comparison \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 6.3,
    "sepal_width": 3.3,
    "petal_length": 6.0,
    "petal_width": 2.5,
    "iterations": 5
  }'

# Response compares all serving methods
{
  "test_configuration": {
    "sample": {...},
    "iterations": 5
  },
  "serving_methods": {
    "direct_onnx": {
      "available": true,
      "response_times_ms": [12.3, 11.8, 12.1, 11.9, 12.0],
      "average_response_time_ms": 12.02,
      "success_rate": 1.0,
      "infrastructure": "direct_onnx_runtime"
    },
    "tensorflow_serving": {
      "available": true,
      "response_times_ms": [25.1, 24.8, 25.3, 24.9, 25.0],
      "average_response_time_ms": 25.02,
      "success_rate": 1.0,
      "infrastructure": "tensorflow_serving"
    },
    "triton_inference": {
      "available": true,
      "response_times_ms": [18.2, 17.9, 18.5, 18.1, 18.0],
      "average_response_time_ms": 18.14,
      "success_rate": 1.0,
      "infrastructure": "triton_inference_server"
    }
  },
  "performance_analysis": {
    "fastest_method": "direct_onnx",
    "fastest_avg_time_ms": 12.02,
    "slowest_method": "tensorflow_serving",
    "slowest_avg_time_ms": 25.02,
    "speedup_factor": 2.08,
    "recommendations": [
      "Direct ONNX is fastest for single requests with minimal overhead",
      "Performance difference: 2.1x between fastest and slowest"
    ]
  }
}
```

## 🔄 Batch Processing

### Create Sample Files
```bash
python scripts/batch_inference.py --create-samples
```

### Process Text File
```bash
python scripts/batch_inference.py --input sample_prompts.txt --output results.json
```

### Process CSV File
```bash
python scripts/batch_inference.py --input sample_prompts.csv --csv --output results.json
```

## 📁 Project Structure

```
ai-backends-py/
├── app.py                    # Main Flask application (Phase 1 & 2 endpoints)
├── grpc_server.py            # Standalone gRPC server for high-performance inference
├── requirements.txt          # Python dependencies (includes LangChain, gRPC)
├── README.md                # This file
├── tests/                   # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py          # Pytest configuration and fixtures
│   ├── test_api_endpoints.py    # Unit/integration tests with pytest
│   └── test_e2e_playwright.py   # End-to-end tests with Playwright
├── pytest.ini              # Pytest configuration
├── run_tests.py            # Test runner script
├── venv/                    # Virtual environment
├── proto/                   # gRPC Protocol Buffer definitions
│   ├── inference.proto      # Service definition for iris classification
│   ├── inference_pb2.py     # Generated Python protobuf classes
│   └── inference_pb2_grpc.py # Generated gRPC service stubs
├── models/                  # Trained model files
│   ├── iris_classifier.onnx # Secure ONNX model
│   ├── iris_classifier.pkl  # Pickle model (demo only)
│   └── malicious_model.pkl  # Security demo (dangerous!)
├── scripts/                 # Utility scripts
│   ├── train_iris_model.py  # Model training with security demo
│   └── batch_inference.py   # Batch processing script
├── sample_prompts.txt       # Sample text input
└── sample_prompts.csv       # Sample CSV input
```

## 🗃️ MLflow Model Registry Management

### Prerequisites for MLflow Integration

**IMPORTANT**: For model artifact serving to work properly, MLflow server must be started with the `--serve-artifacts` flag:

```bash
# ✅ Correct way to start MLflow server (enables artifact serving)
mlflow server --host 0.0.0.0 --port 5004 --serve-artifacts

# ❌ Incorrect - without --serve-artifacts, model downloads will fail
mlflow server --host 0.0.0.0 --port 5004
```

The `--serve-artifacts` flag is essential for:
- Model artifact download from the registry
- Cross-language model serving (Python to TypeScript)
- Production deployment workflows

### Model Registration & Training
```bash
# Train models and register with MLflow (includes both sklearn and ONNX formats)
python scripts/train_iris_model.py

# OR: Train tensor-only ONNX model specifically for TypeScript compatibility
python scripts/train_iris_model_improved.py

# This will:
# 1. Train RandomForestClassifier on Iris dataset  
# 2. Log metrics, parameters, and metadata to MLflow
# 3. Register both sklearn and ONNX models in the registry
# 4. Create tensor-only ONNX models for onnxruntime-node compatibility
# 5. Demonstrate security differences between pickle and ONNX formats
# 6. Set up model aliases for production deployment
```

### Model Management CLI
```bash
# List all registered models with versions, stages, and aliases
python scripts/manage_models.py --list

# Compare performance across model versions
python scripts/manage_models.py --compare iris-classifier-sklearn

# Set aliases for model versions (MLflow 3.x recommended approach)
python scripts/manage_models.py --alias iris-classifier-sklearn 1 staging
python scripts/manage_models.py --alias iris-classifier-sklearn 1 production
python scripts/manage_models.py --alias iris-classifier-sklearn 1 champion

# Show detailed model lineage and metadata
python scripts/manage_models.py --lineage iris-classifier-sklearn 1

# Legacy stage transitions (deprecated in MLflow 3.x - use aliases instead)
python scripts/manage_models.py --transition iris-classifier-sklearn 1 Staging
python scripts/manage_models.py --transition iris-classifier-sklearn 1 Production
```

### MLflow UI Access
```bash
# Access MLflow UI at: http://localhost:5004
# - View experiments and runs
# - Browse model registry
# - Compare model performance
# - Manage model stages and aliases
```

### MLflow 3.x Compatibility Notes

This project is compatible with MLflow 3.x, which introduces important changes:

- **Model Stages Deprecated**: Traditional stages (None, Staging, Production, Archived) are deprecated in favor of **aliases**
- **Recommended Approach**: Use aliases like `staging`, `production`, `champion` for model deployment
- **Backward Compatibility**: Stage-based commands still work but map to aliases internally
- **Migration Path**: Update your workflows to use `--alias` instead of `--transition` commands

```bash
# ✅ Modern MLflow 3.x approach (recommended)
python scripts/manage_models.py --alias iris-classifier-sklearn 1 production

# ⚠️ Legacy approach (deprecated but still works)
python scripts/manage_models.py --transition iris-classifier-sklearn 1 Production
```

### Production Monitoring Workflow
```bash
# 1. Start with some classification requests to generate production data
curl -X POST http://localhost:5001/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# 2. Simulate drift to trigger monitoring alerts
curl -X POST http://localhost:5001/api/v1/classify-shifted \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# 3. Generate drift analysis report
curl -X GET "http://localhost:5001/api/v1/drift-report?limit=50"

# 4. View HTML drift report (generated at data/drift_report.html)
open data/drift_report.html  # macOS

# 5. Use registry-based inference with model aliases (modern approach)
curl -X POST "http://localhost:5001/api/v1/classify-registry?model_format=sklearn&alias=production" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Alternative: Use staging alias for testing
curl -X POST "http://localhost:5001/api/v1/classify-registry?model_format=sklearn&alias=staging" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## 🧪 Testing & CI/CD

### Automated Testing with Pytest/Playwright

The project includes comprehensive test coverage for both Phase 1 and Phase 2 features using **pytest** for unit/integration testing and **Playwright** for end-to-end testing, following our established coding guidelines.

#### Test Structure

```
tests/
├── __init__.py                   # Test package initialization
├── conftest.py                   # Shared pytest fixtures and configuration
├── run_phase2_tests.py          # Comprehensive test runner for Phase 2
├── test_api_endpoints.py         # Unit tests for Phase 1 Flask API endpoints
├── test_phase2_features.py       # Unit tests for Phase 2 features (chat, gRPC, etc.)
├── test_grpc_server.py          # Unit tests for gRPC server functionality
├── test_grpc_integration.py     # Integration tests for gRPC client-server communication
├── test_langchain_memory.py     # Tests for LangChain conversation memory
├── test_serialization.py        # Tests for NumpyEncoder and serialization utilities
└── test_e2e_playwright.py       # End-to-end tests using Playwright (Phase 1 & 2)
```

#### Quick Phase 2 Test Execution

```bash
# Run all Phase 2 tests with service availability checks
python tests/run_phase2_tests.py

# This comprehensive test runner will:
# 1. Check Flask server availability
# 2. Check gRPC server availability  
# 3. Check Ollama service availability
# 4. Run appropriate test suites based on available services
# 5. Provide detailed status reporting and recommendations
```

### GitHub Actions CI/CD Pipeline

The project includes comprehensive GitHub Actions workflows for continuous integration:

- **Full CI Pipeline** (`.github/workflows/ci.yml`) - Runs on `main` and `develop` branches
  - Tests across Python 3.9, 3.10, 3.11, 3.12
  - Security scanning with Bandit and Safety
  - Code quality checks with Black, isort, Flake8, MyPy
  - Coverage reporting with Codecov integration
  - Comprehensive test execution with Ollama and Playwright setup

- **Quick Test Suite** (`.github/workflows/test-only.yml`) - Runs on feature branches
  - Fast feedback for development branches
  - Essential tests with Python 3.11
  - Manual endpoint verification

```bash
# Install test dependencies (if not already installed)
pip install -r requirements.txt
playwright install

# Run all tests
python -m pytest tests/ -v

# Run specific test types
python -m pytest tests/ -m "not e2e"  # Unit/integration tests only
python -m pytest tests/ -m "e2e"      # End-to-end tests only  
python -m pytest tests/ -m "security" # Security tests only

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=html

# Use the test runner script
python run_tests.py --type all --coverage --verbose
```

### Manual Testing

#### Phase 1 Tests
```bash
# Health check
curl http://localhost:5001/health

# Classification test
curl -X POST http://localhost:5001/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Secure generation test
curl -X POST http://localhost:5001/api/v1/generate-secure \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'
```

#### Phase 2 Tests
```bash
# Stateful chat test
curl -X POST http://localhost:5001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, what can you help me with?", "session_id": "test-session-1"}'

# Follow-up chat test (maintains context)
curl -X POST http://localhost:5001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Can you explain that in more detail?", "session_id": "test-session-1"}'

# Detailed classification with serialization demo
curl -X POST http://localhost:5001/api/v1/classify-detailed \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# gRPC classification (requires gRPC server running)
curl -X POST http://localhost:5001/api/v1/classify-grpc \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Performance benchmark (REST vs gRPC)
curl -X POST http://localhost:5001/api/v1/classify-benchmark \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "iterations": 5}'
```

#### Phase 3 Tests
```bash
# Semantic caching test - first call (cache miss)
curl -X POST http://localhost:5001/api/v1/chat-semantic \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?"}'

# Semantic caching test - similar prompt (cache hit)
curl -X POST http://localhost:5001/api/v1/chat-semantic \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is artificial intelligence?"}'

# API versioning test (v2)
curl -X POST http://localhost:5001/api/v2/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, this is API v2 test"}'

# Exact caching test - first call
curl -X POST http://localhost:5001/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Exact caching test - same request (cached response)
curl -X POST http://localhost:5001/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## 🎓 Concepts Demonstrated

### From Module 8 Outline:

#### Phase 1 Concepts:
- **1.2:** Separation of Concerns (API logic vs. Ollama server vs. ONNX model loading)
- **1.3:** Online (API endpoints) vs. Offline (batch script) Inference  
- **2.1:** Model Formats (ONNX vs. pickle security comparison)
- **2.2:** APIs for Inference (REST endpoints)
- **2.3:** Stateless Inference (independent API calls)
- **3.3:** Specialized LLM Frameworks (Ollama integration)
- **5.1:** Input Validation and Security (prompt injection prevention)

#### Phase 2 Concepts:
- **2.2:** APIs for Inference (REST vs. gRPC performance comparison, binary serialization vs JSON)
- **3.3:** Specialized Frameworks for LLMs (using Ollama with LangChain orchestration)
- **4.1:** Model Context Protocol (as a design pattern for stateful conversations)
- **4.2:** Core Components (Prompt Templates, Conversation Buffer Memory)
- **4.3:** Orchestration Frameworks (LangChain integration and memory management)
- **5.2:** Serialization of Complex Data Types (NumPy arrays, custom JSON encoders)

#### Phase 3 Concepts:
- **5.4:** Caching Inferences (both exact and semantic caching with vector embeddings)
- **6.1:** Versioning and Deployment Strategies (API Versioning with Flask Blueprints)

## ⚠️ Security Warnings

1. **Never load pickle files from untrusted sources** - They can execute arbitrary code
2. **Always use ONNX for production models** - Secure and cross-framework compatible  
3. **Implement prompt injection detection** - Protect against malicious inputs
4. **Validate all inputs** - Use proper schema validation for API requests

## 🚀 Next Steps (Phase 3)

The next phase will introduce:
- Containerization with Docker and Docker Compose
- Asynchronous task queues with Celery and Redis
- Production-grade concurrency management
- Load testing and performance optimization

## 📚 Dependencies

### Core Dependencies
- **Flask** - Web framework for API endpoints
- **requests** - HTTP client for Ollama API calls
- **onnxruntime** - ONNX model inference engine
- **scikit-learn** - Machine learning library for training
- **skl2onnx** - Convert sklearn models to ONNX format
- **numpy** - Numerical computing library

### Phase 2 Dependencies
- **langchain** - LLM orchestration and prompt templating framework
- **langchain-community** - Community integrations for LangChain
- **grpcio** - gRPC framework for high-performance communication
- **grpcio-tools** - Protocol Buffers compiler and tools
- **joblib** - Efficient serialization for scikit-learn models

### Phase 3 Dependencies
- **Flask-Caching** - Redis-based caching framework for Flask
- **redis** - Redis client for Python (caching backend)
- **fastembed** - Fast text embeddings for semantic similarity
- **hashlib** - Cryptographic hashing for cache keys (built-in)

### Testing Dependencies
- **pytest** - Unit and integration testing framework
- **pytest-playwright** - End-to-end testing with Playwright
- **pytest-mock** - Mock utilities for testing
- **pytest-cov** - Coverage reporting
- **playwright** - Browser automation for E2E tests

## 🔧 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service (macOS)
brew services start ollama

# Pull TinyLlama model
ollama pull tinyllama
```

### ONNX Model Not Found
```bash
# Ensure model training completed
python scripts/train_iris_model.py

# Check if model files exist
ls -la models/
```

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Redis Connection Issues (Phase 3)
```bash
# Check if Redis is running
redis-cli ping  # Should return PONG

# Start Redis service (macOS)
brew services start redis

# Check Redis connection
redis-cli
> keys *  # List all cached keys
> flushall  # Clear all cache (if needed)
```

### Semantic Caching Issues
```bash
# Test semantic caching manually
curl -X POST http://localhost:5001/api/v1/chat-semantic \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?"}'

# Check cache keys in Redis
redis-cli keys "semantic_cache:*"
redis-cli keys "embeddings:*"
```

## 🎯 Fair Performance Architecture (2025 Update)

### **Architectural Improvement**

This project now implements a **fair network-to-network comparison** between HTTP/REST and gRPC protocols:

**Previous Architecture (Unfair):**
- REST: Direct in-process ONNX inference ⚡ (no network overhead)
- gRPC: Network calls to separate server 🌐 (includes network latency)
- Result: "Apples to oranges" comparison

**Current Architecture (Fair):**
- HTTP/REST: Network calls to HTTP inference server 🌐 (port 5002)
- gRPC: Network calls to gRPC server 🌐 (port 50051)
- Result: **Fair comparison demonstrating gRPC's true performance advantages**

### **How to Test the Fair Comparison**

```bash
# Terminal 1: Start main Flask app
python app.py

# Terminal 2: Start HTTP inference server
python http_server.py

# Terminal 3: Start gRPC server
python grpc_server.py

# Test the fair performance comparison
curl -X POST http://localhost:5001/api/v1/classify-benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
    "iterations": 20
  }'
```

### **Expected Results**

With the fair architecture, you should now see:
- **gRPC 1.1x-2x faster**: For small payloads like Iris classification
- **gRPC 2.5x-10x faster**: For larger payloads, high concurrency, or streaming
- **HTTP/2 advantages**: Binary Protocol Buffers vs JSON serialization
- **Educational alignment**: Matches module learning objectives

## 🏗️ Phase 5: Production Serving Architecture Comparison

### **Serving Infrastructure Evolution**

| Approach | Use Case | Advantages | Disadvantages |
|----------|----------|------------|---------------|
| **Direct ONNX Loading** | Prototypes, Simple Apps | Minimal overhead, Easy setup | No batching, Limited optimization |
| **TensorFlow Serving** | TF-specific production | Built-in versioning, TF optimization | Framework-specific, More complex setup |
| **Triton Inference Server** | High-performance production | Framework-agnostic, Dynamic batching, Concurrent execution | Complex setup, Resource intensive |

### **Performance Characteristics**

- **Latency**: Direct ONNX < Triton < TensorFlow Serving (for single requests)
- **Throughput**: Triton (with batching) > TensorFlow Serving > Direct ONNX
- **Resource Utilization**: Triton > TensorFlow Serving > Direct ONNX
- **Scalability**: Triton > TensorFlow Serving > Direct ONNX

### **Dynamic Batching Benefits**

Run the dynamic batching demo to see Triton's performance advantages:

```bash
# Test dynamic batching performance
python scripts/test_dynamic_batching.py --num-requests 50 --max-workers 10

# Expected results:
# - Sequential requests: ~20 req/s
# - Concurrent requests (batched): ~80-150 req/s
# - Batching improvement: 4x-8x throughput increase
```

### **When to Use Each Approach**

#### **Direct ONNX Loading**
- **Best for**: Prototypes, single-user applications, simple deployments
- **Pros**: Minimal setup, fastest single-request latency
- **Cons**: No automatic batching, limited scalability

#### **TensorFlow Serving**
- **Best for**: TensorFlow-native applications, moderate scale
- **Pros**: Built-in model versioning, TensorFlow optimization
- **Cons**: TensorFlow-specific, moderate setup complexity

#### **Triton Inference Server**
- **Best for**: High-traffic production, multi-framework environments
- **Pros**: Framework-agnostic, dynamic batching, highest throughput
- **Cons**: Complex setup, higher resource requirements