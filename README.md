# AI Back-End Demo: Python/Flask Project - Phase 2

[![CI/CD Pipeline](https://github.com/your-username/ai-backends-py/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/your-username/ai-backends-py/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-blue.svg)](https://docs.pytest.org/)
[![End-to-end tested with Playwright](https://img.shields.io/badge/e2e%20tested%20with-playwright-green.svg)](https://playwright.dev/)

This project demonstrates **Phase 2** of the AI Back-End architecture course, extending Phase 1 with advanced features including stateful LLM interactions, high-performance gRPC communication, and comprehensive orchestration frameworks.

## 🎯 Phase 2 Objectives

**Goal:** Introduce stateful LLM interaction (chat memory), demonstrate gRPC as a high-performance alternative to REST, explore data serialization challenges, and implement Model Context Protocol (MCP) design patterns using LangChain orchestration.

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

# 6. Start the Flask application
python app.py

# 7. (Phase 2) Start the HTTP inference server in a separate terminal
python http_server.py

# 8. (Phase 2) Start the gRPC server in another separate terminal  
python grpc_server.py
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