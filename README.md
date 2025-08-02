# AI Back-End Demo: Python/Flask Project - Phase 1

[![CI/CD Pipeline](https://github.com/your-username/ai-backends-py/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/your-username/ai-backends-py/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-blue.svg)](https://docs.pytest.org/)
[![End-to-end tested with Playwright](https://img.shields.io/badge/e2e%20tested%20with-playwright-green.svg)](https://playwright.dev/)

This project demonstrates **Phase 1** of the AI Back-End architecture course, showcasing core concepts of serving AI models in production.

## 🎯 Phase 1 Objectives

**Goal:** Establish foundational project structure and implement basic, stateless API that serves both a local LLM (TinyLlama) and a traditional ML model in ONNX format.

## 🚀 Features Implemented

### ✅ Completed Tasks

1. **Environment Setup** - Python virtual environment with required dependencies
2. **Model Training & Security Demo** - Iris dataset training with pickle/ONNX comparison  
3. **Basic Flask API** - Production-ready Flask application with proper structure
4. **Stateless LLM Generation** - `/api/v1/generate` endpoint using Ollama API
5. **Security Demonstration** - `/api/v1/generate-secure` with prompt injection prevention
6. **ONNX Model Inference** - `/api/v1/classify` endpoint for iris classification
7. **Batch Processing** - Offline inference script for bulk prompt processing

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
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
# Returns service status and dependency health
```

### LLM Text Generation (Basic)
```bash
POST /api/v1/generate
Content-Type: application/json

{
  "prompt": "What is artificial intelligence?"
}
```

### LLM Text Generation (Secure)
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

### Iris Classification
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
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── tests/                   # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py          # Pytest configuration and fixtures
│   ├── test_api_endpoints.py    # Unit/integration tests with pytest
│   └── test_e2e_playwright.py   # End-to-end tests with Playwright
├── pytest.ini              # Pytest configuration
├── run_tests.py            # Test runner script
├── venv/                    # Virtual environment
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

The project uses **pytest** for unit/integration testing and **Playwright** for end-to-end testing, following our established coding guidelines.

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

## 🎓 Concepts Demonstrated

### From Module 8 Outline:

- **1.2:** Separation of Concerns (API logic vs. Ollama server vs. ONNX model loading)
- **1.3:** Online (API endpoints) vs. Offline (batch script) Inference  
- **2.1:** Model Formats (ONNX vs. pickle security comparison)
- **2.2:** APIs for Inference (REST endpoints)
- **2.3:** Stateless Inference (independent API calls)
- **3.3:** Specialized LLM Frameworks (Ollama integration)
- **5.1:** Input Validation and Security (prompt injection prevention)

## ⚠️ Security Warnings

1. **Never load pickle files from untrusted sources** - They can execute arbitrary code
2. **Always use ONNX for production models** - Secure and cross-framework compatible  
3. **Implement prompt injection detection** - Protect against malicious inputs
4. **Validate all inputs** - Use proper schema validation for API requests

## 🚀 Next Steps (Phase 2)

The next phase will introduce:
- Stateful LLM interactions (chat memory)
- gRPC communication protocols  
- Advanced orchestration with LangChain
- Performance comparisons between protocols

## 📚 Dependencies

### Core Dependencies
- **Flask** - Web framework for API endpoints
- **requests** - HTTP client for Ollama API calls
- **onnxruntime** - ONNX model inference engine
- **scikit-learn** - Machine learning library for training
- **skl2onnx** - Convert sklearn models to ONNX format
- **numpy** - Numerical computing library

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