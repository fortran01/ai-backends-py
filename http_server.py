"""
Standalone HTTP server for Iris classification using ONNX Runtime - Python/Flask

This server provides a fair comparison baseline for REST vs gRPC performance
by ensuring both protocols use network calls rather than direct in-process inference.

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive error handling and logging
- Production-ready server configuration
"""

import logging
import time
from typing import Dict, Any, Optional, List
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app: Flask = Flask(__name__)

# Global variable to cache the ONNX model session
_onnx_session: Optional[ort.InferenceSession] = None

# Iris dataset class names
IRIS_CLASSES: List[str] = ['setosa', 'versicolor', 'virginica']


def get_onnx_session() -> ort.InferenceSession:
    """
    Get or create the ONNX runtime inference session.
    
    Returns:
        ort.InferenceSession: The ONNX runtime inference session
        
    Raises:
        FileNotFoundError: If the ONNX model file is not found
        RuntimeError: If there's an error loading the ONNX model
    """
    global _onnx_session
    
    if _onnx_session is None:
        try:
            model_path: str = "models/iris_classifier_improved.onnx"
            logger.info(f"Loading ONNX model from {model_path}")
            _onnx_session = ort.InferenceSession(model_path)
            logger.info("ONNX model loaded successfully")
            logger.info(f"Input names: {[input.name for input in _onnx_session.get_inputs()]}")
            logger.info(f"Output names: {[output.name for output in _onnx_session.get_outputs()]}")
        except FileNotFoundError:
            logger.error(f"ONNX model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found. Please run 'python scripts/train_iris_model_improved.py' first.")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    return _onnx_session


def validate_iris_features(data: Dict[str, Any]) -> tuple[bool, str, Optional[np.ndarray]]:
    """
    Validate iris classification input features.
    
    Args:
        data: Request JSON data containing iris features
        
    Returns:
        Tuple of (is_valid, error_message, features_array)
    """
    required_fields: List[str] = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}", None
    
    # Validate that all values are numeric
    try:
        features: List[float] = [float(data[field]) for field in required_fields]
    except (ValueError, TypeError) as e:
        return False, f"All feature values must be numeric: {e}", None
    
    # Basic range validation for iris features (reasonable biological limits)
    if not all(0.0 <= feature <= 20.0 for feature in features):
        return False, "Feature values should be between 0.0 and 20.0", None
    
    # Convert to numpy array with correct shape for ONNX model
    features_array: np.ndarray = np.array([features], dtype=np.float32)
    
    return True, "", features_array


def perform_inference(features_array: np.ndarray) -> Dict[str, Any]:
    """
    Perform Iris classification inference using the ONNX model.
    
    Args:
        features_array: Input features as numpy array
        
    Returns:
        Dictionary containing classification results
        
    Raises:
        Exception: If inference fails
    """
    try:
        session = get_onnx_session()
        
        # Get input and output names from the ONNX model
        input_name: str = session.get_inputs()[0].name
        output_names: List[str] = [output.name for output in session.get_outputs()]
        
        # Run inference
        start_time: float = time.time()
        results = session.run(output_names, {input_name: features_array})
        inference_time: float = (time.time() - start_time) * 1000  # Convert to ms
        
        # Handle model with dual outputs (current format)
        if len(results) >= 2:
            # Extract predictions and probabilities
            # Output 0: label (class index), Output 1: probabilities (array)
            label_output = results[0]
            probability_output = results[1]
            
            # Get predicted class index (handle both int and BigInt)
            raw_prediction = label_output[0]
            if isinstance(raw_prediction, (np.integer, int)):
                predicted_class_index: int = int(raw_prediction)
            else:
                predicted_class_index = int(float(raw_prediction))
            
            # Get probabilities from second output
            if len(probability_output.shape) > 1:
                probabilities: np.ndarray = probability_output[0]
            else:
                probabilities = probability_output
                
        else:
            # Fallback for single output models
            predicted_class_index = int(results[0][0])
            # Create uniform probability distribution as fallback
            probabilities = np.ones(len(IRIS_CLASSES)) / len(IRIS_CLASSES)
        
        # Normalize probabilities to ensure they sum to 1.0
        prob_sum: float = float(np.sum(probabilities))
        if prob_sum > 0:
            normalized_probabilities: np.ndarray = probabilities / prob_sum
        else:
            # Fallback if sum is zero - uniform distribution
            normalized_probabilities = np.ones_like(probabilities) / len(probabilities)
        
        # Convert to list for JSON serialization
        prob_list: List[float] = [float(prob) for prob in normalized_probabilities]
        confidence: float = float(np.max(normalized_probabilities))
        predicted_class: str = IRIS_CLASSES[predicted_class_index]
        
        logger.info(f"HTTP Inference completed in {inference_time:.2f}ms - Predicted: {predicted_class} (confidence: {confidence:.4f})")
        
        return {
            "predicted_class": predicted_class,
            "predicted_class_index": predicted_class_index,
            "probabilities": prob_list,
            "confidence": confidence,
            "class_names": IRIS_CLASSES,
            "inference_time_ms": round(inference_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Error during HTTP inference: {e}")
        raise Exception(f"Inference failed: {str(e)}")


@app.route('/health', methods=['GET'])
def health_check() -> Response:
    """
    Health check endpoint for the HTTP inference server.
    
    Returns:
        JSON response with server health status
    """
    try:
        # Check if model is loaded
        model_loaded: bool = _onnx_session is not None
        if not model_loaded:
            try:
                get_onnx_session()
                model_loaded = True
            except Exception:
                model_loaded = False
        
        return jsonify({
            "status": "healthy" if model_loaded else "degraded",
            "service": "HTTP Inference Server",
            "model": "iris_classifier_improved.onnx",
            "model_loaded": model_loaded,
            "timestamp": time.time(),
            "class_names": IRIS_CLASSES
        })
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/classify', methods=['POST'])
def classify() -> Response:
    """
    Iris classification endpoint using ONNX model.
    
    Request Body:
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
    Returns:
        JSON response with classification results:
        {
            "predicted_class": "setosa",
            "predicted_class_index": 0,
            "probabilities": [0.95, 0.03, 0.02],
            "confidence": 0.95,
            "class_names": ["setosa", "versicolor", "virginica"],
            "inference_time_ms": 2.5,
            "input_features": {...}
        }
        
    Raises:
        400: Missing or invalid input features
        500: Model loading or inference error
    """
    try:
        # Validate request data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data: Dict[str, Any] = request.get_json()
        
        # Validate input features
        is_valid, error_message, features_array = validate_iris_features(data)
        if not is_valid:
            return jsonify({
                "error": error_message,
                "code": "INVALID_INPUT"
            }), 400
        
        # Ensure model is loaded
        try:
            get_onnx_session()
        except (FileNotFoundError, RuntimeError) as e:
            return jsonify({
                "error": str(e),
                "code": "MODEL_LOADING_ERROR"
            }), 500
        
        # Perform inference
        try:
            start_time: float = time.time()
            result = perform_inference(features_array)
            total_time: float = (time.time() - start_time) * 1000
            
            # Add input features and total processing time to response
            response_data = {
                **result,
                "model_info": {
                    "format": "HTTP/ONNX",
                    "version": "1.0",
                    "inference_time_ms": total_time
                },
                "input_features": {
                    "sepal_length": float(data['sepal_length']),
                    "sepal_width": float(data['sepal_width']),
                    "petal_length": float(data['petal_length']),
                    "petal_width": float(data['petal_width'])
                }
            }
            
            logger.info(f"Sending HTTP response: {result['predicted_class']} (total time: {total_time:.2f}ms)")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return jsonify({
                "error": str(e),
                "code": "INFERENCE_ERROR"
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in /classify: {e}")
        return jsonify({
            "error": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500


@app.errorhandler(404)
def not_found(error) -> Response:
    """Handle 404 errors with API-appropriate response."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error) -> Response:
    """Handle 405 errors with API-appropriate response."""
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error) -> Response:
    """Handle 500 errors with API-appropriate response."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("🚀 Starting HTTP Inference Server for fair performance comparison")
    print("📋 Available endpoints:")
    print("   GET  /health    - Health check")
    print("   POST /classify  - Iris classification via HTTP/REST")
    print("🎯 Fair Architecture: Network-based HTTP inference for REST vs gRPC comparison")
    print("💡 This server enables fair performance comparison by ensuring both REST and gRPC use network calls")
    
    try:
        # Pre-load the model at startup
        logger.info("Pre-loading ONNX model...")
        get_onnx_session()
        logger.info("Model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")
        logger.info("Model will be loaded on first request")
    
    # Run Flask development server on port 5002 with auto-reload enabled
    app.run(
        host='0.0.0.0', 
        port=5002, 
        debug=True, 
        use_reloader=True,
        threaded=True
    )