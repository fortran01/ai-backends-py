#!/usr/bin/env python3
"""
Local TensorFlow Serving alternative for Mac M1.
Serves SavedModel using Flask REST API compatible with TensorFlow Serving.

Following the coding guidelines: explicit type annotations and comprehensive documentation.
"""
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import logging
from typing import Dict, List, Any, Union
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the SavedModel
MODEL_PATH: str = "./models/iris_tensorflow_savedmodel"
model = None
infer = None

def load_model() -> None:
    """Load the TensorFlow SavedModel."""
    global model, infer
    try:
        model = tf.saved_model.load(MODEL_PATH)
        infer = model.signatures["serving_default"]
        logger.info(f"Model loaded successfully from: {MODEL_PATH}")
        logger.info(f"Available signatures: {list(model.signatures.keys())}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.route('/v1/models/iris:predict', methods=['POST'])
def predict() -> Union[Dict[str, Any], tuple]:
    """
    Predict endpoint compatible with TensorFlow Serving REST API.
    
    Expected input format:
    {
        "instances": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        ]
    }
    
    Returns:
        JSON response with predictions in TensorFlow Serving format
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        # Convert input to tensor format expected by TensorFlow Serving
        if 'instances' in data:
            instances = data['instances']
        else:
            # Handle single instance without 'instances' wrapper
            instances = [data]
        
        if not instances:
            return jsonify({'error': 'No instances provided'}), 400
        
        # Convert to numpy array and then to tensor
        input_data = []
        for inst in instances:
            if isinstance(inst, dict):
                # Handle named input format
                row = [
                    float(inst.get('sepal_length', 0)),
                    float(inst.get('sepal_width', 0)),
                    float(inst.get('petal_length', 0)),
                    float(inst.get('petal_width', 0))
                ]
            elif isinstance(inst, (list, tuple)) and len(inst) == 4:
                # Handle array input format
                row = [float(x) for x in inst]
            else:
                return jsonify({'error': f'Invalid instance format: {inst}'}), 400
            
            input_data.append(row)
        
        input_tensor = tf.constant(input_data, dtype=tf.float32)
        
        # Run inference
        predictions = infer(keras_tensor=input_tensor)
        
        # Format response like TensorFlow Serving
        result = {
            'predictions': []
        }
        
        for i in range(len(instances)):
            # Get the prediction probabilities
            prob_scores = predictions['output_0'][i].numpy()
            predicted_class_index = int(np.argmax(prob_scores))
            
            pred = {
                'output_label': predicted_class_index,
                'output_probability': prob_scores.tolist()
            }
            result['predictions'].append(pred)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/v1/models/iris', methods=['GET'])
def model_status() -> Dict[str, Any]:
    """
    Model status endpoint compatible with TensorFlow Serving.
    
    Returns:
        JSON response with model status information
    """
    return jsonify({
        'model_version_status': [{
            'version': '1',
            'state': 'AVAILABLE',
            'status': {
                'error_code': 'OK',
                'error_message': ''
            }
        }]
    })

@app.route('/v1/models/iris/metadata', methods=['GET'])
def model_metadata() -> Dict[str, Any]:
    """
    Model metadata endpoint compatible with TensorFlow Serving.
    
    Returns:
        JSON response with model metadata
    """
    return jsonify({
        'model_spec': {
            'name': 'iris',
            'signature_name': 'serving_default',
            'version': '1'
        },
        'metadata': {
            'signature_def': {
                'serving_default': {
                    'inputs': {
                        'float_input': {
                            'dtype': 'DT_FLOAT',
                            'tensor_shape': {
                                'dim': [
                                    {'size': '-1', 'name': ''},
                                    {'size': '4', 'name': ''}
                                ]
                            }
                        }
                    },
                    'outputs': {
                        'output_label': {
                            'dtype': 'DT_INT64',
                            'tensor_shape': {
                                'dim': [
                                    {'size': '-1', 'name': ''}
                                ]
                            }
                        },
                        'output_probability': {
                            'dtype': 'DT_FLOAT',
                            'tensor_shape': {
                                'dim': [
                                    {'size': '-1', 'name': ''},
                                    {'size': '3', 'name': ''}
                                ]
                            }
                        }
                    }
                }
            }
        }
    })

@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

def main() -> None:
    """Main function to start the TensorFlow Serving alternative."""
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model path does not exist: {MODEL_PATH}")
        logger.info("Please ensure the SavedModel is created by running: python scripts/keras_export_savedmodel.py")
        return
    
    # Load the model
    load_model()
    
    logger.info("Starting TensorFlow Serving alternative on port 8501")
    logger.info(f"Model loaded from: {MODEL_PATH}")
    logger.info("Available endpoints:")
    logger.info("  POST /v1/models/iris:predict - Make predictions")
    logger.info("  GET  /v1/models/iris - Model status")
    logger.info("  GET  /v1/models/iris/metadata - Model metadata")
    logger.info("  GET  /health - Health check")
    
    # Run Flask development server with auto-reload enabled
    app.run(
        host='0.0.0.0', 
        port=8501, 
        debug=True, 
        use_reloader=True,
        threaded=True
    )

if __name__ == '__main__':
    main()