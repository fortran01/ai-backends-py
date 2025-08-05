#!/usr/bin/env python3
"""
Create TensorFlow SavedModel using Keras export functionality.

This script uses Keras 3's export functionality to create a compatible SavedModel.
Following the coding guidelines: explicit type annotations and comprehensive documentation.
"""

import os
import sys
import logging
from typing import Dict, Any
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_keras_savedmodel(savedmodel_path: str) -> bool:
    """
    Create TensorFlow SavedModel using Keras export.
    
    Args:
        savedmodel_path: Path where the SavedModel will be saved
        
    Returns:
        bool: True if creation successful, False otherwise
    """
    try:
        logger.info("Creating Keras SavedModel...")
        
        # Load Iris dataset
        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.int32)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create model with Input layer
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        logger.info("Training model...")
        model.fit(X_train, y_train, epochs=50, verbose=0, validation_data=(X_test, y_test))
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Export using Keras export
        logger.info(f"Exporting to SavedModel: {savedmodel_path}")
        os.makedirs(savedmodel_path, exist_ok=True)
        
        # Use export method
        model.export(savedmodel_path)
        
        # Save metadata
        metadata = {
            'model_type': 'keras_exported',
            'input_shape': [4],
            'output_classes': iris.target_names.tolist(),
            'feature_names': iris.feature_names,
            'test_accuracy': float(test_accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        metadata_path = os.path.join(savedmodel_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✅ SavedModel exported successfully!")
        logger.info(f"📊 Test accuracy: {test_accuracy:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SavedModel export failed: {str(e)}")
        return False

def validate_savedmodel(savedmodel_path: str) -> bool:
    """
    Validate the SavedModel by loading and testing it.
    
    Args:
        savedmodel_path: Path to the SavedModel directory
        
    Returns:
        bool: True if validation successful, False otherwise
    """
    try:
        logger.info("Validating SavedModel...")
        
        # Load the SavedModel
        loaded_model = tf.saved_model.load(savedmodel_path)
        
        # Get the serving function
        serving_fn = loaded_model.signatures['serving_default']
        
        # Test with sample data
        test_input = tf.constant([
            [5.1, 3.5, 1.4, 0.2],  # Setosa
            [6.4, 3.2, 4.5, 1.5],  # Versicolor
            [6.3, 3.3, 6.0, 2.5]   # Virginica
        ], dtype=tf.float32)
        
        # Make prediction
        result = serving_fn(input_1=test_input)
        
        # Load iris data for class names
        iris = load_iris()
        
        logger.info("✅ Validation successful!")
        logger.info("📊 Test predictions:")
        predictions = result['dense_2'].numpy()
        for i, (sample, prediction) in enumerate(zip(test_input.numpy(), predictions)):
            class_idx = np.argmax(prediction)
            confidence = prediction[class_idx]
            class_name = iris.target_names[class_idx]
            logger.info(f"  Sample {i+1}: {sample} -> {class_name} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        return False

def main() -> None:
    """Main function."""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    savedmodel_path = os.path.join(base_dir, 'models', 'iris_tensorflow_savedmodel')
    
    logger.info("=== Keras Export SavedModel Creation ===")
    
    # Remove existing model
    if os.path.exists(savedmodel_path):
        import shutil
        shutil.rmtree(savedmodel_path)
    
    # Create the model
    success = create_keras_savedmodel(savedmodel_path)
    
    if success:
        validation_success = validate_savedmodel(savedmodel_path)
        
        if validation_success:
            logger.info("\n🎉 SavedModel creation completed successfully!")
            logger.info(f"📁 SavedModel location: {savedmodel_path}")
            logger.info("\n📋 To start TensorFlow Serving:")
            logger.info("1. Start the server:")
            logger.info(f"   tensorflow_model_server --rest_api_port=8501 --model_name=iris --model_base_path={os.path.dirname(savedmodel_path)}")
            logger.info("2. Test with curl:")
            logger.info("   curl -X POST http://localhost:8501/v1/models/iris:predict -H 'Content-Type: application/json' -d '{\"instances\": [[5.1, 3.5, 1.4, 0.2]]}'")
        else:
            sys.exit(1)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()