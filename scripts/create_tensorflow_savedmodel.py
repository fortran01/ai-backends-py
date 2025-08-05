#!/usr/bin/env python3
"""
Create a TensorFlow SavedModel for Iris classification compatible with TensorFlow Serving.

This script creates a trained neural network model that demonstrates TensorFlow Serving capabilities.
Following the coding guidelines: explicit type annotations and comprehensive documentation.
"""

import os
import sys
import logging
from typing import Dict, Any, List
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_iris_tensorflow_model(savedmodel_path: str) -> bool:
    """
    Create and train a TensorFlow model for Iris classification.
    
    Args:
        savedmodel_path: Path where the SavedModel will be saved
        
    Returns:
        bool: True if creation successful, False otherwise
    """
    try:
        logger.info("Creating TensorFlow model for Iris classification...")
        
        # Load and prepare Iris dataset
        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.int32)
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create the model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(4,), name='dense_1'),
            tf.keras.layers.Dropout(0.2, name='dropout_1'),
            tf.keras.layers.Dense(8, activation='relu', name='dense_2'),
            tf.keras.layers.Dense(3, activation='softmax', name='predictions')
        ], name='iris_classifier')
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        logger.info("Training the model...")
        history = model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=16,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Final test accuracy: {test_accuracy:.4f}")
        
        # Save the model using the recommended approach for Keras 3
        logger.info(f"Saving model to: {savedmodel_path}")
        os.makedirs(savedmodel_path, exist_ok=True)
        
        # Save as SavedModel using tf.saved_model.save
        tf.saved_model.save(model, savedmodel_path)
        
        # Save metadata
        metadata = {
            'model_type': 'tensorflow_keras_sequential',
            'input_shape': [4],
            'output_classes': iris.target_names.tolist(),
            'feature_names': iris.feature_names,
            'test_accuracy': float(test_accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs': 150,
            'optimizer': 'Adam',
            'learning_rate': 0.001
        }
        
        metadata_path = os.path.join(savedmodel_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✅ TensorFlow SavedModel created successfully!")
        logger.info(f"📊 Test accuracy: {test_accuracy:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {str(e)}")
        return False

def validate_savedmodel(savedmodel_path: str) -> bool:
    """
    Validate the SavedModel by loading and testing predictions.
    
    Args:
        savedmodel_path: Path to the SavedModel directory
        
    Returns:
        bool: True if validation successful, False otherwise
    """
    try:
        logger.info("Validating SavedModel...")
        
        # Load the model
        loaded_model = tf.keras.models.load_model(savedmodel_path)
        
        # Create test input (sample Iris flower measurements)
        test_samples = np.array([
            [5.1, 3.5, 1.4, 0.2],  # Should be Setosa (class 0)
            [6.4, 3.2, 4.5, 1.5],  # Should be Versicolor (class 1)
            [6.3, 3.3, 6.0, 2.5]   # Should be Virginica (class 2)
        ], dtype=np.float32)
        
        # Make predictions
        predictions = loaded_model.predict(test_samples)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Load Iris dataset for class names
        iris = load_iris()
        
        logger.info("✅ Validation successful!")
        logger.info("📊 Test predictions:")
        for i, (sample, pred_class, probabilities) in enumerate(zip(test_samples, predicted_classes, predictions)):
            logger.info(f"  Sample {i+1}: {sample} -> {iris.target_names[pred_class]} (confidence: {probabilities[pred_class]:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        return False

def main() -> None:
    """Main function to create TensorFlow SavedModel."""
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    savedmodel_path = os.path.join(base_dir, 'models', 'iris_tensorflow_savedmodel')
    
    logger.info("=== TensorFlow SavedModel Creation ===")
    logger.info(f"Target SavedModel: {savedmodel_path}")
    
    # Remove existing model if it exists
    if os.path.exists(savedmodel_path):
        import shutil
        logger.info("Removing existing SavedModel...")
        shutil.rmtree(savedmodel_path)
    
    # Create the model
    success = create_iris_tensorflow_model(savedmodel_path)
    
    if success:
        # Validate the model
        validation_success = validate_savedmodel(savedmodel_path)
        
        if validation_success:
            logger.info("\n🎉 SavedModel creation completed successfully!")
            logger.info("The model is ready for TensorFlow Serving.")
            logger.info(f"📁 SavedModel location: {savedmodel_path}")
            
            # Print serving instructions
            logger.info("\n📋 To start TensorFlow Serving:")
            logger.info("1. Install TensorFlow Serving (if not already installed)")
            logger.info("2. Start the server:")
            logger.info(f"   tensorflow_model_server --rest_api_port=8501 --model_name=iris --model_base_path={os.path.dirname(savedmodel_path)}/iris_tensorflow_savedmodel")
            logger.info("3. Test with curl:")
            logger.info("   curl -X POST http://localhost:8501/v1/models/iris:predict -H 'Content-Type: application/json' -d '{\"instances\": [[5.1, 3.5, 1.4, 0.2]]}'")
        else:
            logger.error("❌ Model validation failed!")
            sys.exit(1)
    else:
        logger.error("❌ Model creation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()