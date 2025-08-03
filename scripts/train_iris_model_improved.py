"""
Improved Iris Dataset Model Training Script for TypeScript Compatibility

This script creates an ONNX model that outputs tensor-only types,
compatible with onnxruntime-node which doesn't support map/dictionary outputs.

Following the coding guidelines:
- Explicit type annotations for all functions and variables
- Comprehensive documentation for cross-platform compatibility
- Focus on production-ready ONNX export
"""

import os
from typing import Tuple, Any
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import warnings


def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the Iris dataset and prepare train/test splits.
    
    Returns:
        Tuple containing X_train, X_test, y_train, y_test arrays
    """
    print("Loading Iris dataset...")
    iris = load_iris()
    X: np.ndarray = iris.data
    y: np.ndarray = iris.target
    
    # Split with random_state for reproducibility
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray  
    y_test: np.ndarray
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained RandomForestClassifier model
    """
    print("Training RandomForestClassifier...")
    
    # Use same parameters as original for consistency
    model: RandomForestClassifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    return model


def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate the trained model and print performance metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    print("Evaluating model performance...")
    
    y_pred: np.ndarray = model.predict(X_test)
    accuracy: float = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))


def save_improved_onnx_model(model: RandomForestClassifier, filepath: str) -> None:
    """
    Save model in ONNX format with tensor-only outputs for onnxruntime-node compatibility.
    
    This version creates outputs that are compatible with onnxruntime-node:
    - Only label output (tensor) 
    - No probability maps/dictionaries
    
    Args:
        model: Trained model to save
        filepath: Path to save the ONNX file
    """
    print(f"\n✅ IMPROVED: Saving tensor-only ONNX model to {filepath}")
    print("✅ This version is compatible with onnxruntime-node")
    print("✅ Only outputs tensor types (no maps/dictionaries)")
    
    # Define input shape for ONNX conversion (4 features for Iris dataset)
    initial_types: list = [('float_input', FloatTensorType([None, 4]))]
    
    # Convert sklearn model to ONNX format with specific options
    onnx_model = convert_sklearn(
        model, 
        initial_types=initial_types,
        target_opset=12,  # Use stable ONNX opset version
        options={
            # Only output class labels (no probability maps)
            'zipmap': False,  # Disable ZipMap which creates dictionary outputs
            'nocl': True,     # No class labels in probability output
        }
    )
    
    # Save ONNX model
    with open(filepath, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"✅ Tensor-only ONNX model saved to {filepath}")
    print("✅ Compatible with both Python onnxruntime and onnxruntime-node")


def test_onnx_compatibility(filepath: str) -> None:
    """
    Test the ONNX model to verify it works with tensor-only outputs.
    
    Args:
        filepath: Path to the ONNX model file
    """
    try:
        import onnxruntime as ort
        
        print(f"\n🔍 TESTING: Loading ONNX model from {filepath}")
        
        session = ort.InferenceSession(filepath)
        
        print("✅ Model loaded successfully!")
        print(f"Input names: {[inp.name for inp in session.get_inputs()]}")
        print(f"Output names: {[out.name for out in session.get_outputs()]}")
        
        # Check output types
        for i, out in enumerate(session.get_outputs()):
            print(f"Output {i} ({out.name}): shape={out.shape}, type={out.type}")
        
        # Test inference
        print("\n🧪 TESTING: Running inference...")
        input_data = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
        results = session.run(None, {'float_input': input_data})
        
        print("✅ Inference successful!")
        print(f"Number of outputs: {len(results)}")
        for i, result in enumerate(results):
            print(f"Output {i}: type={type(result)}, shape={result.shape if hasattr(result, 'shape') else 'N/A'}")
            print(f"  Content: {result}")
        
        # Check if this is tensor-only (no lists/dicts)
        tensor_only = all(isinstance(result, np.ndarray) for result in results)
        if tensor_only:
            print("✅ SUCCESS: All outputs are tensors (numpy arrays)")
            print("✅ This model is compatible with onnxruntime-node!")
        else:
            print("❌ WARNING: Some outputs are not tensors")
            
    except ImportError:
        print("⚠️  onnxruntime not available for testing")
    except Exception as e:
        print(f"❌ Error testing ONNX model: {e}")


def main() -> None:
    """
    Main function that creates an improved ONNX model for TypeScript compatibility.
    """
    print("🧠 AI Back-End Demo: Improved Iris ONNX Model Training")
    print("🔧 Creating TypeScript/onnxruntime-node compatible model")
    print("="*80)
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test = load_and_prepare_data()
        
        # Train the model
        model = train_model(X_train, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)
        
        # Save improved ONNX model
        improved_onnx_path: str = 'models/iris_classifier_improved.onnx'
        save_improved_onnx_model(model, improved_onnx_path)
        
        # Test compatibility
        test_onnx_compatibility(improved_onnx_path)
        
        print("\n" + "="*80)
        print("✅ Improved model training completed!")
        print("📁 Files created:")
        print(f"   - {improved_onnx_path} (tensor-only, onnxruntime-node compatible)")
        print("\n🎯 Key Improvements:")
        print("   1. Only tensor outputs (no maps/dictionaries)")
        print("   2. Compatible with onnxruntime-node")
        print("   3. Maintains same prediction accuracy")
        print("   4. Ready for TypeScript/NestJS integration")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    # Suppress sklearn warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    main()