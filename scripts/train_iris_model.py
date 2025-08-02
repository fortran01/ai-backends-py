"""
Iris Dataset Model Training Script with Security Demonstration

This script demonstrates:
1. Training a RandomForestClassifier on the Iris dataset
2. Saving models in both pickle (unsafe) and ONNX (safe) formats
3. Security vulnerabilities of pickle files
4. Model format comparison for production deployment

Following the coding guidelines:
- Explicit type annotations for all functions and variables
- Comprehensive documentation for security demonstration
- Educational security warnings about pickle risks
"""

import pickle
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
    
    # Split with random_state for reproducibility as specified in the plan
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
    
    # Use random_state=42 as specified in the implementation plan
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


def save_pickle_model(model: RandomForestClassifier, filepath: str) -> None:
    """
    Save model in pickle format with security warnings.
    
    SECURITY WARNING: Pickle files can execute arbitrary code when loaded!
    This demonstrates why pickle should never be used with untrusted sources.
    
    Args:
        model: Trained model to save
        filepath: Path to save the pickle file
    """
    print(f"\n🚨 SECURITY DEMONSTRATION: Saving model as pickle to {filepath}")
    print("⚠️  WARNING: Pickle files can execute arbitrary code when loaded!")
    print("⚠️  NEVER load pickle files from untrusted sources in production!")
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Safe pickle model saved to {filepath}")


def create_malicious_pickle(filepath: str) -> None:
    """
    Create a malicious pickle file to demonstrate security risks.
    
    EDUCATIONAL PURPOSE ONLY: This shows how an attacker could embed
    malicious code in a pickle file that gets executed when loaded.
    
    Args:
        filepath: Path to save the malicious pickle file
    """
    print(f"\n🔥 SECURITY DEMONSTRATION: Creating malicious pickle at {filepath}")
    print("⚠️  This demonstrates how pickle can be exploited by attackers!")
    
    # Create a malicious class that executes code when unpickled
    class MaliciousModel:
        def __reduce__(self) -> Tuple[Any, Tuple[str]]:
            # This will execute when the pickle is loaded!
            # In a real attack, this could download malware, steal data, etc.
            return (print, ("🚨 SECURITY BREACH: Pickle executed arbitrary code! This could have been malware! 🚨",))
    
    malicious_obj: MaliciousModel = MaliciousModel()
    
    with open(filepath, 'wb') as f:
        pickle.dump(malicious_obj, f)
    
    print(f"💀 Malicious pickle created at {filepath}")
    print("💀 When this file is loaded, it will execute arbitrary code!")
    print("💀 This is why ONNX format is preferred for production!")


def save_onnx_model(model: RandomForestClassifier, filepath: str) -> None:
    """
    Save model in secure ONNX format.
    
    ONNX is the recommended format for production as it:
    - Cannot execute arbitrary code
    - Is framework-agnostic
    - Provides better security and portability
    
    Args:
        model: Trained model to save
        filepath: Path to save the ONNX file
    """
    print(f"\n✅ SECURE: Saving model as ONNX to {filepath}")
    print("✅ ONNX format is safe - it cannot execute arbitrary code!")
    print("✅ ONNX provides cross-framework compatibility and better security!")
    
    # Define input shape for ONNX conversion (4 features for Iris dataset)
    initial_types: list = [('float_input', FloatTensorType([None, 4]))]
    
    # Convert sklearn model to ONNX format
    onnx_model = convert_sklearn(
        model, 
        initial_types=initial_types,
        target_opset=12  # Use a stable ONNX opset version
    )
    
    # Save ONNX model
    with open(filepath, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"✅ Secure ONNX model saved to {filepath}")


def demonstrate_pickle_loading_risk(safe_pickle_path: str, malicious_pickle_path: str) -> None:
    """
    Demonstrate the difference between loading safe and malicious pickle files.
    
    Args:
        safe_pickle_path: Path to the legitimate model pickle
        malicious_pickle_path: Path to the malicious pickle file
    """
    print("\n" + "="*80)
    print("🔍 SECURITY COMPARISON: Loading pickle files")
    print("="*80)
    
    # Load the safe pickle file
    print("\n1. Loading SAFE pickle file (legitimate model):")
    try:
        with open(safe_pickle_path, 'rb') as f:
            safe_model = pickle.load(f)
        print("✅ Safe model loaded successfully - this is the legitimate RandomForest model")
        print(f"✅ Model type: {type(safe_model)}")
    except Exception as e:
        print(f"❌ Error loading safe pickle: {e}")
    
    # Load the malicious pickle file  
    print("\n2. Loading MALICIOUS pickle file (contains embedded code):")
    print("⚠️  Watch for the security breach message...")
    try:
        with open(malicious_pickle_path, 'rb') as f:
            # This will execute the malicious code!
            malicious_obj = pickle.load(f)
        print("💀 Malicious pickle loaded - arbitrary code was executed!")
    except Exception as e:
        print(f"❌ Error loading malicious pickle: {e}")
    
    print("\n📋 SECURITY LESSON:")
    print("   - The safe pickle contained a legitimate model")
    print("   - The malicious pickle executed arbitrary code when loaded")
    print("   - In a real attack scenario, this could:")
    print("     * Download and execute malware")
    print("     * Steal sensitive data or credentials") 
    print("     * Compromise the entire system")
    print("   - This is why ONNX format is recommended for production!")


def main() -> None:
    """
    Main function that orchestrates the complete model training and security demonstration.
    """
    print("🧠 AI Back-End Demo: Iris Classification Model Training")
    print("🔒 Security Demonstration: Pickle vs ONNX formats")
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
        
        # Save models in different formats
        safe_pickle_path: str = 'models/iris_classifier.pkl'
        malicious_pickle_path: str = 'models/malicious_model.pkl'
        onnx_path: str = 'models/iris_classifier.onnx'
        
        # Save the legitimate model as pickle (for comparison)
        save_pickle_model(model, safe_pickle_path)
        
        # Create malicious pickle (for security demonstration)
        create_malicious_pickle(malicious_pickle_path)
        
        # Save the model in secure ONNX format
        save_onnx_model(model, onnx_path)
        
        # Demonstrate the security risks of pickle loading
        demonstrate_pickle_loading_risk(safe_pickle_path, malicious_pickle_path)
        
        print("\n" + "="*80)
        print("✅ Model training and security demonstration completed!")
        print("📁 Files created:")
        print(f"   - {safe_pickle_path} (legitimate model, but unsafe format)")
        print(f"   - {malicious_pickle_path} (malicious pickle - DO NOT DISTRIBUTE!)")
        print(f"   - {onnx_path} (secure, production-ready format)")
        print("\n🎯 Key Takeaways:")
        print("   1. Never load pickle files from untrusted sources")
        print("   2. Use ONNX format for production model deployment")
        print("   3. ONNX provides better security and cross-framework compatibility")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    # Suppress sklearn warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    main()