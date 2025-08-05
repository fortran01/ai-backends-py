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
import datetime
from typing import Tuple, Any, Dict
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import warnings

# Phase 4: MLflow imports for model registry
import mlflow
import mlflow.sklearn
import mlflow.onnx
import onnx


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


def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the trained model and return performance metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing performance metrics
    """
    print("Evaluating model performance...")
    
    y_pred: np.ndarray = model.predict(X_test)
    
    # Calculate comprehensive metrics
    accuracy: float = accuracy_score(y_test, y_pred)
    precision: float = precision_score(y_test, y_pred, average='weighted')
    recall: float = recall_score(y_test, y_pred, average='weighted')
    f1: float = f1_score(y_test, y_pred, average='weighted')
    
    metrics: Dict[str, float] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Model precision: {precision:.4f}")
    print(f"Model recall: {recall:.4f}")
    print(f"Model F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))
    
    return metrics


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
    Save model in secure ONNX format with tensor-only outputs for onnxruntime-node compatibility.
    
    ONNX is the recommended format for production as it:
    - Cannot execute arbitrary code
    - Is framework-agnostic
    - Provides better security and portability
    - Compatible with both Python and TypeScript/Node.js runtimes
    
    Args:
        model: Trained model to save
        filepath: Path to save the ONNX file
    """
    print(f"\n✅ SECURE: Saving tensor-only ONNX model to {filepath}")
    print("✅ ONNX format is safe - it cannot execute arbitrary code!")
    print("✅ ONNX provides cross-framework compatibility and better security!")
    print("✅ Tensor-only outputs for onnxruntime-node compatibility!")
    
    # Define input shape for ONNX conversion (4 features for Iris dataset)
    initial_types: list = [('float_input', FloatTensorType([None, 4]))]
    
    # Convert sklearn model to ONNX format with tensor-only outputs
    onnx_model = convert_sklearn(
        model, 
        initial_types=initial_types,
        target_opset=12,  # Use a stable ONNX opset version
        options={
            # Only output tensor types (no maps/dictionaries) for onnxruntime-node compatibility
            'zipmap': False,  # Disable ZipMap which creates dictionary outputs
            'nocl': True,     # No class labels in probability output
        }
    )
    
    # Save ONNX model
    with open(filepath, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"✅ Secure tensor-only ONNX model saved to {filepath}")


def register_models_with_mlflow(
    model: RandomForestClassifier, 
    onnx_model_path: str,
    metrics: Dict[str, float],
    X_train: np.ndarray,
    dataset_version: str = "1.0"
) -> None:
    """
    Register both pickle and ONNX models in MLflow Model Registry with comprehensive metadata.
    
    Args:
        model: Trained sklearn model
        onnx_model_path: Path to the ONNX model file
        metrics: Dictionary of model performance metrics
        X_train: Training data for signature inference
        dataset_version: Version of the training dataset
    """
    print("\n🚀 Phase 4: Registering models with MLflow Model Registry...")
    
    # Configure MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5004")
    mlflow.set_experiment("iris-classification")
    
    with mlflow.start_run() as run:
        # Log hyperparameters
        params = {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "random_state": model.random_state,
            "dataset_version": dataset_version,
            "training_timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log additional metadata
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("dataset", "iris")
        mlflow.set_tag("purpose", "educational_demo")
        mlflow.set_tag("security_demo", "pickle_vs_onnx")
        
        # Register sklearn model (pickle format)
        sklearn_model_name = "iris-classifier-sklearn"
        print(f"📝 Registering sklearn model as '{sklearn_model_name}'...")
        
        sklearn_model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn_model",
            registered_model_name=sklearn_model_name,
            signature=mlflow.models.infer_signature(X_train)
        )
        
        print(f"✅ Sklearn model registered: {sklearn_model_info.model_uri}")
        
        # Register ONNX model (secure format)
        onnx_model_name = "iris-classifier-onnx"
        print(f"📝 Registering ONNX model as '{onnx_model_name}'...")
        
        # Load ONNX model properly using onnx.load()
        import onnx
        onnx_model = onnx.load(onnx_model_path)
        
        # Create signature for ONNX model
        y_pred = model.predict(X_train)
        signature = mlflow.models.infer_signature(X_train, y_pred)
        
        onnx_model_info = mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path="onnx_model", 
            registered_model_name=onnx_model_name,
            signature=signature,
            input_example=X_train[:5]  # Add example input
        )
        
        print(f"✅ ONNX model registered: {onnx_model_info.model_uri}")
        
        # Log model comparison notes
        mlflow.log_text(
            """
            Model Format Comparison:
            
            1. Sklearn (Pickle) Model:
               - Pros: Native Python format, preserves all sklearn functionality
               - Cons: Security risk - can execute arbitrary code, Python-specific
               - Use case: Internal development and testing only
               
            2. ONNX Model:
               - Pros: Secure (no code execution), cross-framework compatibility
               - Cons: Limited to inference only, some functionality loss
               - Use case: Production deployment (RECOMMENDED)
               
            Security Recommendation: Always use ONNX format for production deployment.
            """,
            "model_comparison.txt"
        )
        
        print(f"📊 MLflow run completed: {run.info.run_id}")
        print("🔗 View models in MLflow UI at: http://localhost:5004")
        print("📋 Model Registry accessible via MLflow UI > Models tab")


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
        
        # Evaluate the model and capture metrics
        metrics: Dict[str, float] = evaluate_model(model, X_test, y_test)
        
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
        
        # Phase 4: Register models with MLflow Model Registry
        try:
            register_models_with_mlflow(model, onnx_path, metrics, X_train)
        except Exception as mlflow_error:
            print(f"⚠️  MLflow registration failed: {mlflow_error}")
            print("💡 Make sure MLflow server is running: mlflow server --host 0.0.0.0 --port 5004")
            print("🔄 Continuing with local model saving...")
        
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