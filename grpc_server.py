"""
gRPC Server for AI Back-End Demo - Phase 2

This standalone gRPC server demonstrates:
- High-performance binary protocol communication
- Efficient iris classification service
- Protocol Buffers serialization
- Performance comparison baseline vs REST

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive error handling
- Proper logging and monitoring
- Production-ready server configuration
"""

import logging
import time
import sys
import os
import threading
from concurrent import futures
from typing import Any, Optional

import grpc
import numpy as np
import onnxruntime as ort
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import generated gRPC files
import proto.inference_pb2 as inference_pb2
import proto.inference_pb2_grpc as inference_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to cache the ONNX model session
_onnx_session: Optional[ort.InferenceSession] = None

# Iris dataset class names
IRIS_CLASSES = ['setosa', 'versicolor', 'virginica']

# Global flag for server restart
_restart_server = False


class FileChangeHandler(FileSystemEventHandler):
    """Handler for file system events to restart server on changes."""
    
    def on_modified(self, event):
        """Handle file modification events."""
        global _restart_server
        
        if event.is_directory:
            return
        
        # Only restart on Python file changes, excluding __pycache__
        if (event.src_path.endswith('.py') and 
            '__pycache__' not in event.src_path and
            '.pyc' not in event.src_path):
            file_name = os.path.basename(event.src_path)
            logger.info(f"🔄 File changed: {file_name} - Restarting server...")
            _restart_server = True


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
            model_path: str = "models/iris_classifier.onnx"
            logger.info(f"Loading ONNX model from {model_path}")
            _onnx_session = ort.InferenceSession(model_path)
            logger.info("ONNX model loaded successfully")
        except FileNotFoundError:
            logger.error(f"ONNX model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found. Please run 'python scripts/train_iris_model.py' first.")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    return _onnx_session


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """
    gRPC servicer for iris classification inference.
    
    This class implements the InferenceService defined in the proto file,
    providing high-performance binary protocol access to the iris classifier.
    """
    
    def Classify(self, request: inference_pb2.ClassifyRequest, context: grpc.ServicerContext) -> inference_pb2.ClassifyResponse:
        """
        Classify iris flowers using the ONNX model via gRPC.
        
        Args:
            request: ClassifyRequest containing iris features
            context: gRPC context for the request
            
        Returns:
            ClassifyResponse with prediction results
        """
        start_time: float = time.time()
        
        try:
            logger.info(f"gRPC Classification request: sepal_length={request.sepal_length:.2f}, "
                       f"sepal_width={request.sepal_width:.2f}, petal_length={request.petal_length:.2f}, "
                       f"petal_width={request.petal_width:.2f}")
            
            # Validate input features
            features = [request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]
            
            # Basic range validation
            if not all(0.0 <= feature <= 10.0 for feature in features):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Feature values should be between 0.0 and 10.0")
                return inference_pb2.ClassifyResponse()
            
            # Convert to numpy array with correct shape for ONNX model
            features_array: np.ndarray = np.array([features], dtype=np.float32)
            
            # Get ONNX model session
            try:
                session = get_onnx_session()
            except (FileNotFoundError, RuntimeError) as e:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Model loading error: {str(e)}")
                return inference_pb2.ClassifyResponse()
            
            # Perform inference
            try:
                # Get input and output names from the ONNX model
                input_name: str = session.get_inputs()[0].name
                output_names = [output.name for output in session.get_outputs()]
                
                # Run inference
                results = session.run(output_names, {input_name: features_array})
                
                # Parse results from ONNX model with dual outputs
                predicted_class_index: int = int(results[0][0])  # Class prediction
                raw_probabilities: np.ndarray = results[1][0]  # Probability array
                
                # Normalize probabilities to ensure they sum to 1.0
                prob_sum: float = float(np.sum(raw_probabilities))
                if prob_sum > 0:
                    normalized_probabilities: np.ndarray = raw_probabilities / prob_sum
                else:
                    # Fallback if sum is zero - uniform distribution
                    normalized_probabilities = np.ones_like(raw_probabilities) / len(raw_probabilities)
                
                # Convert to list for protobuf serialization
                prob_list = [float(prob) for prob in normalized_probabilities]
                confidence: float = float(np.max(normalized_probabilities))
                predicted_class: str = IRIS_CLASSES[predicted_class_index]
                
                processing_time: float = time.time() - start_time
                logger.info(f"gRPC Classification result: {predicted_class} (confidence: {confidence:.3f}, "
                           f"time: {processing_time*1000:.2f}ms)")
                
                # Create and return response
                response = inference_pb2.ClassifyResponse()
                response.predicted_class = predicted_class
                response.predicted_class_index = predicted_class_index
                response.probabilities.extend(prob_list)
                response.confidence = confidence
                response.all_classes.extend(IRIS_CLASSES)
                
                # Echo back input features
                response.input_features.sepal_length = request.sepal_length
                response.input_features.sepal_width = request.sepal_width
                response.input_features.petal_length = request.petal_length
                response.input_features.petal_width = request.petal_width
                
                return response
                
            except Exception as e:
                logger.error(f"Error during gRPC inference: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Model inference failed: {str(e)}")
                return inference_pb2.ClassifyResponse()
                
        except Exception as e:
            logger.error(f"Unexpected error in gRPC Classify: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return inference_pb2.ClassifyResponse()


def serve_with_reload() -> None:
    """
    Start the gRPC server with file watching for auto-reload.
    
    The server runs on port 50051 and provides high-performance
    binary protocol access to the iris classification model.
    """
    global _restart_server
    
    while True:
        _restart_server = False
        
        # Set up file watcher
        event_handler = FileChangeHandler()
        observer = Observer()
        observer.schedule(event_handler, path='.', recursive=True)
        observer.start()
        
        # Create gRPC server with thread pool
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # Add our servicer to the server
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)
        
        # Listen on port 50051
        listen_addr = '[::]:50051'
        server.add_insecure_port(listen_addr)
        
        # Start the server
        server.start()
        
        print("🚀 gRPC Inference Server started (with auto-reload)")
        print(f"📡 Listening on {listen_addr}")
        print("🔌 Available services:")
        print("   - InferenceService.Classify - High-performance iris classification")
        print("🎯 Features demonstrated:")
        print("   - Protocol Buffers binary serialization")
        print("   - High-performance gRPC communication")
        print("   - Concurrent request handling with thread pool")
        print("   - Production-ready error handling and logging") 
        print("   - Auto-reload on file changes 🔄")
        print("💡 Test with gRPC client or performance comparison endpoint")
        print("⏹️  Press Ctrl+C to stop the server")
        
        try:
            # Keep checking for restart flag
            while not _restart_server:
                time.sleep(0.1)
            
            # Stop the server for restart
            print("\n🔄 Restarting server...")
            server.stop(grace=2.0)
            observer.stop()
            observer.join()
            
            # Clear the ONNX session cache for fresh reload
            global _onnx_session
            _onnx_session = None
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping gRPC server...")
            server.stop(grace=5.0)
            observer.stop()
            observer.join()
            print("✅ gRPC server stopped")
            break


def serve() -> None:
    """Legacy serve function for backward compatibility."""
    serve_with_reload()


if __name__ == '__main__':
    serve()