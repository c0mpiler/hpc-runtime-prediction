#!/usr/bin/env python3
"""
gRPC Client wrapper for RT Predictor UI service.
Handles communication with the RT Predictor API service.
"""

import grpc
import time
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent import futures

# Import generated gRPC modules
from proto import rt_predictor_pb2
from proto import rt_predictor_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Data class for prediction results."""
    predicted_runtime: float
    confidence_interval: Tuple[float, float]
    model_version: str
    prediction_time: datetime
    features_used: Dict[str, Any]
    runtime_category: str = "medium"
    confidence_score: float = 0.8
    
    def to_dict(self):
        """Convert to dictionary for display."""
        return {
            'predicted_runtime': self.predicted_runtime,
            'confidence_lower': self.confidence_interval[0],
            'confidence_upper': self.confidence_interval[1],
            'model_version': self.model_version,
            'prediction_time': self.prediction_time.isoformat(),
            'runtime_category': self.runtime_category,
            'confidence_score': self.confidence_score,
            'features_used': self.features_used
        }


class RTPredictorClient:
    """
    gRPC client wrapper for RT Predictor service.
    Provides high-level methods for prediction requests.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 50051, timeout: int = 30):
        """
        Initialize the gRPC client.
        
        Args:
            host: API service hostname
            port: API service port
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self._connect()
    
    def _connect(self):
        """Establish connection to the gRPC service."""
        try:
            # Create channel with retry policy
            options = [
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
            ]
            
            self.channel = grpc.insecure_channel(
                f'{self.host}:{self.port}',
                options=options
            )
            
            # Add timeout for connection check
            grpc.channel_ready_future(self.channel).result(timeout=5)
            
            self.stub = rt_predictor_pb2_grpc.JobRuntimePredictorStub(self.channel)
            logger.info(f"Connected to RT Predictor API at {self.host}:{self.port}")
            
        except grpc.FutureTimeoutError:
            logger.error(f"Failed to connect to RT Predictor API at {self.host}:{self.port}")
            raise ConnectionError(f"Cannot connect to API service at {self.host}:{self.port}")

    def predict_single(self, job_params: Dict[str, Any]) -> PredictionResult:
        """
        Make a single job runtime prediction.
        
        Args:
            job_params: Dictionary containing job parameters
            
        Returns:
            PredictionResult object
        """
        try:
            # Create JobFeatures message
            features = rt_predictor_pb2.JobFeatures(
                processors_req=int(job_params.get('cpus_req', 1)),
                nodes_req=int(job_params.get('nodes_req', 1)),
                mem_req=float(job_params.get('mem_req', 1.0)),  # Convert MB to GB
                wallclock_req=float(job_params.get('walltime_req', 3600)),
                partition=job_params.get('partition', 'default'),
                qos=job_params.get('qos', 'normal'),
                submit_time=int(job_params.get('submit_time', int(time.time())))
            )
            
            # Create request
            request = rt_predictor_pb2.PredictRequest(
                features=features,
                request_id=f"ui-{int(time.time())}",
                metadata={
                    'source': 'streamlit-ui',
                    'user': job_params.get('username', 'unknown')
                }
            )
            
            # Make prediction
            response = self.stub.Predict(request, timeout=self.timeout)
            
            # Process response
            result = PredictionResult(
                predicted_runtime=response.predicted_runtime_seconds,
                confidence_interval=(
                    response.confidence_lower,
                    response.confidence_upper
                ),
                model_version=response.model_version,
                prediction_time=datetime.fromtimestamp(response.prediction_timestamp),
                runtime_category=response.runtime_category,
                confidence_score=response.prediction_confidence,
                features_used=job_params
            )
            
            logger.info(f"Prediction successful: {result.predicted_runtime:.2f} seconds")
            return result
        
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()} - {e.details()}")
            raise RuntimeError(f"Prediction failed: {e.details()}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, job_params_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """
        Make batch predictions for multiple jobs.
        
        Args:
            job_params_list: List of job parameter dictionaries
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        # Use thread pool for concurrent requests
        with futures.ThreadPoolExecutor(max_workers=min(10, len(job_params_list))) as executor:
            future_to_params = {
                executor.submit(self.predict_single, params): params 
                for params in job_params_list
            }
            
            for future in futures.as_completed(future_to_params):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    params = future_to_params[future]
                    logger.error(f"Failed to predict for params {params}: {str(e)}")
                    # Add a failed result
                    results.append(None)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        try:
            # Create empty request (no specific model info request in proto)
            request = rt_predictor_pb2.PredictRequest(
                features=rt_predictor_pb2.JobFeatures(
                    processors_req=1,
                    nodes_req=1,
                    mem_req=1.0,
                    wallclock_req=1.0,
                    partition="info",
                    qos="normal",
                    submit_time=int(time.time())
                ),
                request_id="model-info",
                metadata={'request_type': 'model_info'}
            )
            
            response = self.stub.Predict(request, timeout=5)
            
            return {
                'version': response.model_version,
                'type': response.model_metadata.get('model_type', 'ensemble'),
                'last_updated': response.model_metadata.get('last_updated', 'unknown')
            }
        except:
            # Return default info if service doesn't support model info
            return {
                'version': '2.0',
                'type': 'ensemble',
                'last_updated': 'unknown'
            }
    
    def close(self):
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()
            logger.info("Closed connection to RT Predictor API")


def format_runtime(seconds: float) -> str:
    """
    Format runtime in seconds to human-readable format.
    
    Args:
        seconds: Runtime in seconds
        
    Returns:
        Formatted string (e.g., "2h 30m 15s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours < 24:
            return f"{hours}h {minutes}m {secs}s"
        else:
            days = hours // 24
            hours = hours % 24
            return f"{days}d {hours}h {minutes}m"


def calculate_efficiency(predicted_runtime: float, requested_walltime: float) -> float:
    """
    Calculate job efficiency based on predicted vs requested time.
    
    Args:
        predicted_runtime: Predicted runtime in seconds
        requested_walltime: Requested walltime in seconds
        
    Returns:
        Efficiency percentage (0-100)
    """
    if requested_walltime <= 0:
        return 0.0
    
    efficiency = (predicted_runtime / requested_walltime) * 100
    # Cap at 100% (can't be more than 100% efficient)
    return min(efficiency, 100.0)
