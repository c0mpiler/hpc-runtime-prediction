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
