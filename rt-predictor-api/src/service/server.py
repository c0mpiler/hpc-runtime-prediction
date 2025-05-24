#!/usr/bin/env python
"""Enhanced gRPC Server for RT Predictor API Service with performance optimizations."""

import grpc
from concurrent import futures
import logging
import time
import sys
import signal
import os
from pathlib import Path
from typing import Iterator, Optional, Dict, Any
import asyncio
from functools import wraps, lru_cache
import hashlib
import json
from threading import Lock
from collections import deque
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import generated protobuf code
sys.path.insert(0, str(Path(__file__).parent.parent / 'proto'))
import rt_predictor_pb2
import rt_predictor_pb2_grpc

from service.predictor import PredictorService
from utils.config import load_config
from utils.logger import setup_logger
from prometheus_client import start_http_server, Counter, Histogram, Gauge, Summary
import structlog

# Setup logging
logger = setup_logger('rt_predictor_server')
struct_logger = structlog.get_logger()

# Enhanced Prometheus metrics
REQUEST_COUNT = Counter('rt_predictor_requests_total', 'Total prediction requests', ['method', 'status'])
REQUEST_LATENCY = Histogram('rt_predictor_request_duration_seconds', 'Request latency', ['method'], 
                           buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5])
ACTIVE_CONNECTIONS = Gauge('rt_predictor_active_connections', 'Active gRPC connections')
MODEL_LOAD_TIME = Histogram('rt_predictor_model_load_seconds', 'Model loading time')
PREDICTION_LATENCY = Histogram('rt_predictor_prediction_duration_seconds', 'Prediction computation time',
                              buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25])
CACHE_HITS = Counter('rt_predictor_cache_hits_total', 'Cache hit count')
CACHE_MISSES = Counter('rt_predictor_cache_misses_total', 'Cache miss count')
RETRY_COUNT = Counter('rt_predictor_retries_total', 'Retry count', ['method'])
CIRCUIT_BREAKER_STATE = Gauge('rt_predictor_circuit_breaker_state', 'Circuit breaker state (0=closed, 1=open)')
BATCH_SIZE = Summary('rt_predictor_batch_size', 'Batch request sizes')
QUEUE_SIZE = Gauge('rt_predictor_queue_size', 'Request queue size')


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = Lock()
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'half-open'
                else:
                    CIRCUIT_BREAKER_STATE.set(1)
                    raise Exception("Circuit breaker is open")
                    
        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failure_count = 0
                    CIRCUIT_BREAKER_STATE.set(0)
            return result
            
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                    CIRCUIT_BREAKER_STATE.set(1)
                    
            raise e


def retry_with_backoff(retries: int = 3, backoff: float = 0.1):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if i < retries - 1:
                        sleep_time = backoff * (2 ** i)
                        time.sleep(sleep_time)
                        RETRY_COUNT.labels(method=func.__name__).inc()
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator


class PredictionCache:
    """LRU cache for prediction results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
        self.access_order = deque(maxlen=max_size)
        self._lock = Lock()
        
    def _hash_features(self, features: rt_predictor_pb2.JobFeatures) -> str:
        """Generate hash for job features."""
        feature_dict = {
            'processors_req': features.processors_req,
            'nodes_req': features.nodes_req,
            'mem_req': features.mem_req,
            'wallclock_req': features.wallclock_req,
            'partition': features.partition,
            'qos': features.qos
        }
        return hashlib.md5(json.dumps(feature_dict, sort_keys=True).encode()).hexdigest()
        
    def get(self, features: rt_predictor_pb2.JobFeatures) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available and not expired."""
        key = self._hash_features(features)
        
        with self._lock:
            if key not in self.cache:
                CACHE_MISSES.inc()
                return None
                
            # Check if expired
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                CACHE_MISSES.inc()
                return None
                
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            
            CACHE_HITS.inc()
            return self.cache[key]
            
    def put(self, features: rt_predictor_pb2.JobFeatures, prediction: Dict[str, Any]):
        """Store prediction in cache."""
        key = self._hash_features(features)
        
        with self._lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest = self.access_order.popleft()
                del self.cache[oldest]
                del self.timestamps[oldest]
                
            self.cache[key] = prediction
            self.timestamps[key] = time.time()
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)


class RTPredictorServicer(rt_predictor_pb2_grpc.JobRuntimePredictorServicer):
    """Enhanced gRPC service implementation for RT Predictor."""
    
    def __init__(self, predictor_service: PredictorService, config: dict):
        """Initialize the gRPC service."""
        self.predictor = predictor_service
        self.logger = struct_logger.bind(service='rt_predictor_grpc')
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('circuit_breaker_threshold', 5),
            recovery_timeout=config.get('circuit_breaker_timeout', 60.0)
        )
        self.cache = PredictionCache(
            max_size=config.get('cache_size', 1000),
            ttl_seconds=config.get('cache_ttl', 300)
        )
        self.predictor.start_time = time.time()
        self.request_queue = deque(maxlen=config.get('max_queue_size', 10000))
        
    def _validate_features(self, features: rt_predictor_pb2.JobFeatures) -> bool:
        """Validate job features."""
        if features.processors_req <= 0 or features.nodes_req <= 0:
            return False
        if features.mem_req < 0 or features.wallclock_req <= 0:
            return False
        if not features.partition or not features.qos:
            return False
        return True
        
    @retry_with_backoff(retries=3)
    def _make_prediction(self, features: rt_predictor_pb2.JobFeatures) -> Dict[str, Any]:
        """Make prediction with retry logic."""
        # Check cache first
        cached_result = self.cache.get(features)
        if cached_result:
            return cached_result
            
        # Make actual prediction
        prediction = self.circuit_breaker.call(self.predictor.predict_single, features)
        
        # Cache the result
        self.cache.put(features, prediction)
        
        return prediction
        
    def Predict(self, request: rt_predictor_pb2.PredictRequest, 
                context: grpc.ServicerContext) -> rt_predictor_pb2.PredictResponse:
        """Handle single prediction requests with enhanced error handling."""
        start_time = time.time()
        method = 'Predict'
        
        try:
            ACTIVE_CONNECTIONS.inc()
            
            # Validate request
            if not self._validate_features(request.features):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Invalid job features provided')
                REQUEST_COUNT.labels(method=method, status='invalid').inc()
                raise grpc.RpcError()
            
            self.logger.info("predict_request_received", 
                           processors=request.features.processors_req,
                           nodes=request.features.nodes_req,
                           partition=request.features.partition,
                           request_id=request.request_id)
            
            # Make prediction with retry logic
            with PREDICTION_LATENCY.time():
                prediction = self._make_prediction(request.features)
            
            # Create response
            response = rt_predictor_pb2.PredictResponse(
                predicted_runtime_seconds=prediction['predicted_runtime'],
                confidence_lower=prediction['confidence_lower'],
                confidence_upper=prediction['confidence_upper'],
                model_version=prediction['model_version'],
                prediction_timestamp=int(time.time()),
                runtime_category=prediction.get('runtime_category', 'medium'),
                prediction_confidence=prediction.get('confidence_score', 0.8),
                feature_importance=prediction.get('feature_importance', {}),
                request_id=request.request_id
            )
            
            # Record metrics
            REQUEST_COUNT.labels(method=method, status='success').inc()
            REQUEST_LATENCY.labels(method=method).observe(time.time() - start_time)
            
            self.logger.info("predict_request_completed", 
                           predicted_runtime=prediction['predicted_runtime'],
                           duration=time.time() - start_time,
                           request_id=request.request_id)
            
            return response
            
        except grpc.RpcError:
            raise
        except Exception as e:
            REQUEST_COUNT.labels(method=method, status='error').inc()
            self.logger.error("predict_request_failed", 
                            error=str(e), 
                            request_id=request.request_id,
                            exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Prediction failed: {str(e)}')
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()
    
    def PredictBatch(self, request: rt_predictor_pb2.PredictBatchRequest,
                     context: grpc.ServicerContext) -> rt_predictor_pb2.PredictBatchResponse:
        """Handle batch prediction requests with parallel processing."""
        start_time = time.time()
        method = 'PredictBatch'
        batch_size = len(request.jobs)
        
        try:
            ACTIVE_CONNECTIONS.inc()
            BATCH_SIZE.observe(batch_size)
            
            self.logger.info("batch_predict_request_received", 
                           batch_size=batch_size,
                           batch_id=request.batch_id)
            
            # Validate all features
            for i, job in enumerate(request.jobs):
                if not self._validate_features(job):
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(f'Invalid job features at index {i}')
                    REQUEST_COUNT.labels(method=method, status='invalid').inc()
                    raise grpc.RpcError()
            
            # Use thread pool for parallel predictions
            max_parallel = request.max_parallel or min(batch_size, 10)
            with futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                # Submit all predictions
                future_to_index = {
                    executor.submit(self._make_prediction, job): i 
                    for i, job in enumerate(request.jobs)
                }
                
                # Collect results
                predictions = [None] * batch_size
                failed_indices = []
                errors = []
                
                for future in futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        predictions[index] = future.result()
                    except Exception as e:
                        failed_indices.append(index)
                        errors.append(f"Job {index}: {str(e)}")
                        predictions[index] = {
                            'predicted_runtime': 0,
                            'confidence_lower': 0,
                            'confidence_upper': 0,
                            'model_version': 'error',
                            'runtime_category': 'error',
                            'confidence_score': 0
                        }
            
            # Create responses
            responses = []
            for pred in predictions:
                if pred:
                    responses.append(rt_predictor_pb2.PredictResponse(
                        predicted_runtime_seconds=pred['predicted_runtime'],
                        confidence_lower=pred['confidence_lower'],
                        confidence_upper=pred['confidence_upper'],
                        model_version=pred['model_version'],
                        prediction_timestamp=int(time.time()),
                        runtime_category=pred.get('runtime_category', 'medium'),
                        prediction_confidence=pred.get('confidence_score', 0.8),
                        feature_importance=pred.get('feature_importance', {})
                    ))
            
            response = rt_predictor_pb2.PredictBatchResponse(
                predictions=responses,
                batch_id=request.batch_id,
                total_predictions=batch_size,
                failed_predictions=len(failed_indices),
                errors=errors,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
            
            # Record metrics
            status = 'partial' if failed_indices else 'success'
            REQUEST_COUNT.labels(method=method, status=status).inc()
            REQUEST_LATENCY.labels(method=method).observe(time.time() - start_time)
            
            self.logger.info("batch_predict_request_completed", 
                           batch_size=batch_size,
                           failed=len(failed_indices),
                           duration=time.time() - start_time,
                           batch_id=request.batch_id)
            
            return response
            
        except grpc.RpcError:
            raise
        except Exception as e:
            REQUEST_COUNT.labels(method=method, status='error').inc()
            self.logger.error("batch_predict_request_failed", 
                            error=str(e), 
                            batch_id=request.batch_id,
                            exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Batch prediction failed: {str(e)}')
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()
    
    def PredictStream(self, request_iterator: Iterator[rt_predictor_pb2.PredictRequest],
                      context: grpc.ServicerContext) -> Iterator[rt_predictor_pb2.PredictResponse]:
        """Handle streaming prediction requests with backpressure."""
        method = 'PredictStream'
        request_count = 0
        
        try:
            ACTIVE_CONNECTIONS.inc()
            self.logger.info("stream_predict_started")
            
            for request in request_iterator:
                start_time = time.time()
                request_count += 1
                
                # Add to queue for monitoring
                self.request_queue.append(time.time())
                QUEUE_SIZE.set(len(self.request_queue))
                
                try:
                    # Validate features
                    if not self._validate_features(request.features):
                        self.logger.warning("stream_predict_invalid_features", 
                                          request_num=request_count)
                        continue
                    
                    # Make prediction
                    with PREDICTION_LATENCY.time():
                        prediction = self._make_prediction(request.features)
                    
                    # Create response
                    response = rt_predictor_pb2.PredictResponse(
                        predicted_runtime_seconds=prediction['predicted_runtime'],
                        confidence_lower=prediction['confidence_lower'],
                        confidence_upper=prediction['confidence_upper'],
                        model_version=prediction['model_version'],
                        prediction_timestamp=int(time.time()),
                        runtime_category=prediction.get('runtime_category', 'medium'),
                        prediction_confidence=prediction.get('confidence_score', 0.8),
                        feature_importance=prediction.get('feature_importance', {}),
                        request_id=request.request_id
                    )
                    
                    REQUEST_LATENCY.labels(method=method).observe(time.time() - start_time)
                    
                    yield response
                    
                except Exception as e:
                    self.logger.error("stream_predict_item_failed", 
                                    error=str(e), 
                                    request_num=request_count)
                    # Continue processing stream
                    
            REQUEST_COUNT.labels(method=method, status='success').inc()
            self.logger.info("stream_predict_completed", total_requests=request_count)
            
        except Exception as e:
            REQUEST_COUNT.labels(method=method, status='error').inc()
            self.logger.error("stream_predict_failed", error=str(e), exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Stream prediction failed: {str(e)}')
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()
    
    def GetModelInfo(self, request: rt_predictor_pb2.GetModelInfoRequest,
                     context: grpc.ServicerContext) -> rt_predictor_pb2.GetModelInfoResponse:
        """Get information about the loaded model."""
        try:
            info = self.predictor.get_model_info()
            
            # Calculate average latency
            total_latency = sum(REQUEST_LATENCY.labels(method='Predict')._sum.get() or 0)
            total_requests = sum(REQUEST_COUNT.labels(method='Predict', status='success')._value.get() or 0)
            avg_latency_ms = (total_latency / total_requests * 1000) if total_requests > 0 else 0
            
            return rt_predictor_pb2.GetModelInfoResponse(
                model_version=info['version'],
                model_type=info['type'],
                training_date=info['training_date'],
                num_features=info['feature_count'],
                training_samples=info.get('training_samples', 0),
                validation_mae=info['metrics'].get('val_mae', 0),
                validation_mape=info['metrics'].get('val_mape', 0),
                test_mae=info['metrics'].get('test_mae', 0),
                test_mape=info['metrics'].get('test_mape', 0),
                feature_names=info.get('feature_names', []),
                is_loaded=True,
                predictions_served=int(total_requests),
                average_latency_ms=avg_latency_ms,
                uptime_seconds=int(time.time() - self.predictor.start_time),
                ensemble_weights=info.get('ensemble_weights', {})
            )
        except Exception as e:
            self.logger.error("get_model_info_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Failed to get model info: {str(e)}')
            raise
    
    def HealthCheck(self, request: rt_predictor_pb2.HealthCheckRequest,
                    context: grpc.ServicerContext) -> rt_predictor_pb2.HealthCheckResponse:
        """Health check endpoint with detailed status."""
        try:
            is_healthy = self.predictor.is_healthy()
            
            # Get system metrics
            import psutil
            process = psutil.Process()
            memory_usage_mb = process.memory_info().rss / 1024 / 1024
            cpu_usage_percent = process.cpu_percent(interval=0.1)
            
            # Check circuit breaker state
            cb_state = 'open' if self.circuit_breaker.state == 'open' else 'closed'
            
            status = rt_predictor_pb2.HealthCheckResponse.SERVING if is_healthy else rt_predictor_pb2.HealthCheckResponse.NOT_SERVING
            
            return rt_predictor_pb2.HealthCheckResponse(
                status=status,
                message=f'Service is {"operational" if is_healthy else "degraded"}',
                uptime_seconds=int(time.time() - self.predictor.start_time),
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent,
                model_status={
                    'circuit_breaker': cb_state,
                    'cache_size': str(len(self.cache.cache)),
                    'queue_size': str(len(self.request_queue))
                }
            )
        except Exception as e:
            return rt_predictor_pb2.HealthCheckResponse(
                status=rt_predictor_pb2.HealthCheckResponse.NOT_SERVING,
                message=str(e)
            )


def serve(config: dict):
    """Start the enhanced gRPC server."""
    # Initialize predictor service
    logger.info("Loading predictor service...")
    start_load = time.time()
    
    # Pass full config to PredictorService (it needs model config)
    predictor_service = PredictorService(config)
    
    MODEL_LOAD_TIME.observe(time.time() - start_load)
    logger.info(f"Predictor service loaded in {time.time() - start_load:.2f}s")
    
    # Extract server config for server-specific settings
    server_config = config.get('server', {})
    
    # Start Prometheus metrics server
    metrics_port = server_config.get('metrics_port', 8181)
    start_http_server(metrics_port)
    logger.info(f"Prometheus metrics server started on port {metrics_port}")
    
    # Create gRPC server with optimized settings
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=server_config.get('max_workers', 20)),
        options=[
            ('grpc.max_send_message_length', server_config.get('max_message_length', 10 * 1024 * 1024)),
            ('grpc.max_receive_message_length', server_config.get('max_message_length', 10 * 1024 * 1024)),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.max_ping_strikes', 0),
        ],
        maximum_concurrent_rpcs=server_config.get('max_concurrent_rpcs', 100)
    )
    
    # Add service (pass server_config for cache/circuit breaker settings)
    rt_predictor_pb2_grpc.add_JobRuntimePredictorServicer_to_server(
        RTPredictorServicer(predictor_service, server_config), server
    )
    
    # Add insecure port
    port = server_config.get('port', 50051)
    server.add_insecure_port(f'[::]:{port}')
    
    # Start server
    server.start()
    logger.info(f"Enhanced gRPC server started on port {port}")
    
    # Setup graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down server...")
        server.stop(grace=10)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep server running
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(grace=10)


def main():
    """Main entry point."""
    # Load configuration
    config = load_config()
    
    # Set enhanced server defaults
    server_config = config.get('server', {})
    server_config.setdefault('max_workers', 20)
    server_config.setdefault('max_concurrent_rpcs', 100)
    server_config.setdefault('cache_size', 1000)
    server_config.setdefault('cache_ttl', 300)
    server_config.setdefault('circuit_breaker_threshold', 5)
    server_config.setdefault('circuit_breaker_timeout', 60.0)
    server_config.setdefault('max_queue_size', 10000)
    config['server'] = server_config  # Update config with defaults
    
    # Start server with full config
    serve(config)


if __name__ == '__main__':
    main()
