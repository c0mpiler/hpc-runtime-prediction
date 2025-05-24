#!/usr/bin/env python
"""gRPC Server for RT Predictor API Service."""

import grpc
from concurrent import futures
import logging
import time
import sys
import signal
import os
from pathlib import Path
from typing import Iterator, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import generated protobuf code
sys.path.insert(0, str(Path(__file__).parent.parent / 'proto'))
import rt_predictor_pb2
import rt_predictor_pb2_grpc

from service.predictor import PredictorService
from utils.config import load_config
from utils.logger import setup_logger
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import structlog

# Setup logging
logger = setup_logger('rt_predictor_server')
struct_logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('rt_predictor_requests_total', 'Total prediction requests', ['method', 'status'])
REQUEST_LATENCY = Histogram('rt_predictor_request_duration_seconds', 'Request latency', ['method'])
ACTIVE_CONNECTIONS = Gauge('rt_predictor_active_connections', 'Active gRPC connections')
MODEL_LOAD_TIME = Histogram('rt_predictor_model_load_seconds', 'Model loading time')
PREDICTION_LATENCY = Histogram('rt_predictor_prediction_duration_seconds', 'Prediction computation time')


class RTPredictorServicer(rt_predictor_pb2_grpc.JobRuntimePredictorServicer):
    """gRPC service implementation for RT Predictor."""
    
    def __init__(self, predictor_service: PredictorService):
        """Initialize the gRPC service."""
        self.predictor = predictor_service
        self.logger = struct_logger.bind(service='rt_predictor_grpc')
        
    def Predict(self, request: rt_predictor_pb2.PredictRequest, 
                context: grpc.ServicerContext) -> rt_predictor_pb2.PredictResponse:
        """Handle single prediction requests."""
        start_time = time.time()
        method = 'Predict'
        
        try:
            ACTIVE_CONNECTIONS.inc()
            self.logger.info("predict_request_received", 
                           processors=request.features.processors_req,
                           nodes=request.features.nodes_req,
                           partition=request.features.partition)
            
            # Make prediction
            with PREDICTION_LATENCY.time():
                prediction = self.predictor.predict_single(request.features)
            
            # Create response
            response = rt_predictor_pb2.PredictResponse(
                predicted_runtime_seconds=prediction['predicted_runtime'],
                confidence_lower=prediction['confidence_lower'],
                confidence_upper=prediction['confidence_upper'],
                model_version=prediction['model_version'],
                prediction_timestamp=int(time.time()),
                runtime_category=prediction.get('runtime_category', 'medium'),
                prediction_confidence=prediction.get('confidence_score', 0.8),
                model_metadata={'features_used': str(prediction.get('features_used', {}))}
            )
            
            # Record metrics
            REQUEST_COUNT.labels(method=method, status='success').inc()
            REQUEST_LATENCY.labels(method=method).observe(time.time() - start_time)
            
            self.logger.info("predict_request_completed", 
                           predicted_runtime=prediction['predicted_runtime'],
                           duration=time.time() - start_time)
            
            return response
            
        except Exception as e:
            REQUEST_COUNT.labels(method=method, status='error').inc()
            self.logger.error("predict_request_failed", error=str(e), exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Prediction failed: {str(e)}')
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()
    
    def PredictBatch(self, request: rt_predictor_pb2.PredictBatchRequest,
                     context: grpc.ServicerContext) -> rt_predictor_pb2.PredictBatchResponse:
        """Handle batch prediction requests."""
        start_time = time.time()
        method = 'BatchPredict'
        
        try:
            ACTIVE_CONNECTIONS.inc()
            self.logger.info("batch_predict_request_received", 
                           batch_size=len(request.jobs))
            
            # Make batch predictions
            with PREDICTION_LATENCY.time():
                predictions = self.predictor.predict_batch([req.features for req in request.jobs])
            
            # Create response
            responses = []
            for pred in predictions:
                responses.append(rt_predictor_pb2.PredictResponse(
                    predicted_runtime_seconds=pred['predicted_runtime'],
                    confidence_lower=pred['confidence_lower'],
                    confidence_upper=pred['confidence_upper'],
                    model_version=pred['model_version'],
                    prediction_timestamp=int(time.time()),
                    runtime_category=pred.get('runtime_category', 'medium'),
                    prediction_confidence=pred.get('confidence_score', 0.8),
                    model_metadata={'features_used': str(pred.get('features_used', {}))}
                ))
            
            response = rt_predictor_pb2.PredictBatchResponse(predictions=responses)
            
            # Record metrics
            REQUEST_COUNT.labels(method=method, status='success').inc()
            REQUEST_LATENCY.labels(method=method).observe(time.time() - start_time)
            
            self.logger.info("batch_predict_request_completed", 
                           batch_size=len(request.jobs),
                           duration=time.time() - start_time)
            
            return response
            
        except Exception as e:
            REQUEST_COUNT.labels(method=method, status='error').inc()
            self.logger.error("batch_predict_request_failed", error=str(e), exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Batch prediction failed: {str(e)}')
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()
    
    def PredictStream(self, request_iterator: Iterator[rt_predictor_pb2.PredictRequest],
                      context: grpc.ServicerContext) -> Iterator[rt_predictor_pb2.PredictResponse]:
        """Handle streaming prediction requests."""
        method = 'StreamPredict'
        request_count = 0
        
        try:
            ACTIVE_CONNECTIONS.inc()
            self.logger.info("stream_predict_started")
            
            for request in request_iterator:
                start_time = time.time()
                request_count += 1
                
                try:
                    # Make prediction
                    with PREDICTION_LATENCY.time():
                        prediction = self.predictor.predict_single(request.features)
                    
                    # Create response
                    response = rt_predictor_pb2.PredictResponse(
                        predicted_runtime_seconds=prediction['predicted_runtime'],
                        confidence_lower=prediction['confidence_lower'],
                        confidence_upper=prediction['confidence_upper'],
                        model_version=prediction['model_version'],
                        prediction_timestamp=int(time.time()),
                        runtime_category=prediction.get('runtime_category', 'medium'),
                        prediction_confidence=prediction.get('confidence_score', 0.8),
                        model_metadata={'features_used': str(prediction.get('features_used', {}))}
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
            
            return rt_predictor_pb2.GetModelInfoResponse(
                model_version=info['version'],
                model_type=info['type'],
                training_date=info['training_date'],
                num_features=info['feature_count'],
                validation_mae=info['metrics'].get('test_mae', 0),
                validation_mape=info['metrics'].get('test_mape', 0),
                test_mae=info['metrics'].get('test_mae', 0),
                test_mape=info['metrics'].get('test_mape', 0),
                is_loaded=True,
                predictions_served=int(REQUEST_COUNT.labels(method='Predict', status='success')._value.get()),
                average_latency_ms=0.0,  # This would need proper tracking
                uptime_seconds=int(time.time() - self.predictor.start_time)
            )
        except Exception as e:
            self.logger.error("get_model_info_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Failed to get model info: {str(e)}')
            raise
    
    def HealthCheck(self, request: rt_predictor_pb2.HealthCheckRequest,
                    context: grpc.ServicerContext) -> rt_predictor_pb2.HealthCheckResponse:
        """Health check endpoint."""
        try:
            is_healthy = self.predictor.is_healthy()
            
            return rt_predictor_pb2.HealthCheckResponse(
                status='healthy' if is_healthy else 'unhealthy',
                timestamp=int(time.time()),
                message='Service is operational' if is_healthy else 'Service is not operational'
            )
        except Exception as e:
            return rt_predictor_pb2.HealthCheckResponse(
                status='unhealthy',
                timestamp=int(time.time()),
                message=str(e)
            )


def serve(config: dict):
    """Start the gRPC server."""
    # Initialize predictor service
    logger.info("Loading predictor service...")
    start_load = time.time()
    
    predictor_service = PredictorService(config)
    
    MODEL_LOAD_TIME.observe(time.time() - start_load)
    logger.info(f"Predictor service loaded in {time.time() - start_load:.2f}s")
    
    # Start Prometheus metrics server
    metrics_port = config.get('metrics_port', 8181)
    start_http_server(metrics_port)
    logger.info(f"Prometheus metrics server started on port {metrics_port}")
    
    # Create gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.get('max_workers', 10)),
        options=[
            ('grpc.max_send_message_length', config.get('max_message_length', 10 * 1024 * 1024)),
            ('grpc.max_receive_message_length', config.get('max_message_length', 10 * 1024 * 1024)),
        ]
    )
    
    # Add service
    rt_predictor_pb2_grpc.add_JobRuntimePredictorServicer_to_server(
        RTPredictorServicer(predictor_service), server
    )
    
    # Add insecure port
    port = config.get('port', 50051)
    server.add_insecure_port(f'[::]:{port}')
    
    # Start server
    server.start()
    logger.info(f"gRPC server started on port {port}")
    
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
    
    # Get server config
    server_config = config.get('server', {})
    
    # Start server
    serve(server_config)


if __name__ == '__main__':
    main()
