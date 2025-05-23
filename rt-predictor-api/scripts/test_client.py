#!/usr/bin/env python
"""Test client for RT Predictor gRPC service."""

import grpc
import sys
import time
import argparse
from pathlib import Path

# Add proto directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'proto'))
import rt_predictor_pb2
import rt_predictor_pb2_grpc


def test_single_prediction(stub):
    """Test single prediction."""
    print("\n=== Testing Single Prediction ===")
    
    request = rt_predictor_pb2.PredictRequest(
        processors_req=32,
        nodes_req=4,
        mem_req=128000,
        time_req=3600,
        partition="normal",
        qos="normal",
        gpus_req=0,
        user="testuser",
        account="testacct",
        job_name="test_job"
    )
    
    try:
        start_time = time.time()
        response = stub.Predict(request)
        elapsed = time.time() - start_time
        
        print(f"✓ Prediction successful (took {elapsed:.3f}s)")
        print(f"  Predicted runtime: {response.predicted_runtime:.2f} seconds")
        print(f"  Confidence interval: [{response.confidence_lower:.2f}, {response.confidence_upper:.2f}]")
        print(f"  Model version: {response.model_version}")
        print(f"  Features used: {response.features_used}")
        
        # Check for overestimation
        overestimation_ratio = request.time_req / response.predicted_runtime
        if overestimation_ratio > 1.5:
            print(f"  ⚠️  Overestimation alert! Requested/Predicted ratio: {overestimation_ratio:.2f}")
        
    except grpc.RpcError as e:
        print(f"✗ Prediction failed: {e.code()}: {e.details()}")


def test_batch_prediction(stub):
    """Test batch prediction."""
    print("\n=== Testing Batch Prediction ===")
    
    # Create multiple requests
    requests = []
    for i in range(5):
        requests.append(rt_predictor_pb2.PredictRequest(
            processors_req=8 * (i + 1),
            nodes_req=i + 1,
            mem_req=16000 * (i + 1),
            time_req=1800 * (i + 1),
            partition="normal" if i % 2 == 0 else "gpu",
            qos="normal",
            gpus_req=0 if i < 3 else 2,
            user=f"user{i}",
            account="batchtest",
            job_name=f"batch_job_{i}"
        ))
    
    batch_request = rt_predictor_pb2.BatchPredictRequest(requests=requests)
    
    try:
        start_time = time.time()
        batch_response = stub.BatchPredict(batch_request)
        elapsed = time.time() - start_time
        
        print(f"✓ Batch prediction successful (took {elapsed:.3f}s)")
        print(f"  Processed {len(batch_response.responses)} predictions")
        
        for i, response in enumerate(batch_response.responses):
            print(f"\n  Job {i}:")
            print(f"    Predicted runtime: {response.predicted_runtime:.2f} seconds")
            print(f"    Confidence: [{response.confidence_lower:.2f}, {response.confidence_upper:.2f}]")
            
    except grpc.RpcError as e:
        print(f"✗ Batch prediction failed: {e.code()}: {e.details()}")


def test_stream_prediction(stub):
    """Test streaming prediction."""
    print("\n=== Testing Stream Prediction ===")
    
    def request_generator():
        """Generate streaming requests."""
        for i in range(10):
            yield rt_predictor_pb2.PredictRequest(
                processors_req=16,
                nodes_req=2,
                mem_req=32000,
                time_req=7200,
                partition="gpu" if i % 3 == 0 else "normal",
                qos="high",
                gpus_req=1 if i % 3 == 0 else 0,
                user="streamuser",
                account="streamtest",
                job_name=f"stream_job_{i}"
            )
            time.sleep(0.1)  # Simulate streaming delay
    
    try:
        response_count = 0
        start_time = time.time()
        
        for response in stub.StreamPredict(request_generator()):
            response_count += 1
            print(f"  Response {response_count}: {response.predicted_runtime:.2f}s "
                  f"(confidence: [{response.confidence_lower:.2f}, {response.confidence_upper:.2f}])")
        
        elapsed = time.time() - start_time
        print(f"\n✓ Stream prediction completed (took {elapsed:.3f}s)")
        print(f"  Processed {response_count} predictions")
        
    except grpc.RpcError as e:
        print(f"✗ Stream prediction failed: {e.code()}: {e.details()}")


def test_model_info(stub):
    """Test getting model information."""
    print("\n=== Testing Model Info ===")
    
    request = rt_predictor_pb2.ModelInfoRequest()
    
    try:
        response = stub.GetModelInfo(request)
        
        print("✓ Model info retrieved:")
        print(f"  Model version: {response.model_version}")
        print(f"  Model type: {response.model_type}")
        print(f"  Training date: {response.training_date}")
        print(f"  Feature count: {response.feature_count}")
        print(f"  Metrics:")
        print(f"    MAE: {response.metrics.get('mae', 0):.2f}")
        print(f"    RMSE: {response.metrics.get('rmse', 0):.2f}")
        print(f"    R²: {response.metrics.get('r2', 0):.4f}")
        
    except grpc.RpcError as e:
        print(f"✗ Model info failed: {e.code()}: {e.details()}")


def test_health_check(stub):
    """Test health check."""
    print("\n=== Testing Health Check ===")
    
    request = rt_predictor_pb2.HealthCheckRequest()
    
    try:
        response = stub.HealthCheck(request)
        
        print(f"✓ Health check: {response.status}")
        print(f"  Timestamp: {response.timestamp}")
        if response.message:
            print(f"  Message: {response.message}")
        
    except grpc.RpcError as e:
        print(f"✗ Health check failed: {e.code()}: {e.details()}")


def main():
    """Main test client."""
    parser = argparse.ArgumentParser(description='Test RT Predictor gRPC service')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=50051, help='Server port')
    parser.add_argument('--test', choices=['all', 'single', 'batch', 'stream', 'info', 'health'],
                        default='all', help='Test to run')
    
    args = parser.parse_args()
    
    # Create channel and stub
    channel = grpc.insecure_channel(f'{args.host}:{args.port}')
    stub = rt_predictor_pb2_grpc.RTPredictor


(channel)
    
    print(f"Connecting to RT Predictor service at {args.host}:{args.port}")
    
    try:
        if args.test == 'all' or args.test == 'health':
            test_health_check(stub)
        
        if args.test == 'all' or args.test == 'info':
            test_model_info(stub)
        
        if args.test == 'all' or args.test == 'single':
            test_single_prediction(stub)
        
        if args.test == 'all' or args.test == 'batch':
            test_batch_prediction(stub)
        
        if args.test == 'all' or args.test == 'stream':
            test_stream_prediction(stub)
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        sys.exit(1)
    finally:
        channel.close()


if __name__ == '__main__':
    main()
