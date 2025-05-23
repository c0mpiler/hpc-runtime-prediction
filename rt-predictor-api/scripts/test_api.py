#!/usr/bin/env python3
"""
Test script for RT Predictor API Service
"""

import grpc
import time
import sys
from pathlib import Path

# Add proto to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from proto import rt_predictor_pb2
from proto import rt_predictor_pb2_grpc


def test_api_connection(host='localhost', port=50051):
    """Test connection to the API service."""
    print(f"Testing connection to {host}:{port}...")
    
    try:
        # Create channel
        channel = grpc.insecure_channel(f'{host}:{port}')
        
        # Create stub
        stub = rt_predictor_pb2_grpc.JobRuntimePredictorStub(channel)
        
        # Test health check
        request = rt_predictor_pb2.HealthCheckRequest()
        response = stub.HealthCheck(request, timeout=5)
        
        if response.status == rt_predictor_pb2.HealthCheckResponse.SERVING:
            print("✅ API service is healthy!")
            return True
        else:
            print("❌ API service is not serving")
            return False
            
    except grpc.RpcError as e:
        print(f"❌ gRPC error: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    finally:
        if 'channel' in locals():
            channel.close()


def test_prediction(host='localhost', port=50051):
    """Test making a prediction."""
    print("\nTesting prediction...")
    
    try:
        # Create channel and stub
        channel = grpc.insecure_channel(f'{host}:{port}')
        stub = rt_predictor_pb2_grpc.JobRuntimePredictorStub(channel)
        
        # Create test job features
        features = rt_predictor_pb2.JobFeatures(
            processors_req=16,
            nodes_req=1,
            mem_req=32.0,  # GB
            wallclock_req=3600.0,  # seconds
            partition="compute",
            qos="normal",
            submit_time=int(time.time())
        )
        
        # Create request
        request = rt_predictor_pb2.PredictRequest(
            features=features,
            request_id="test-123"
        )
        
        # Make prediction
        response = stub.Predict(request, timeout=10)
        
        print(f"✅ Prediction successful!")
        print(f"   Predicted runtime: {response.predicted_runtime_seconds:.0f} seconds")
        print(f"   Confidence: [{response.confidence_lower:.0f}, {response.confidence_upper:.0f}]")
        print(f"   Category: {response.runtime_category}")
        print(f"   Model version: {response.model_version}")
        
        return True
        
    except grpc.RpcError as e:
        print(f"❌ gRPC error: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    finally:
        if 'channel' in locals():
            channel.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RT Predictor API")
    parser.add_argument('--host', default='localhost', help='API host')
    parser.add_argument('--port', type=int, default=50051, help='API port')
    
    args = parser.parse_args()
    
    # Run tests
    connection_ok = test_api_connection(args.host, args.port)
    
    if connection_ok:
        test_prediction(args.host, args.port)
    else:
        print("\n⚠️  Cannot test prediction - API not connected")
        sys.exit(1)
