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
