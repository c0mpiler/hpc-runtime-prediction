# RT Predictor API Service

High-performance gRPC API service for HPC job runtime predictions using ensemble ML models.

## Overview

This service provides real-time runtime predictions for HPC jobs via a gRPC interface. It supports:
- Single predictions with <10ms latency
- Batch predictions for multiple jobs
- Streaming predictions for continuous workloads
- Confidence intervals for all predictions
- Prometheus metrics for monitoring
- Health checks and model info endpoints

## Features

- **High Performance**: Optimized for M2 Max and multi-core systems
- **Ensemble Models**: Combines XGBoost, LightGBM, and CatBoost
- **Feature Engineering**: 44+ engineered features for accurate predictions
- **Production Ready**: Health checks, metrics, logging, error handling
- **Scalable**: Supports batch and streaming predictions
- **Observable**: Prometheus metrics and structured logging

## Architecture

```
├── src/
│   ├── proto/          # Protocol buffer definitions
│   ├── service/        # gRPC server and predictor logic
│   ├── features/       # Feature engineering (shared with training)
│   ├── model/          # Model loading utilities
│   └── utils/          # Configuration and logging
├── scripts/            # Helper scripts
├── configs/            # Configuration files
├── models/             # Trained model artifacts
└── tests/              # Test suite
```

## Quick Start

### Prerequisites

- Python 3.11+
- Trained model artifacts in `models/production/`
- Docker (optional)

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate proto files:
```bash
./scripts/generate_proto.sh
```

3. Place trained model in `models/production/`

4. Start the server:
```bash
python src/service/server.py
```

5. Test the service:
```bash
python scripts/test_client.py
```

### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up -d
```

2. With monitoring stack:
```bash
docker-compose --profile monitoring up -d
```

3. M2 Max optimized deployment:
```bash
# Uses docker-compose.m2max.yml with resource limits:
# - 4 CPU cores
# - 8GB RAM
make start-m2max
```

## API Reference

### Single Prediction
```python
request = PredictRequest(
    processors_req=32,
    nodes_req=4,
    mem_req=128000,
    time_req=3600,
    partition="normal",
    qos="normal"
)
response = stub.Predict(request)
# Returns: predicted_runtime, confidence_lower, confidence_upper
```

### Batch Prediction
```python
batch_request = BatchPredictRequest(requests=[req1, req2, req3])
batch_response = stub.BatchPredict(batch_request)
# Returns: list of predictions
```

### Streaming Prediction
```python
def request_generator():
    for job in job_queue:
        yield create_predict_request(job)

for response in stub.StreamPredict(request_generator()):
    process_prediction(response)
```

## Configuration

Edit `configs/config.toml`:

```toml
[server]
port = 50051
max_workers = 10
metrics_port = 8181

[model]
path = "models/production"

[features.optimization]
chunk_size = 100000
enable_caching = true
n_jobs = -1
```

## Monitoring

### Prometheus Metrics

Available at `http://localhost:8181/metrics`:
- `rt_predictor_requests_total`: Total requests by method
- `rt_predictor_request_duration_seconds`: Request latency
- `rt_predictor_prediction_duration_seconds`: Model inference time
- `rt_predictor_active_connections`: Current active connections

### Grafana Dashboard

Access at `http://localhost:3000` (admin/admin) when using monitoring profile.

## Performance

- Single prediction latency: <10ms (P99)
- Batch prediction throughput: 10,000+ predictions/second
- Streaming rate: 1,000+ predictions/second
- Model accuracy: MAE ~5,878 seconds (~1.6 hours)

## Troubleshooting

### Service won't start
- Check model files exist in `models/production/`
- Verify proto files are generated
- Check port 50051 is available
- Ensure ensemble_config.json has `models` key (not `model_names`)

### Predictions failing
- Ensure feature engineering matches training
- Check model compatibility
- Review logs for missing features

### High latency
- Enable model caching
- Increase worker threads
- Use batch predictions for multiple jobs

### Common Errors and Fixes

1. **KeyError: 'model_names'**
   - The ensemble config expects `models` key, not `model_names`
   - Ensure your trained models use the correct config format

2. **Proto compilation errors**
   - Message names have been standardized:
     - Use `PredictBatchRequest` (not `BatchPredictRequest`)
     - Use `request.jobs` (not `request.requests`)
     - Use `PredictStream` (not `StreamPredict`)
     - Use `GetModelInfoRequest` (not `ModelInfoRequest`)

## Development

### Running Tests
```bash
pytest tests/
```

### Updating Proto Files
```bash
# Edit src/proto/rt_predictor.proto
./scripts/generate_proto.sh
```

### Adding New Features
1. Update feature engineering in training service
2. Retrain model with new features
3. Deploy new model to API service
4. No code changes needed in API!

## License

See LICENSE file in root directory.
