# RT Predictor Microservices

A production-ready microservices architecture for HPC job runtime prediction, featuring machine learning models trained on the NREL Eagle dataset.

## Architecture Overview

The system consists of three main microservices:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Training       │────▶│  API Service    │◀────│  UI Service     │
│  Service        │     │  (gRPC)         │     │  (Streamlit)    │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
   ┌──────────┐           ┌──────────┐           ┌──────────┐
   │  Models  │           │ Metrics  │           │  Users   │
   │  Volume  │           │  (Prom)  │           │          │
   └──────────┘           └──────────┘           └──────────┘
```

## Services

### 1. RT Predictor Training Service
- Trains ensemble ML models (XGBoost, LightGBM, CatBoost)
- Processes 11M+ Eagle HPC job records
- Generates optimized feature engineering pipeline
- Outputs model artifacts to shared volume

### 2. RT Predictor API Service
- High-performance gRPC service
- Serves predictions with <10ms latency
- Handles single, batch, and streaming requests
- Exposes Prometheus metrics

### 3. RT Predictor UI Service
- Modern Streamlit web interface
- Single and batch prediction capabilities
- Real-time analytics dashboard
- Overestimation alerts

## Quick Start

### Prerequisites
- Docker and Docker Compose
- 16GB+ RAM recommended
- 10GB+ disk space

### 1. Clone and Setup
```bash
cd /Users/c0mpiler/sandbox/IBM/EDAaaS/edaaas-dev/rt-predictor/microservices
```

### 2. Prepare Training Data
```bash
# Option A: Use provided data (requires Git LFS)
git lfs pull  # Download data files
./copy_data.sh  # Copy to training directory

# Option B: Generate synthetic data
python rt-predictor-training/scripts/generate_synthetic_data.py
```

See [DATA.md](DATA.md) for detailed instructions.

### 3. Train Models (First Time)
```bash
# Run training service
docker-compose --profile training up rt-predictor-training
```

### 3. Start All Services
```bash
# Start API, UI, and monitoring
docker-compose up -d
```

### 4. Access Services
- UI: http://localhost:8501
- API: localhost:50051 (gRPC)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Configuration

Each service has its own configuration in `configs/config.toml`:

```toml
# Example: API Service Configuration
[server]
host = "0.0.0.0"
port = 50051
workers = 4

[model]
path = "/app/models/best_model.pkl"
feature_engineer_path = "/app/models/feature_engineer.pkl"
```

## Development

### Local Development

1. **Training Service**:
```bash
cd rt-predictor-training
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

2. **API Service**:
```bash
cd rt-predictor-api
./scripts/generate_proto.sh
python src/service/server.py
```

3. **UI Service**:
```bash
cd rt-predictor-ui
streamlit run src/app.py
```

### Testing

Each service includes tests:
```bash
# Run all tests
docker-compose run rt-predictor-api pytest
docker-compose run rt-predictor-ui pytest
```

## Monitoring

### Metrics Available
- Prediction latency (p50, p95, p99)
- Request rate and errors
- Model performance metrics
- Resource utilization

### Custom Dashboards
Import provided Grafana dashboards:
1. RT Predictor Overview
2. API Performance
3. Model Metrics

## Production Deployment

### Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/
```

### Scaling
- API: Horizontal scaling with load balancer
- UI: Multiple replicas behind reverse proxy
- Training: Scheduled jobs with resource limits

## Performance

- **Training Time**: ~5 minutes on 11M records
- **Prediction Latency**: <10ms (p95)
- **Throughput**: 10K+ predictions/second
- **Model Accuracy**: MAE ~1.6 hours

## Security

- Input validation on all endpoints
- gRPC with TLS support
- API authentication ready
- Secure configuration management

## Troubleshooting

### Common Issues

1. **Models not found**:
```bash
# Check shared volume
docker volume inspect microservices_shared-models
```

2. **Connection refused**:
```bash
# Check service health
docker-compose ps
docker-compose logs rt-predictor-api
```

3. **High latency**:
- Check resource limits
- Enable caching
- Scale API replicas

## Contributing

1. Follow microservice boundaries
2. Add tests for new features
3. Update documentation
4. Use conventional commits

## License

See LICENSE file in root directory.
