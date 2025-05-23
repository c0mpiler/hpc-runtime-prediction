# RT Predictor UI Service

A modern Streamlit-based web interface for the RT Predictor system, providing intuitive access to HPC job runtime predictions.

## Features

- **Single Job Prediction**: Quick runtime predictions for individual jobs
- **Batch Prediction**: Upload CSV files for bulk predictions
- **Analytics Dashboard**: Visualize prediction accuracy and system performance
- **Real-time Monitoring**: Track prediction metrics and model performance
- **Overestimation Alerts**: Highlight when requested walltime significantly exceeds predicted runtime

## Architecture

The UI service connects to the RT Predictor API service via gRPC, providing a clean separation of concerns:

```
User → Streamlit UI → gRPC Client → RT Predictor API → ML Models
```

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- RT Predictor API service running

## Quick Start

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Generate proto files**:
```bash
./scripts/generate_proto.sh
```

3. **Run the UI**:
```bash
# Set API connection (optional, defaults to localhost:50051)
export RT_PREDICTOR_API_HOST=localhost
export RT_PREDICTOR_API_PORT=50051

# Start Streamlit
streamlit run src/app.py
```

### Docker Deployment

1. **Build the image**:
```bash
docker build -t rt-predictor-ui .
```

2. **Run with docker-compose**:
```bash
docker-compose up -d
```

## Configuration

### Environment Variables

- `RT_PREDICTOR_API_HOST`: API service hostname (default: `localhost`)
- `RT_PREDICTOR_API_PORT`: API service port (default: `50051`)
- `STREAMLIT_SERVER_PORT`: UI port (default: `8501`)

### Streamlit Configuration

Edit `.streamlit/config.toml` for Streamlit-specific settings:

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true

[browser]
gatherUsageStats = false
```

## UI Pages

### 1. Single Prediction
- Input job parameters manually
- Get instant runtime predictions
- View confidence intervals
- See overestimation alerts

### 2. Batch Prediction
- Upload CSV files with job parameters
- Download results with predictions
- Visualize prediction distribution
- Export enhanced datasets

### 3. Analytics
- Model performance metrics
- Prediction accuracy trends
- Feature importance visualization
- System health monitoring

## Development

### Project Structure

```
rt-predictor-ui/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── pages/              # UI pages
│   │   ├── single_prediction.py
│   │   ├── batch_prediction.py
│   │   └── analytics.py
│   ├── utils/              # Utilities
│   │   └── grpc_client.py  # gRPC client wrapper
│   └── proto/              # Generated protobuf files
├── configs/                # Configuration files
├── scripts/                # Utility scripts
├── tests/                  # Test files
├── requirements.txt        # Dependencies
├── Dockerfile              # Container definition
└── docker-compose.yml      # Service orchestration
```

### Adding New Features

1. Create new page in `src/pages/`
2. Import in `app.py`
3. Add to navigation menu
4. Update gRPC client if needed

### Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/ --api-host=localhost
```

## Monitoring

The UI service exposes metrics on port 9090 (when configured):

- Request count and latency
- Active sessions
- Error rates
- Resource usage

## Troubleshooting

### Connection Issues

1. **Check API service**:
```bash
# Verify API is running
docker ps | grep rt-predictor-api

# Test connection
grpcurl -plaintext localhost:50051 list
```

2. **Check logs**:
```bash
docker logs rt-predictor-ui
```

### Performance Issues

1. **Enable caching**:
```python
@st.cache_data
def expensive_computation():
    ...
```

2. **Optimize queries**:
- Use batch predictions for multiple jobs
- Enable connection pooling

## Security

- Input validation on all forms
- Sanitized error messages
- No sensitive data in logs
- HTTPS support via reverse proxy

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## License

See LICENSE file in root directory.
