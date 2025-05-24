# Changelog

All notable changes to the RT Predictor Microservices project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2025-05-24

### Fixed

#### UI Service Navigation
- Fixed Streamlit auto-detection of `pages/` directory causing empty pages
- Resolved navigation issue where batch prediction and analytics pages were not displaying
- All three UI pages (Single Prediction, Batch Prediction, Analytics) now working correctly
- Verified successful end-to-end predictions with 3m 10s runtime (80% confidence)

### Changed

#### Documentation
- Updated README.md with navigation fix information
- Added troubleshooting section for Streamlit multi-page issues
- Updated UI README.md with known issues and fixes

## [1.2.0] - 2025-05-24

### Added

#### M2 Max Optimization (Apple Silicon)
- Created `docker-compose.m2max.yml` with optimized resource limits
- Added `config.m2max.toml` with Apple Silicon optimized settings:
  - 10 CPU cores allocation (83% utilization)
  - 48GB RAM limit (75% of 64GB total)
  - 5x larger chunk sizes (500k records)
  - Deeper trees and more iterations for all models
- New make commands: `train-m2max`, `start-m2max`, `fresh-start-m2max`
- Added `optimize-m2max.sh` quick setup script
- Created comprehensive `M2MAX_OPTIMIZATION.md` documentation
- 2-3x faster training times (5-8 minutes vs 15-20 minutes)

### Updated

#### Documentation
- Main README.md: Added M2 Max quick start and performance sections
- Training README.md: Added M2 Max configuration details
- API README.md: Added resource allocation notes
- Makefile help: Added M2 Max commands section

## [1.1.0] - 2025-05-23

### Fixed

#### Training Service
- Fixed import error: Changed `FeatureEngineer` to `OptimizedFeatureEngineer` in trainer.py
- Fixed missing imports in `utils/__init__.py` 
- Fixed `train_all_models` method to accept correct parameters
- Fixed model saving to use shared volume correctly

#### API Service  
- Fixed protobuf message name mismatches:
  - `BatchPredictRequest` → `PredictBatchRequest`
  - `request.requests` → `request.jobs` in batch predictions
  - `StreamPredict` → `PredictStream` for streaming RPC
  - `ModelInfoRequest` → `GetModelInfoRequest`
- Fixed `KeyError: 'model_names'` - changed to use `models` key in ensemble_config.json
- Fixed model path configuration to correctly use `/app/models` from shared volume

#### UI Service
- Fixed missing `except` block syntax error in `grpc_client.py`
- Fixed incomplete `single_prediction.py` file - added missing exception handling and prediction display logic
- Fixed incomplete `app.py` file - added missing main content area and page routing logic
- Added `health_check` method to gRPC client for better error handling
- Removed health check requirement from initial connection to improve startup reliability
- Fixed blank UI issue by completing the Streamlit app structure

### Added

#### API Service
- Enhanced server implementation with:
  - Response caching for repeated predictions
  - Circuit breaker pattern for fault tolerance
  - Retry logic with exponential backoff
  - Connection pooling for better performance
  - Comprehensive Prometheus metrics
  - Health check endpoint
  - Request/response interceptors for monitoring

#### Documentation
- Updated all README files with troubleshooting sections
- Added migration notes from monolithic version
- Documented ensemble config format
- Added common error fixes and solutions

#### Monitoring
- Created comprehensive Grafana dashboard configuration
- Added Prometheus metrics collection across all services

### Changed

#### Training Service
- Ensemble config now uses `models` key instead of `model_names`
- Standardized on `OptimizedFeatureEngineer` throughout

#### API Service
- Improved error handling and logging
- Better model loading with fallback options
- Standardized protobuf message naming conventions

## [1.0.0] - 2025-05-20

### Added
- Initial microservices architecture implementation
- Training service for ML model development
- gRPC API service for predictions
- Streamlit UI for user interaction
- Docker Compose orchestration
- Prometheus and Grafana monitoring stack
- Comprehensive documentation

### Features
- Support for XGBoost, LightGBM, and CatBoost ensemble models
- Single, batch, and streaming prediction capabilities
- 44+ engineered features for accurate predictions
- Real-time monitoring and alerting
- Optimized for M2 Max and multi-core systems
