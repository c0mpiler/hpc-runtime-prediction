# RT Predictor Training Service

This microservice handles model training for the RT Predictor system.

## Overview

The training service:
- Loads and preprocesses HPC job data
- Engineers features using advanced techniques
- Trains ensemble models (XGBoost, LightGBM, CatBoost)
- Saves model artifacts for the prediction API

## Quick Start

### Using Docker (Recommended)

**Standard Setup:**
```bash
make train
```

**M2 Max Optimized (Apple Silicon):**
```bash
# 2-3x faster training with optimized settings
make train-m2max
```

### Local Development

1. **Setup environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Prepare data:**

   **Option A - Use provided Eagle dataset:**
   ```bash
   # Run data setup script from microservices root
   cd .. && ./copy_data.sh
   ```
   
   **Option B - Generate synthetic data:**
   ```bash
   python scripts/generate_synthetic_data.py --output data/raw/eagle_data.parquet
   ```
   
   See [DATA.md](../DATA.md) for detailed data setup instructions.
3. **Train models:**
```bash
# Quick test with sample
python src/train.py --sample-size 10000

# Full training
python src/train.py --config configs/config.toml
```

### Docker

1. **Build image:**
```bash
docker build -t rt-predictor-training .
```

2. **Run training:**
```bash
docker run -v $(pwd)/data:/app/data rt-predictor-training
```

## Configuration

Edit `configs/config.toml` to customize:
- Model hyperparameters
- Feature engineering settings
- Data paths
- Training parameters

### M2 Max Optimization

For Apple Silicon M2 Max, use `configs/config.m2max.toml` which includes:
- **CPU cores**: 10 (out of 12 available)
- **Memory**: Up to 48GB allocation
- **Chunk size**: 500k records (5x larger)
- **Tree depth**: 12 (increased from 10)
- **LightGBM leaves**: 255 (increased from 127)
- **Max bins**: 511 (better accuracy)

To apply M2 Max optimization:
```bash
# Option 1: Use make command
make train-m2max

# Option 2: Copy config manually
cp configs/config.m2max.toml configs/config.toml
```

## Output

Training produces these artifacts in `data/models/`:
- `xgboost_model.pkl` - XGBoost model
- `lightgbm_model.pkl` - LightGBM model
- `catboost_model.pkl` - CatBoost model
- `feature_engineering.pkl` - Feature transformer
- `ensemble_config.json` - Ensemble weights (uses `models` key)
- `model_info.json` - Model metadata
- `training_metrics.json` - Performance metrics

### Ensemble Config Format

The `ensemble_config.json` uses the following structure:
```json
{
  "weights": {
    "xgboost": 0.33,
    "lightgbm": 0.33,
    "catboost": 0.34
  },
  "models": ["xgboost", "lightgbm", "catboost"],
  "combine_method": "weighted_average"
}
```

## Important Notes

### Feature Engineering
- Uses `OptimizedFeatureEngineer` class (not the legacy `FeatureEngineer`)
- Supports parallel processing and caching
- Handles 44+ engineered features

## Monitoring

Training progress is logged to:
- Console (with colors)
- `logs/` directory (timestamped files)

### Performance Monitoring

**Standard training:**
```bash
docker stats rt-predictor-training
```

**M2 Max optimized:**
- CPU usage: ~1000% (10 cores)
- Memory: Up to 48GB
- Training time: 5-8 minutes (vs 10-15 minutes standard)

## Testing

Run tests:
```bash
pytest tests/ -v --cov=src
```
