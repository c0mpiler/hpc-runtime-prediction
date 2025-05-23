# RT Predictor Training Service

This microservice handles model training for the RT Predictor system.

## Overview

The training service:
- Loads and preprocesses HPC job data
- Engineers features using advanced techniques
- Trains ensemble models (XGBoost, LightGBM, CatBoost)
- Saves model artifacts for the prediction API

## Quick Start

### Local Development

1. **Setup environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Prepare data:**

   **Option A - Use existing Eagle dataset:**
   ```bash
   # Copy from monolithic app if available
   cp ../../../ml/eagle-jobs/data/full-data/*.parquet data/raw/
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

## Output

Training produces these artifacts in `data/models/`:
- `xgboost_model.pkl` - XGBoost model
- `lightgbm_model.pkl` - LightGBM model
- `catboost_model.pkl` - CatBoost model
- `feature_engineering.pkl` - Feature transformer
- `ensemble_config.json` - Ensemble weights
- `model_info.json` - Model metadata
- `training_metrics.json` - Performance metrics

## Monitoring

Training progress is logged to:
- Console (with colors)
- `logs/` directory (timestamped files)

## Testing

Run tests:
```bash
pytest tests/ -v --cov=src
```
