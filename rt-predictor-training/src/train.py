#!/usr/bin/env python3
"""
Main training script for RT Predictor Model Training Service.

This script handles the complete training pipeline from raw data to model artifacts.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib
import toml
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features import FeatureEngineer, DataPreprocessor
from src.model import ModelTrainer
from src.utils.config import Config
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger('training')


def load_config(config_path: str) -> Config:
    """Load configuration from TOML file."""
    logger.info(f"Loading configuration from {config_path}")
    config = Config(config_path)
    return config


def load_and_preprocess_data(config: Config) -> pd.DataFrame:
    """Load and preprocess raw data."""
    logger.info("Loading raw data...")
    
    data_path = Path(config.get('data.raw_data_path'))
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows from {data_path}")
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    df_clean = preprocessor.clean_data(df)
    logger.info(f"After cleaning: {len(df_clean):,} rows")
    
    return df_clean


def split_data(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    logger.info("Splitting data...")
    
    # Get split ratios
    test_size = config.get('training.test_size', 0.2)
    val_size = config.get('training.validation_size', 0.1)
    random_state = config.get('training.random_state', 42)
    
    # Sort by submit_time for temporal split
    df = df.sort_values('submit_time')
    
    # Calculate split indices
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # Split
    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx:test_idx]
    test_df = df.iloc[test_idx:]
    
    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    return train_df, val_df, test_df


def engineer_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                     config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureEngineer]:
    """Apply feature engineering to all datasets."""
    logger.info("Engineering features...")
    
    # Initialize feature engineer
    use_advanced = config.get('features.use_advanced_features', True)
    chunk_size = config.get('features.chunk_size', 100000)
    
    feature_engineer = FeatureEngineer(
        use_advanced_features=use_advanced,
        chunk_size=chunk_size,
        enable_caching=True
    )
    
    # Fit on training data
    logger.info("Fitting feature engineering on training data...")
    X_train = feature_engineer.fit_transform(train_df)
    
    # Transform validation and test
    logger.info("Transforming validation data...")
    X_val = feature_engineer.transform(val_df)
    
    logger.info("Transforming test data...")
    X_test = feature_engineer.transform(test_df)
    
    return X_train, X_val, X_test, feature_engineer


def train_models(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                feature_engineer: FeatureEngineer, config: Config) -> Dict[str, Any]:
    """Train all models and create ensemble."""
    logger.info("Training models...")
    
    # Get target column
    target_col = config.get('training.target_column', 'run_time')
    use_log = config.get('training.use_log_transform', True)
    
    # Prepare target
    if use_log:
        y_train = np.log1p(train_df[target_col])
        y_val = np.log1p(val_df[target_col])
        y_test = np.log1p(test_df[target_col])
    else:
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        y_test = test_df[target_col]
    
    # Get feature columns
    feature_cols = feature_engineer.get_feature_columns()
    available_cols = [col for col in feature_cols if col in X_train.columns]
    
    X_train_features = X_train[available_cols]
    X_val_features = X_val[available_cols]
    X_test_features = X_test[available_cols]
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train models
    models = trainer.train_all_models(
        X_train_features, y_train,
        X_val_features, y_val,
        X_test_features, y_test
    )
    
    return models


def save_artifacts(models: Dict[str, Any], feature_engineer: FeatureEngineer, 
                  config: Config, training_time: float):
    """Save all training artifacts."""
    logger.info("Saving artifacts...")
    
    output_dir = Path(config.get('data.model_output_path'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    for name, model in models['models'].items():
        model_path = output_dir / f"{name}_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved {name} model to {model_path}")
    
    # Save feature engineering
    fe_path = output_dir / "feature_engineering.pkl"
    feature_engineer.save(fe_path)
    logger.info(f"Saved feature engineering to {fe_path}")
    
    # Save ensemble config
    ensemble_config = {
        'weights': models['ensemble_weights'],
        'models': list(models['models'].keys()),
        'combine_method': 'weighted_average'
    }
    
    import json
    with open(output_dir / "ensemble_config.json", 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    # Save model info
    model_info = {
        'model_type': 'ensemble',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_time_seconds': training_time,
        'config': config.to_dict(),
        'ensemble_weights': models['ensemble_weights'],
        'version': '2.0'
    }
    
    with open(output_dir / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save metrics
    with open(output_dir / "training_metrics.json", 'w') as f:
        json.dump(models['metrics'], f, indent=2)
    
    # Save feature importance
    if 'feature_importance' in models:
        with open(output_dir / "feature_importance.json", 'w') as f:
            json.dump(models['feature_importance'], f, indent=2)
    
    logger.info(f"All artifacts saved to {output_dir}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train RT Predictor models')
    parser.add_argument('--config', type=str, default='configs/config.toml',
                       help='Path to configuration file')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Use a sample of the data for quick testing')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation set creation')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load and preprocess data
        df = load_and_preprocess_data(config)
        
        # Sample if requested
        if args.sample_size:
            logger.info(f"Sampling {args.sample_size} rows for testing...")
            df = df.sample(n=args.sample_size, random_state=42)
        
        # Split data
        train_df, val_df, test_df = split_data(df, config)
        
        # Engineer features
        X_train, X_val, X_test, feature_engineer = engineer_features(
            train_df, val_df, test_df, config
        )
        
        # Train models
        models = train_models(
            X_train, X_val, X_test,
            train_df, val_df, test_df,
            feature_engineer, config
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Save artifacts
        save_artifacts(models, feature_engineer, config, training_time)
        
        logger.info(f"Training completed successfully in {training_time:.2f} seconds!")
        
        # Print final metrics
        logger.info("\nFinal Test Metrics:")
        for model_name, metrics in models['metrics'].items():
            if 'test_mae' in metrics:
                logger.info(f"{model_name}: MAE = {metrics['test_mae']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
