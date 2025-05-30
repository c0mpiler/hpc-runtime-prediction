#!/usr/bin/env python
"""Predictor Service for RT Predictor API."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json
import time
from typing import Dict, Any, List, Optional
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import feature engineering
from src.features.feature_engineering_optimized import OptimizedFeatureEngineer
from src.utils.logger import setup_logger

logger = setup_logger('predictor_service')


class PredictorService:
    """Service for making runtime predictions."""
    
    def __init__(self, config: dict):
        """Initialize predictor service."""
        self.config = config
        self.model = None
        self.feature_engineer = None
        self.model_info = {}
        self.feature_columns = None
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load trained model and artifacts."""
        # Get model path from config - handle both flat and nested structures
        model_config = self.config.get('model', {})
        model_path = Path(model_config.get('path', self.config.get('model_path', 'models/production')))
        
        if not model_path.exists():
            raise ValueError(f"Model path {model_path} does not exist")
        
        logger.info(f"Loading model from {model_path}")
        
        # Load model info
        info_path = model_path / 'model_info.json'
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.model_info = json.load(f)
        
        # Load metrics
        metrics_path = model_path / 'training_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
                self.model_info['metrics'] = metrics_data.get('metrics', {}).get(
                    metrics_data.get('best_model', 'ensemble'), {}
                )
        
        # Load feature importance
        importance_path = model_path / 'feature_importance.json'
        if importance_path.exists():
            with open(importance_path, 'r') as f:
                self.model_info['feature_importance'] = json.load(f)
        
        # Determine model type and load accordingly
        model_type = self.model_info.get('model_type', 'ensemble')
        
        if model_type == 'ensemble':
            # Load ensemble configuration
            ensemble_config_path = model_path / 'ensemble_config.json'
            with open(ensemble_config_path, 'r') as f:
                ensemble_config = json.load(f)
            
            # Load component models
            models = {}
            for model_name in ensemble_config['models']:
                model_file = model_path / f"{model_name}_model.pkl"
                if model_file.exists():
                    models[model_name] = joblib.load(model_file)
                    logger.info(f"Loaded {model_name} model")
            
            # Create ensemble predictor
            from src.model.trainer import EnsemblePredictor
            self.model = EnsemblePredictor(models, ensemble_config['weights'])
            
        else:
            # Load single model
            model_file = model_path / f"{model_type}_model.pkl"
            self.model = joblib.load(model_file)
            logger.info(f"Loaded {model_type} model")
        
        # Initialize feature engineer
        fe_path = model_path / 'feature_engineering.pkl'
        logger.info(f"Looking for feature engineer at: {fe_path}")
        try:
            # Load the saved artifacts and reconstruct the feature engineer
            artifacts = joblib.load(fe_path)
            
            # Check if it's already a feature engineer object (backward compatibility)
            if hasattr(artifacts, 'transform'):
                self.feature_engineer = artifacts
            else:
                # It's a dictionary of artifacts, reconstruct the feature engineer
                self.feature_engineer = OptimizedFeatureEngineer(
                    use_advanced_features=artifacts.get('use_advanced_features', True),
                    chunk_size=artifacts.get('chunk_size', 100000),
                    n_jobs=artifacts.get('n_jobs', -1),
                    enable_caching=artifacts.get('enable_caching', True)
                )
                
                # Load the saved state
                self.feature_engineer.scalers = artifacts['scalers']
                self.feature_engineer.encoders = artifacts['encoders']
                self.feature_engineer.feature_stats = artifacts['feature_stats']
                self.feature_engineer.user_stats = artifacts.get('user_stats', {})
                self.feature_engineer.queue_stats = artifacts.get('queue_stats', {})
                self.feature_engineer.is_fitted = artifacts['is_fitted']
                
                # Re-compile regex patterns (if method exists)
                if hasattr(self.feature_engineer, '_compile_regex_patterns'):
                    self.feature_engineer._compile_regex_patterns()
                
            logger.info("Successfully loaded saved feature engineer")
        except Exception as e:
            logger.error(f"Failed to load feature engineer from {fe_path}: {str(e)}")
            raise ValueError(f"Could not load feature engineer. Training artifacts may be incomplete.")
        
        # Get feature columns from model
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_columns = list(self.model.feature_names_in_)
        elif hasattr(self.model, 'models'):
            # For ensemble, get from first model
            first_model = next(iter(self.model.models.values()))
            if hasattr(first_model, 'feature_names_in_'):
                self.feature_columns = list(first_model.feature_names_in_)
        
        logger.info(f"Model loaded successfully. Type: {model_type}, Features: {len(self.feature_columns) if self.feature_columns else 'unknown'}")
    
    def predict_single(self, features: Any) -> Dict[str, Any]:
        """Make a single prediction."""
        try:
            data = {
                'processors_req': features.processors_req,
                'nodes_req': features.nodes_req,
                'mem_req': features.mem_req,
                'wallclock_req': features.wallclock_req,  # Match feature engineer
                'partition': features.partition,
                'qos': features.qos,
                'gpus_req': 0,  # Not in proto, default to 0
                # Add default values for missing columns
                'user': 'unknown',
                'account': 'default',
                'name': 'job',
                'job_id': 0,  # Dummy value
                'state': 'PENDING',
                'submit_time': pd.Timestamp.now(),
                'start_time': pd.Timestamp.now(),
                'run_time': 0  # Will be predicted
            }
            
            df = pd.DataFrame([data])
            
            # Engineer features
            features_df = self.feature_engineer.transform(df)
            
            # Select only the features used in training
            if self.feature_columns:
                available_features = [col for col in self.feature_columns if col in features_df.columns]
                if len(available_features) < len(self.feature_columns):
                    missing = set(self.feature_columns) - set(available_features)
                    logger.warning(f"Missing features: {missing}")
                    # Add missing features with default values
                    for col in missing:
                        features_df[col] = 0
                
                X = features_df[self.feature_columns]
            else:
                X = features_df
            
            # Make prediction
            y_pred_log = self.model.predict(X)
            y_pred = np.expm1(y_pred_log[0])  # Convert from log space
            
            # Calculate confidence intervals (simple approach)
            # In production, you might want to use quantile regression or other methods
            confidence_factor = 0.2  # 20% confidence interval
            confidence_lower = y_pred * (1 - confidence_factor)
            confidence_upper = y_pred * (1 + confidence_factor)
            
            return {
                'predicted_runtime': float(y_pred),
                'confidence_lower': float(confidence_lower),
                'confidence_upper': float(confidence_upper),
                'model_version': self.model_info.get('version', '1.0'),
                'features_used': len(X.columns)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise
    
    def predict_batch(self, requests: List[Any]) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        try:
            # Convert requests to dataframe
            data_list = []
            for i, job_features in enumerate(requests):
                data = {
                    'processors_req': job_features.processors_req,
                    'nodes_req': job_features.nodes_req,
                    'mem_req': job_features.mem_req,
                    'wallclock_req': job_features.wallclock_req,  # Match feature engineer
                    'partition': job_features.partition,
                    'qos': job_features.qos,
                    'gpus_req': 0,  # Not in proto, default to 0
                    'user': 'unknown',
                    'account': 'default',
                    'name': 'job',
                    'job_id': i,  # Use index as dummy job_id
                    'state': 'PENDING',
                    'submit_time': pd.Timestamp.now(),
                    'start_time': pd.Timestamp.now(),
                    'run_time': 0
                }
                data_list.append(data)
            
            df = pd.DataFrame(data_list)
            
            # Engineer features
            features_df = self.feature_engineer.transform(df)
            
            # Select features
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in features_df.columns:
                        features_df[col] = 0
                X = features_df[self.feature_columns]
            else:
                X = features_df
            
            # Make predictions
            y_pred_log = self.model.predict(X)
            y_pred = np.expm1(y_pred_log)
            
            # Create responses
            responses = []
            for i, pred in enumerate(y_pred):
                confidence_lower = pred * 0.8
                confidence_upper = pred * 1.2
                
                responses.append({
                    'predicted_runtime': float(pred),
                    'confidence_lower': float(confidence_lower),
                    'confidence_upper': float(confidence_upper),
                    'model_version': self.model_info.get('version', '1.0'),
                    'features_used': len(X.columns)
                })
            
            return responses
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'version': self.model_info.get('version', '1.0'),
            'type': self.model_info.get('model_type', 'unknown'),
            'training_date': self.model_info.get('training_date', 'unknown'),
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'metrics': self.model_info.get('metrics', {})
        }
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self.model is not None and self.feature_engineer is not None
