"""Model training utilities for runtime prediction - Enhanced Version."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.ensemble import VotingRegressor
import joblib
from pathlib import Path
import logging
import time
import json
import warnings

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate ML models for runtime prediction - with ensemble support."""
    
    def __init__(self, config: dict):
        """Initialize model trainer with configuration."""
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        self.feature_importance = {}
        self.ensemble_model = None
        self.use_ensemble = config.get('use_ensemble', True)
        self.ensemble_weights = config.get('ensemble_weights', None)
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBRegressor:
        """Train XGBoost model optimized for M2 Max."""
        logger.info("Training XGBoost model...")
        start_time = time.time()
        
        # Get XGBoost parameters from config
        params = self.config.get('xgb_params', {}).copy()  # Make a copy
        
        # Extract early_stopping_rounds for fit method
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)
        
        # Optimize for M2 Max
        params.update({
            'tree_method': 'hist',  # Histogram-based algorithm
            'n_jobs': -1,  # Use all CPU cores
            'random_state': 42
        })
        
        # Remove predictor parameter (deprecated)
        params.pop('predictor', None)
        
        # Create and train model
        # In newer XGBoost versions, early_stopping_rounds goes in the constructor
        model = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=early_stopping_rounds,
            enable_categorical=True
        )
        
        # Train with early stopping
        eval_set = [(X_val, y_val)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100 if self.config.get('verbose', False) else 0
        )
        
        # Get feature importance
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)        
        metrics = self._calculate_metrics(y_train, train_pred, y_val, val_pred)
        self.metrics['xgboost'] = metrics
        
        logger.info(f"XGBoost trained in {time.time() - start_time:.2f}s - Val MAE: {metrics['val_mae']:.2f}, Val R2: {metrics['val_r2']:.4f}")
        
        return model
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMRegressor:
        """Train LightGBM model optimized for M2 Max."""
        logger.info("Training LightGBM model...")
        start_time = time.time()
        
        # Get LightGBM parameters from config
        params = self.config.get('lgb_params', {}).copy()  # Make a copy
        
        # Extract early_stopping_rounds for fit method
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)
        
        # Optimize for M2 Max
        params.update({
            'device': 'cpu',  # M2 GPU support is limited
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            'force_row_wise': True  # Avoid warnings
        })
        
        # Create and train model
        model = lgb.LGBMRegressor(**params)
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(100 if self.config.get('verbose', False) else 0)
            ]
        )
        
        # Get feature importance
        self.feature_importance['lightgbm'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        metrics = self._calculate_metrics(y_train, train_pred, y_val, val_pred)
        self.metrics['lightgbm'] = metrics
        
        logger.info(f"LightGBM trained in {time.time() - start_time:.2f}s - Val MAE: {metrics['val_mae']:.2f}, Val R2: {metrics['val_r2']:.4f}")
        
        return model
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       categorical_features: Optional[List[str]] = None) -> 'cb.CatBoostRegressor':
        """Train CatBoost model - handles categoricals well."""
        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not installed. Skipping...")
            return None
            
        logger.info("Training CatBoost model...")
        start_time = time.time()
        
        # Get CatBoost parameters from config
        params = self.config.get('cb_params', {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'task_type': 'CPU',
            'thread_count': -1,
            'early_stopping_rounds': 50,
            'use_best_model': True
        })
        
        # Create model
        model = cb.CatBoostRegressor(**params)
        
        # Prepare categorical features
        if categorical_features is None:
            # Identify categorical columns
            categorical_features = []
            for col in X_train.columns:
                if 'encoded' in col or col in ['partition', 'qos', 'user']:
                    categorical_features.append(col)        
        # Train model
        model.fit(
            X_train, y_train,
            cat_features=categorical_features if categorical_features else None,
            eval_set=(X_val, y_val),
            verbose=100 if self.config.get('verbose', False) else 0
        )
        
        # Get feature importance
        self.feature_importance['catboost'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        metrics = self._calculate_metrics(y_train, train_pred, y_val, val_pred)
        self.metrics['catboost'] = metrics
        
        logger.info(f"CatBoost trained in {time.time() - start_time:.2f}s - Val MAE: {metrics['val_mae']:.2f}, Val R2: {metrics['val_r2']:.4f}")
        
        return model
    
    def create_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Create an ensemble of models."""
        logger.info("Creating ensemble model...")
        
        # Determine weights based on validation performance
        if self.ensemble_weights is None:
            # Calculate weights based on validation MAE (inverse)
            mae_scores = {name: metrics['val_mae'] for name, metrics in self.metrics.items()}
            total_inverse_mae = sum(1 / mae for mae in mae_scores.values())
            
            self.ensemble_weights = {
                name: (1 / mae) / total_inverse_mae 
                for name, mae in mae_scores.items()
            }
            
        logger.info(f"Ensemble weights: {self.ensemble_weights}")
        
        # Create weighted ensemble predictions
        val_predictions = []
        train_predictions = []        
        for name, model in self.models.items():
            if model is not None:
                weight = self.ensemble_weights.get(name, 0)
                val_predictions.append(model.predict(X_val) * weight)
                train_predictions.append(model.predict(X_train) * weight)
        
        # Sum weighted predictions
        ensemble_val_pred = np.sum(val_predictions, axis=0)
        ensemble_train_pred = np.sum(train_predictions, axis=0)
        
        # Calculate ensemble metrics
        ensemble_metrics = self._calculate_metrics(
            y_train, ensemble_train_pred, 
            y_val, ensemble_val_pred
        )
        
        self.metrics['ensemble'] = ensemble_metrics
        logger.info(f"Ensemble - Val MAE: {ensemble_metrics['val_mae']:.2f}, Val R2: {ensemble_metrics['val_r2']:.4f}")
        
        # Create ensemble model wrapper
        self.ensemble_model = EnsemblePredictor(
            models=self.models,
            weights=self.ensemble_weights
        )
        
        return ensemble_metrics
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train all configured models and optionally create ensemble."""
        logger.info("Training all models...")
        start_time = time.time()
        
        # Get list of models to train from config
        models_to_train = self.config.get('models', ['xgboost', 'lightgbm', 'catboost'])
        
        # Train XGBoost
        if 'xgboost' in models_to_train:
            self.models['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Train LightGBM
        if 'lightgbm' in models_to_train:
            self.models['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Train CatBoost
        if 'catboost' in models_to_train and CATBOOST_AVAILABLE:
            self.models['catboost'] = self.train_catboost(X_train, y_train, X_val, y_val)        
        # Create ensemble if enabled
        if self.use_ensemble and len(self.models) > 1:
            self.create_ensemble(X_train, y_train, X_val, y_val)
            self.best_model_name = 'ensemble'
            self.best_model = self.ensemble_model
        else:
            # Select best individual model based on validation MAE
            best_mae = float('inf')
            for name, metrics in self.metrics.items():
                if name != 'ensemble' and metrics['val_mae'] < best_mae:
                    best_mae = metrics['val_mae']
                    self.best_model_name = name
                    self.best_model = self.models[name]
        
        logger.info(f"Best model: {self.best_model_name}")
        logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")
        
        # Log feature importance for best model
        if self.best_model_name in self.feature_importance:
            logger.info("\nTop 10 Most Important Features:")
            top_features = self.feature_importance[self.best_model_name].head(10)
            for _, row in top_features.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.models
    
    def _calculate_metrics(self, y_train: pd.Series, y_train_pred: np.ndarray,
                          y_val: pd.Series, y_val_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics with additional metrics."""
        # Convert from log space back to original scale
        y_train_orig = np.expm1(y_train)
        y_train_pred_orig = np.expm1(y_train_pred)
        y_val_orig = np.expm1(y_val)
        y_val_pred_orig = np.expm1(y_val_pred)
        
        # Calculate MAPE with protection against division by zero
        train_mape = np.mean(np.abs((y_train_orig - y_train_pred_orig) / 
                                   np.maximum(y_train_orig, 1))) * 100
        val_mape = np.mean(np.abs((y_val_orig - y_val_pred_orig) / 
                                 np.maximum(y_val_orig, 1))) * 100        
        return {
            'train_mae': mean_absolute_error(y_train_orig, y_train_pred_orig),
            'train_rmse': np.sqrt(mean_squared_error(y_train_orig, y_train_pred_orig)),
            'train_r2': r2_score(y_train_orig, y_train_pred_orig),
            'train_mape': train_mape,
            'val_mae': mean_absolute_error(y_val_orig, y_val_pred_orig),
            'val_rmse': np.sqrt(mean_squared_error(y_val_orig, y_val_pred_orig)),
            'val_r2': r2_score(y_val_orig, y_val_pred_orig),
            'val_mape': val_mape
        }
    
    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the best model on test set."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        y_pred = self.best_model.predict(X_test)
        
        # Convert from log space
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
        
        # Calculate comprehensive metrics
        test_metrics = {
            'test_mae': mean_absolute_error(y_test_orig, y_pred_orig),
            'test_rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'test_r2': r2_score(y_test_orig, y_pred_orig),
            'test_mape': np.mean(np.abs((y_test_orig - y_pred_orig) / 
                                      np.maximum(y_test_orig, 1))) * 100
        }
        
        # Add percentile errors
        errors = np.abs(y_test_orig - y_pred_orig)
        test_metrics['test_p50_error'] = np.percentile(errors, 50)
        test_metrics['test_p90_error'] = np.percentile(errors, 90)
        test_metrics['test_p95_error'] = np.percentile(errors, 95)
        
        logger.info(f"Test set performance:")
        logger.info(f"  MAE: {test_metrics['test_mae']:.2f} seconds")
        logger.info(f"  MAPE: {test_metrics['test_mape']:.2f}%")
        logger.info(f"  R2: {test_metrics['test_r2']:.4f}")
        logger.info(f"  P50 Error: {test_metrics['test_p50_error']:.2f} seconds")
        logger.info(f"  P90 Error: {test_metrics['test_p90_error']:.2f} seconds")
        
        return test_metrics    
    def save_model(self, model_dir: Path):
        """Save the best model and training artifacts."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model(s)
        if self.best_model_name == 'ensemble':
            # Save all component models
            for name, model in self.models.items():
                if model is not None:
                    model_path = model_dir / f"{name}_model.pkl"
                    joblib.dump(model, model_path)
            
            # Save ensemble configuration
            ensemble_config = {
                'weights': self.ensemble_weights,
                'model_names': list(self.models.keys())
            }
            with open(model_dir / 'ensemble_config.json', 'w') as f:
                json.dump(ensemble_config, f, indent=2)
        else:
            # Save single model
            model_path = model_dir / f"{self.best_model_name}_model.pkl"
            joblib.dump(self.best_model, model_path)
        
        logger.info(f"Model saved to {model_dir}")
        
        # Save metrics
        metrics_path = model_dir / "training_metrics.json"
        # Convert numpy types to Python types for JSON serialization
        metrics_serializable = {}
        for model_name, model_metrics in self.metrics.items():
            metrics_serializable[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in model_metrics.items()
            }
        
        with open(metrics_path, 'w') as f:
            json.dump({
                'best_model': self.best_model_name,
                'metrics': metrics_serializable
            }, f, indent=2)        
        # Save model info
        info_path = model_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump({
                'model_type': self.best_model_name,
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': self.config,
                'ensemble_weights': self.ensemble_weights if self.best_model_name == 'ensemble' else None,
                'version': '2.0'  # Version tracking
            }, f, indent=2)
        
        # Save feature importance
        if self.feature_importance:
            importance_path = model_dir / "feature_importance.json"
            importance_data = {}
            for model_name, importance_df in self.feature_importance.items():
                importance_data[model_name] = importance_df.to_dict('records')
            
            with open(importance_path, 'w') as f:
                json.dump(importance_data, f, indent=2)
        
        logger.info(f"Model artifacts saved to {model_dir}")


class EnsemblePredictor:
    """Wrapper for ensemble predictions."""
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]):
        """Initialize ensemble predictor."""
        self.models = models
        self.weights = weights
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        
        for name, model in self.models.items():
            if model is not None and name in self.weights:
                weight = self.weights[name]
                pred = model.predict(X)
                predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def __getstate__(self):
        """For pickling."""
        return {'models': self.models, 'weights': self.weights}
    
    def __setstate__(self, state):
        """For unpickling."""
        self.models = state['models']
        self.weights = state['weights']