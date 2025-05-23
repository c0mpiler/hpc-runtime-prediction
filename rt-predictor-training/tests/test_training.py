#!/usr/bin/env python3
"""
Tests for RT Predictor Training Service
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from features.feature_engineering_optimized import OptimizedFeatureEngineer
from features.preprocessing import DataPreprocessor
from model.trainer import ModelTrainer


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def test_feature_engineer_init(self):
        """Test feature engineer initialization."""
        fe = OptimizedFeatureEngineer()
        assert fe is not None
        assert hasattr(fe, 'feature_names')
    
    def test_feature_creation(self):
        """Test feature creation with sample data."""
        # Create sample data
        data = pd.DataFrame({
            'nodes_req': [1, 2, 4],
            'processors_req': [16, 32, 64],
            'mem_req': [32000, 64000, 128000],
            'wallclock_req': [3600, 7200, 14400],
            'partition': ['compute', 'gpu', 'compute'],
            'qos': ['normal', 'high', 'normal'],
            'run_time': [3000, 6500, 13000]
        })
        
        fe = OptimizedFeatureEngineer()
        features = fe.fit_transform(data)
        
        assert features is not None
        assert len(features) == len(data)
        assert features.shape[1] > 7  # More features than input


class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        prep = DataPreprocessor()
        assert prep is not None
    
    def test_outlier_removal(self):
        """Test outlier removal."""
        # Create data with outliers
        data = pd.DataFrame({
            'run_time': [100, 200, 300, 10000, 150],  # 10000 is outlier
            'wallclock_req': [200, 300, 400, 500, 250]
        })
        
        prep = DataPreprocessor()
        cleaned = prep.remove_outliers(data, 'run_time')
        
        assert len(cleaned) < len(data)
        assert 10000 not in cleaned['run_time'].values


class TestModelTrainer:
    """Test model training functionality."""
    
    def test_trainer_init(self):
        """Test trainer initialization."""
        trainer = ModelTrainer(model_type='xgboost')
        assert trainer is not None
        assert trainer.model_type == 'xgboost'
    
    @pytest.mark.slow
    def test_model_training(self):
        """Test model training with synthetic data."""
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples)
        })
        
        # Target with some correlation to features
        y = (X['feature_1'] * 2 + 
             X['feature_2'] * 1.5 + 
             X['feature_3'] * 0.5 + 
             np.random.randn(n_samples) * 0.1)
        
        trainer = ModelTrainer(model_type='xgboost')
        trainer.train(X, y, X, y)  # Using same data for train/val for test
        
        assert trainer.model is not None
        assert hasattr(trainer, 'best_params')
        
        # Test prediction
        predictions = trainer.predict(X)
        assert len(predictions) == len(X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
