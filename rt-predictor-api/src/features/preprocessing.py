"""Data preprocessing utilities for job runtime prediction."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data preprocessing and cleaning."""
    
    def __init__(self, config: dict = None):
        """Initialize preprocessor with configuration."""
        self.config = config or {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by removing invalid entries.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        initial_rows = len(df)
        logger.info(f"Starting data cleaning with {initial_rows} rows")
        
        # Remove jobs with negative or zero runtime
        df = df[df['run_time'] > 0]
        logger.info(f"Removed {initial_rows - len(df)} jobs with invalid runtime")
        
        # Remove extremely short jobs (likely errors)
        min_runtime = self.config.get('min_runtime_seconds', 10)
        df = df[df['run_time'] >= min_runtime]
        
        # Remove extremely long jobs (outliers)
        max_runtime = self.config.get('max_runtime_seconds', 604800)  # 1 week
        df = df[df['run_time'] <= max_runtime]
        
        # Remove jobs with invalid resource requests
        df = df[df['processors_req'] > 0]
        df = df[df['nodes_req'] > 0]
        df = df[df['mem_req'] >= 0]
        
        # Remove outliers in CPU and memory
        cpu_percentile = self.config.get('cpu_outlier_percentile', 99.5)
        mem_percentile = self.config.get('memory_outlier_percentile', 99.5)
        
        cpu_threshold = df['processors_req'].quantile(cpu_percentile / 100)
        mem_threshold = df['mem_req'].quantile(mem_percentile / 100)
        
        df = df[df['processors_req'] <= cpu_threshold]
        df = df[df['mem_req'] <= mem_threshold]
        
        # Keep only completed jobs for training
        if 'state' in df.columns:
            df = df[df['state'] == 'COMPLETED']
        
        logger.info(f"Data cleaning complete. Rows: {initial_rows} -> {len(df)}")
        return df
    
    def split_data(self, df: pd.DataFrame, 
                   target_col: str = 'run_time') -> Tuple[pd.DataFrame, ...]:
        """Split data into train, validation, and test sets.
        
        Args:
            df: Preprocessed dataframe
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Sort by time for temporal split
        if 'submit_time' in df.columns:
            df = df.sort_values('submit_time')
        
        # Get train/val/test sizes from config
        train_size = self.config.get('train_size', 0.7)
        val_size = self.config.get('val_size', 0.15)
        test_size = self.config.get('test_size', 0.15)
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        # Split data temporally
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def prepare_features_target(self, df: pd.DataFrame, 
                                feature_cols: list,
                                target_col: str = 'run_time') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling.
        
        Args:
            df: Dataframe with all columns
            feature_cols: List of feature column names
            target_col: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        # Ensure all feature columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) < len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            logger.warning(f"Missing feature columns: {missing}")
        
        X = df[available_cols]
        y = df[target_col]
        
        # Log transform target for better distribution
        y_log = np.log1p(y)
        
        return X, y_log
