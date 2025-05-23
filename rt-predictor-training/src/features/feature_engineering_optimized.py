"""Feature engineering for job runtime prediction - Optimized Version V2.

This version includes:
- Chunked processing for large datasets
- Parallel processing for independent operations
- Vectorized operations for string processing
- Intelligent caching for expensive computations
- Memory-efficient data types
- Progress tracking with tqdm
- Fixed categorical encoding issues
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
import joblib
from pathlib import Path
import logging
import re
import time
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OptimizedFeatureEngineer:
    """Optimized feature engineering pipeline for job data with M2 Max optimizations."""
    
    def __init__(self, 
                 use_advanced_features: bool = True,
                 chunk_size: int = 100000,
                 n_jobs: int = None,
                 enable_caching: bool = True,
                 cache_dir: str = "data/processed/cache"):
        """Initialize optimized feature engineering components.
        
        Args:
            use_advanced_features: Whether to use advanced feature engineering
            chunk_size: Size of chunks for processing large datasets
            n_jobs: Number of parallel jobs (default: CPU count - 1)
            enable_caching: Whether to cache expensive computations
            cache_dir: Directory for cache files
        """
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        self.user_stats = {}
        self.queue_stats = {}
        self.is_fitted = False
        self.use_advanced_features = use_advanced_features
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
        self.enable_caching = enable_caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-compile regex patterns for job name features
        self._compile_regex_patterns()
        
        logger.info(f"Initialized OptimizedFeatureEngineer with:")
        logger.info(f"  - Chunk size: {self.chunk_size:,}")
        logger.info(f"  - Parallel jobs: {self.n_jobs}")
        logger.info(f"  - Caching: {'enabled' if self.enable_caching else 'disabled'}")
        logger.info(f"  - Advanced features: {'enabled' if self.use_advanced_features else 'disabled'}")
    
    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.regex_patterns = {
            'mpi': re.compile(r'mpi|MPI', re.IGNORECASE),
            'gpu': re.compile(r'gpu|cuda|GPU|CUDA', re.IGNORECASE),
            'array': re.compile(r'\[\d+\]|\d+of\d+', re.IGNORECASE),
            'test': re.compile(r'test|TEST|debug|DEBUG', re.IGNORECASE),
            'prod': re.compile(r'prod|PROD|production', re.IGNORECASE),
            'numbers': re.compile(r'\d'),
            'array_size': re.compile(r'\[(\d+)-(\d+)\]')
        }
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to memory-efficient dtypes."""
        logger.info("Optimizing data types for memory efficiency...")
        
        # Convert int64 to smaller types where possible
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            if col in df.columns:
                max_val = df[col].max()
                min_val = df[col].min()
                
                if min_val >= 0:  # Unsigned
                    if max_val < 255:
                        df[col] = df[col].astype('uint8')
                    elif max_val < 65535:
                        df[col] = df[col].astype('uint16')
                    elif max_val < 4294967295:
                        df[col] = df[col].astype('uint32')
                else:  # Signed
                    if min_val > -128 and max_val < 127:
                        df[col] = df[col].astype('int8')
                    elif min_val > -32768 and max_val < 32767:
                        df[col] = df[col].astype('int16')
                    elif min_val > -2147483648 and max_val < 2147483647:
                        df[col] = df[col].astype('int32')
        
        # Convert object to category for low-cardinality columns
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if col in df.columns:
                num_unique = df[col].nunique()
                num_total = len(df[col])
                if num_unique / num_total < 0.5:  # Less than 50% unique
                    df[col] = df[col].astype('category')
        
        # Convert float64 to float32 where precision allows
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype('float32')
        
        return df
    
    def _process_in_chunks(self, df: pd.DataFrame, func, desc: str = "Processing") -> pd.DataFrame:
        """Process large dataframes in chunks with progress tracking."""
        if len(df) <= self.chunk_size:
            return func(df)
        
        chunks = []
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        
        with tqdm(total=total_chunks, desc=desc) as pbar:
            for start in range(0, len(df), self.chunk_size):
                chunk = df.iloc[start:start + self.chunk_size].copy()
                processed_chunk = func(chunk)
                chunks.append(processed_chunk)
                pbar.update(1)
        
        return pd.concat(chunks, ignore_index=True)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features with optimizations."""
        start_time = time.time()
        logger.info(f"Starting optimized feature engineering on {len(df):,} rows")
        logger.info(f"Advanced features: {'enabled' if self.use_advanced_features else 'disabled'}")
        
        # Create a copy and optimize memory
        df = df.copy()
        df = self._optimize_dtypes(df)
        
        # Phase 1: Basic transformations (can be done in chunks)
        logger.info("Phase 1: Basic transformations...")
        df = self._handle_missing_values(df)
        df = self._process_in_chunks(df, self._create_time_features, "Time features")
        df = self._process_in_chunks(df, self._create_resource_features, "Resource features")
        
        # Phase 2: Advanced features (require global statistics)
        if self.use_advanced_features:
            logger.info("Phase 2: Advanced features...")
            
            # User features with caching
            df = self._create_user_features_optimized(df, fit=True)
            
            # Queue features with optimized computation
            df = self._create_queue_features_optimized(df, fit=True)
            
            # Job name features (vectorized)
            df = self._process_in_chunks(df, self._create_job_name_features_vectorized, 
                                       "Job name features")
            
            # Advanced interaction features
            df = self._process_in_chunks(df, self._create_advanced_interaction_features,
                                       "Interaction features")
        
        # Phase 3: Encoding and scaling
        logger.info("Phase 3: Encoding and scaling...")
        df = self._encode_categoricals_optimized(df, fit=True)
        df = self._scale_features_optimized(df, fit=True)
        
        # Store feature statistics
        self._compute_feature_stats(df)
        self.is_fitted = True
        
        elapsed = time.time() - start_time
        logger.info(f"Feature engineering complete in {elapsed:.2f}s. Shape: {df.shape}")
        logger.info(f"Processing rate: {len(df)/elapsed:.0f} rows/second")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        start_time = time.time()
        logger.info(f"Transforming {len(df):,} rows...")
        
        df = df.copy()
        df = self._optimize_dtypes(df)
        
        df = self._handle_missing_values(df)
        df = self._process_in_chunks(df, self._create_time_features, "Time features")
        df = self._process_in_chunks(df, self._create_resource_features, "Resource features")
        
        if self.use_advanced_features:
            df = self._create_user_features_optimized(df, fit=False)
            df = self._create_queue_features_optimized(df, fit=False)
            df = self._process_in_chunks(df, self._create_job_name_features_vectorized,
                                       "Job name features")
            df = self._process_in_chunks(df, self._create_advanced_interaction_features,
                                       "Interaction features")
        
        df = self._encode_categoricals_optimized(df, fit=False)
        df = self._scale_features_optimized(df, fit=False)
        
        elapsed = time.time() - start_time
        logger.info(f"Transform complete in {elapsed:.2f}s")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if 'wallclock_req' in df.columns and df['wallclock_req'].isnull().any():
            median_wallclock = df['wallclock_req'].median()
            df['wallclock_req'] = df['wallclock_req'].fillna(median_wallclock)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features (vectorized)."""
        if 'submit_time' in df.columns:
            submit_time = pd.to_datetime(df['submit_time'])
            df['hour_of_day'] = submit_time.dt.hour
            df['day_of_week'] = submit_time.dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype('uint8')
            df['month'] = submit_time.dt.month
            df['quarter'] = submit_time.dt.quarter
            
            if self.use_advanced_features:
                # Vectorized operations
                df['is_night'] = (
                    (df['hour_of_day'] >= 20) | (df['hour_of_day'] < 6)
                ).astype('uint8')
                
                df['is_business_hours'] = (
                    (df['hour_of_day'].between(9, 16)) & 
                    (~df['is_weekend'].astype(bool))
                ).astype('uint8')
                
                df['day_of_month'] = submit_time.dt.day
                df['week_of_year'] = submit_time.dt.isocalendar().week.astype('uint8')
        
        return df
    
    def _create_resource_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create resource-based interaction features (vectorized)."""
        # Avoid division by zero with np.maximum
        df['cpu_memory_ratio'] = df['processors_req'] / np.maximum(df['mem_req'], 1)
        df['resource_intensity'] = np.log1p(df['processors_req'] * df['nodes_req'])
        df['memory_per_node'] = df['mem_req'] / np.maximum(df['nodes_req'], 1)
        df['total_compute_units'] = df['processors_req'] * df['nodes_req']
        df['walltime_resource_ratio'] = df['wallclock_req'] / np.maximum(df['total_compute_units'], 1)
        
        if self.use_advanced_features:
            df['memory_per_cpu'] = df['mem_req'] / np.maximum(df['processors_req'], 1)
            
            # Use quantiles for thresholds (more efficient)
            mem_per_cpu_q75 = df['memory_per_cpu'].quantile(0.75)
            total_compute_q90 = df['total_compute_units'].quantile(0.9)
            
            df['is_high_memory_job'] = (df['memory_per_cpu'] > mem_per_cpu_q75).astype('uint8')
            df['is_large_job'] = (df['total_compute_units'] > total_compute_q90).astype('uint8')
            
            cpu_mem_median = df['cpu_memory_ratio'].median()
            df['resource_balance'] = np.abs(df['cpu_memory_ratio'] - cpu_mem_median)
        
        return df
    
    def _create_user_features_optimized(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create user behavior features with caching and optimization."""
        if 'user' not in df.columns:
            return df
        
        cache_path = self.cache_dir / "user_stats_cache.pkl"
        
        if fit:
            if self.enable_caching and cache_path.exists():
                logger.info("Loading cached user statistics...")
                self.user_stats = joblib.load(cache_path)
            else:
                logger.info("Computing user statistics (optimized)...")
                
                # Don't convert to category globally - just for groupby
                with tqdm(desc="User statistics") as pbar:
                    # Use category type only for groupby efficiency
                    user_groups = df.groupby(df['user'].astype('category'), observed=True)
                    
                    user_agg = user_groups.agg({
                        'run_time': ['mean', 'std', 'count', 'median'],
                        'wallclock_req': ['mean', 'median'],
                        'processors_req': ['mean', 'median'],
                        'mem_req': ['mean', 'median']
                    })
                    pbar.update(1)
                    
                    # Calculate accuracy efficiently
                    accuracy = user_groups.apply(
                        lambda x: np.mean(x['wallclock_req'] / np.maximum(x['run_time'], 1))
                    )
                    user_agg['accuracy'] = accuracy
                    pbar.update(1)
                
                self.user_stats = user_agg.fillna(0)
                
                if self.enable_caching:
                    joblib.dump(self.user_stats, cache_path)
                    logger.info(f"Cached user statistics to {cache_path}")
        
        # Vectorized mapping using merge
        if hasattr(self, 'user_stats') and len(self.user_stats) > 0:
            # Create a temporary dataframe for merging
            user_stats_flat = pd.DataFrame({
                'user': self.user_stats.index,
                'user_mean_runtime': self.user_stats[('run_time', 'mean')].values,
                'user_job_count': self.user_stats[('run_time', 'count')].values,
                'user_accuracy_score': self.user_stats['accuracy'].values
            })
            
            # Efficient merge
            df = df.merge(user_stats_flat, on='user', how='left')
            
            # Fill missing values with defaults
            df['user_mean_runtime'] = df['user_mean_runtime'].fillna(df['run_time'].mean())
            df['user_job_count'] = df['user_job_count'].fillna(1)
            df['user_accuracy_score'] = df['user_accuracy_score'].fillna(1.0)
            
            # Vectorized feature creation
            df['user_overestimate_tendency'] = (df['user_accuracy_score'] > 1.5).astype('uint8')
            df['user_underestimate_tendency'] = (df['user_accuracy_score'] < 0.8).astype('uint8')
            
            job_count_q75 = df['user_job_count'].quantile(0.75)
            df['user_is_frequent'] = (df['user_job_count'] > job_count_q75).astype('uint8')
        
        return df
    
    def _create_queue_features_optimized(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create queue congestion features with optimization."""
        if 'partition' not in df.columns or 'submit_time' not in df.columns:
            return df
        
        cache_path = self.cache_dir / "queue_stats_cache.pkl"
        
        # Extract time components efficiently
        submit_dt = pd.to_datetime(df['submit_time'])
        df['submit_hour'] = submit_dt.dt.hour
        df['submit_date'] = submit_dt.dt.date
        
        if fit:
            if self.enable_caching and cache_path.exists():
                logger.info("Loading cached queue statistics...")
                self.queue_stats = joblib.load(cache_path)
            else:
                logger.info("Computing queue statistics (optimized)...")
                
                # Use category type only for groupby
                with tqdm(desc="Queue statistics") as pbar:
                    self.queue_stats = df.groupby(
                        [df['partition'].astype('category'), 'submit_hour'], 
                        observed=True
                    ).agg({
                        'job_id': 'count',
                        'run_time': ['mean', 'median'],
                        'processors_req': 'sum'
                    }).fillna(0)
                    pbar.update(1)
                
                if self.enable_caching:
                    joblib.dump(self.queue_stats, cache_path)
                    logger.info(f"Cached queue statistics to {cache_path}")
        
        # Vectorized queue load mapping
        if hasattr(self, 'queue_stats') and len(self.queue_stats) > 0:
            # Create lookup dataframe
            queue_load_df = self.queue_stats[('job_id', 'count')].reset_index()
            queue_load_df.columns = ['partition', 'submit_hour', 'queue_load']
            
            # Efficient merge
            df = df.merge(queue_load_df, on=['partition', 'submit_hour'], how='left')
            df['queue_load'] = df['queue_load'].fillna(0)
            
            # Queue wait time (vectorized)
            if 'start_time' in df.columns:
                start_dt = pd.to_datetime(df['start_time'])
                wait_seconds = (start_dt - submit_dt).dt.total_seconds()
                df['queue_wait_time'] = np.maximum(wait_seconds.fillna(0), 0)
        
        return df
    
    def _create_job_name_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from job names using vectorized operations."""
        if 'name' not in df.columns:
            return df
        
        # Handle categorical dtype and ensure string type
        if df['name'].dtype.name == 'category':
            # Convert to string first to avoid category issues
            # Replace NaN with empty string before converting
            name_series = df['name'].cat.add_categories(['']).fillna('').astype(str)
        else:
            name_series = df['name'].fillna('').astype(str)
        
        # Basic features
        df['name_length'] = name_series.str.len()
        
        # Use pre-compiled regex patterns with str.contains (vectorized)
        df['name_has_numbers'] = name_series.str.contains(
            self.regex_patterns['numbers'], regex=True
        ).astype('uint8')
        
        df['is_mpi_job'] = name_series.str.contains(
            self.regex_patterns['mpi'], regex=True
        ).astype('uint8')
        
        df['is_gpu_job'] = name_series.str.contains(
            self.regex_patterns['gpu'], regex=True
        ).astype('uint8')
        
        df['is_array_job'] = name_series.str.contains(
            self.regex_patterns['array'], regex=True
        ).astype('uint8')
        
        df['is_test_job'] = name_series.str.contains(
            self.regex_patterns['test'], regex=True
        ).astype('uint8')
        
        df['is_prod_job'] = name_series.str.contains(
            self.regex_patterns['prod'], regex=True
        ).astype('uint8')
        
        # Extract array job size (vectorized)
        array_matches = name_series.str.extract(self.regex_patterns['array_size'])
        df['array_job_size'] = 0
        valid_matches = array_matches[0].notna() & array_matches[1].notna()
        if valid_matches.any():
            df.loc[valid_matches, 'array_job_size'] = (
                array_matches.loc[valid_matches, 1].astype(int) - 
                array_matches.loc[valid_matches, 0].astype(int) + 1
            )
        
        return df
    
    def _create_advanced_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced interaction features (vectorized)."""
        # Resource allocation efficiency
        # Avoid division by zero
        df['cpu_efficiency_score'] = df['processors_req'] / np.maximum(df['nodes_req'] * 32, 1)
        
        # Job complexity indicators
        df['job_complexity_score'] = (
            df['resource_intensity'] * df['wallclock_req'] / 3600
        )
        
        # Time-resource interaction
        if 'is_business_hours' in df.columns and 'is_large_job' in df.columns:
            df['peak_hour_large_job'] = (
                df['is_business_hours'].astype('uint8') * 
                df['is_large_job'].astype('uint8')
            )
        
        # User-partition affinity (only if both columns exist)
        if 'user' in df.columns and 'partition' in df.columns:
            df['user_partition_pair'] = (
                df['user'].astype(str) + '_' + df['partition'].astype(str)
            )
        
        return df
    
    def _encode_categoricals_optimized(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables with optimization."""
        categorical_columns = ['partition', 'qos']
        
        if self.use_advanced_features and 'user' in df.columns:
            categorical_columns.append('user')
        
        for col in categorical_columns:
            if col in df.columns:
                # Store original dtype
                original_dtype = df[col].dtype
                
                # Convert category to string for encoding if needed
                if original_dtype.name == 'category':
                    df[col] = df[col].astype(str)
                
                if fit:
                    if col == 'user' and self.use_advanced_features:
                        # Frequency encoding for users
                        user_counts = df[col].value_counts()
                        self.encoders[col] = user_counts.to_dict()
                    else:
                        # Label encoding for others
                        self.encoders[col] = LabelEncoder()
                        # Fit on unique values (excluding NaN)
                        unique_vals = df[col].dropna().unique()
                        self.encoders[col].fit(unique_vals)
                
                # Apply encoding
                if col == 'user' and self.use_advanced_features:
                    df[f'{col}_frequency'] = df[col].map(self.encoders[col]).fillna(1)
                else:
                    # Create encoded column
                    if hasattr(self.encoders[col], 'classes_'):
                        # Create mapping dictionary
                        class_to_int = {
                            cls: idx for idx, cls in enumerate(self.encoders[col].classes_)
                        }
                        # Map values and handle unknown values
                        df[f'{col}_encoded'] = df[col].map(class_to_int)
                        # Fill NaN with -1 for unknown categories
                        df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(-1).astype('int16')
                    else:
                        # If encoder not fitted properly
                        df[f'{col}_encoded'] = -1
        
        return df
    
    def _scale_features_optimized(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features with parallel processing."""
        # Define numerical features
        numerical_features = [
            'processors_req', 'nodes_req', 'mem_req', 'wallclock_req',
            'cpu_memory_ratio', 'resource_intensity', 'memory_per_node',
            'total_compute_units', 'walltime_resource_ratio'
        ]
        
        if self.use_advanced_features:
            advanced_numerical = [
                'memory_per_cpu', 'resource_balance', 'user_mean_runtime',
                'user_job_count', 'user_accuracy_score', 'queue_load',
                'name_length', 'array_job_size', 'cpu_efficiency_score',
                'job_complexity_score'
            ]
            numerical_features.extend([f for f in advanced_numerical if f in df.columns])
            
            if 'queue_wait_time' in df.columns:
                numerical_features.append('queue_wait_time')
        
        # Process features in parallel batches
        def scale_feature(feature):
            if feature not in df.columns:
                return None
                
            if fit:
                scaler = RobustScaler()
                # Handle infinity and NaN values
                feature_values = df[[feature]].copy()
                feature_values.replace([np.inf, -np.inf], np.nan, inplace=True)
                feature_values.fillna(feature_values.median(), inplace=True)
                scaled_values = scaler.fit_transform(feature_values)
                return feature, scaler, scaled_values
            else:
                if feature in self.scalers:
                    # Handle infinity and NaN values
                    feature_values = df[[feature]].copy()
                    feature_values.replace([np.inf, -np.inf], np.nan, inplace=True)
                    feature_values.fillna(df[feature].median(), inplace=True)
                    scaled_values = self.scalers[feature].transform(feature_values)
                    return feature, None, scaled_values
                return None
        
        # Use ThreadPoolExecutor for I/O bound scaling operations
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(tqdm(
                executor.map(scale_feature, numerical_features),
                total=len(numerical_features),
                desc="Scaling features"
            ))
        
        # Apply results
        for result in results:
            if result is not None:
                feature, scaler, scaled_values = result
                df[f'{feature}_scaled'] = scaled_values
                if fit and scaler is not None:
                    self.scalers[feature] = scaler
        
        return df
    
    def _compute_feature_stats(self, df: pd.DataFrame):
        """Compute and store feature statistics for monitoring."""
        numeric_dtypes = ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 
                         'uint8', 'uint16', 'uint32', 'uint64']
        
        for col in df.columns:
            if df[col].dtype in numeric_dtypes:
                self.feature_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q50': float(df[col].quantile(0.50)),
                    'q75': float(df[col].quantile(0.75))
                }
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns after engineering."""
        base_features = [
            'processors_req_scaled', 'nodes_req_scaled', 'mem_req_scaled',
            'wallclock_req_scaled', 'cpu_memory_ratio_scaled',
            'resource_intensity_scaled', 'memory_per_node_scaled',
            'total_compute_units_scaled', 'walltime_resource_ratio_scaled',
            'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'quarter',
            'partition_encoded', 'qos_encoded'
        ]
        
        if self.use_advanced_features:
            advanced_features = [
                'is_night', 'is_business_hours', 'day_of_month', 'week_of_year',
                'memory_per_cpu_scaled', 'is_high_memory_job', 'is_large_job',
                'resource_balance_scaled', 'cpu_efficiency_score_scaled',
                'job_complexity_score_scaled', 'peak_hour_large_job',
                'user_mean_runtime_scaled', 'user_job_count_scaled', 
                'user_accuracy_score_scaled', 'user_overestimate_tendency',
                'user_underestimate_tendency', 'user_is_frequent', 'user_frequency',
                'queue_load_scaled',
                'name_length_scaled', 'name_has_numbers', 'is_mpi_job', 'is_gpu_job',
                'is_array_job', 'is_test_job', 'is_prod_job', 'array_job_size_scaled'
            ]
            
            if 'queue_wait_time_scaled' in self.feature_stats:
                advanced_features.append('queue_wait_time_scaled')
            
            base_features.extend(advanced_features)
        
        return base_features
    
    def save(self, path: Path):
        """Save feature engineering artifacts."""
        artifacts = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_stats': self.feature_stats,
            'user_stats': self.user_stats if hasattr(self, 'user_stats') else {},
            'queue_stats': self.queue_stats if hasattr(self, 'queue_stats') else {},
            'is_fitted': self.is_fitted,
            'use_advanced_features': self.use_advanced_features,
            'chunk_size': self.chunk_size,
            'n_jobs': self.n_jobs,
            'enable_caching': self.enable_caching,
            'version': '3.1-optimized-fixed'
        }
        joblib.dump(artifacts, path)
        logger.info(f"Optimized feature engineering artifacts saved to {path}")
    
    def load(self, path: Path):
        """Load feature engineering artifacts."""
        artifacts = joblib.load(path)
        self.scalers = artifacts['scalers']
        self.encoders = artifacts['encoders']
        self.feature_stats = artifacts['feature_stats']
        self.user_stats = artifacts.get('user_stats', {})
        self.queue_stats = artifacts.get('queue_stats', {})
        self.is_fitted = artifacts['is_fitted']
        self.use_advanced_features = artifacts.get('use_advanced_features', False)
        self.chunk_size = artifacts.get('chunk_size', 100000)
        self.n_jobs = artifacts.get('n_jobs', multiprocessing.cpu_count() - 1)
        self.enable_caching = artifacts.get('enable_caching', True)
        
        # Re-compile regex patterns
        self._compile_regex_patterns()
        
        logger.info(f"Optimized feature engineering artifacts loaded from {path}")
        logger.info(f"Version: {artifacts.get('version', '1.0')}")


# Alias for backward compatibility
FeatureEngineer = OptimizedFeatureEngineer
