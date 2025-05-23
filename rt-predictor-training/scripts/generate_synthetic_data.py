#!/usr/bin/env python3
"""
Generate synthetic Eagle HPC dataset for development and testing.
This creates realistic synthetic data matching the Eagle schema.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path


def generate_synthetic_eagle_data(n_records=100000, seed=42):
    """
    Generate synthetic HPC job data matching Eagle schema.
    
    Args:
        n_records: Number of records to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic Eagle data
    """
    np.random.seed(seed)
    
    print(f"Generating {n_records:,} synthetic job records...")
    
    # Base timestamp for job submissions
    base_time = datetime(2024, 1, 1)
    
    # Generate job IDs
    job_ids = [f"job_{i:08d}" for i in range(1, n_records + 1)]
    
    # Generate user distribution (power law - few users submit many jobs)
    n_users = min(100, n_records // 100)
    user_weights = np.power(np.arange(1, n_users + 1), -1.5)
    user_weights /= user_weights.sum()
    users = [f"user_{i:03d}" for i in range(n_users)]
    
    # Generate accounts (projects)
    n_accounts = min(20, n_records // 500)
    accounts = [f"proj_{i:02d}" for i in range(n_accounts)]
    
    # Job names (common patterns)
    job_name_patterns = [
        "simulation", "analysis", "ml_train", "data_proc", 
        "optimization", "monte_carlo", "fluid_sim", "genome_align"
    ]
