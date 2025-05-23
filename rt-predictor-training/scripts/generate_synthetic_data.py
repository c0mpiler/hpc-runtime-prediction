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
    
    # Generate data
    data = pd.DataFrame({
        'job_id': job_ids,
        'user': np.random.choice(users, n_records, p=user_weights),
        'account': np.random.choice(accounts, n_records),
        'name': [f"{np.random.choice(job_name_patterns)}_{i%100:03d}.sh" for i in range(n_records)],
        'state': np.random.choice(['COMPLETED', 'FAILED', 'TIMEOUT'], 
                                 n_records, p=[0.85, 0.10, 0.05]),
        'partition': np.random.choice(['compute', 'gpu', 'debug', 'bigmem', 'standard'], 
                                     n_records, p=[0.50, 0.20, 0.10, 0.10, 0.10]),
        'qos': np.random.choice(['normal', 'high', 'low', 'debug'], 
                               n_records, p=[0.60, 0.20, 0.15, 0.05]),
    })
    
    # Generate submit times (spread over 6 months)
    submit_times = []
    for i in range(n_records):
        days_offset = np.random.randint(0, 180)
        hours_offset = np.random.randint(0, 24)
        submit_time = base_time + timedelta(days=days_offset, hours=hours_offset)
        submit_times.append(int(submit_time.timestamp()))
    data['submit_time'] = submit_times
    
    # Generate resource requirements with correlations
    # Nodes and processors are correlated
    data['nodes_req'] = np.random.choice([1, 2, 4, 8, 16, 32], 
                                        n_records, p=[0.40, 0.25, 0.20, 0.10, 0.04, 0.01])
    data['processors_req'] = data['nodes_req'] * np.random.choice([16, 24, 32, 48], n_records)
    
    # Memory correlates with nodes
    data['mem_req'] = data['nodes_req'] * np.random.choice([32000, 64000, 128000], n_records)
    
    # Walltime requests (in seconds)
    walltime_choices = [
        3600,    # 1 hour
        7200,    # 2 hours
        14400,   # 4 hours
        28800,   # 8 hours
        43200,   # 12 hours
        86400,   # 24 hours
        172800,  # 48 hours
    ]
    data['wallclock_req'] = np.random.choice(walltime_choices, 
                                             n_records, p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.04, 0.01])
    
    # GPU requirements (most jobs don't use GPUs)
    gpu_prob = np.where(data['partition'] == 'gpu', 0.9, 0.01)
    data['gpus_req'] = np.where(
        np.random.random(n_records) < gpu_prob,
        np.random.choice([1, 2, 4, 8], n_records),
        0
    )
    
    # Generate realistic runtimes
    # Runtime is a fraction of requested time, with some noise
    efficiency = np.random.beta(2, 5, n_records)  # Most jobs use 20-60% of requested time
    data['run_time'] = (data['wallclock_req'] * efficiency).astype(int)
    
    # Add some outliers (jobs that run very short or very long)
    outlier_mask = np.random.random(n_records) < 0.05
    data.loc[outlier_mask, 'run_time'] = np.random.choice([60, 120, 300], sum(outlier_mask))
    
    # Start time is submit time plus queue wait
    queue_wait = np.random.exponential(1800, n_records)  # Average 30 min wait
    data['start_time'] = data['submit_time'] + queue_wait.astype(int)
    
    # End time
    data['end_time'] = data['start_time'] + data['run_time']
    
    # Exit codes
    data['exit_code'] = np.where(data['state'] == 'COMPLETED', 0, 
                                 np.random.choice([1, 2, 127, 255], n_records))
    data['derived_exit_code'] = data['exit_code']
    
    # Sort by submit time
    data = data.sort_values('submit_time').reset_index(drop=True)
    
    print(f"✓ Generated {len(data):,} synthetic job records")
    print(f"  Time range: {pd.to_datetime(data['submit_time'].min(), unit='s')} to {pd.to_datetime(data['submit_time'].max(), unit='s')}")
    print(f"  Partitions: {data['partition'].value_counts().to_dict()}")
    print(f"  GPU jobs: {(data['gpus_req'] > 0).sum():,}")
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Eagle HPC dataset")
    parser.add_argument('--rows', type=int, default=100000, help='Number of records to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='data/raw/eagle_data.parquet', 
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    df = generate_synthetic_eagle_data(args.rows, args.seed)
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
