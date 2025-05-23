# RT Predictor - Data Guide

This document describes the data requirements, sources, and setup for the RT Predictor system.

## Dataset Overview

The RT Predictor is trained on the **NREL Eagle HPC System Dataset**, containing over 11 million job records from the National Renewable Energy Laboratory's Eagle supercomputer.

### Dataset Characteristics
- **Size**: 11M+ job records
- **Time Period**: Historical HPC job submissions and executions
- **Format**: Parquet files for efficient storage and processing
- **Features**: 18 columns including resource requests, runtimes, and job metadata

## Data Schema

The Eagle dataset contains the following columns:

```
- job_id: Unique job identifier
- user: Anonymized user ID
- account: Account/project identifier
- name: Job name (often script name)
- state: Job completion state (COMPLETED, FAILED, TIMEOUT, etc.)
- partition: Compute partition (compute, gpu, debug, etc.)
- qos: Quality of Service level
- submit_time: Job submission timestamp
- start_time: Job start timestamp
- end_time: Job completion timestamp
- run_time: Actual runtime in seconds
- wallclock_req: Requested walltime in seconds
- nodes_req: Number of nodes requested
- processors_req: Number of processors requested
- mem_req: Memory requested in MB
- gpus_req: Number of GPUs requested (0 for most jobs)
- exit_code: Job exit code
- derived_exit_code: Processed exit status
```

## Data Location

### Original Data Source
The original Eagle dataset is typically obtained from:
1. **NREL HPC Data Repository** (if publicly available)
2. **Direct from NREL** (may require data use agreement)
3. **Academic repositories** hosting HPC workload traces

### Local Data Setup

1. **Expected location for training service**:
   ```
   microservices/rt-predictor-training/data/raw/eagle_data.parquet
   ```

2. **If you have the data elsewhere** (e.g., from the monolithic app):
   ```bash
   # Copy from original location
   cp /Users/c0mpiler/sandbox/IBM/EDAaaS/edaaas-dev/rt-predictor/ml/eagle-jobs/data/full-data/*.parquet \
      /Users/c0mpiler/sandbox/IBM/EDAaaS/edaaas-dev/rt-predictor/microservices/rt-predictor-training/data/raw/
   ```

3. **Docker volume mapping**:
   ```bash
   # Map your local data directory when running training
   docker run -v /path/to/your/eagle/data:/app/data/raw rt-predictor-training
   ```

## Data Preparation

### Option 1: Use Existing Processed Data
If you already have the Eagle dataset from the monolithic RT Predictor:

```bash
# Navigate to microservices directory
cd /Users/c0mpiler/sandbox/IBM/EDAaaS/edaaas-dev/rt-predictor/microservices

# Create data directory
mkdir -p rt-predictor-training/data/raw

# Copy existing data
cp ../ml/eagle-jobs/data/full-data/*.parquet rt-predictor-training/data/raw/
```

### Option 2: Download Sample Data
For testing purposes, you can use a sample dataset:

```bash
# Create sample data script
cd rt-predictor-training
python scripts/create_sample_data.py --rows 100000 --output data/raw/eagle_data_sample.parquet
```

### Option 3: Generate Synthetic Data
For development without access to real data:

```python
# scripts/generate_synthetic_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_eagle_data(n_records=100000):
    """Generate synthetic HPC job data matching Eagle schema."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    data = {
        'job_id': range(1, n_records + 1),
        'user': np.random.choice([f'user_{i}' for i in range(100)], n_records),
        'account': np.random.choice([f'proj_{i}' for i in range(20)], n_records),
        'partition': np.random.choice(['compute', 'gpu', 'debug', 'bigmem'], 
                                     n_records, p=[0.6, 0.2, 0.1, 0.1]),
        'qos': np.random.choice(['normal', 'high', 'low'], n_records, p=[0.7, 0.2, 0.1]),
        'nodes_req': np.random.choice([1, 2, 4, 8, 16], n_records, p=[0.5, 0.25, 0.15, 0.07, 0.03]),
        'processors_req': np.random.choice([16, 32, 48, 64], n_records),
        'mem_req': np.random.choice([32000, 64000, 128000, 256000], n_records),
        'wallclock_req': np.random.choice([3600, 7200, 14400, 28800], n_records),
        'gpus_req': np.random.choice([0, 0, 0, 0, 1, 2, 4], n_records),
    }
    
    # Generate realistic runtimes (correlated with requested time)
    data['run_time'] = [
        int(req * np.random.uniform(0.1, 0.9)) 
        for req in data['wallclock_req']
    ]
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_synthetic_eagle_data()
    df.to_parquet('data/raw/eagle_data.parquet')
    print(f"Generated {len(df)} synthetic records")
```

## Data Quality Checks

Before training, the data undergoes several quality checks:

1. **Missing Values**: Check for and handle missing values
2. **Outliers**: Remove extreme outliers (e.g., runtime > 30 days)
3. **Data Types**: Ensure correct data types for all columns
4. **Temporal Consistency**: Verify start_time > submit_time, end_time > start_time
5. **Resource Bounds**: Check reasonable resource requests

## Data Splits

The training pipeline automatically splits data:
- **Training**: 70% (temporal split)
- **Validation**: 15% (temporal split)
- **Test**: 15% (temporal split)

Temporal splitting ensures the model is evaluated on future jobs, mimicking production usage.

## Privacy and Security

- User IDs are anonymized in the dataset
- No personally identifiable information (PII)
- Job names may contain script names but no sensitive data
- Account names are project identifiers

## Troubleshooting

### "File not found" Error
```bash
# Check if data exists
ls -la rt-predictor-training/data/raw/

# Create directory if missing
mkdir -p rt-predictor-training/data/raw

# Verify file permissions
chmod -R 755 rt-predictor-training/data/
```

### "Invalid parquet file" Error
```bash
# Verify file is valid parquet
python -c "import pandas as pd; df = pd.read_parquet('path/to/file.parquet'); print(df.shape)"
```

### Memory Issues with Large Dataset
```bash
# Use sample for development
python src/train.py --sample-size 100000

# Or increase Docker memory
docker run -m 16g -v $(pwd)/data:/app/data rt-predictor-training
```

## Next Steps

1. **Obtain the data** using one of the methods above
2. **Place in correct location**: `rt-predictor-training/data/raw/`
3. **Run training**: `make train` or `docker-compose --profile training up`
4. **Verify model output**: Check `data/models/` for saved artifacts

## References

- NREL Eagle System: https://www.nrel.gov/hpc/eagle-system.html
- HPC Workload Archives: https://www.cs.huji.ac.il/labs/parallel/workload/
- Parquet Format: https://parquet.apache.org/
