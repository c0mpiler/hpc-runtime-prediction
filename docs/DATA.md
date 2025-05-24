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

### Repository Data Structure

The Eagle dataset is included in the repository using Git LFS:

```
microservices/raw-data/
├── full/
│   ├── eagle_data.parquet      # Full dataset (241MB)
│   └── eagle_data.csv.bz2      # Compressed CSV (110MB)
└── sample/
    ├── sample_eagle_data.csv   # Sample for development (192KB)
    ├── sample_eagle_data.json  # JSON format sample
    └── sample_eagle_data.pkl   # Pickle format sample
```

### Training Data Setup

1. **Clone with Git LFS**:
   ```bash
   # Ensure Git LFS is installed
   git lfs install
   
   # Clone repository (will download LFS files)
   git clone <repository-url>
   
   # Or if already cloned, pull LFS files
   git lfs pull
   ```

2. **Copy data to training directory**:
   ```bash
   # Run the data setup script
   ./copy_data.sh
   ```

3. **Docker volume mapping**:
   ```bash
   # The docker-compose.yml handles this automatically
   docker-compose --profile training up rt-predictor-training
   ```

## Data Preparation Options

### Option 1: Use Provided Data
The repository includes the Eagle dataset:

```bash
# Navigate to microservices directory
cd rt-predictor-microservices

# Copy data to training directory
./copy_data.sh
```

### Option 2: Generate Synthetic Data
For development or testing without the full dataset:

```bash
# Generate synthetic data
python rt-predictor-training/scripts/generate_synthetic_data.py \
    --rows 100000 \
    --output rt-predictor-training/data/raw/eagle_data.parquet
```

The synthetic data generator creates realistic HPC job patterns including:
- Power-law user distribution
- Temporal patterns in job submissions
- Correlated resource requirements
- GPU job distribution
- Realistic runtime efficiencies

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

## Working with Different Data Formats

### Parquet Files (Recommended)
```python
import pandas as pd
df = pd.read_parquet('raw-data/full/eagle_data.parquet')
```

### Compressed CSV Files
```python
import pandas as pd
df = pd.read_csv('raw-data/full/eagle_data.csv.bz2', compression='bz2')
# Consider converting to parquet for better performance
df.to_parquet('eagle_data.parquet')
```

### Sample Data
```python
# For quick development iterations
df = pd.read_csv('raw-data/sample/sample_eagle_data.csv')
```

## Privacy and Security

- User IDs are anonymized in the dataset
- No personally identifiable information (PII)
- Job names may contain script names but no sensitive data
- Account names are project identifiers

## Troubleshooting

### Git LFS Issues
```bash
# Check LFS status
git lfs status

# Re-download LFS files
git lfs fetch --all
git lfs checkout
```

### "File not found" Error
```bash
# Check if data exists
ls -la raw-data/

# Run data setup
./copy_data.sh

# Verify training data directory
ls -la rt-predictor-training/data/raw/
```

### Memory Issues with Large Dataset
```bash
# Use sample for development
cd rt-predictor-training
python src/train.py --sample-size 100000

# Or increase Docker memory
docker run -m 16g rt-predictor-training
```

### Converting Between Formats
```python
# CSV to Parquet
import pandas as pd
df = pd.read_csv('data.csv')
df.to_parquet('data.parquet')

# Compressed CSV to Parquet
df = pd.read_csv('data.csv.bz2', compression='bz2')
df.to_parquet('data.parquet')
```

## Performance Tips

1. **Use Parquet Format**: 2-3x faster loading than CSV
2. **Sample During Development**: Use `--sample-size` flag
3. **Enable Compression**: Parquet has built-in compression
4. **Batch Processing**: Process data in chunks for large files

## Next Steps

1. **Setup data**: Run `./copy_data.sh`
2. **Verify data**: Check `rt-predictor-training/data/raw/`
3. **Run training**: `make train` or use docker-compose
4. **Monitor progress**: Check logs in `rt-predictor-training/logs/`

## References

- NREL Eagle System: https://www.nrel.gov/hpc/eagle-system.html
- HPC Workload Archives: https://www.cs.huji.ac.il/labs/parallel/workload/
- Parquet Format: https://parquet.apache.org/
- Git LFS: https://git-lfs.github.com/
