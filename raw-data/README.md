# RT Predictor Dataset

This directory contains the NREL Eagle HPC dataset used for training the RT Predictor models.

## Directory Structure

```
raw-data/
├── full/                        # Full dataset (11M+ records)
│   ├── eagle_data.parquet      # Parquet format (241MB) - RECOMMENDED
│   └── eagle_data.csv.bz2      # Compressed CSV (110MB)
└── sample/                      # Sample datasets for development
    ├── sample_eagle_data.csv    # Small CSV sample (192KB)
    ├── sample_eagle_data.json   # JSON format sample (384KB)
    └── sample_eagle_data.pkl    # Pickle format sample (136KB)
```

## Usage

### For Training
```bash
# From microservices root directory
./copy_data.sh
```

This will copy the appropriate data files to the training service directory.

### For Development
Use the sample data for quick development iterations:
```bash
cd rt-predictor-training
python src/train.py --data-path ../raw-data/sample/sample_eagle_data.csv --sample-size 1000
```

## Data Format

See [DATA.md](../DATA.md) for detailed schema information.

## Git LFS

These files are tracked using Git LFS. Ensure you have Git LFS installed:
```bash
git lfs install
git lfs pull
```
