#!/bin/bash
# Script to copy Eagle dataset to training service

# Source and destination paths
SOURCE_DIR="raw-data"  # Data is now within microservices directory
DEST_DIR="rt-predictor-training/data/raw"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "RT Predictor Data Setup Script"
echo "=============================="
echo ""

# Function to copy data files
copy_data() {
    local source_type=$1
    local pattern=$2
    
    echo "Checking $source_type data..."
    
    if [ -d "$SOURCE_DIR/$source_type" ]; then
        # Create destination directory
        mkdir -p "$DEST_DIR"
        
        # Count available files
        local available=$(find "$SOURCE_DIR/$source_type" -name "$pattern" | wc -l)
        
        if [ $available -gt 0 ]; then
            echo "  Found $available $pattern file(s)"
            # Copy files matching pattern
            find "$SOURCE_DIR/$source_type" -name "$pattern" -exec cp -v {} "$DEST_DIR/" \;
            
            # Count copied files
            local copied=$(find "$DEST_DIR" -name "$pattern" | wc -l)
            if [ $copied -gt 0 ]; then
                echo -e "  ${GREEN}✓ Copied $copied file(s)${NC}"
                return 0
            fi
        fi
    fi
    return 1
}

# Check if source directory exists
if [ -d "$SOURCE_DIR" ]; then
    echo -e "${GREEN}✓ Found data directory at: $SOURCE_DIR${NC}"
    echo ""
    
    # Try to copy full dataset first (prefer parquet)
    if copy_data "full" "*.parquet"; then
        echo ""
        echo -e "${GREEN}✓ Full dataset ready for training!${NC}"
        DATASET_TYPE="full"
    # Try compressed CSV if no parquet
    elif copy_data "full" "*.csv.bz2"; then
        echo ""
        echo -e "${GREEN}✓ Full dataset (compressed CSV) ready!${NC}"
        echo -e "${YELLOW}Note: Consider converting to parquet for better performance${NC}"
        DATASET_TYPE="full-csv"
    # If no full dataset, try sample
    elif copy_data "sample" "*.csv"; then
        echo ""
        echo -e "${YELLOW}⚠ Using sample dataset (for development/testing)${NC}"
        DATASET_TYPE="sample"
    else
        echo -e "${RED}✗ No suitable data files found${NC}"
        exit 1
    fi
    
    echo ""
    echo "Next steps:"
    echo "1. Run training:"
    echo "   make train"
    echo "   # or"
    echo "   docker-compose --profile training up rt-predictor-training"
    echo ""
    echo "2. For development with sample data:"
    echo "   cd rt-predictor-training"
    echo "   python src/train.py --sample-size 10000"
    
else
    echo -e "${RED}✗ Data directory not found at: $SOURCE_DIR${NC}"
    echo ""
    echo "Please ensure the raw-data directory exists with:"
    echo "- raw-data/full/eagle_data.parquet (or .csv.bz2)"
    echo "- raw-data/sample/sample_eagle_data.csv"
    echo ""
    echo "Alternative: Generate synthetic data:"
    echo "  python rt-predictor-training/scripts/generate_synthetic_data.py"
    exit 1
fi
