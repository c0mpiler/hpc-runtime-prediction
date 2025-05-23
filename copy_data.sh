#!/bin/bash
# Script to copy Eagle dataset from monolithic RT Predictor to microservices

# Source and destination paths
SOURCE_DIR="../ml/eagle-jobs/data/full-data"
DEST_DIR="rt-predictor-training/data/raw"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "RT Predictor Data Migration Script"
echo "=================================="
echo ""

# Check if source directory exists
if [ -d "$SOURCE_DIR" ]; then
    echo -e "${GREEN}✓ Found Eagle dataset at: $SOURCE_DIR${NC}"
    
    # Create destination directory
    mkdir -p "$DEST_DIR"
    
    # Count parquet files
    FILE_COUNT=$(find "$SOURCE_DIR" -name "*.parquet" | wc -l)
    echo "  Found $FILE_COUNT parquet files"
    
    # Copy files
    echo ""
    echo "Copying files to $DEST_DIR..."
    cp -v "$SOURCE_DIR"/*.parquet "$DEST_DIR/" 2>/dev/null
    
    # Verify copy
    COPIED_COUNT=$(find "$DEST_DIR" -name "*.parquet" | wc -l)
    
    if [ $COPIED_COUNT -gt 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Successfully copied $COPIED_COUNT files${NC}"
        echo ""
        echo "Data is ready for training! Run:"
        echo "  make train"
        echo "or"
        echo "  docker-compose --profile training up rt-predictor-training"
    else
        echo -e "${RED}✗ No files were copied${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Eagle dataset not found at: $SOURCE_DIR${NC}"
    echo ""
    echo "Options:"
    echo "1. Generate synthetic data:"
    echo "   python rt-predictor-training/scripts/generate_synthetic_data.py"
    echo ""
    echo "2. Download the Eagle dataset and place in:"
    echo "   $DEST_DIR"
    echo ""
    echo "See DATA.md for more information."
    exit 1
fi
