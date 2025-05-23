#!/bin/bash

# Script to generate Python code from Protocol Buffer definitions

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Ensure we're in the project root
cd "$PROJECT_ROOT"

echo "Generating Python code from proto files..."

# Create proto output directory if it doesn't exist
mkdir -p src/proto

# Generate Python code
python -m grpc_tools.protoc \
    -I./src/proto \
    --python_out=./src/proto \
    --grpc_python_out=./src/proto \
    ./src/proto/rt_predictor.proto

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo "✓ Proto files generated successfully!"
    echo "Generated files:"
    ls -la src/proto/*_pb2*.py
else
    echo "✗ Failed to generate proto files"
    exit 1
fi

# Fix imports in generated files (grpc_tools generates absolute imports)
echo "Fixing imports in generated files..."

# Fix the import in the grpc file
if [ -f "src/proto/rt_predictor_pb2_grpc.py" ]; then
    # On macOS, use -i '' for in-place editing
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/import rt_predictor_pb2/from . import rt_predictor_pb2/' src/proto/rt_predictor_pb2_grpc.py
    else
        sed -i 's/import rt_predictor_pb2/from . import rt_predictor_pb2/' src/proto/rt_predictor_pb2_grpc.py
    fi
    echo "✓ Fixed imports"
fi

echo "Proto generation complete!"
