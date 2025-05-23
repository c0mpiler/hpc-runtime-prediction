#!/bin/bash
# Generate Python code from proto file

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Create output directory
mkdir -p src/proto

# Generate Python code
python -m grpc_tools.protoc \
    -I../rt-predictor-api/src/proto \
    --python_out=src/proto \
    --grpc_python_out=src/proto \
    ../rt-predictor-api/src/proto/rt_predictor.proto

# Fix imports in generated files
sed -i '' 's/import rt_predictor_pb2/from . import rt_predictor_pb2/g' src/proto/rt_predictor_pb2_grpc.py

echo "Proto files generated successfully!"
