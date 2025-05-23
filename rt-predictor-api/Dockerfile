# RT Predictor API Service Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Make scripts executable
RUN chmod +x scripts/*.sh

# Generate proto files
RUN python -m grpc_tools.protoc \
    -I./src/proto \
    --python_out=./src/proto \
    --grpc_python_out=./src/proto \
    ./src/proto/rt_predictor.proto

# Fix imports in generated files
RUN sed -i 's/import rt_predictor_pb2/from . import rt_predictor_pb2/' src/proto/rt_predictor_pb2_grpc.py

# Create directories for models and logs
RUN mkdir -p models logs

# Expose ports
EXPOSE 50051 8181

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python scripts/test_client.py --test health || exit 1

# Run the server
CMD ["python", "src/service/server.py"]
