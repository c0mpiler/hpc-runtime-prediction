FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
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
    ./src/proto/rt_predictor.proto && \
    sed -i 's/import rt_predictor_pb2/from . import rt_predictor_pb2/g' src/proto/rt_predictor_pb2_grpc.py

# Create necessary directories
RUN mkdir -p /app/logs /app/.streamlit

# Streamlit configuration
RUN echo '\
[server]\n\
port = 8501\n\
address = "0.0.0.0"\n\
headless = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /app/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application with PYTHONPATH set
CMD ["sh", "-c", "PYTHONPATH=/app/src:$PYTHONPATH streamlit run src/app.py --server.maxUploadSize=100"]
