#!/bin/bash
# Enable enhanced server for RT Predictor API

echo "Enabling enhanced RT Predictor API server..."

# Backup original server
if [ ! -f "src/service/server_original.py" ]; then
    cp src/service/server.py src/service/server_original.py
    echo "✓ Backed up original server to server_original.py"
fi

# Copy enhanced server
cp src/service/server_enhanced.py src/service/server.py
echo "✓ Enabled enhanced server with:"
echo "  - Connection pooling and optimized gRPC settings"
echo "  - Request caching (LRU with TTL)"
echo "  - Circuit breaker pattern for fault tolerance"
echo "  - Retry logic with exponential backoff"
echo "  - Enhanced Prometheus metrics"
echo "  - Request validation and parallel batch processing"

echo ""
echo "To revert to original server, run:"
echo "  cp src/service/server_original.py src/service/server.py"
echo ""
echo "Rebuild and restart the API service:"
echo "  docker-compose build rt-predictor-api"
echo "  docker-compose restart rt-predictor-api"
