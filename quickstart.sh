#!/bin/bash
# RT Predictor Microservices - Quick Start Script
# This script sets up and runs the entire system from scratch

set -e  # Exit on error

echo "============================================="
echo "RT Predictor Microservices - Quick Start"
echo "============================================="
echo ""

# Check if DEV environment variable is set
if [ -z "$DEV" ]; then
    echo "Setting DEV environment variable to parent directory..."
    export DEV="$(cd "$(dirname "$0")/../.." && pwd)"
    echo "DEV set to: $DEV"
fi

# Check prerequisites
echo "Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed. Aborting." >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "❌ Git is required but not installed. Aborting." >&2; exit 1; }

echo "✅ All prerequisites installed"
echo ""

# Clean up any existing Docker resources
echo "Cleaning up any existing resources..."
docker-compose down -v --remove-orphans 2>/dev/null || true
# Remove orphaned networks
docker network ls | grep microservices | awk '{print $1}' | xargs -r docker network rm 2>/dev/null || true
echo "✅ Cleanup complete"
echo ""

# Check if Git LFS is installed and pull data
echo "Checking for Git LFS data..."
if command -v git-lfs >/dev/null 2>&1; then
    echo "Pulling Git LFS data..."
    git lfs pull
else
    echo "⚠️  Git LFS not installed. You may need to download data manually."
fi
echo ""

# Run the fresh start
echo "Starting complete setup..."
echo "This will:"
echo "1. Clean any existing containers/volumes"
echo "2. Set up the environment"
echo "3. Build all Docker images"
echo "4. Train ML models (~5-10 minutes)"
echo "5. Start all services"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Use make commands for consistency
    echo "Running fresh start..."
    make fresh-start
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "============================================="
        echo "✅ Setup Complete!"
        echo "============================================="
        echo ""
        echo "Access points:"
        echo "- UI: http://localhost:8501"
        echo "- API Metrics: http://localhost:8181/metrics"
        echo "- Prometheus: http://localhost:9090"
        echo "- Grafana: http://localhost:3000 (admin/admin)"
        echo ""
        echo "Useful commands:"
        echo "- View logs: make logs"
        echo "- Stop services: make stop"
        echo "- Restart services: make restart"
        echo "- Check status: make status"
        echo ""
    else
        echo ""
        echo "❌ Setup failed. Please check the errors above."
        echo "Common issues:"
        echo "1. Docker daemon not running"
        echo "2. Port conflicts (8501, 50051, 8181, 9090, 3000)"
        echo "3. Insufficient disk space"
        echo ""
        echo "To retry: ./quickstart.sh"
    fi
else
    echo "Setup cancelled."
fi
