#!/bin/bash
# M2 Max Optimization Setup Script

set -e

echo "🚀 M2 Max Optimization for RT Predictor"
echo "======================================="
echo ""
echo "This will configure the system to use:"
echo "- 10 CPU cores (out of 12 available)"
echo "- Up to 48GB RAM (out of 64GB available)"
echo "- Optimized batch sizes and parallelism"
echo ""

# Stop current training if running
echo "Stopping current training (if any)..."
docker stop rt-predictor-training 2>/dev/null || true

# Apply optimizations
echo "Applying M2 Max optimizations..."

# Backup current config
cp rt-predictor-training/configs/config.toml rt-predictor-training/configs/config.toml.backup 2>/dev/null || true

# Use optimized config
cp rt-predictor-training/configs/config.m2max.toml rt-predictor-training/configs/config.toml

echo ""
echo "✅ Configuration updated!"
echo ""
echo "To start training with M2 Max optimization, run:"
echo "  make train-m2max"
echo ""
echo "For a complete fresh start with optimization:"
echo "  make fresh-start-m2max"
echo ""
echo "To use these optimizations by default, you can also:"
echo "  export COMPOSE_FILE=docker-compose.m2max.yml"
echo ""
