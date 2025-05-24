#!/bin/bash
# Test script to verify the complete setup works from scratch

set -e

echo "Testing RT Predictor Microservices Setup"
echo "========================================"
echo ""

# Test 1: Check if all required files exist
echo "Test 1: Checking required files..."
required_files=(
    "docker-compose.yml"
    "Makefile"
    "quickstart.sh"
    ".env.example"
    "SETUP.md"
    "README.md"
    "CHANGELOG.md"
    "rt-predictor-training/Dockerfile"
    "rt-predictor-api/Dockerfile"
    "rt-predictor-ui/Dockerfile"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file is missing"
        exit 1
    fi
done
echo ""

# Test 2: Check if proto files are set up correctly
echo "Test 2: Checking proto setup..."
if [ -f "rt-predictor-ui/src/proto/__init__.py" ]; then
    echo "✅ UI proto __init__.py exists"
else
    echo "❌ UI proto __init__.py is missing"
    exit 1
fi
echo ""

# Test 3: Check Makefile targets
echo "Test 3: Testing Makefile targets..."
make help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Makefile is valid"
else
    echo "❌ Makefile has errors"
    exit 1
fi
echo ""

echo "All tests passed! ✅"
echo ""
echo "To run the complete setup, use:"
echo "  ./quickstart.sh"
echo "or"
echo "  make fresh-start"
