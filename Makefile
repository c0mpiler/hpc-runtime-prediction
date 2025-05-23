# RT Predictor Microservices Makefile

.PHONY: help build train start stop restart logs clean test

# Default target
help:
	@echo "RT Predictor Microservices Management"
	@echo "====================================="
	@echo "make build       - Build all Docker images"
	@echo "make train       - Run training service"
	@echo "make start       - Start all services"
	@echo "make stop        - Stop all services"
	@echo "make restart     - Restart all services"
	@echo "make logs        - Show logs for all services"
	@echo "make clean       - Clean up containers and volumes"
	@echo "make test        - Run all tests"
	@echo "make status      - Show service status"

# Build all images
build:
	@echo "Building all services..."
	docker-compose build

# Run training
train:
	@echo "Running training service..."
	docker-compose --profile training up rt-predictor-training

# Start services
start:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Services started!"
	@echo "UI: http://localhost:8501"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000"

# Stop services
stop:
	@echo "Stopping all services..."
	docker-compose down

# Restart services
restart: stop start

# Show logs
logs:
	docker-compose logs -f

# Clean up
clean:
	@echo "Cleaning up..."
	docker-compose down -v
	@echo "Cleaned up containers and volumes"

# Run tests
test:
	@echo "Running tests..."
	cd rt-predictor-training && python -m pytest tests/
	cd rt-predictor-api && python -m pytest tests/
	cd rt-predictor-ui && python -m pytest tests/

# Show status
status:
	@docker-compose ps
