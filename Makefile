# RT Predictor Microservices Makefile

.PHONY: help setup build train start stop restart logs clean test status proto-gen dev-setup fresh-start

# Default target
help:
	@echo "RT Predictor Microservices Management"
	@echo "====================================="
	@echo "make setup       - Initial setup (install dependencies, generate protos)"
	@echo "make build       - Build all Docker images"
	@echo "make train       - Run training service"
	@echo "make start       - Start all services"
	@echo "make stop        - Stop all services"
	@echo "make restart     - Restart all services"
	@echo "make logs        - Show logs for all services"
	@echo "make clean       - Clean up containers and volumes"
	@echo "make test        - Run all tests"
	@echo "make status      - Show service status"
	@echo "make fresh-start - Complete setup from scratch (clean + setup + train + start)"
	@echo "make proto-gen   - Generate protobuf files for all services"

# Initial setup
setup: proto-gen
	@echo "Setting up RT Predictor Microservices..."
	@echo "Checking Docker..."
	@docker --version || (echo "Docker not installed" && exit 1)
	@docker-compose --version || (echo "Docker Compose not installed" && exit 1)
	@echo "Checking data..."
	@if [ ! -f rt-predictor-training/data/raw/eagle_data.parquet ]; then \
		echo "Setting up training data..."; \
		./copy_data.sh || echo "Warning: Could not copy data. You may need to run 'git lfs pull' first."; \
	fi
	@echo "Creating .env file if not exists..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "Setup complete!"

# Generate protobuf files
proto-gen:
	@echo "Generating protobuf files..."
	@# Generate for API service (if running locally)
	@if [ -f rt-predictor-api/scripts/generate_proto.sh ]; then \
		cd rt-predictor-api && chmod +x scripts/generate_proto.sh && ./scripts/generate_proto.sh || true; \
	fi
	@# Copy proto file to UI service
	@cp rt-predictor-api/src/proto/rt_predictor.proto rt-predictor-ui/src/proto/
	@# Create __init__.py for UI proto module
	@echo "# RT Predictor Protocol Buffer Definitions" > rt-predictor-ui/src/proto/__init__.py
	@echo "Proto files prepared!"

# Build all images
build: proto-gen
	@echo "Building all services..."
	docker-compose build

# Run training
train:
	@echo "Running training service..."
	@echo "This will train models on the Eagle dataset (may take 5-10 minutes)..."
	docker-compose --profile training up rt-predictor-training

# Start services
start:
	@echo "Starting all services..."
	docker-compose up -d
	@sleep 5
	@echo "\nServices started!"
	@echo "=================="
	@echo "UI: http://localhost:8501"
	@echo "API Metrics: http://localhost:8181/metrics"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "=================="
	@make status

# Stop services
stop:
	@echo "Stopping all services..."
	docker-compose down

# Restart services
restart: stop start

# Show logs
logs:
	docker-compose logs -f

# Show specific service logs
logs-%:
	docker-compose logs -f rt-predictor-$*

# Clean up
clean:
	@echo "Cleaning up..."
	docker-compose down -v
	@echo "Cleaned up containers and volumes"

# Run tests
test:
	@echo "Running tests..."
	@if [ -d rt-predictor-training/venv ]; then \
		cd rt-predictor-training && source venv/bin/activate && python -m pytest tests/ -v; \
	else \
		echo "Training service tests skipped (no venv found)"; \
	fi
	@if [ -d rt-predictor-api/venv ]; then \
		cd rt-predictor-api && source venv/bin/activate && python -m pytest tests/ -v; \
	else \
		echo "API service tests skipped (no venv found)"; \
	fi
	@if [ -d rt-predictor-ui/venv ]; then \
		cd rt-predictor-ui && source venv/bin/activate && python -m pytest tests/ -v; \
	else \
		echo "UI service tests skipped (no venv found)"; \
	fi

# Show status
status:
	@echo "\nService Status:"
	@echo "==============="
	@docker-compose ps
	@echo "\nVolumes:"
	@docker volume ls | grep microservices || echo "No volumes found"
	@echo "\nNetworks:"
	@docker network ls | grep microservices || echo "No networks found"

# Fresh start from scratch
fresh-start:
	@echo "Starting fresh setup..."
	@make clean
	@make setup
	@make build
	@make train
	@make start
	@echo "\nFresh setup complete! All services are running."

# Development setup (for local development)
dev-setup:
	@echo "Setting up development environment..."
	@echo "Creating Python virtual environments..."
	@cd rt-predictor-training && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
	@cd rt-predictor-api && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
	@cd rt-predictor-ui && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
	@echo "Development setup complete!"
