# RT Predictor Microservices Makefile

.PHONY: help setup build train start stop restart logs clean clean-all test status fresh-start dev-setup

# Default target
help:
	@echo "RT Predictor Microservices Management"
	@echo "====================================="
	@echo "make setup       - Initial setup (check prerequisites, prepare data)"
	@echo "make build       - Build all Docker images"
	@echo "make train       - Run training service"
	@echo "make start       - Start all services"
	@echo "make stop        - Stop all services"
	@echo "make restart     - Restart all services"
	@echo "make logs        - Show logs for all services"
	@echo "make clean       - Clean up containers, volumes, and networks"
	@echo "make clean-all   - Deep clean including Docker system prune"
	@echo "make test        - Run all tests"
	@echo "make status      - Show service status"
	@echo "make fresh-start - Complete setup from scratch (clean + setup + train + start)"
	@echo ""
	@echo "M2 Max Optimization Commands (Apple Silicon):"
	@echo "============================================="
	@echo "make train-m2max       - Run optimized training (2-3x faster)"
	@echo "make start-m2max       - Start services with resource limits"
	@echo "make fresh-start-m2max - Complete setup with M2 Max optimization"

# Initial setup
setup:
	@echo "Setting up RT Predictor Microservices..."
	@echo "Checking Docker..."
	@docker --version || (echo "Docker not installed" && exit 1)
	@docker-compose --version || (echo "Docker Compose not installed" && exit 1)
	@echo "Checking data..."
	@if [ ! -f rt-predictor-training/data/raw/eagle_data.parquet ]; then \
		echo "Setting up training data..."; \
		if [ -f copy_data.sh ]; then \
			chmod +x copy_data.sh && ./copy_data.sh || echo "Warning: Could not copy data. You may need to run 'git lfs pull' first."; \
		else \
			echo "Warning: copy_data.sh not found. Please ensure training data is in rt-predictor-training/data/raw/"; \
		fi; \
	fi
	@echo "Creating .env file if not exists..."
	@if [ ! -f .env ]; then cp .env.example .env 2>/dev/null || echo "Warning: .env.example not found"; fi
	@# Ensure proto files are in place for UI service
	@if [ -f rt-predictor-api/src/proto/rt_predictor.proto ]; then \
		mkdir -p rt-predictor-ui/src/proto; \
		cp rt-predictor-api/src/proto/rt_predictor.proto rt-predictor-ui/src/proto/ 2>/dev/null || true; \
		echo "# RT Predictor Protocol Buffer Definitions" > rt-predictor-ui/src/proto/__init__.py; \
	fi
	@echo "Setup complete!"

# Build all images
build:
	@echo "Building all services..."
	@# Ensure proto file is copied before build
	@if [ -f rt-predictor-api/src/proto/rt_predictor.proto ]; then \
		mkdir -p rt-predictor-ui/src/proto; \
		cp rt-predictor-api/src/proto/rt_predictor.proto rt-predictor-ui/src/proto/ 2>/dev/null || true; \
	fi
	docker-compose build --no-cache

# Run training
train:
	@echo "Running training service..."
	@echo "This will train models on the Eagle dataset (may take 5-10 minutes)..."
	@# Ensure clean state for training
	@docker-compose --profile training down --remove-orphans 2>/dev/null || true
	docker-compose --profile training up rt-predictor-training

# Start services
start:
	@echo "Starting all services..."
	@# Ensure clean state
	@docker-compose down --remove-orphans 2>/dev/null || true
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
	docker-compose down --remove-orphans

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
	@docker-compose down -v --remove-orphans 2>/dev/null || true
	@# Remove any orphaned networks
	@docker network ls | grep microservices | awk '{print $1}' | xargs -r docker network rm 2>/dev/null || true
	@echo "Cleaned up containers, volumes, and networks"

# Deep clean
clean-all: clean
	@echo "Performing deep clean..."
	@docker system prune -f
	@echo "Deep clean complete"

# Run tests
test:
	@echo "Running tests..."
	@echo "Tests should be run inside containers or with proper environment setup"
	@echo "To run tests in containers:"
	@echo "  docker-compose run --rm rt-predictor-training pytest tests/"
	@echo "  docker-compose run --rm rt-predictor-api pytest tests/"
	@echo "  docker-compose run --rm rt-predictor-ui pytest tests/"

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
	@make clean-all
	@make setup
	@make build
	@make train
	@make start
	@echo "\nFresh setup complete! All services are running."

# M2 Max optimized training
train-m2max:
	@echo "Running M2 Max optimized training..."
	@echo "Using 10 CPU cores and up to 48GB RAM..."
	@# Copy optimized config
	@cp rt-predictor-training/configs/config.m2max.toml rt-predictor-training/configs/config.toml
	@# Ensure clean state for training
	@docker-compose --profile training down --remove-orphans 2>/dev/null || true
	docker-compose -f docker-compose.m2max.yml --profile training up rt-predictor-training

# M2 Max optimized start
start-m2max:
	@echo "Starting services with M2 Max optimization..."
	@docker-compose down --remove-orphans 2>/dev/null || true
	docker-compose -f docker-compose.m2max.yml up -d
	@sleep 5
	@echo "\nServices started with M2 Max optimization!"
	@echo "=================="
	@echo "UI: http://localhost:8501"
	@echo "API Metrics: http://localhost:8181/metrics"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "=================="
	@make status

# Fresh start with M2 Max optimization
fresh-start-m2max:
	@echo "Starting fresh setup with M2 Max optimization..."
	@make clean-all
	@make setup
	@make build
	@make train-m2max
	@make start-m2max
	@echo "\nFresh setup complete with M2 Max optimization!"

# Development setup (for local development)
dev-setup:
	@echo "Setting up development environment..."
	@echo "Creating Python virtual environments..."
	@cd rt-predictor-training && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
	@cd rt-predictor-api && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
	@cd rt-predictor-ui && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
	@echo "Development setup complete!"
