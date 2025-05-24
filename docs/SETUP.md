# Environment Setup Guide

This guide helps you set up your development environment for the RT Predictor Microservices project.

## Required Environment Variables

### 1. DEV - Development Directory Path

The `$DEV` environment variable should point to your local development directory where the project is cloned.

#### Setting up DEV (Choose one method):

**Option A - Temporary (current session only):**
```bash
export DEV="/path/to/your/development/directory"
```

**Option B - Permanent (add to shell profile):**

For bash (~/.bashrc or ~/.bash_profile):
```bash
echo 'export DEV="/path/to/your/development/directory"' >> ~/.bashrc
source ~/.bashrc
```

For zsh (~/.zshrc):
```bash
echo 'export DEV="/path/to/your/development/directory"' >> ~/.zshrc
source ~/.zshrc
```

**Option C - Using direnv (recommended for project-specific variables):**
```bash
# Install direnv first
brew install direnv  # macOS
# or
sudo apt-get install direnv  # Ubuntu/Debian

# Create .envrc in project root
echo 'export DEV="$(pwd)/.."' > .envrc
direnv allow
```

### 2. Verify Setup

After setting up, verify the environment variable:
```bash
echo $DEV
# Should output: /path/to/your/development/directory

# Navigate to project
cd $DEV/rt-predictor/microservices
```

## Optional Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` to customize:
- `RT_PREDICTOR_API_HOST` - API service hostname
- `RT_PREDICTOR_API_PORT` - API service port
- `STREAMLIT_SERVER_PORT` - UI service port
- `PROMETHEUS_PORT` - Prometheus metrics port
- `GRAFANA_PORT` - Grafana dashboard port

## Docker Environment

When using Docker Compose, most environment variables are pre-configured. You only need to set `$DEV` if you're referencing local paths in scripts or documentation.

## Troubleshooting

1. **"$DEV: unbound variable" error**
   - Ensure you've exported the DEV variable
   - Check if you're in the correct shell where it was set

2. **Path not found errors**
   - Verify $DEV points to the correct directory
   - Ensure the full path exists: `ls $DEV/rt-predictor/microservices`

3. **Permission denied**
   - Check directory permissions
   - Ensure Docker has access to the directories

## Next Steps

After setting up the environment:
1. Continue with the main [README.md](README.md) instructions
2. Run the training service to generate models
3. Start all services with Docker Compose
