"""Configuration management for RT Predictor API Service."""

import os
import toml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from TOML file and environment variables.
    
    Args:
        config_path: Path to config file. If None, uses CONFIG_PATH env var or default.
        
    Returns:
        Configuration dictionary
    """
    # Determine config path
    if config_path is None:
        config_path = os.getenv('CONFIG_PATH', 'configs/config.toml')
    
    config_path = Path(config_path)
    
    # Load base configuration
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = toml.load(f)
    else:
        # Default configuration
        config = {
            'server': {
                'port': 50051,
                'max_workers': 10,
                'max_message_length': 10 * 1024 * 1024,  # 10MB
                'metrics_port': 8181
            },
            'model': {
                'path': 'models/production'
            },
            'features': {
                'optimization': {
                    'chunk_size': 100000,
                    'enable_caching': True,
                    'n_jobs': -1
                }
            },
            'logging': {
                'level': 'INFO',
                'format': 'json'
            }
        }
    
    # Override with environment variables
    if os.getenv('GRPC_PORT'):
        config['server']['port'] = int(os.getenv('GRPC_PORT'))
    
    if os.getenv('MODEL_PATH'):
        config['model']['path'] = os.getenv('MODEL_PATH')
    
    if os.getenv('METRICS_PORT'):
        config['server']['metrics_port'] = int(os.getenv('METRICS_PORT'))
    
    if os.getenv('LOG_LEVEL'):
        config['logging']['level'] = os.getenv('LOG_LEVEL')
    
    return config


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model-specific configuration."""
    return config.get('model', {})


def get_server_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract server-specific configuration."""
    return config.get('server', {})
