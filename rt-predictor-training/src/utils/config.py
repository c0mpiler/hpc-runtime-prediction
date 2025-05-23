"""Configuration management for RT Predictor training service."""

import toml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager for training service."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration from TOML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.toml"
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        self._config = toml.load(self.config_path)
        
        # Convert relative paths to absolute
        self._resolve_paths()
    
    def _resolve_paths(self):
        """Convert relative paths in config to absolute paths."""
        base_dir = self.config_path.parent.parent
        
        # Resolve data paths
        if 'data' in self._config:
            for key, value in self._config['data'].items():
                if 'path' in key and isinstance(value, str):
                    path = Path(value)
                    if not path.is_absolute():
                        self._config['data'][key] = str(base_dir / path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
