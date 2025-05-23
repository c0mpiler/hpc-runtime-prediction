#!/usr/bin/env python3
"""
Unit tests for RT Predictor API Service
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class TestAPIService:
    """Test API service components."""
    
    def test_imports(self):
        """Test that all modules can be imported."""
        try:
            from service import server
            from service import predictor
            from utils import config
            from utils import logger
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_config_loading(self):
        """Test configuration loading."""
        from utils.config import Config
        
        config = Config()
        assert config is not None
        assert hasattr(config, 'get')
    
    def test_logger_creation(self):
        """Test logger creation."""
        from utils.logger import setup_logger
        
        logger = setup_logger('test')
        assert logger is not None
        logger.info("Test log message")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
