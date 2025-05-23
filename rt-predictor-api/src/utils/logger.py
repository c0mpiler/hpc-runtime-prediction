"""Logging configuration for RT Predictor API Service."""

import logging
import structlog
import sys
from typing import Optional


def setup_logger(name: str = 'rt_predictor_api', 
                 level: str = 'INFO',
                 log_format: str = 'json') -> logging.Logger:
    """Setup structured logging.
    
    Args:
        name: Logger name
        level: Logging level
        log_format: Output format ('json' or 'console')
        
    Returns:
        Configured logger
    """
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if log_format == 'json':
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Get logger
    logger = logging.getLogger(name)
    
    return logger


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger
    """
    return structlog.get_logger(name)
