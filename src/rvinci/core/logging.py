import logging
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger with the specified name.
    
    Args:
        name: The name of the logger, typically __name__.
        level: Optional logging level override.
        
    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, assume it's configured
    if logger.handlers:
        return logger
        
    # Default configuration if not configured
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    if level is not None:
        logger.setLevel(level)
    else:
        # Default to INFO if not set
        logger.setLevel(logging.INFO)
        
    return logger
