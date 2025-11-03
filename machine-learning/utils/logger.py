"""
Logging utility for the forecasting system
"""
import logging
import os
from datetime import datetime
from config.settings import LOGS_DIR

def setup_logger(name='stock_forecast', log_file=None):
    """
    Setup logger with both file and console handlers
    
    Args:
        name: Logger name
        log_file: Log file path (optional)
    
    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = os.path.join(LOGS_DIR, f'forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
