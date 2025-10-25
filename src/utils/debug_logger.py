"""
Debug logging system for Auto-Wall that outputs to both console and files.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler

class DebugLogger:
    def __init__(self, name="auto_wall", level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler - always show debug output in terminal
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler - rotating log file
        try:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, "debug.log")
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1024*1024, backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.warning(f"Could not set up file logging: {e}")
    
    def debug(self, message):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)

# Global logger instance
debug_logger = DebugLogger()

def log_debug(message):
    """Convenience function for debug logging."""
    debug_logger.debug(message)

def log_info(message):
    """Convenience function for info logging."""
    debug_logger.info(message)

def log_warning(message):
    """Convenience function for warning logging."""
    debug_logger.warning(message)

def log_error(message):
    """Convenience function for error logging."""
    debug_logger.error(message)