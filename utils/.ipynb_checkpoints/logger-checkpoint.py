"""
Logger System for Chaotic Speaker Recognition Project
Provides unified logging functionality with file and console output,
configurable log levels, and experiment-specific log management.
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

def setup_logger(
    name: str, 
    log_file: Optional[str] = None, 
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
    
class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m'   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            log_message = f"{self.COLORS[record.levelname]}{log_message}{self.RESET}"
        return log_message


class ExperimentLogger:
    """
    Centralized logging system for chaotic speaker recognition experiments
    Supports multiple output streams, structured logging, and experiment tracking
    """
    
    def __init__(self, 
                 name: str = "chaotic_sr", 
                 log_dir: str = "logs",
                 experiment_name: Optional[str] = None,
                 console_level: str = "INFO",
                 file_level: str = "DEBUG",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Initialize the experiment logger
        
        Args:
            name: Logger name
            log_dir: Directory to store log files
            experiment_name: Specific experiment identifier
            console_level: Console logging level
            file_level: File logging level
            max_file_size: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.console_level = getattr(logging, console_level.upper())
        self.file_level = getattr(logging, file_level.upper())
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_log_dir = self.log_dir / self.experiment_name
        self.experiment_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplication
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handlers()
        
        # Log experiment start
        self.info(f"Experiment logger initialized: {self.experiment_name}")
        self.info(f"Log directory: {self.experiment_log_dir}")
    
    def _setup_console_handler(self):
        """Setup colored console handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        
        # Use colored formatter for console
        console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        console_formatter = ColoredFormatter(
            fmt=console_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self):
        """Setup file handlers for different log types"""
        # Main log file with rotation
        main_log_file = self.experiment_log_dir / "main.log"
        main_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        main_handler.setLevel(self.file_level)
        
        # Error log file
        error_log_file = self.experiment_log_dir / "error.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        # File formatter (without colors)
        file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        file_formatter = logging.Formatter(
            fmt=file_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        main_handler.setFormatter(file_formatter)
        error_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(main_handler)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, extra=kwargs)
    
    def log_experiment_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        config_file = self.experiment_log_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.info("Experiment configuration saved")
        self.info(f"Config: {json.dumps(config, indent=2, default=str)}")
    
    def log_metrics(self, metrics: Dict[str, Any], epoch: Optional[int] = None):
        """Log training/evaluation metrics"""
        metrics_file = self.experiment_log_dir / "metrics.jsonl"
        
        # Add timestamp and epoch
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            **metrics
        }
        
        # Append to JSONL file
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, default=str) + '\n')
        
        # Also log to console
        metric_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        epoch_str = f"Epoch {epoch} | " if epoch is not None else ""
        self.info(f"{epoch_str}Metrics: {metric_str}")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model architecture and parameters"""
        model_file = self.experiment_log_dir / "model_info.json"
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        self.info("Model information saved")
        self.info(f"Model info: {json.dumps(model_info, indent=2, default=str)}")
    
    def log_system_info(self):
        """Log system information"""
        import platform
        import psutil
        import torch
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            system_info["cuda_version"] = torch.version.cuda
            system_info["gpu_names"] = [torch.cuda.get_device_name(i) 
                                       for i in range(torch.cuda.device_count())]
        
        system_file = self.experiment_log_dir / "system_info.json"
        with open(system_file, 'w', encoding='utf-8') as f:
            json.dump(system_info, f, indent=2, default=str)
        
        self.info("System information logged")
        for key, value in system_info.items():
            self.info(f"System {key}: {value}")
    
    def create_child_logger(self, child_name: str) -> logging.Logger:
        """Create a child logger for specific components"""
        child_logger_name = f"{self.name}.{child_name}"
        child_logger = logging.getLogger(child_logger_name)
        return child_logger
    
    def close(self):
        """Close all handlers and cleanup"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# Global logger instance
_global_logger: Optional[ExperimentLogger] = None


def get_logger(name: str = "chaotic_sr") -> ExperimentLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ExperimentLogger(name=name)
    return _global_logger


def setup_experiment_logger(experiment_name: str, 
                          log_dir: str = "logs",
                          console_level: str = "INFO",
                          file_level: str = "DEBUG") -> ExperimentLogger:
    """Setup a new experiment logger"""
    global _global_logger
    _global_logger = ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level
    )
    return _global_logger


def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.debug(f"Calling function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {str(e)}")
            raise
    return wrapper


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    logger = setup_experiment_logger("test_experiment")
    
    # Log system information
    logger.log_system_info()
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test configuration logging
    test_config = {
        "model_type": "chaotic_network",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
    logger.log_experiment_config(test_config)
    
    # Test metrics logging
    test_metrics = {
        "train_loss": 0.123,
        "val_loss": 0.456,
        "accuracy": 0.789
    }
    logger.log_metrics(test_metrics, epoch=1)
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.exception("Caught an exception during testing")
    
    logger.info("Logger testing completed")
    logger.close()