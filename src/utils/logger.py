"""
Logging utilities for InfraOpt simulation.

This module provides centralized logging configuration and utilities
for the simulation platform.
"""

import logging
import sys
from typing import Optional
from datetime import datetime
import os


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        log_format: Format string for log messages
    """
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("pandas").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class SimulationLogger:
    """Custom logger for simulation-specific logging."""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        """
        Initialize simulation logger.
        
        Args:
            name: Logger name
            log_file: Optional log file path
        """
        self.logger = get_logger(name)
        self.log_file = log_file
        self.start_time = datetime.now()
        
        if log_file:
            # Create file handler for simulation logs
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_simulation_start(self, config: dict) -> None:
        """Log simulation start with configuration."""
        self.logger.info("=" * 60)
        self.logger.info("INFRAOPT SIMULATION STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Start time: {self.start_time}")
        self.logger.info(f"Configuration: {config}")
    
    def log_simulation_end(self, results: dict) -> None:
        """Log simulation end with results summary."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.logger.info("=" * 60)
        self.logger.info("INFRAOPT SIMULATION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"End time: {end_time}")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Results summary: {results}")
    
    def log_optimization_step(self, step: str, metrics: dict) -> None:
        """Log optimization step with metrics."""
        self.logger.info(f"Optimization step: {step}")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_risk_assessment(self, risk_type: str, metrics: dict) -> None:
        """Log risk assessment results."""
        self.logger.info(f"Risk assessment - {risk_type}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log error with context."""
        self.logger.error(f"Error in {context}: {str(error)}")
        self.logger.error(f"Error type: {type(error).__name__}")
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log performance metric."""
        self.logger.info(f"Performance - {metric_name}: {value} {unit}")
    
    def log_resource_utilization(self, utilization: dict) -> None:
        """Log resource utilization metrics."""
        self.logger.info("Resource utilization:")
        for resource, util in utilization.items():
            self.logger.info(f"  {resource}: {util:.2%}")


# Global logger instance
_simulation_logger: Optional[SimulationLogger] = None


def get_simulation_logger(name: str = "infraopt.simulation") -> SimulationLogger:
    """
    Get global simulation logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Simulation logger instance
    """
    global _simulation_logger
    if _simulation_logger is None:
        _simulation_logger = SimulationLogger(name)
    return _simulation_logger


def log_simulation_event(event_type: str, message: str, **kwargs) -> None:
    """
    Log a simulation event.
    
    Args:
        event_type: Type of event
        message: Event message
        **kwargs: Additional event data
    """
    logger = get_simulation_logger()
    logger.logger.info(f"[{event_type.upper()}] {message}")
    if kwargs:
        for key, value in kwargs.items():
            logger.logger.info(f"  {key}: {value}")


# Performance monitoring decorator
def log_performance(func):
    """Decorator to log function performance."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"{func.__name__} failed after {duration:.2f} seconds: {str(e)}")
            raise
    
    return wrapper


class DummyLogger:
    def info(self, msg, **kwargs):
        print(f"[INFO] {msg}", kwargs)

def setup_logger():
    return DummyLogger() 