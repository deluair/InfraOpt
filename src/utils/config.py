"""
Configuration management for InfraOpt simulation.

This module handles loading, validation, and management of simulation
configuration parameters.
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging


@dataclass
class SimulationConfig:
    """Configuration class for simulation parameters."""
    
    # Simulation parameters
    simulation_time_horizon: int = 365  # days
    time_step_hours: int = 1
    random_seed: int = 42
    
    # Data center parameters
    num_data_centers: int = 10
    data_center_types: list = field(default_factory=lambda: ["hyperscale", "enterprise", "edge"])
    
    # Workload parameters
    num_workloads: int = 20
    workload_types: list = field(default_factory=lambda: ["training", "inference", "research", "commercial"])
    
    # Optimization parameters
    optimization_objectives: list = field(default_factory=lambda: ["cost", "energy", "performance"])
    optimization_timeout_seconds: int = 300
    optimization_tolerance: float = 1e-6
    
    # Risk assessment parameters
    risk_simulations: int = 1000
    confidence_level: float = 0.95
    
    # Economic parameters
    inflation_rate: float = 0.03
    interest_rate: float = 0.05
    carbon_tax_rate: float = 50.0
    
    # Performance parameters
    parallel_processing: bool = True
    max_workers: int = 4
    cache_results: bool = True
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration from dictionary."""
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        if self.simulation_time_horizon <= 0:
            errors.append("simulation_time_horizon must be positive")
        
        if self.num_data_centers <= 0:
            errors.append("num_data_centers must be positive")
        
        if self.num_workloads <= 0:
            errors.append("num_workloads must be positive")
        
        if self.risk_simulations <= 0:
            errors.append("risk_simulations must be positive")
        
        if not (0 <= self.confidence_level <= 1):
            errors.append("confidence_level must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
        
        return True
    
    @classmethod
    def from_file(cls, file_path: str) -> 'SimulationConfig':
        """Load configuration from YAML file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Default configuration templates
DEFAULT_CONFIGS = {
    "basic": {
        "simulation_time_horizon": 30,
        "num_data_centers": 5,
        "num_workloads": 10,
        "optimization_objectives": ["cost"],
        "risk_simulations": 100
    },
    "standard": {
        "simulation_time_horizon": 365,
        "num_data_centers": 10,
        "num_workloads": 20,
        "optimization_objectives": ["cost", "energy", "performance"],
        "risk_simulations": 1000
    },
    "comprehensive": {
        "simulation_time_horizon": 1095,  # 3 years
        "num_data_centers": 25,
        "num_workloads": 50,
        "optimization_objectives": ["cost", "energy", "performance", "sustainability"],
        "risk_simulations": 5000,
        "parallel_processing": True,
        "max_workers": 8
    }
}


def load_config(config_name: str = "standard") -> SimulationConfig:
    """Load a predefined configuration."""
    if config_name not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}")
    
    return SimulationConfig(DEFAULT_CONFIGS[config_name])


def create_config_file(file_path: str, config_name: str = "standard") -> None:
    """Create a configuration file with default settings."""
    config = load_config(config_name)
    config.save_to_file(file_path) 