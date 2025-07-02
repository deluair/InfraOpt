"""
Infrastructure models for data centers and computing resources.

This module defines the data structures for representing physical
infrastructure assets in the simulation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import numpy as np


class GPUType(Enum):
    """Types of GPU accelerators available."""
    H100 = "H100"
    A100 = "A100"
    V100 = "V100"
    RTX_4090 = "RTX_4090"
    CUSTOM = "CUSTOM"


@dataclass
class ComputingResource:
    """Represents a computing resource (GPU, CPU, etc.)."""
    
    id: str
    resource_type: str  # "GPU", "CPU", "Memory", "Storage"
    model: str
    capacity: float  # TFLOPS for GPU, cores for CPU, GB for memory
    power_consumption: float  # Watts
    cost_per_unit: float  # USD
    availability: float = 0.995  # 99.5% default availability
    efficiency_factor: float = 1.0  # Performance efficiency multiplier
    
    def __post_init__(self):
        """Calculate derived properties."""
        self.annual_power_cost = self.power_consumption * 24 * 365 * 0.08 / 1000  # $0.08/kWh
        self.total_cost = self.cost_per_unit + self.annual_power_cost


@dataclass
class DataCenter:
    """Represents a data center with infrastructure specifications."""
    
    name: str
    location: str
    power_capacity: float  # MW
    pue: float  # Power Usage Effectiveness
    electricity_cost: float  # $/kWh
    cooling_efficiency: float  # Cooling system efficiency
    geographic_zone: str
    regulatory_environment: str
    construction_cost: float  # $/MW
    operational_cost_per_mw: float  # $/MW/year
    
    # Computed properties
    annual_cost: float = field(init=False)
    effective_power: float = field(init=False)
    carbon_intensity: float = field(init=False)
    
    # Resource inventory
    gpus: List[ComputingResource] = field(default_factory=list)
    cpus: List[ComputingResource] = field(default_factory=list)
    memory: List[ComputingResource] = field(default_factory=list)
    storage: List[ComputingResource] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived properties."""
        # Calculate effective power capacity considering PUE
        self.effective_power = self.power_capacity / self.pue
        
        # Calculate annual operational costs
        power_cost = self.power_capacity * 1000 * 24 * 365 * self.electricity_cost
        operational_cost = self.power_capacity * self.operational_cost_per_mw
        self.annual_cost = power_cost + operational_cost
        
        # Estimate carbon intensity based on geographic zone
        carbon_factors = {
            "US-East": 0.4,  # kg CO2/kWh
            "US-West": 0.3,
            "Europe": 0.2,
            "Asia-Pacific": 0.6,
            "Middle-East": 0.7
        }
        self.carbon_intensity = carbon_factors.get(self.geographic_zone, 0.5)
    
    def add_resource(self, resource: ComputingResource) -> None:
        """Add a computing resource to the data center."""
        if resource.resource_type == "GPU":
            self.gpus.append(resource)
        elif resource.resource_type == "CPU":
            self.cpus.append(resource)
        elif resource.resource_type == "Memory":
            self.memory.append(resource)
        elif resource.resource_type == "Storage":
            self.storage.append(resource)
    
    def get_total_compute_capacity(self) -> float:
        """Get total compute capacity in TFLOPS."""
        return sum(gpu.capacity for gpu in self.gpus)
    
    def get_total_power_consumption(self) -> float:
        """Get total power consumption in MW."""
        total_watts = sum(
            sum(r.power_consumption for r in resources)
            for resources in [self.gpus, self.cpus, self.memory, self.storage]
        )
        return total_watts / 1_000_000  # Convert to MW
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get resource utilization percentages."""
        total_resources = {
            "GPU": len(self.gpus),
            "CPU": len(self.cpus),
            "Memory": len(self.memory),
            "Storage": len(self.storage)
        }
        
        # Simulate utilization (in real scenario, this would come from monitoring)
        utilization = {}
        for resource_type, count in total_resources.items():
            if count > 0:
                # Simulate utilization between 60-95%
                utilization[resource_type] = np.random.uniform(0.6, 0.95)
            else:
                utilization[resource_type] = 0.0
        
        return utilization
    
    def calculate_carbon_footprint(self) -> float:
        """Calculate annual carbon footprint in metric tons CO2."""
        annual_energy_kwh = self.power_capacity * 1000 * 24 * 365
        return annual_energy_kwh * self.carbon_intensity / 1000  # Convert to metric tons


@dataclass
class NetworkInfrastructure:
    """Represents network connectivity infrastructure."""
    
    bandwidth_capacity: float  # Gbps
    latency_ms: float  # milliseconds
    network_type: str  # "InfiniBand", "Ethernet", "Custom"
    redundancy_level: int  # Number of redundant connections
    cost_per_gbps: float  # $/Gbps/year


@dataclass
class CoolingSystem:
    """Represents cooling system infrastructure."""
    
    cooling_type: str  # "Air", "Liquid", "Immersion"
    efficiency: float  # Cooling efficiency factor
    power_consumption: float  # MW
    water_consumption: float  # L/hour
    maintenance_cost: float  # $/year


# Predefined GPU configurations based on real-world specifications
GPU_CONFIGS = {
    GPUType.H100: ComputingResource(
        id="h100_80gb",
        resource_type="GPU",
        model="NVIDIA H100 80GB",
        capacity=989.0,  # TFLOPS (FP16)
        power_consumption=700,  # Watts
        cost_per_unit=25000,  # USD
        efficiency_factor=1.0
    ),
    GPUType.A100: ComputingResource(
        id="a100_80gb",
        resource_type="GPU",
        model="NVIDIA A100 80GB",
        capacity=312.0,  # TFLOPS (FP16)
        power_consumption=400,  # Watts
        cost_per_unit=10000,  # USD
        efficiency_factor=0.95
    ),
    GPUType.V100: ComputingResource(
        id="v100_32gb",
        resource_type="GPU",
        model="NVIDIA V100 32GB",
        capacity=112.0,  # TFLOPS (FP16)
        power_consumption=250,  # Watts
        cost_per_unit=3000,  # USD
        efficiency_factor=0.85
    ),
    GPUType.RTX_4090: ComputingResource(
        id="rtx_4090_24gb",
        resource_type="GPU",
        model="NVIDIA RTX 4090 24GB",
        capacity=83.0,  # TFLOPS (FP16)
        power_consumption=450,  # Watts
        cost_per_unit=1600,  # USD
        efficiency_factor=0.8
    )
}


# Predefined data center configurations
DATA_CENTER_TEMPLATES = {
    "hyperscale": {
        "power_capacity_range": (50, 500),  # MW
        "pue_range": (1.1, 1.3),
        "electricity_cost_range": (0.05, 0.12),  # $/kWh
        "construction_cost": 8000000,  # $/MW
        "operational_cost_per_mw": 200000  # $/MW/year
    },
    "enterprise": {
        "power_capacity_range": (5, 50),  # MW
        "pue_range": (1.4, 1.8),
        "electricity_cost_range": (0.08, 0.15),  # $/kWh
        "construction_cost": 10000000,  # $/MW
        "operational_cost_per_mw": 300000  # $/MW/year
    },
    "edge": {
        "power_capacity_range": (0.5, 5),  # MW
        "pue_range": (1.6, 2.0),
        "electricity_cost_range": (0.10, 0.20),  # $/kWh
        "construction_cost": 15000000,  # $/MW
        "operational_cost_per_mw": 500000  # $/MW/year
    }
} 