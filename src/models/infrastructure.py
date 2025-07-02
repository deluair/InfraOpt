from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum

class GPUType(str, Enum):
    """Enumeration for different types of GPUs."""
    H100 = "H100"
    A100 = "A100"
    V100 = "V100"
    RTX4090 = "RTX4090"

class GPU(BaseModel):
    """Represents a single GPU resource."""
    id: str
    gpu_type: GPUType
    cost_per_hour: float = Field(..., description="Operational cost per hour of a running GPU.")
    capacity_tflops: float = Field(..., description="Computational capacity in TFLOPS.")
    power_consumption_kw: float = Field(..., description="Power consumption in kilowatts when active.")
    utilization: float = Field(0.0, ge=0.0, le=1.0, description="Current utilization percentage from 0.0 to 1.0.")
    assigned_workload_id: str | None = None

class DataCenter(BaseModel):
    """Represents a single data center."""
    id: str
    location: str = Field(..., description="Geographic location, e.g., 'us-east-1'")
    pue: float = Field(..., description="Power Usage Effectiveness of the data center.")
    electricity_cost_per_kwh: float = Field(..., description="Cost of electricity in $/kWh.")
    resources: List[GPU] = []

    @property
    def total_it_power_kw(self) -> float:
        """Calculates the total power consumption of the IT equipment (GPUs)."""
        return sum(res.power_consumption_kw * res.utilization for res in self.resources)

    @property
    def total_facility_power_kw(self) -> float:
        """Calculates the total facility power, including cooling and overhead, based on PUE."""
        return self.total_it_power_kw * self.pue

    @property
    def operational_cost_per_hour(self) -> float:
        """Calculates the total operational cost per hour, including GPU and energy costs."""
        gpu_cost = sum(gpu.cost_per_hour for gpu in self.resources if gpu.utilization > 0)
        energy_cost = self.total_facility_power_kw * self.electricity_cost_per_kwh
        return gpu_cost + energy_cost
    
    def get_available_gpus(self) -> List[GPU]:
        """Returns a list of GPUs that are not currently utilized."""
        return [gpu for gpu in self.resources if gpu.utilization == 0.0] 