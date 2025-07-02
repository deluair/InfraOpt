import random
import uuid
from typing import List
import pandas as pd

from src.models.infrastructure import DataCenter, GPU, GPUType
from src.models.workloads import Workload, WorkloadType
from src.models.economics import GPU_HOURLY_COST, ELECTRICITY_COST_PER_KWH, PUE_BY_LOCATION

# Pre-defined configurations for different GPU types
GPU_SPECS = {
    GPUType.H100: {"capacity_tflops": 989.0, "power_consumption_kw": 0.7},
    GPUType.A100: {"capacity_tflops": 312.0, "power_consumption_kw": 0.4},
    GPUType.V100: {"capacity_tflops": 112.0, "power_consumption_kw": 0.3},
    GPUType.RTX4090: {"capacity_tflops": 83.0, "power_consumption_kw": 0.45},
}

class DataGenerator:
    """Generates synthetic infrastructure and workload data for the simulation."""

    def generate_infrastructure(self, num_datacenters: int = 2, gpus_per_dc: int = 10) -> List[DataCenter]:
        """Generates a list of data centers, each populated with GPUs."""
        datacenters = []
        gpu_id_counter = 1
        
        locations = list(ELECTRICITY_COST_PER_KWH.keys())

        for i in range(num_datacenters):
            location = locations[i % len(locations)]
            dc = DataCenter(
                id=f"dc-{i+1}",
                location=location,
                pue=PUE_BY_LOCATION[location],
                electricity_cost_per_kwh=ELECTRICITY_COST_PER_KWH[location],
                resources=[]
            )

            for _ in range(gpus_per_dc):
                gpu_type = random.choice(list(GPUType))
                spec = GPU_SPECS[gpu_type]
                gpu = GPU(
                    id=f"gpu-{gpu_id_counter}",
                    gpu_type=gpu_type,
                    cost_per_hour=GPU_HOURLY_COST[gpu_type.value],
                    capacity_tflops=spec["capacity_tflops"],
                    power_consumption_kw=spec["power_consumption_kw"],
                )
                dc.resources.append(gpu)
                gpu_id_counter += 1
            
            datacenters.append(dc)
            
        return datacenters

    def generate_workloads(self, num_workloads: int = 20) -> List[Workload]:
        """Generates a list of workloads with random requirements."""
        workloads = []
        for i in range(num_workloads):
            workload_type = random.choice([WorkloadType.TRAINING, WorkloadType.INFERENCE])
            workload = Workload(
                id=str(uuid.uuid4()),
                workload_type=workload_type,
                required_tflops=random.uniform(50, 500),  # Required processing power
                duration_hours=random.uniform(1, 12),  # Estimated job duration
            )
            workloads.append(workload)
            
        return workloads 