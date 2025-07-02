from typing import List
from src.models.infrastructure import DataCenter, GPU
from src.models.workloads import Workload

class Scheduler:
    """A simple scheduler to assign workloads to available resources."""

    def find_first_available_gpu(self, workload: Workload, datacenters: List[DataCenter]) -> GPU | None:
        """
        Finds the first available GPU across all data centers that can handle the workload.

        Args:
            workload: The workload to be scheduled.
            datacenters: The list of data centers to search within.

        Returns:
            The first available GPU that meets the requirements, or None if none is found.
        """
        for dc in datacenters:
            for gpu in dc.resources:
                # Check if the GPU is available and has enough capacity
                if gpu.utilization == 0.0 and gpu.capacity_tflops >= workload.required_tflops:
                    return gpu
        return None 