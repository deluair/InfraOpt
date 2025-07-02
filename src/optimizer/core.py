from typing import List
from src.models.infrastructure import DataCenter, GPU
from src.models.workloads import Workload
from src.scheduler.core import Scheduler

class Optimizer:
    """Decides the best placement for workloads based on a defined strategy."""

    def __init__(self):
        self.scheduler = Scheduler()

    def find_best_placement(self, workload: Workload, datacenters: List[DataCenter]) -> GPU | None:
        """
        Finds the best GPU for a workload.
        
        For this basic implementation, "best" is simply the first one available.
        Future implementations could optimize for cost, energy, latency, etc.

        Args:
            workload: The workload to place.
            datacenters: The list of data centers to consider.

        Returns:
            The optimal GPU for the workload, or None if no placement is possible.
        """
        # Simple strategy: find the first available GPU.
        return self.scheduler.find_first_available_gpu(workload, datacenters) 