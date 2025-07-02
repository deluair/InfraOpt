"""
Resource scheduler for dynamic workload allocation.

This module implements intelligent scheduling algorithms for
distributing workloads across data center resources.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import heapq
import logging

from ..models.infrastructure import DataCenter, ComputingResource
from ..models.workloads import WorkloadPattern, WorkloadType
from ..utils.config import SimulationConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SchedulingDecision:
    """Represents a scheduling decision."""
    
    workload_id: str
    data_center_id: str
    start_time: datetime
    end_time: datetime
    resource_allocation: Dict[str, float]
    priority: int
    estimated_cost: float
    estimated_energy: float


@dataclass
class SchedulingResult:
    """Result of scheduling process."""
    
    decisions: List[SchedulingDecision]
    total_cost: float
    total_energy: float
    resource_utilization: Dict[str, float]
    scheduling_time: float
    success_rate: float
    metadata: Dict[str, Any]


class ResourceScheduler:
    """
    Intelligent resource scheduler for workload distribution.
    
    This class implements various scheduling algorithms including:
    - Priority-based scheduling
    - Load balancing
    - Energy-aware scheduling
    - Multi-objective optimization
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the scheduler.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.logger = logger
        
        # Scheduling parameters
        self.scheduling_window_hours = 24
        self.resource_reservation_factor = 0.1  # 10% resource reservation
        
        self.logger.info("ResourceScheduler initialized")
    
    def schedule_resources(
        self,
        data_centers: List[DataCenter],
        workloads: List[WorkloadPattern],
        optimization_results: Dict[str, Any]
    ) -> SchedulingResult:
        """
        Schedule workloads across data centers.
        
        Args:
            data_centers: Available data centers
            workloads: Workloads to schedule
            optimization_results: Results from optimization phase
            
        Returns:
            Scheduling result
        """
        self.logger.info("Starting resource scheduling...")
        
        start_time = datetime.now()
        
        # Sort workloads by priority (higher priority first)
        sorted_workloads = sorted(workloads, key=lambda w: w.priority, reverse=True)
        
        # Initialize scheduling decisions
        decisions = []
        
        # Track resource utilization
        resource_utilization = self._initialize_resource_tracking(data_centers)
        
        # Schedule each workload
        successful_schedules = 0
        
        for workload in sorted_workloads:
            decision = self._schedule_workload(
                workload, data_centers, resource_utilization, optimization_results
            )
            
            if decision:
                decisions.append(decision)
                successful_schedules += 1
                self._update_resource_utilization(decision, resource_utilization)
            else:
                self.logger.warning(f"Failed to schedule workload: {workload.id}")
        
        # Calculate metrics
        total_cost = sum(d.estimated_cost for d in decisions)
        total_energy = sum(d.estimated_energy for d in decisions)
        success_rate = successful_schedules / len(workloads)
        
        scheduling_time = (datetime.now() - start_time).total_seconds()
        
        result = SchedulingResult(
            decisions=decisions,
            total_cost=total_cost,
            total_energy=total_energy,
            resource_utilization=self._calculate_final_utilization(resource_utilization),
            scheduling_time=scheduling_time,
            success_rate=success_rate,
            metadata={
                "scheduled_workloads": successful_schedules,
                "total_workloads": len(workloads),
                "scheduling_algorithm": "priority_based"
            }
        )
        
        self.logger.info(f"Scheduling completed: {successful_schedules}/{len(workloads)} workloads scheduled")
        return result
    
    def _schedule_workload(
        self,
        workload: WorkloadPattern,
        data_centers: List[DataCenter],
        resource_utilization: Dict[str, Dict[str, float]],
        optimization_results: Dict[str, Any]
    ) -> Optional[SchedulingDecision]:
        """
        Schedule a single workload.
        
        Args:
            workload: Workload to schedule
            data_centers: Available data centers
            resource_utilization: Current resource utilization
            optimization_results: Optimization results
            
        Returns:
            Scheduling decision or None if scheduling failed
        """
        # Get optimization-based allocation if available
        allocation_matrix = optimization_results.get("multi_objective", {}).get("allocation_matrix")
        
        if allocation_matrix is not None:
            # Use optimization results to guide scheduling
            return self._schedule_with_optimization(
                workload, data_centers, resource_utilization, allocation_matrix
            )
        else:
            # Use heuristic scheduling
            return self._schedule_with_heuristics(
                workload, data_centers, resource_utilization
            )
    
    def _schedule_with_optimization(
        self,
        workload: WorkloadPattern,
        data_centers: List[DataCenter],
        resource_utilization: Dict[str, Dict[str, float]],
        allocation_matrix: np.ndarray
    ) -> Optional[SchedulingDecision]:
        """Schedule using optimization results."""
        # Find workload index
        workload_idx = None
        for i, w in enumerate(self._get_all_workloads()):
            if w.id == workload.id:
                workload_idx = i
                break
        
        if workload_idx is None:
            return None
        
        # Find best data center based on allocation matrix
        best_dc_idx = np.argmax(allocation_matrix[workload_idx, :])
        best_dc = data_centers[best_dc_idx]
        
        # Check if data center has capacity
        if self._check_capacity(workload, best_dc, resource_utilization[best_dc.name]):
            return self._create_scheduling_decision(workload, best_dc)
        
        return None
    
    def _schedule_with_heuristics(
        self,
        workload: WorkloadPattern,
        data_centers: List[DataCenter],
        resource_utilization: Dict[str, Dict[str, float]]
    ) -> Optional[SchedulingDecision]:
        """Schedule using heuristic algorithms."""
        # Score each data center
        dc_scores = []
        
        for dc in data_centers:
            score = self._calculate_dc_score(workload, dc, resource_utilization[dc.name])
            dc_scores.append((score, dc))
        
        # Sort by score (higher is better)
        dc_scores.sort(reverse=True)
        
        # Try to schedule on best available data center
        for score, dc in dc_scores:
            if self._check_capacity(workload, dc, resource_utilization[dc.name]):
                return self._create_scheduling_decision(workload, dc)
        
        return None
    
    def _calculate_dc_score(
        self,
        workload: WorkloadPattern,
        dc: DataCenter,
        utilization: Dict[str, float]
    ) -> float:
        """Calculate score for data center suitability."""
        score = 0.0
        
        # Compute capacity score
        available_compute = dc.get_total_compute_capacity() * (1 - utilization.get("compute", 0))
        compute_score = min(1.0, available_compute / workload.compute_intensity)
        score += compute_score * 0.4
        
        # Memory capacity score
        available_memory = sum(r.capacity for r in dc.memory) * (1 - utilization.get("memory", 0))
        memory_score = min(1.0, available_memory / workload.memory_requirement)
        score += memory_score * 0.3
        
        # Storage capacity score
        available_storage = sum(r.capacity for r in dc.storage) * (1 - utilization.get("storage", 0))
        storage_score = min(1.0, available_storage / workload.storage_requirement)
        score += storage_score * 0.2
        
        # Network capacity score
        network_score = 1.0 - utilization.get("network", 0)
        score += network_score * 0.1
        
        # Penalty for high utilization
        avg_utilization = np.mean(list(utilization.values()))
        score *= (1 - avg_utilization * 0.5)
        
        return score
    
    def _check_capacity(
        self,
        workload: WorkloadPattern,
        dc: DataCenter,
        utilization: Dict[str, float]
    ) -> bool:
        """Check if data center has capacity for workload."""
        # Check compute capacity
        available_compute = dc.get_total_compute_capacity() * (1 - utilization.get("compute", 0))
        if workload.compute_intensity > available_compute:
            return False
        
        # Check memory capacity
        available_memory = sum(r.capacity for r in dc.memory) * (1 - utilization.get("memory", 0))
        if workload.memory_requirement > available_memory:
            return False
        
        # Check storage capacity
        available_storage = sum(r.capacity for r in dc.storage) * (1 - utilization.get("storage", 0))
        if workload.storage_requirement > available_storage:
            return False
        
        return True
    
    def _create_scheduling_decision(
        self,
        workload: WorkloadPattern,
        dc: DataCenter
    ) -> SchedulingDecision:
        """Create scheduling decision for workload on data center."""
        # Calculate timing
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=workload.duration_hours)
        
        # Calculate resource allocation
        resource_allocation = {
            "compute": workload.compute_intensity,
            "memory": workload.memory_requirement,
            "storage": workload.storage_requirement,
            "network": workload.network_bandwidth
        }
        
        # Estimate costs
        estimated_cost = self._estimate_workload_cost(workload, dc)
        estimated_energy = self._estimate_workload_energy(workload, dc)
        
        return SchedulingDecision(
            workload_id=workload.id,
            data_center_id=dc.name,
            start_time=start_time,
            end_time=end_time,
            resource_allocation=resource_allocation,
            priority=workload.priority,
            estimated_cost=estimated_cost,
            estimated_energy=estimated_energy
        )
    
    def _estimate_workload_cost(self, workload: WorkloadPattern, dc: DataCenter) -> float:
        """Estimate cost for workload on data center."""
        # Base cost from data center
        base_cost_per_hour = dc.annual_cost / 8760  # Annual cost / hours per year
        
        # Workload-specific costs
        compute_cost = workload.compute_intensity * 0.1  # $0.1 per TFLOPS-hour
        memory_cost = workload.memory_requirement * 0.01  # $0.01 per GB-hour
        storage_cost = workload.storage_requirement * 0.005  # $0.005 per GB-hour
        network_cost = workload.network_bandwidth * 0.05  # $0.05 per Gbps-hour
        
        total_cost_per_hour = compute_cost + memory_cost + storage_cost + network_cost
        
        return total_cost_per_hour * workload.duration_hours
    
    def _estimate_workload_energy(self, workload: WorkloadPattern, dc: DataCenter) -> float:
        """Estimate energy consumption for workload on data center."""
        # Base energy from compute requirements
        compute_energy = workload.compute_intensity * dc.pue * workload.duration_hours
        
        # Additional energy for other resources
        memory_energy = workload.memory_requirement * 0.001 * workload.duration_hours
        storage_energy = workload.storage_requirement * 0.002 * workload.duration_hours
        network_energy = workload.network_bandwidth * 0.01 * workload.duration_hours
        
        return compute_energy + memory_energy + storage_energy + network_energy
    
    def _initialize_resource_tracking(self, data_centers: List[DataCenter]) -> Dict[str, Dict[str, float]]:
        """Initialize resource utilization tracking."""
        tracking = {}
        
        for dc in data_centers:
            tracking[dc.name] = {
                "compute": 0.0,
                "memory": 0.0,
                "storage": 0.0,
                "network": 0.0
            }
        
        return tracking
    
    def _update_resource_utilization(
        self,
        decision: SchedulingDecision,
        resource_utilization: Dict[str, Dict[str, float]]
    ) -> None:
        """Update resource utilization after scheduling decision."""
        dc_name = decision.data_center_id
        allocation = decision.resource_allocation
        
        # Update utilization for each resource type
        for resource_type, amount in allocation.items():
            if resource_type in resource_utilization[dc_name]:
                resource_utilization[dc_name][resource_type] += amount
    
    def _calculate_final_utilization(
        self,
        resource_utilization: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate final resource utilization across all data centers."""
        final_utilization = {
            "compute": 0.0,
            "memory": 0.0,
            "storage": 0.0,
            "network": 0.0
        }
        
        total_resources = {
            "compute": 0.0,
            "memory": 0.0,
            "storage": 0.0,
            "network": 0.0
        }
        
        # Calculate total resources and utilization
        for dc_util in resource_utilization.values():
            for resource_type, utilization in dc_util.items():
                final_utilization[resource_type] += utilization
                total_resources[resource_type] += 1.0  # Normalize by number of DCs
        
        # Convert to percentages
        for resource_type in final_utilization:
            if total_resources[resource_type] > 0:
                final_utilization[resource_type] /= total_resources[resource_type]
        
        return final_utilization
    
    def _get_all_workloads(self) -> List[WorkloadPattern]:
        """Get all workloads (placeholder - should be passed from simulator)."""
        # This is a placeholder - in real implementation, workloads should be passed
        return []
    
    def schedule_with_load_balancing(
        self,
        data_centers: List[DataCenter],
        workloads: List[WorkloadPattern]
    ) -> SchedulingResult:
        """
        Schedule workloads with load balancing.
        
        Args:
            data_centers: Available data centers
            workloads: Workloads to schedule
            
        Returns:
            Scheduling result
        """
        self.logger.info("Starting load-balanced scheduling...")
        
        start_time = datetime.now()
        
        # Initialize resource tracking
        resource_utilization = self._initialize_resource_tracking(data_centers)
        decisions = []
        
        # Sort workloads by resource requirements (largest first)
        sorted_workloads = sorted(
            workloads,
            key=lambda w: w.compute_intensity + w.memory_requirement + w.storage_requirement,
            reverse=True
        )
        
        successful_schedules = 0
        
        for workload in sorted_workloads:
            # Find data center with lowest utilization
            best_dc = None
            best_score = float('inf')
            
            for dc in data_centers:
                if self._check_capacity(workload, dc, resource_utilization[dc.name]):
                    # Calculate load balancing score (lower is better)
                    score = np.mean(list(resource_utilization[dc.name].values()))
                    
                    if score < best_score:
                        best_score = score
                        best_dc = dc
            
            if best_dc:
                decision = self._create_scheduling_decision(workload, best_dc)
                decisions.append(decision)
                self._update_resource_utilization(decision, resource_utilization)
                successful_schedules += 1
            else:
                self.logger.warning(f"Failed to schedule workload: {workload.id}")
        
        # Calculate metrics
        total_cost = sum(d.estimated_cost for d in decisions)
        total_energy = sum(d.estimated_energy for d in decisions)
        success_rate = successful_schedules / len(workloads)
        
        scheduling_time = (datetime.now() - start_time).total_seconds()
        
        result = SchedulingResult(
            decisions=decisions,
            total_cost=total_cost,
            total_energy=total_energy,
            resource_utilization=self._calculate_final_utilization(resource_utilization),
            scheduling_time=scheduling_time,
            success_rate=success_rate,
            metadata={
                "scheduled_workloads": successful_schedules,
                "total_workloads": len(workloads),
                "scheduling_algorithm": "load_balancing"
            }
        )
        
        self.logger.info(f"Load-balanced scheduling completed: {successful_schedules}/{len(workloads)} workloads scheduled")
        return result
    
    def schedule_with_energy_awareness(
        self,
        data_centers: List[DataCenter],
        workloads: List[WorkloadPattern]
    ) -> SchedulingResult:
        """
        Schedule workloads with energy awareness.
        
        Args:
            data_centers: Available data centers
            workloads: Workloads to schedule
            
        Returns:
            Scheduling result
        """
        self.logger.info("Starting energy-aware scheduling...")
        
        start_time = datetime.now()
        
        # Initialize resource tracking
        resource_utilization = self._initialize_resource_tracking(data_centers)
        decisions = []
        
        # Sort workloads by energy efficiency requirements
        sorted_workloads = sorted(
            workloads,
            key=lambda w: w.compute_intensity * w.duration_hours,  # Energy consumption
            reverse=True
        )
        
        successful_schedules = 0
        
        for workload in sorted_workloads:
            # Find most energy-efficient data center
            best_dc = None
            best_energy_score = float('inf')
            
            for dc in data_centers:
                if self._check_capacity(workload, dc, resource_utilization[dc.name]):
                    # Calculate energy efficiency score
                    energy_score = dc.pue * self._estimate_workload_energy(workload, dc)
                    
                    if energy_score < best_energy_score:
                        best_energy_score = energy_score
                        best_dc = dc
            
            if best_dc:
                decision = self._create_scheduling_decision(workload, best_dc)
                decisions.append(decision)
                self._update_resource_utilization(decision, resource_utilization)
                successful_schedules += 1
            else:
                self.logger.warning(f"Failed to schedule workload: {workload.id}")
        
        # Calculate metrics
        total_cost = sum(d.estimated_cost for d in decisions)
        total_energy = sum(d.estimated_energy for d in decisions)
        success_rate = successful_schedules / len(workloads)
        
        scheduling_time = (datetime.now() - start_time).total_seconds()
        
        result = SchedulingResult(
            decisions=decisions,
            total_cost=total_cost,
            total_energy=total_energy,
            resource_utilization=self._calculate_final_utilization(resource_utilization),
            scheduling_time=scheduling_time,
            success_rate=success_rate,
            metadata={
                "scheduled_workloads": successful_schedules,
                "total_workloads": len(workloads),
                "scheduling_algorithm": "energy_aware"
            }
        )
        
        self.logger.info(f"Energy-aware scheduling completed: {successful_schedules}/{len(workloads)} workloads scheduled")
        return result 