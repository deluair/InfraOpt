"""
Multi-objective optimization engine for InfraOpt.

This module implements optimization algorithms for cost, energy,
performance, and sustainability objectives.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import cvxpy as cp
from scipy.optimize import minimize, differential_evolution
import logging

from ..models.infrastructure import DataCenter, ComputingResource
from ..models.workloads import WorkloadPattern
from ..utils.config import SimulationConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    
    objective_value: float
    decision_variables: Dict[str, Any]
    constraints_satisfied: bool
    optimization_time: float
    convergence_status: str
    metadata: Dict[str, Any]


class CostOptimizer:
    """
    Multi-objective optimization engine for AI infrastructure.
    
    This class implements various optimization algorithms to find
    optimal resource allocation strategies.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the optimizer.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.logger = logger
        
        # Optimization parameters
        self.timeout = config.optimization_timeout_seconds
        self.tolerance = config.optimization_tolerance
        
        self.logger.info("CostOptimizer initialized")
    
    def optimize_cost(
        self,
        data_centers: List[DataCenter],
        workloads: List[WorkloadPattern]
    ) -> OptimizationResult:
        """
        Optimize for minimum total cost.
        
        Args:
            data_centers: List of available data centers
            workloads: List of workloads to schedule
            
        Returns:
            Optimization result
        """
        self.logger.info("Starting cost optimization...")
        
        # Define decision variables
        # x[i,j] = fraction of workload i assigned to data center j
        n_workloads = len(workloads)
        n_dcs = len(data_centers)
        
        x = cp.Variable((n_workloads, n_dcs), nonneg=True)
        
        # Objective: minimize total cost
        total_cost = 0
        for i, workload in enumerate(workloads):
            for j, dc in enumerate(data_centers):
                # Cost per unit of workload
                cost_per_unit = self._calculate_workload_cost(workload, dc)
                total_cost += cp.sum(x[i, j] * cost_per_unit)
        
        objective = cp.Minimize(total_cost)
        
        # Constraints
        constraints = []
        
        # Each workload must be fully assigned
        for i in range(n_workloads):
            constraints.append(cp.sum(x[i, :]) == 1)
        
        # Data center capacity constraints
        for j, dc in enumerate(data_centers):
            capacity = dc.get_total_compute_capacity()
            for i, workload in enumerate(workloads):
                constraints.append(x[i, j] * workload.compute_intensity <= capacity)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        
        try:
            result = problem.solve(timeout=self.timeout, eps_abs=self.tolerance)
            
            return OptimizationResult(
                objective_value=problem.value,
                decision_variables={"allocation_matrix": x.value},
                constraints_satisfied=problem.status == "optimal",
                optimization_time=problem.solver_stats.solve_time,
                convergence_status=problem.status,
                metadata={
                    "solver": problem.solver_stats.solver_name,
                    "iterations": problem.solver_stats.num_iters
                }
            )
            
        except Exception as e:
            self.logger.error(f"Cost optimization failed: {str(e)}")
            return OptimizationResult(
                objective_value=float('inf'),
                decision_variables={},
                constraints_satisfied=False,
                optimization_time=0,
                convergence_status="failed",
                metadata={"error": str(e)}
            )
    
    def optimize_energy(
        self,
        data_centers: List[DataCenter],
        workloads: List[WorkloadPattern]
    ) -> OptimizationResult:
        """
        Optimize for minimum energy consumption.
        
        Args:
            data_centers: List of available data centers
            workloads: List of workloads to schedule
            
        Returns:
            Optimization result
        """
        self.logger.info("Starting energy optimization...")
        
        # Define decision variables
        n_workloads = len(workloads)
        n_dcs = len(data_centers)
        
        x = cp.Variable((n_workloads, n_dcs), nonneg=True)
        
        # Objective: minimize total energy consumption
        total_energy = 0
        for i, workload in enumerate(workloads):
            for j, dc in enumerate(data_centers):
                # Energy consumption per unit of workload
                energy_per_unit = self._calculate_workload_energy(workload, dc)
                total_energy += cp.sum(x[i, j] * energy_per_unit)
        
        objective = cp.Minimize(total_energy)
        
        # Constraints (same as cost optimization)
        constraints = []
        
        for i in range(n_workloads):
            constraints.append(cp.sum(x[i, :]) == 1)
        
        for j, dc in enumerate(data_centers):
            capacity = dc.get_total_compute_capacity()
            for i, workload in enumerate(workloads):
                constraints.append(x[i, j] * workload.compute_intensity <= capacity)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        
        try:
            result = problem.solve(timeout=self.timeout, eps_abs=self.tolerance)
            
            return OptimizationResult(
                objective_value=problem.value,
                decision_variables={"allocation_matrix": x.value},
                constraints_satisfied=problem.status == "optimal",
                optimization_time=problem.solver_stats.solve_time,
                convergence_status=problem.status,
                metadata={
                    "solver": problem.solver_stats.solver_name,
                    "iterations": problem.solver_stats.num_iters
                }
            )
            
        except Exception as e:
            self.logger.error(f"Energy optimization failed: {str(e)}")
            return OptimizationResult(
                objective_value=float('inf'),
                decision_variables={},
                constraints_satisfied=False,
                optimization_time=0,
                convergence_status="failed",
                metadata={"error": str(e)}
            )
    
    def optimize_performance(
        self,
        data_centers: List[DataCenter],
        workloads: List[WorkloadPattern]
    ) -> OptimizationResult:
        """
        Optimize for maximum performance.
        
        Args:
            data_centers: List of available data centers
            workloads: List of workloads to schedule
            
        Returns:
            Optimization result
        """
        self.logger.info("Starting performance optimization...")
        
        # Define decision variables
        n_workloads = len(workloads)
        n_dcs = len(data_centers)
        
        x = cp.Variable((n_workloads, n_dcs), nonneg=True)
        
        # Objective: maximize total performance (minimize negative performance)
        total_performance = 0
        for i, workload in enumerate(workloads):
            for j, dc in enumerate(data_centers):
                # Performance per unit of workload
                performance_per_unit = self._calculate_workload_performance(workload, dc)
                total_performance += cp.sum(x[i, j] * performance_per_unit)
        
        objective = cp.Maximize(total_performance)
        
        # Constraints
        constraints = []
        
        for i in range(n_workloads):
            constraints.append(cp.sum(x[i, :]) == 1)
        
        for j, dc in enumerate(data_centers):
            capacity = dc.get_total_compute_capacity()
            for i, workload in enumerate(workloads):
                constraints.append(x[i, j] * workload.compute_intensity <= capacity)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        
        try:
            result = problem.solve(timeout=self.timeout, eps_abs=self.tolerance)
            
            return OptimizationResult(
                objective_value=problem.value,
                decision_variables={"allocation_matrix": x.value},
                constraints_satisfied=problem.status == "optimal",
                optimization_time=problem.solver_stats.solve_time,
                convergence_status=problem.status,
                metadata={
                    "solver": problem.solver_stats.solver_name,
                    "iterations": problem.solver_stats.num_iters
                }
            )
            
        except Exception as e:
            self.logger.error(f"Performance optimization failed: {str(e)}")
            return OptimizationResult(
                objective_value=float('-inf'),
                decision_variables={},
                constraints_satisfied=False,
                optimization_time=0,
                convergence_status="failed",
                metadata={"error": str(e)}
            )
    
    def optimize_multi_objective(
        self,
        data_centers: List[DataCenter],
        workloads: List[WorkloadPattern],
        weights: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """
        Multi-objective optimization with weighted objectives.
        
        Args:
            data_centers: List of available data centers
            workloads: List of workloads to schedule
            weights: Weights for different objectives
            
        Returns:
            Optimization result
        """
        self.logger.info("Starting multi-objective optimization...")
        
        if weights is None:
            weights = {"cost": 0.4, "energy": 0.3, "performance": 0.3}
        
        # Define decision variables
        n_workloads = len(workloads)
        n_dcs = len(data_centers)
        
        x = cp.Variable((n_workloads, n_dcs), nonneg=True)
        
        # Multi-objective function
        total_objective = 0
        
        # Cost component
        if weights.get("cost", 0) > 0:
            cost_component = 0
            for i, workload in enumerate(workloads):
                for j, dc in enumerate(data_centers):
                    cost_per_unit = self._calculate_workload_cost(workload, dc)
                    cost_component += cp.sum(x[i, j] * cost_per_unit)
            total_objective += weights["cost"] * cost_component
        
        # Energy component
        if weights.get("energy", 0) > 0:
            energy_component = 0
            for i, workload in enumerate(workloads):
                for j, dc in enumerate(data_centers):
                    energy_per_unit = self._calculate_workload_energy(workload, dc)
                    energy_component += cp.sum(x[i, j] * energy_per_unit)
            total_objective += weights["energy"] * energy_component
        
        # Performance component (negative because we minimize)
        if weights.get("performance", 0) > 0:
            performance_component = 0
            for i, workload in enumerate(workloads):
                for j, dc in enumerate(data_centers):
                    performance_per_unit = self._calculate_workload_performance(workload, dc)
                    performance_component += cp.sum(x[i, j] * performance_per_unit)
            total_objective -= weights["performance"] * performance_component
        
        objective = cp.Minimize(total_objective)
        
        # Constraints
        constraints = []
        
        for i in range(n_workloads):
            constraints.append(cp.sum(x[i, :]) == 1)
        
        for j, dc in enumerate(data_centers):
            capacity = dc.get_total_compute_capacity()
            for i, workload in enumerate(workloads):
                constraints.append(x[i, j] * workload.compute_intensity <= capacity)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        
        try:
            result = problem.solve(timeout=self.timeout, eps_abs=self.tolerance)
            
            return OptimizationResult(
                objective_value=problem.value,
                decision_variables={"allocation_matrix": x.value},
                constraints_satisfied=problem.status == "optimal",
                optimization_time=problem.solver_stats.solve_time,
                convergence_status=problem.status,
                metadata={
                    "solver": problem.solver_stats.solver_name,
                    "iterations": problem.solver_stats.num_iters,
                    "weights": weights
                }
            )
            
        except Exception as e:
            self.logger.error(f"Multi-objective optimization failed: {str(e)}")
            return OptimizationResult(
                objective_value=float('inf'),
                decision_variables={},
                constraints_satisfied=False,
                optimization_time=0,
                convergence_status="failed",
                metadata={"error": str(e), "weights": weights}
            )
    
    def _calculate_workload_cost(self, workload: WorkloadPattern, dc: DataCenter) -> float:
        """Calculate cost per unit of workload on data center."""
        # Base cost from data center
        base_cost = dc.annual_cost / (dc.power_capacity * 8760)  # Cost per kWh
        
        # Workload-specific cost factors
        compute_cost = workload.compute_intensity * base_cost
        memory_cost = workload.memory_requirement * 0.01  # $0.01 per GB
        storage_cost = workload.storage_requirement * 0.05  # $0.05 per GB
        network_cost = workload.network_bandwidth * 0.1  # $0.1 per Gbps
        
        return compute_cost + memory_cost + storage_cost + network_cost
    
    def _calculate_workload_energy(self, workload: WorkloadPattern, dc: DataCenter) -> float:
        """Calculate energy consumption per unit of workload on data center."""
        # Base energy from compute requirements
        compute_energy = workload.compute_intensity * dc.pue
        
        # Additional energy for memory, storage, and network
        memory_energy = workload.memory_requirement * 0.001  # kWh per GB
        storage_energy = workload.storage_requirement * 0.002  # kWh per GB
        network_energy = workload.network_bandwidth * 0.01  # kWh per Gbps
        
        return compute_energy + memory_energy + storage_energy + network_energy
    
    def _calculate_workload_performance(self, workload: WorkloadPattern, dc: DataCenter) -> float:
        """Calculate performance per unit of workload on data center."""
        # Performance based on available compute capacity
        available_capacity = dc.get_total_compute_capacity()
        
        # Performance efficiency factors
        compute_efficiency = min(1.0, available_capacity / workload.compute_intensity)
        memory_efficiency = min(1.0, sum(r.capacity for r in dc.memory) / workload.memory_requirement)
        storage_efficiency = min(1.0, sum(r.capacity for r in dc.storage) / workload.storage_requirement)
        
        # Overall performance score
        performance_score = (compute_efficiency * 0.6 + 
                           memory_efficiency * 0.2 + 
                           storage_efficiency * 0.2)
        
        return performance_score
    
    def generate_pareto_frontier(
        self,
        data_centers: List[DataCenter],
        workloads: List[WorkloadPattern],
        num_points: int = 20
    ) -> List[OptimizationResult]:
        """
        Generate Pareto frontier for multi-objective optimization.
        
        Args:
            data_centers: List of available data centers
            workloads: List of workloads to schedule
            num_points: Number of points on Pareto frontier
            
        Returns:
            List of optimization results
        """
        self.logger.info(f"Generating Pareto frontier with {num_points} points...")
        
        pareto_results = []
        
        for i in range(num_points):
            # Vary weights to explore Pareto frontier
            cost_weight = i / (num_points - 1)
            energy_weight = (1 - cost_weight) / 2
            performance_weight = (1 - cost_weight) / 2
            
            weights = {
                "cost": cost_weight,
                "energy": energy_weight,
                "performance": performance_weight
            }
            
            result = self.optimize_multi_objective(data_centers, workloads, weights)
            pareto_results.append(result)
        
        self.logger.info(f"Generated {len(pareto_results)} Pareto frontier points")
        return pareto_results 