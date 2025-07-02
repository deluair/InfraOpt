"""
Main simulation controller for InfraOpt.

This module orchestrates the entire simulation process, including data generation,
optimization, scheduling, and result analysis.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..models.infrastructure import DataCenter, ComputingResource
from ..models.economics import EconomicEnvironment
from ..models.workloads import WorkloadPattern
from ..data.generator import SyntheticDataGenerator
from ..analytics.reporter import SimulationReporter
from ..utils.config import SimulationConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class InfraOptSimulator:
    """
    Main simulation controller for AI infrastructure economics optimization.
    
    This class orchestrates the entire simulation process, including:
    - Data generation and validation
    - Multi-objective optimization
    - Resource scheduling
    - Result analysis and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the simulation controller.
        
        Args:
            config: Configuration dictionary for simulation parameters
        """
        self.config = SimulationConfig(config or {})
        self.logger = logger
        
        # Initialize components
        self.data_generator = SyntheticDataGenerator(self.config)
        self.reporter = SimulationReporter()
        
        # Simulation state
        self.data_centers: List[DataCenter] = []
        self.economic_env: Optional[EconomicEnvironment] = None
        self.workloads: List[WorkloadPattern] = []
        self.results: Dict[str, Any] = {}
        
        self.logger.info("InfraOptSimulator initialized")
    
    def setup_simulation(self) -> None:
        """Set up the simulation environment with synthetic data."""
        self.logger.info("Setting up simulation environment...")
        
        # Generate synthetic data
        self.data_centers = self.data_generator.generate_data_centers()
        self.economic_env = self.data_generator.generate_economic_environment()
        self.workloads = self.data_generator.generate_workload_patterns()
        
        self.logger.info(f"Generated {len(self.data_centers)} data centers")
        self.logger.info(f"Generated {len(self.workloads)} workload patterns")
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Run the multi-objective optimization process.
        
        Returns:
            Dictionary containing optimization results
        """
        from .optimizer import CostOptimizer
        
        self.logger.info("Starting optimization process...")
        
        optimizer = CostOptimizer(self.config)
        
        # Run optimization with different objectives
        results = {
            "cost_optimization": optimizer.optimize_cost(self.data_centers, self.workloads),
            "energy_optimization": optimizer.optimize_energy(self.data_centers, self.workloads),
            "performance_optimization": optimizer.optimize_performance(self.data_centers, self.workloads),
            "multi_objective": optimizer.optimize_multi_objective(self.data_centers, self.workloads)
        }
        
        self.logger.info("Optimization completed")
        return results
    
    def run_scheduling(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run resource scheduling based on optimization results.
        
        Args:
            optimization_results: Results from the optimization process
            
        Returns:
            Dictionary containing scheduling results
        """
        from .scheduler import ResourceScheduler
        
        self.logger.info("Starting resource scheduling...")
        
        scheduler = ResourceScheduler(self.config)
        
        # Schedule resources based on optimization results
        scheduling_results = scheduler.schedule_resources(
            self.data_centers,
            self.workloads,
            optimization_results
        )
        
        self.logger.info("Resource scheduling completed")
        return scheduling_results
    
    def run_risk_assessment(self) -> Dict[str, Any]:
        """
        Run Monte Carlo risk assessment simulations.
        
        Returns:
            Dictionary containing risk assessment results
        """
        self.logger.info("Starting risk assessment...")
        
        # Monte Carlo simulation parameters
        n_simulations = self.config.get("risk_assessment.simulations", 1000)
        time_horizon = self.config.get("simulation.time_horizon", 365)
        
        risk_results = {
            "financial_risk": self._assess_financial_risk(n_simulations, time_horizon),
            "operational_risk": self._assess_operational_risk(n_simulations, time_horizon),
            "supply_chain_risk": self._assess_supply_chain_risk(n_simulations, time_horizon)
        }
        
        self.logger.info("Risk assessment completed")
        return risk_results
    
    def _assess_financial_risk(self, n_simulations: int, time_horizon: int) -> Dict[str, float]:
        """Assess financial risk using Monte Carlo simulation."""
        # Simulate cost variations
        base_costs = np.array([dc.annual_cost for dc in self.data_centers])
        cost_volatility = 0.15  # 15% cost volatility
        
        simulated_costs = []
        for _ in range(n_simulations):
            # Add random cost variations
            cost_shock = np.random.normal(0, cost_volatility, len(base_costs))
            simulated_cost = base_costs * (1 + cost_shock)
            simulated_costs.append(np.sum(simulated_cost))
        
        simulated_costs = np.array(simulated_costs)
        
        return {
            "var_95": np.percentile(simulated_costs, 5),  # 95% VaR
            "var_99": np.percentile(simulated_costs, 1),  # 99% VaR
            "expected_loss": np.mean(simulated_costs),
            "max_loss": np.max(simulated_costs),
            "volatility": np.std(simulated_costs)
        }
    
    def _assess_operational_risk(self, n_simulations: int, time_horizon: int) -> Dict[str, float]:
        """Assess operational risk using Monte Carlo simulation."""
        # Simulate availability and performance variations
        base_availability = 0.995  # 99.5% base availability
        availability_volatility = 0.01  # 1% availability volatility
        
        simulated_availability = []
        for _ in range(n_simulations):
            availability_shock = np.random.normal(0, availability_volatility)
            simulated_avail = max(0.8, min(1.0, base_availability + availability_shock))
            simulated_availability.append(simulated_avail)
        
        simulated_availability = np.array(simulated_availability)
        
        return {
            "min_availability": np.min(simulated_availability),
            "avg_availability": np.mean(simulated_availability),
            "availability_volatility": np.std(simulated_availability),
            "downtime_risk": 1 - np.mean(simulated_availability)
        }
    
    def _assess_supply_chain_risk(self, n_simulations: int, time_horizon: int) -> Dict[str, float]:
        """Assess supply chain risk using Monte Carlo simulation."""
        # Simulate supply chain disruptions
        base_lead_time = 90  # days
        lead_time_volatility = 30  # days
        
        simulated_lead_times = []
        for _ in range(n_simulations):
            lead_time_shock = np.random.normal(0, lead_time_volatility)
            simulated_lt = max(30, base_lead_time + lead_time_shock)
            simulated_lead_times.append(simulated_lt)
        
        simulated_lead_times = np.array(simulated_lead_times)
        
        return {
            "avg_lead_time": np.mean(simulated_lead_times),
            "max_lead_time": np.max(simulated_lead_times),
            "lead_time_volatility": np.std(simulated_lead_times),
            "disruption_probability": np.sum(simulated_lead_times > 120) / n_simulations
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete simulation process.
        
        Returns:
            Dictionary containing all simulation results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting InfraOpt simulation at {start_time}")
        
        try:
            # Setup simulation environment
            self.setup_simulation()
            
            # Run optimization
            optimization_results = self.run_optimization()
            
            # Run scheduling
            scheduling_results = self.run_scheduling(optimization_results)
            
            # Run risk assessment
            risk_results = self.run_risk_assessment()
            
            # Compile results
            self.results = {
                "optimization": optimization_results,
                "scheduling": scheduling_results,
                "risk_assessment": risk_results,
                "metadata": {
                    "start_time": start_time,
                    "end_time": datetime.now(),
                    "data_centers_count": len(self.data_centers),
                    "workloads_count": len(self.workloads),
                    "config": self.config.to_dict()
                }
            }
            
            self.logger.info("Simulation completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            raise
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive simulation report.
        
        Args:
            output_path: Optional path for report output
            
        Returns:
            Path to generated report
        """
        if not self.results:
            raise ValueError("No simulation results available. Run simulation first.")
        
        return self.reporter.generate_report(self.results, output_path)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics from simulation results.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.results:
            return {}
        
        total_cost = sum(dc.annual_cost for dc in self.data_centers)
        total_power = sum(dc.power_capacity for dc in self.data_centers)
        avg_pue = np.mean([dc.pue for dc in self.data_centers])
        
        return {
            "total_infrastructure_cost": total_cost,
            "total_power_capacity_mw": total_power,
            "average_pue": avg_pue,
            "data_centers_count": len(self.data_centers),
            "optimization_objectives": list(self.results.get("optimization", {}).keys()),
            "risk_metrics": list(self.results.get("risk_assessment", {}).keys())
        } 