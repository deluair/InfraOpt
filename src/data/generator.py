"""
Synthetic data generator for InfraOpt simulation.

This module generates realistic synthetic data for data centers,
workloads, and economic environments based on real-world parameters.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random
import os
import json

from ..models.infrastructure import (
    DataCenter, ComputingResource, GPUType, GPU_CONFIGS, DATA_CENTER_TEMPLATES
)
from ..models.economics import (
    EconomicEnvironment, EnergyMarket, SupplyChain, ENERGY_MARKET_CONFIGS, SUPPLY_CHAIN_CONFIGS
)
from ..models.workloads import (
    WorkloadPattern, TrainingJob, InferenceRequest, ResearchWorkload, CommercialWorkload,
    WorkloadType, ModelType, WORKLOAD_TEMPLATES
)
from ..utils.config import SimulationConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticDataGenerator:
    """
    Generator for synthetic simulation data.
    
    This class creates realistic synthetic data for:
    - Data centers with varying specifications
    - Economic environments with market dynamics
    - Workload patterns with different characteristics
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the data generator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.logger = logger
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
        # Geographic regions for data centers
        self.regions = [
            "US-East", "US-West", "US-Central", "Europe-West", "Europe-East",
            "Asia-Pacific", "Middle-East", "South-America", "Africa", "Australia"
        ]
        
        # Data center locations with realistic coordinates
        self.locations = {
            "US-East": ["New York", "Virginia", "Georgia", "Florida"],
            "US-West": ["California", "Oregon", "Washington", "Nevada"],
            "US-Central": ["Texas", "Illinois", "Ohio", "Michigan"],
            "Europe-West": ["London", "Frankfurt", "Amsterdam", "Paris"],
            "Europe-East": ["Warsaw", "Prague", "Budapest", "Vienna"],
            "Asia-Pacific": ["Tokyo", "Singapore", "Seoul", "Hong Kong"],
            "Middle-East": ["Dubai", "Riyadh", "Doha", "Abu Dhabi"],
            "South-America": ["São Paulo", "Rio de Janeiro", "Buenos Aires", "Santiago"],
            "Africa": ["Johannesburg", "Cape Town", "Lagos", "Nairobi"],
            "Australia": ["Sydney", "Melbourne", "Perth", "Brisbane"]
        }
        
        self.logger.info("SyntheticDataGenerator initialized")
    
    def generate_data_centers(self) -> List[DataCenter]:
        """
        Generate synthetic data centers.
        
        Returns:
            List of DataCenter objects
        """
        self.logger.info("Generating synthetic data centers...")
        
        data_centers = []
        num_dcs = self.config.num_data_centers
        
        for i in range(num_dcs):
            # Select random region and location
            region = random.choice(self.regions)
            location = random.choice(self.locations[region])
            
            # Select data center type
            dc_type = random.choice(self.config.data_center_types)
            template = DATA_CENTER_TEMPLATES[dc_type]
            
            # Generate specifications within template ranges
            power_capacity = np.random.uniform(
                template["power_capacity_range"][0],
                template["power_capacity_range"][1]
            )
            
            pue = np.random.uniform(
                template["pue_range"][0],
                template["pue_range"][1]
            )
            
            electricity_cost = np.random.uniform(
                template["electricity_cost_range"][0],
                template["electricity_cost_range"][1]
            )
            
            # Create data center
            dc = DataCenter(
                name=f"{location}-DC-{i+1:03d}",
                location=location,
                power_capacity=power_capacity,
                pue=pue,
                electricity_cost=electricity_cost,
                cooling_efficiency=np.random.uniform(0.7, 0.95),
                geographic_zone=region,
                regulatory_environment=self._get_regulatory_environment(region),
                construction_cost=template["construction_cost"],
                operational_cost_per_mw=template["operational_cost_per_mw"]
            )
            
            # Add computing resources
            self._add_computing_resources(dc)
            
            data_centers.append(dc)
        
        self.logger.info(f"Generated {len(data_centers)} data centers")
        return data_centers
    
    def _add_computing_resources(self, dc: DataCenter) -> None:
        """Add computing resources to data center."""
        # Calculate available power for IT equipment
        it_power_mw = dc.power_capacity / dc.pue
        
        # Add GPUs (60% of IT power)
        gpu_power_mw = it_power_mw * 0.6
        self._add_gpus(dc, gpu_power_mw)
        
        # Add CPUs (20% of IT power)
        cpu_power_mw = it_power_mw * 0.2
        self._add_cpus(dc, cpu_power_mw)
        
        # Add Memory (10% of IT power)
        memory_power_mw = it_power_mw * 0.1
        self._add_memory(dc, memory_power_mw)
        
        # Add Storage (10% of IT power)
        storage_power_mw = it_power_mw * 0.1
        self._add_storage(dc, storage_power_mw)
    
    def _add_gpus(self, dc: DataCenter, available_power_mw: float) -> None:
        """Add GPU resources to data center."""
        gpu_types = list(GPU_CONFIGS.keys())
        total_gpu_power_w = available_power_mw * 1_000_000  # Convert to watts
        
        while total_gpu_power_w > 0:
            # Select random GPU type
            gpu_type = random.choice(gpu_types)
            gpu_config = GPU_CONFIGS[gpu_type]
            
            # Calculate number of GPUs that can fit
            max_gpus = int(total_gpu_power_w / gpu_config.power_consumption)
            if max_gpus <= 0:
                break
            
            # Add 1-10 GPUs of this type
            num_gpus = min(random.randint(1, 10), max_gpus)
            
            for j in range(num_gpus):
                gpu = ComputingResource(
                    id=f"{dc.name}-GPU-{gpu_type.value}-{j+1}",
                    resource_type="GPU",
                    model=gpu_config.model,
                    capacity=gpu_config.capacity,
                    power_consumption=gpu_config.power_consumption,
                    cost_per_unit=gpu_config.cost_per_unit,
                    availability=np.random.uniform(0.98, 0.999),
                    efficiency_factor=gpu_config.efficiency_factor
                )
                dc.add_resource(gpu)
            
            total_gpu_power_w -= num_gpus * gpu_config.power_consumption
    
    def _add_cpus(self, dc: DataCenter, available_power_mw: float) -> None:
        """Add CPU resources to data center."""
        cpu_models = [
            ("Intel Xeon Platinum 8380", 32, 270, 8000),
            ("AMD EPYC 7763", 64, 280, 7000),
            ("Intel Xeon Gold 6338", 32, 205, 3000),
            ("AMD EPYC 7443", 24, 180, 2000)
        ]
        
        total_cpu_power_w = available_power_mw * 1_000_000
        
        while total_cpu_power_w > 0:
            model, cores, power_w, cost = random.choice(cpu_models)
            max_cpus = int(total_cpu_power_w / power_w)
            
            if max_cpus <= 0:
                break
            
            num_cpus = min(random.randint(1, 5), max_cpus)
            
            for j in range(num_cpus):
                cpu = ComputingResource(
                    id=f"{dc.name}-CPU-{j+1}",
                    resource_type="CPU",
                    model=model,
                    capacity=cores,
                    power_consumption=power_w,
                    cost_per_unit=cost,
                    availability=np.random.uniform(0.99, 0.9999)
                )
                dc.add_resource(cpu)
            
            total_cpu_power_w -= num_cpus * power_w
    
    def _add_memory(self, dc: DataCenter, available_power_mw: float) -> None:
        """Add memory resources to data center."""
        memory_configs = [
            (128, 15, 200),   # 128GB, 15W, $200
            (256, 25, 400),   # 256GB, 25W, $400
            (512, 45, 800),   # 512GB, 45W, $800
            (1024, 80, 1500)  # 1TB, 80W, $1500
        ]
        
        total_memory_power_w = available_power_mw * 1_000_000
        
        while total_memory_power_w > 0:
            capacity_gb, power_w, cost = random.choice(memory_configs)
            max_modules = int(total_memory_power_w / power_w)
            
            if max_modules <= 0:
                break
            
            num_modules = min(random.randint(1, 3), max_modules)
            
            for j in range(num_modules):
                memory = ComputingResource(
                    id=f"{dc.name}-MEM-{j+1}",
                    resource_type="Memory",
                    model=f"{capacity_gb}GB DDR4",
                    capacity=capacity_gb,
                    power_consumption=power_w,
                    cost_per_unit=cost,
                    availability=np.random.uniform(0.999, 0.9999)
                )
                dc.add_resource(memory)
            
            total_memory_power_w -= num_modules * power_w
    
    def _add_storage(self, dc: DataCenter, available_power_mw: float) -> None:
        """Add storage resources to data center."""
        storage_configs = [
            (1000, 10, 100),   # 1TB SSD, 10W, $100
            (2000, 15, 180),   # 2TB SSD, 15W, $180
            (4000, 20, 350),   # 4TB SSD, 20W, $350
            (8000, 25, 650)    # 8TB SSD, 25W, $650
        ]
        
        total_storage_power_w = available_power_mw * 1_000_000
        
        while total_storage_power_w > 0:
            capacity_gb, power_w, cost = random.choice(storage_configs)
            max_drives = int(total_storage_power_w / power_w)
            
            if max_drives <= 0:
                break
            
            num_drives = min(random.randint(1, 4), max_drives)
            
            for j in range(num_drives):
                storage = ComputingResource(
                    id=f"{dc.name}-STOR-{j+1}",
                    resource_type="Storage",
                    model=f"{capacity_gb}GB SSD",
                    capacity=capacity_gb,
                    power_consumption=power_w,
                    cost_per_unit=cost,
                    availability=np.random.uniform(0.9995, 0.9999)
                )
                dc.add_resource(storage)
            
            total_storage_power_w -= num_drives * power_w
    
    def _get_regulatory_environment(self, region: str) -> str:
        """Get regulatory environment for region."""
        regulatory_map = {
            "US-East": "GDPR-compliant",
            "US-West": "CCPA-compliant",
            "US-Central": "HIPAA-compliant",
            "Europe-West": "GDPR-compliant",
            "Europe-East": "GDPR-compliant",
            "Asia-Pacific": "Local-data-sovereignty",
            "Middle-East": "Local-data-sovereignty",
            "South-America": "LGPD-compliant",
            "Africa": "Local-data-sovereignty",
            "Australia": "Privacy-Act-compliant"
        }
        return regulatory_map.get(region, "Standard-compliance")
    
    def generate_economic_environment(self) -> EconomicEnvironment:
        """
        Generate synthetic economic environment.
        
        Returns:
            EconomicEnvironment object
        """
        self.logger.info("Generating economic environment...")
        
        # Create energy markets for each region
        energy_markets = []
        for region in self.regions:
            if region in ENERGY_MARKET_CONFIGS:
                energy_markets.append(ENERGY_MARKET_CONFIGS[region])
            else:
                # Create default energy market
                energy_market = EnergyMarket(
                    region=region,
                    base_electricity_cost=np.random.uniform(0.08, 0.15),
                    time_of_use_rates={"peak": 1.3, "off_peak": 0.7},
                    renewable_energy_availability=np.random.uniform(0.1, 0.4),
                    carbon_tax_rate=np.random.uniform(30, 100),
                    grid_stability_factor=np.random.uniform(0.8, 0.95),
                    volatility_factor=np.random.uniform(0.05, 0.15)
                )
                energy_markets.append(energy_market)
        
        # Create supply chains
        supply_chains = list(SUPPLY_CHAIN_CONFIGS.values())
        
        # Create labor markets
        labor_markets = []
        for region in self.regions:
            labor_market = {
                "region": region,
                "skilled_technician_salary": np.random.uniform(60000, 120000),
                "availability_factor": np.random.uniform(0.7, 0.95),
                "training_cost_per_employee": np.random.uniform(5000, 15000),
                "turnover_rate": np.random.uniform(0.05, 0.20)
            }
            labor_markets.append(labor_market)
        
        # Create economic environment
        economic_env = EconomicEnvironment(
            energy_markets=energy_markets,
            supply_chains=supply_chains,
            labor_markets=labor_markets,
            inflation_rate=self.config.inflation_rate,
            interest_rate=self.config.interest_rate,
            carbon_tax_rate=self.config.carbon_tax_rate
        )
        
        self.logger.info("Economic environment generated")
        return economic_env
    
    def generate_workload_patterns(self) -> List[WorkloadPattern]:
        """
        Generate synthetic workload patterns.
        
        Returns:
            List of WorkloadPattern objects
        """
        self.logger.info("Generating workload patterns...")
        
        workloads = []
        num_workloads = self.config.num_workloads
        
        # Use predefined templates as base
        template_names = list(WORKLOAD_TEMPLATES.keys())
        
        for i in range(num_workloads):
            # Select random template
            template_name = random.choice(template_names)
            template = WORKLOAD_TEMPLATES[template_name]
            
            # Create workload with variations
            workload = self._create_workload_variation(template, i)
            workloads.append(workload)
        
        self.logger.info(f"Generated {len(workloads)} workload patterns")
        return workloads
    
    def _create_workload_variation(self, template: WorkloadPattern, index: int) -> WorkloadPattern:
        """Create a variation of a workload template."""
        # Add random variations to template parameters
        variation_factor = np.random.uniform(0.8, 1.2)
        
        if isinstance(template, TrainingJob):
            return TrainingJob(
                id=f"{template.id}_var_{index}",
                name=f"{template.name} (Variation {index})",
                workload_type=template.workload_type,
                model_type=template.model_type,
                priority=template.priority,
                sla_requirements=template.sla_requirements,
                compute_intensity=template.compute_intensity * variation_factor,
                memory_requirement=template.memory_requirement * variation_factor,
                storage_requirement=template.storage_requirement * variation_factor,
                network_bandwidth=template.network_bandwidth * variation_factor,
                duration_hours=template.duration_hours * variation_factor,
                frequency_per_day=template.frequency_per_day,
                time_critical=template.time_critical,
                model_size_billions=template.model_size_billions * variation_factor,
                training_data_size_tb=template.training_data_size_tb * variation_factor,
                convergence_epochs=template.convergence_epochs,
                checkpoint_frequency=template.checkpoint_frequency,
                distributed_training=template.distributed_training
            )
        
        elif isinstance(template, InferenceRequest):
            return InferenceRequest(
                id=f"{template.id}_var_{index}",
                name=f"{template.name} (Variation {index})",
                workload_type=template.workload_type,
                model_type=template.model_type,
                priority=template.priority,
                sla_requirements=template.sla_requirements,
                compute_intensity=template.compute_intensity * variation_factor,
                memory_requirement=template.memory_requirement * variation_factor,
                storage_requirement=template.storage_requirement * variation_factor,
                network_bandwidth=template.network_bandwidth * variation_factor,
                duration_hours=template.duration_hours,
                frequency_per_day=template.frequency_per_day,
                time_critical=template.time_critical,
                requests_per_second=template.requests_per_second * variation_factor,
                latency_requirement_ms=template.latency_requirement_ms,
                throughput_requirement=template.throughput_requirement * variation_factor,
                model_loading_time_seconds=template.model_loading_time_seconds
            )
        
        elif isinstance(template, CommercialWorkload):
            return CommercialWorkload(
                id=f"{template.id}_var_{index}",
                name=f"{template.name} (Variation {index})",
                workload_type=template.workload_type,
                model_type=template.model_type,
                priority=template.priority,
                sla_requirements=template.sla_requirements,
                compute_intensity=template.compute_intensity * variation_factor,
                memory_requirement=template.memory_requirement * variation_factor,
                storage_requirement=template.storage_requirement * variation_factor,
                network_bandwidth=template.network_bandwidth * variation_factor,
                duration_hours=template.duration_hours,
                frequency_per_day=template.frequency_per_day,
                time_critical=template.time_critical,
                revenue_per_request=template.revenue_per_request * variation_factor,
                sla_penalty_cost=template.sla_penalty_cost,
                peak_load_multiplier=template.peak_load_multiplier,
                seasonal_variation=template.seasonal_variation
            )
        
        else:  # ResearchWorkload
            return ResearchWorkload(
                id=f"{template.id}_var_{index}",
                name=f"{template.name} (Variation {index})",
                workload_type=template.workload_type,
                model_type=template.model_type,
                priority=template.priority,
                sla_requirements=template.sla_requirements,
                compute_intensity=template.compute_intensity * variation_factor,
                memory_requirement=template.memory_requirement * variation_factor,
                storage_requirement=template.storage_requirement * variation_factor,
                network_bandwidth=template.network_bandwidth * variation_factor,
                duration_hours=template.duration_hours * variation_factor,
                frequency_per_day=template.frequency_per_day,
                time_critical=template.time_critical,
                experimental_phase=template.experimental_phase,
                success_probability=template.success_probability,
                iteration_count=template.iteration_count,
                parallel_experiments=template.parallel_experiments
            )
    
    def generate_time_series_data(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "1H"
    ) -> pd.DataFrame:
        """
        Generate time series data for simulation.
        
        Args:
            start_date: Start date
            end_date: End date
            frequency: Time frequency
            
        Returns:
            DataFrame with time series data
        """
        self.logger.info("Generating time series data...")
        
        # Create time index
        time_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Generate synthetic time series
        data = {
            'timestamp': time_index,
            'energy_price': self._generate_energy_price_series(time_index),
            'workload_demand': self._generate_workload_demand_series(time_index),
            'network_traffic': self._generate_network_traffic_series(time_index),
            'temperature': self._generate_temperature_series(time_index),
            'carbon_intensity': self._generate_carbon_intensity_series(time_index)
        }
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated time series data with {len(df)} records")
        return df
    
    def _generate_energy_price_series(self, time_index: pd.DatetimeIndex) -> pd.Series:
        """Generate synthetic energy price time series."""
        # Base price with daily and weekly patterns
        base_price = 0.10
        
        # Daily pattern (higher during day)
        daily_pattern = np.sin(2 * np.pi * time_index.hour / 24) * 0.02
        
        # Weekly pattern (lower on weekends)
        weekly_pattern = np.where(time_index.weekday < 5, 0, -0.01)
        
        # Random noise
        noise = np.random.normal(0, 0.005, len(time_index))
        
        return pd.Series(base_price + daily_pattern + weekly_pattern + noise, index=time_index)
    
    def _generate_workload_demand_series(self, time_index: pd.DatetimeIndex) -> pd.Series:
        """Generate synthetic workload demand time series."""
        # Base demand
        base_demand = 1000
        
        # Daily pattern (higher during business hours)
        daily_pattern = np.sin(2 * np.pi * time_index.hour / 24) * 200
        
        # Weekly pattern (lower on weekends)
        weekly_pattern = np.where(time_index.weekday < 5, 0, -300)
        
        # Random noise
        noise = np.random.normal(0, 50, len(time_index))
        
        return pd.Series(base_demand + daily_pattern + weekly_pattern + noise, index=time_index)
    
    def _generate_network_traffic_series(self, time_index: pd.DatetimeIndex) -> pd.Series:
        """Generate synthetic network traffic time series."""
        # Base traffic
        base_traffic = 100  # Gbps
        
        # Daily pattern
        daily_pattern = np.sin(2 * np.pi * time_index.hour / 24) * 20
        
        # Weekly pattern
        weekly_pattern = np.where(time_index.weekday < 5, 0, -30)
        
        # Random noise
        noise = np.random.normal(0, 5, len(time_index))
        
        return pd.Series(base_traffic + daily_pattern + weekly_pattern + noise, index=time_index)
    
    def _generate_temperature_series(self, time_index: pd.DatetimeIndex) -> pd.Series:
        """Generate synthetic temperature time series."""
        # Base temperature (20°C)
        base_temp = 20
        
        # Daily pattern
        daily_pattern = np.sin(2 * np.pi * time_index.hour / 24) * 3
        
        # Seasonal pattern
        seasonal_pattern = np.sin(2 * np.pi * time_index.dayofyear / 365) * 10
        
        # Random noise
        noise = np.random.normal(0, 1, len(time_index))
        
        return pd.Series(base_temp + daily_pattern + seasonal_pattern + noise, index=time_index)
    
    def _generate_carbon_intensity_series(self, time_index: pd.DatetimeIndex) -> pd.Series:
        """Generate synthetic carbon intensity time series."""
        # Base carbon intensity
        base_intensity = 0.5  # kg CO2/kWh
        
        # Daily pattern (lower at night due to renewable energy)
        daily_pattern = np.sin(2 * np.pi * time_index.hour / 24) * 0.1
        
        # Seasonal pattern (lower in summer due to solar)
        seasonal_pattern = np.sin(2 * np.pi * time_index.dayofyear / 365) * 0.2
        
        # Random noise
        noise = np.random.normal(0, 0.05, len(time_index))
        
        return pd.Series(base_intensity + daily_pattern + seasonal_pattern + noise, index=time_index)

class DataGenerator:
    def generate_all(self, num_datacenters=50, num_gpus=10000):
        # Placeholder: Generate fake data
        return {
            'datacenters': [{'id': i, 'capacity': 100 + i} for i in range(num_datacenters)],
            'gpus': [{'id': i, 'type': 'H100' if i % 2 == 0 else 'A100'} for i in range(num_gpus)]
        }

    def save_data(self, data, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2) 