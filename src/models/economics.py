"""
Economic models for market dynamics and financial calculations.

This module defines the data structures for representing economic
variables and market conditions in the simulation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


@dataclass
class EnergyMarket:
    """Represents energy market conditions and pricing."""
    
    region: str
    base_electricity_cost: float  # $/kWh
    time_of_use_rates: Dict[str, float]  # Peak/Off-peak multipliers
    renewable_energy_availability: float  # Percentage
    carbon_tax_rate: float  # $/ton CO2
    grid_stability_factor: float  # 0-1 scale
    volatility_factor: float  # Price volatility
    
    def get_hourly_rate(self, hour: int, day_type: str = "weekday") -> float:
        """Get electricity rate for specific hour and day type."""
        # Time-of-use pricing logic
        if 6 <= hour <= 22:  # Peak hours
            multiplier = self.time_of_use_rates.get("peak", 1.2)
        else:  # Off-peak hours
            multiplier = self.time_of_use_rates.get("off_peak", 0.8)
        
        # Add some randomness for market volatility
        volatility = np.random.normal(0, self.volatility_factor)
        return self.base_electricity_cost * multiplier * (1 + volatility)


@dataclass
class SupplyChain:
    """Represents supply chain conditions and constraints."""
    
    component_type: str  # "GPU", "CPU", "Memory", "Storage"
    lead_time_days: int
    cost_per_unit: float
    availability_factor: float  # 0-1 scale
    supplier_reliability: float  # 0-1 scale
    geopolitical_risk: float  # 0-1 scale
    
    def get_effective_cost(self, quantity: int) -> float:
        """Calculate effective cost considering supply chain risks."""
        # Add risk premium based on geopolitical and supplier risks
        risk_premium = (self.geopolitical_risk + (1 - self.supplier_reliability)) * 0.1
        return self.cost_per_unit * quantity * (1 + risk_premium)
    
    def get_availability_probability(self) -> float:
        """Get probability of component availability."""
        return self.availability_factor * self.supplier_reliability


@dataclass
class LaborMarket:
    """Represents labor market conditions."""
    
    region: str
    skilled_technician_salary: float  # $/year
    availability_factor: float  # 0-1 scale
    training_cost_per_employee: float  # $/employee
    turnover_rate: float  # Annual turnover percentage
    
    def get_annual_labor_cost(self, num_technicians: int) -> float:
        """Calculate annual labor cost including training and turnover."""
        base_cost = num_technicians * self.skilled_technician_salary
        training_cost = num_technicians * self.training_cost_per_employee * self.turnover_rate
        return base_cost + training_cost


@dataclass
class CurrencyExchange:
    """Represents currency exchange rates and volatility."""
    
    base_currency: str
    target_currency: str
    exchange_rate: float
    volatility_factor: float
    last_updated: datetime
    
    def get_current_rate(self) -> float:
        """Get current exchange rate with volatility."""
        volatility = np.random.normal(0, self.volatility_factor)
        return self.exchange_rate * (1 + volatility)


@dataclass
class EconomicEnvironment:
    """Represents the overall economic environment."""
    
    # Market conditions
    energy_markets: List[EnergyMarket] = field(default_factory=list)
    supply_chains: List[SupplyChain] = field(default_factory=list)
    labor_markets: List[LaborMarket] = field(default_factory=list)
    currency_rates: List[CurrencyExchange] = field(default_factory=list)
    
    # Economic indicators
    inflation_rate: float = 0.03  # 3% annual inflation
    interest_rate: float = 0.05  # 5% interest rate
    gdp_growth_rate: float = 0.025  # 2.5% GDP growth
    market_volatility: float = 0.15  # 15% market volatility
    
    # Regulatory environment
    carbon_tax_rate: float = 50.0  # $/ton CO2
    data_sovereignty_requirements: Dict[str, Any] = field(default_factory=dict)
    ai_governance_regulations: Dict[str, Any] = field(default_factory=dict)
    
    def get_energy_cost(self, region: str, hour: int, day_type: str = "weekday") -> float:
        """Get energy cost for specific region and time."""
        for market in self.energy_markets:
            if market.region == region:
                return market.get_hourly_rate(hour, day_type)
        return 0.10  # Default rate
    
    def get_supply_chain_cost(self, component_type: str, quantity: int) -> float:
        """Get supply chain cost for specific component."""
        for chain in self.supply_chains:
            if chain.component_type == component_type:
                return chain.get_effective_cost(quantity)
        return 0.0
    
    def get_labor_cost(self, region: str, num_technicians: int) -> float:
        """Get labor cost for specific region."""
        for market in self.labor_markets:
            if market.region == region:
                return market.get_annual_labor_cost(num_technicians)
        return 0.0
    
    def calculate_inflation_adjustment(self, base_cost: float, years: int) -> float:
        """Calculate inflation-adjusted cost."""
        return base_cost * (1 + self.inflation_rate) ** years
    
    def get_currency_rate(self, from_currency: str, to_currency: str) -> float:
        """Get current exchange rate between currencies."""
        for rate in self.currency_rates:
            if (rate.base_currency == from_currency and 
                rate.target_currency == to_currency):
                return rate.get_current_rate()
        return 1.0  # Default 1:1 rate


# Predefined economic scenarios
ECONOMIC_SCENARIOS = {
    "baseline": {
        "inflation_rate": 0.03,
        "interest_rate": 0.05,
        "gdp_growth_rate": 0.025,
        "market_volatility": 0.15,
        "carbon_tax_rate": 50.0
    },
    "high_inflation": {
        "inflation_rate": 0.08,
        "interest_rate": 0.08,
        "gdp_growth_rate": 0.01,
        "market_volatility": 0.25,
        "carbon_tax_rate": 75.0
    },
    "recession": {
        "inflation_rate": 0.01,
        "interest_rate": 0.02,
        "gdp_growth_rate": -0.02,
        "market_volatility": 0.30,
        "carbon_tax_rate": 30.0
    },
    "green_transition": {
        "inflation_rate": 0.04,
        "interest_rate": 0.06,
        "gdp_growth_rate": 0.03,
        "market_volatility": 0.20,
        "carbon_tax_rate": 100.0
    }
}


# Predefined energy market configurations
ENERGY_MARKET_CONFIGS = {
    "US-East": EnergyMarket(
        region="US-East",
        base_electricity_cost=0.08,
        time_of_use_rates={"peak": 1.3, "off_peak": 0.7},
        renewable_energy_availability=0.25,
        carbon_tax_rate=50.0,
        grid_stability_factor=0.9,
        volatility_factor=0.05
    ),
    "US-West": EnergyMarket(
        region="US-West",
        base_electricity_cost=0.12,
        time_of_use_rates={"peak": 1.4, "off_peak": 0.6},
        renewable_energy_availability=0.40,
        carbon_tax_rate=75.0,
        grid_stability_factor=0.85,
        volatility_factor=0.08
    ),
    "Europe": EnergyMarket(
        region="Europe",
        base_electricity_cost=0.15,
        time_of_use_rates={"peak": 1.5, "off_peak": 0.5},
        renewable_energy_availability=0.35,
        carbon_tax_rate=100.0,
        grid_stability_factor=0.95,
        volatility_factor=0.10
    ),
    "Asia-Pacific": EnergyMarket(
        region="Asia-Pacific",
        base_electricity_cost=0.10,
        time_of_use_rates={"peak": 1.2, "off_peak": 0.8},
        renewable_energy_availability=0.15,
        carbon_tax_rate=30.0,
        grid_stability_factor=0.80,
        volatility_factor=0.12
    )
}


# Predefined supply chain configurations
SUPPLY_CHAIN_CONFIGS = {
    "GPU": SupplyChain(
        component_type="GPU",
        lead_time_days=90,
        cost_per_unit=15000,
        availability_factor=0.7,
        supplier_reliability=0.8,
        geopolitical_risk=0.3
    ),
    "CPU": SupplyChain(
        component_type="CPU",
        lead_time_days=45,
        cost_per_unit=2000,
        availability_factor=0.9,
        supplier_reliability=0.95,
        geopolitical_risk=0.1
    ),
    "Memory": SupplyChain(
        component_type="Memory",
        lead_time_days=30,
        cost_per_unit=500,
        availability_factor=0.85,
        supplier_reliability=0.9,
        geopolitical_risk=0.2
    ),
    "Storage": SupplyChain(
        component_type="Storage",
        lead_time_days=60,
        cost_per_unit=1000,
        availability_factor=0.8,
        supplier_reliability=0.85,
        geopolitical_risk=0.15
    )
}

# Defines the operational cost per hour for different GPU types.
GPU_HOURLY_COST = {
    "H100": 2.50,  # Approximate cost for H100
    "A100": 1.50,  # Approximate cost for A100
    "V100": 0.75,  # Approximate cost for V100
    "RTX4090": 0.40, # Approximate cost for RTX4090
}

# Defines the electricity cost in $/kWh for different geographic locations.
ELECTRICITY_COST_PER_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.10,
    "eu-central-1": 0.18,
    "asia-pacific-1": 0.15,
}

# Defines the Power Usage Effectiveness (PUE) for different locations.
PUE_BY_LOCATION = {
    "us-east-1": 1.15,
    "us-west-2": 1.20,
    "eu-central-1": 1.12,
    "asia-pacific-1": 1.25,
} 