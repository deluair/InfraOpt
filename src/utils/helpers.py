"""
Helper functions for InfraOpt simulation.

This module contains utility functions for common calculations,
validations, and data processing operations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import math


def calculate_tco(
    capex: float,
    opex_annual: float,
    lifetime_years: int = 5,
    discount_rate: float = 0.05
) -> float:
    """
    Calculate Total Cost of Ownership (TCO).
    
    Args:
        capex: Capital expenditure
        opex_annual: Annual operational expenditure
        lifetime_years: Expected lifetime in years
        discount_rate: Discount rate for present value calculation
        
    Returns:
        Total cost of ownership
    """
    # Calculate present value of operational costs
    pv_opex = 0
    for year in range(1, lifetime_years + 1):
        pv_opex += opex_annual / ((1 + discount_rate) ** year)
    
    return capex + pv_opex


def estimate_roi(
    initial_investment: float,
    annual_revenue: float,
    annual_costs: float,
    lifetime_years: int = 5,
    discount_rate: float = 0.05
) -> Dict[str, float]:
    """
    Estimate Return on Investment (ROI) metrics.
    
    Args:
        initial_investment: Initial investment amount
        annual_revenue: Annual revenue
        annual_costs: Annual operational costs
        lifetime_years: Investment lifetime
        discount_rate: Discount rate
        
    Returns:
        Dictionary containing ROI metrics
    """
    annual_profit = annual_revenue - annual_costs
    
    # Calculate Net Present Value (NPV)
    npv = -initial_investment
    for year in range(1, lifetime_years + 1):
        npv += annual_profit / ((1 + discount_rate) ** year)
    
    # Calculate Internal Rate of Return (IRR) approximation
    total_profit = annual_profit * lifetime_years
    simple_roi = (total_profit - initial_investment) / initial_investment
    
    # Calculate payback period
    payback_years = initial_investment / annual_profit if annual_profit > 0 else float('inf')
    
    return {
        "npv": npv,
        "simple_roi": simple_roi,
        "payback_years": payback_years,
        "annual_profit": annual_profit,
        "total_profit": total_profit
    }


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check required fields
    required_fields = ["simulation_time_horizon", "num_data_centers", "num_workloads"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(config[field], (int, float)) or config[field] <= 0:
            errors.append(f"Invalid value for {field}: must be positive number")
    
    # Check optimization objectives
    if "optimization_objectives" in config:
        valid_objectives = ["cost", "energy", "performance", "sustainability"]
        for obj in config["optimization_objectives"]:
            if obj not in valid_objectives:
                errors.append(f"Invalid optimization objective: {obj}")
    
    # Check risk assessment parameters
    if "risk_simulations" in config:
        if not isinstance(config["risk_simulations"], int) or config["risk_simulations"] <= 0:
            errors.append("risk_simulations must be positive integer")
    
    if "confidence_level" in config:
        if not (0 <= config["confidence_level"] <= 1):
            errors.append("confidence_level must be between 0 and 1")
    
    return errors


def calculate_pue(
    it_power: float,
    total_power: float,
    cooling_efficiency: float = 0.8
) -> float:
    """
    Calculate Power Usage Effectiveness (PUE).
    
    Args:
        it_power: IT equipment power consumption
        total_power: Total facility power consumption
        cooling_efficiency: Cooling system efficiency
        
    Returns:
        PUE value
    """
    if it_power <= 0:
        return float('inf')
    
    return total_power / it_power


def estimate_carbon_footprint(
    power_consumption_kwh: float,
    carbon_intensity: float = 0.5  # kg CO2/kWh
) -> float:
    """
    Estimate carbon footprint from power consumption.
    
    Args:
        power_consumption_kwh: Power consumption in kWh
        carbon_intensity: Carbon intensity in kg CO2/kWh
        
    Returns:
        Carbon footprint in kg CO2
    """
    return power_consumption_kwh * carbon_intensity


def calculate_resource_efficiency(
    actual_utilization: float,
    theoretical_max: float,
    overhead_factor: float = 0.1
) -> float:
    """
    Calculate resource efficiency.
    
    Args:
        actual_utilization: Actual resource utilization
        theoretical_max: Theoretical maximum utilization
        overhead_factor: System overhead factor
        
    Returns:
        Efficiency percentage
    """
    if theoretical_max <= 0:
        return 0.0
    
    effective_max = theoretical_max * (1 - overhead_factor)
    return min(1.0, actual_utilization / effective_max)


def generate_time_series(
    start_date: datetime,
    end_date: datetime,
    frequency: str = "1H"
) -> pd.DatetimeIndex:
    """
    Generate time series index for simulation.
    
    Args:
        start_date: Start date
        end_date: End date
        frequency: Time frequency (e.g., "1H", "1D")
        
    Returns:
        Datetime index
    """
    return pd.date_range(start=start_date, end=end_date, freq=frequency)


def calculate_load_balancing_score(
    resource_utilizations: List[float]
) -> float:
    """
    Calculate load balancing efficiency score.
    
    Args:
        resource_utilizations: List of resource utilization percentages
        
    Returns:
        Load balancing score (0-1, higher is better)
    """
    if not resource_utilizations:
        return 0.0
    
    mean_util = np.mean(resource_utilizations)
    std_util = np.std(resource_utilizations)
    
    # Score based on low standard deviation (good balance)
    if mean_util == 0:
        return 0.0
    
    coefficient_of_variation = std_util / mean_util
    return max(0.0, 1.0 - coefficient_of_variation)


def estimate_network_latency(
    distance_km: float,
    network_type: str = "fiber"
) -> float:
    """
    Estimate network latency based on distance and type.
    
    Args:
        distance_km: Distance in kilometers
        network_type: Network type ("fiber", "copper", "wireless")
        
    Returns:
        Latency in milliseconds
    """
    # Speed of light in fiber: ~200,000 km/s
    # Speed of light in air: ~300,000 km/s
    # Copper: ~230,000 km/s
    
    speeds = {
        "fiber": 200000,
        "copper": 230000,
        "wireless": 300000
    }
    
    speed = speeds.get(network_type.lower(), 200000)
    
    # Calculate one-way latency
    latency_seconds = distance_km / speed
    
    # Add processing overhead
    processing_overhead = 0.001  # 1ms
    
    return (latency_seconds * 1000) + processing_overhead


def calculate_reliability_score(
    availability: float,
    mttf: float,  # Mean Time To Failure (hours)
    mttr: float   # Mean Time To Repair (hours)
) -> float:
    """
    Calculate system reliability score.
    
    Args:
        availability: System availability (0-1)
        mttf: Mean time to failure in hours
        mttr: Mean time to repair in hours
        
    Returns:
        Reliability score (0-1)
    """
    # Availability component
    availability_score = availability
    
    # Reliability component based on MTTF/MTTR ratio
    reliability_ratio = mttf / mttr if mttr > 0 else float('inf')
    reliability_score = min(1.0, reliability_ratio / 1000)  # Normalize
    
    # Combined score (weighted average)
    return 0.7 * availability_score + 0.3 * reliability_score


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format currency amount for display.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    elif currency == "EUR":
        return f"€{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format percentage value for display.
    
    Args:
        value: Value to format (0-1)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calculate percentile value.
    
    Args:
        values: List of values
        percentile: Percentile (0-100)
        
    Returns:
        Percentile value
    """
    if not values:
        return 0.0
    
    return np.percentile(values, percentile)


def normalize_values(values: List[float], method: str = "minmax") -> List[float]:
    """
    Normalize values using specified method.
    
    Args:
        values: List of values to normalize
        method: Normalization method ("minmax", "zscore", "robust")
        
    Returns:
        Normalized values
    """
    if not values:
        return []
    
    values_array = np.array(values)
    
    if method == "minmax":
        min_val = np.min(values_array)
        max_val = np.max(values_array)
        if max_val == min_val:
            return [0.5] * len(values)
        return ((values_array - min_val) / (max_val - min_val)).tolist()
    
    elif method == "zscore":
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        if std_val == 0:
            return [0.0] * len(values)
        return ((values_array - mean_val) / std_val).tolist()
    
    elif method == "robust":
        median_val = np.median(values_array)
        mad = np.median(np.abs(values_array - median_val))
        if mad == 0:
            return [0.0] * len(values)
        return ((values_array - median_val) / mad).tolist()
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for numerical columns.
    
    Args:
        data: Input dataframe
        
    Returns:
        Correlation matrix
    """
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    return data[numerical_cols].corr()


def detect_outliers(
    values: List[float],
    method: str = "iqr",
    threshold: float = 1.5
) -> List[bool]:
    """
    Detect outliers in data.
    
    Args:
        values: List of values
        method: Detection method ("iqr", "zscore")
        threshold: Detection threshold
        
    Returns:
        List of boolean flags indicating outliers
    """
    if not values:
        return []
    
    values_array = np.array(values)
    
    if method == "iqr":
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (values_array < lower_bound) | (values_array > upper_bound)
    
    elif method == "zscore":
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        if std_val == 0:
            return [False] * len(values)
        z_scores = np.abs((values_array - mean_val) / std_val)
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}") 