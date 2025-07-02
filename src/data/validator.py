"""
Data validation utilities for InfraOpt simulation.

This module provides validation functions to ensure data quality
and consistency across the simulation platform.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Data validator for ensuring data quality and consistency.
    
    This class provides comprehensive validation functions for:
    - Data type validation
    - Range validation
    - Consistency checks
    - Completeness validation
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.logger = logger
        self.validation_results = {}
        
        self.logger.info("DataValidator initialized")
    
    def validate_data_center_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data center specifications.
        
        Args:
            data: DataFrame containing data center data
            
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        try:
            # Required columns
            required_columns = [
                "name", "location", "power_capacity", "pue", 
                "electricity_cost", "geographic_zone"
            ]
            
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                results["errors"].append(f"Missing required columns: {missing_columns}")
                results["valid"] = False
            
            # Data type validation
            if "power_capacity" in data.columns:
                if not pd.api.types.is_numeric_dtype(data["power_capacity"]):
                    results["errors"].append("power_capacity must be numeric")
                    results["valid"] = False
            
            if "pue" in data.columns:
                if not pd.api.types.is_numeric_dtype(data["pue"]):
                    results["errors"].append("pue must be numeric")
                    results["valid"] = False
            
            if "electricity_cost" in data.columns:
                if not pd.api.types.is_numeric_dtype(data["electricity_cost"]):
                    results["errors"].append("electricity_cost must be numeric")
                    results["valid"] = False
            
            # Range validation
            if "power_capacity" in data.columns:
                if (data["power_capacity"] <= 0).any():
                    results["errors"].append("power_capacity must be positive")
                    results["valid"] = False
                
                if (data["power_capacity"] > 1000).any():
                    results["warnings"].append("Some power capacities exceed 1000 MW")
            
            if "pue" in data.columns:
                if (data["pue"] < 1.0).any():
                    results["errors"].append("PUE cannot be less than 1.0")
                    results["valid"] = False
                
                if (data["pue"] > 3.0).any():
                    results["warnings"].append("Some PUE values exceed 3.0")
            
            if "electricity_cost" in data.columns:
                if (data["electricity_cost"] <= 0).any():
                    results["errors"].append("electricity_cost must be positive")
                    results["valid"] = False
                
                if (data["electricity_cost"] > 1.0).any():
                    results["warnings"].append("Some electricity costs exceed $1.0/kWh")
            
            # Uniqueness validation
            if "name" in data.columns:
                if data["name"].duplicated().any():
                    results["errors"].append("Data center names must be unique")
                    results["valid"] = False
            
            # Summary statistics
            results["summary"] = {
                "total_data_centers": len(data),
                "avg_power_capacity": data["power_capacity"].mean() if "power_capacity" in data.columns else None,
                "avg_pue": data["pue"].mean() if "pue" in data.columns else None,
                "avg_electricity_cost": data["electricity_cost"].mean() if "electricity_cost" in data.columns else None
            }
            
        except Exception as e:
            results["errors"].append(f"Validation failed: {str(e)}")
            results["valid"] = False
        
        self.validation_results["data_centers"] = results
        return results
    
    def validate_workload_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate workload pattern data.
        
        Args:
            data: DataFrame containing workload data
            
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        try:
            # Required columns
            required_columns = [
                "id", "name", "workload_type", "compute_intensity",
                "memory_requirement", "storage_requirement", "duration_hours"
            ]
            
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                results["errors"].append(f"Missing required columns: {missing_columns}")
                results["valid"] = False
            
            # Data type validation
            numeric_columns = ["compute_intensity", "memory_requirement", "storage_requirement", "duration_hours"]
            for col in numeric_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        results["errors"].append(f"{col} must be numeric")
                        results["valid"] = False
            
            # Range validation
            if "compute_intensity" in data.columns:
                if (data["compute_intensity"] <= 0).any():
                    results["errors"].append("compute_intensity must be positive")
                    results["valid"] = False
            
            if "memory_requirement" in data.columns:
                if (data["memory_requirement"] <= 0).any():
                    results["errors"].append("memory_requirement must be positive")
                    results["valid"] = False
            
            if "storage_requirement" in data.columns:
                if (data["storage_requirement"] <= 0).any():
                    results["errors"].append("storage_requirement must be positive")
                    results["valid"] = False
            
            if "duration_hours" in data.columns:
                if (data["duration_hours"] <= 0).any():
                    results["errors"].append("duration_hours must be positive")
                    results["valid"] = False
            
            # Workload type validation
            if "workload_type" in data.columns:
                valid_types = ["training", "inference", "research", "commercial"]
                invalid_types = set(data["workload_type"]) - set(valid_types)
                if invalid_types:
                    results["errors"].append(f"Invalid workload types: {invalid_types}")
                    results["valid"] = False
            
            # Uniqueness validation
            if "id" in data.columns:
                if data["id"].duplicated().any():
                    results["errors"].append("Workload IDs must be unique")
                    results["valid"] = False
            
            # Summary statistics
            results["summary"] = {
                "total_workloads": len(data),
                "workload_types": data["workload_type"].value_counts().to_dict() if "workload_type" in data.columns else None,
                "avg_compute_intensity": data["compute_intensity"].mean() if "compute_intensity" in data.columns else None,
                "avg_duration": data["duration_hours"].mean() if "duration_hours" in data.columns else None
            }
            
        except Exception as e:
            results["errors"].append(f"Validation failed: {str(e)}")
            results["valid"] = False
        
        self.validation_results["workloads"] = results
        return results
    
    def validate_economic_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate economic environment data.
        
        Args:
            data: DataFrame containing economic data
            
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        try:
            # Required columns for energy markets
            if "energy_markets" in data.columns or any("energy" in col.lower() for col in data.columns):
                energy_columns = ["region", "base_electricity_cost", "carbon_tax_rate"]
                missing_energy = set(energy_columns) - set(data.columns)
                if missing_energy:
                    results["warnings"].append(f"Missing energy market columns: {missing_energy}")
            
            # Data type validation
            numeric_columns = ["base_electricity_cost", "carbon_tax_rate", "inflation_rate", "interest_rate"]
            for col in numeric_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        results["errors"].append(f"{col} must be numeric")
                        results["valid"] = False
            
            # Range validation
            if "base_electricity_cost" in data.columns:
                if (data["base_electricity_cost"] <= 0).any():
                    results["errors"].append("base_electricity_cost must be positive")
                    results["valid"] = False
            
            if "carbon_tax_rate" in data.columns:
                if (data["carbon_tax_rate"] < 0).any():
                    results["errors"].append("carbon_tax_rate cannot be negative")
                    results["valid"] = False
            
            if "inflation_rate" in data.columns:
                if (data["inflation_rate"] < -0.5).any() or (data["inflation_rate"] > 1.0).any():
                    results["warnings"].append("Some inflation rates are outside expected range")
            
            # Summary statistics
            results["summary"] = {
                "total_regions": len(data),
                "avg_electricity_cost": data["base_electricity_cost"].mean() if "base_electricity_cost" in data.columns else None,
                "avg_carbon_tax": data["carbon_tax_rate"].mean() if "carbon_tax_rate" in data.columns else None
            }
            
        except Exception as e:
            results["errors"].append(f"Validation failed: {str(e)}")
            results["valid"] = False
        
        self.validation_results["economics"] = results
        return results
    
    def validate_simulation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate simulation configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        try:
            # Required configuration parameters
            required_params = ["simulation_time_horizon", "num_data_centers", "num_workloads"]
            for param in required_params:
                if param not in config:
                    results["errors"].append(f"Missing required parameter: {param}")
                    results["valid"] = False
                elif not isinstance(config[param], (int, float)) or config[param] <= 0:
                    results["errors"].append(f"{param} must be a positive number")
                    results["valid"] = False
            
            # Range validation
            if "simulation_time_horizon" in config:
                if config["simulation_time_horizon"] > 3650:  # 10 years
                    results["warnings"].append("Simulation time horizon exceeds 10 years")
            
            if "num_data_centers" in config:
                if config["num_data_centers"] > 100:
                    results["warnings"].append("Number of data centers exceeds 100")
            
            if "num_workloads" in config:
                if config["num_workloads"] > 1000:
                    results["warnings"].append("Number of workloads exceeds 1000")
            
            # Optimization parameters
            if "optimization_objectives" in config:
                valid_objectives = ["cost", "energy", "performance", "sustainability"]
                invalid_objectives = set(config["optimization_objectives"]) - set(valid_objectives)
                if invalid_objectives:
                    results["errors"].append(f"Invalid optimization objectives: {invalid_objectives}")
                    results["valid"] = False
            
            # Risk assessment parameters
            if "risk_simulations" in config:
                if config["risk_simulations"] > 10000:
                    results["warnings"].append("Number of risk simulations exceeds 10,000")
            
            if "confidence_level" in config:
                if not (0 <= config["confidence_level"] <= 1):
                    results["errors"].append("confidence_level must be between 0 and 1")
                    results["valid"] = False
            
            # Summary
            results["summary"] = {
                "config_parameters": len(config),
                "optimization_objectives": len(config.get("optimization_objectives", [])),
                "has_risk_assessment": "risk_simulations" in config
            }
            
        except Exception as e:
            results["errors"].append(f"Configuration validation failed: {str(e)}")
            results["valid"] = False
        
        self.validation_results["config"] = results
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation results.
        
        Returns:
            Summary of validation results
        """
        summary = {
            "overall_valid": True,
            "total_errors": 0,
            "total_warnings": 0,
            "validation_details": {}
        }
        
        for dataset_name, results in self.validation_results.items():
            summary["validation_details"][dataset_name] = {
                "valid": results["valid"],
                "error_count": len(results["errors"]),
                "warning_count": len(results["warnings"])
            }
            
            if not results["valid"]:
                summary["overall_valid"] = False
            
            summary["total_errors"] += len(results["errors"])
            summary["total_warnings"] += len(results["warnings"])
        
        return summary
    
    def generate_validation_report(self) -> str:
        """
        Generate a comprehensive validation report.
        
        Returns:
            Formatted validation report
        """
        summary = self.get_validation_summary()
        
        report = f"""
InfraOpt Data Validation Report
{'=' * 50}

Overall Status: {'✓ VALID' if summary['overall_valid'] else '✗ INVALID'}
Total Errors: {summary['total_errors']}
Total Warnings: {summary['total_warnings']}

Detailed Results:
"""
        
        for dataset_name, details in summary["validation_details"].items():
            status = "✓ VALID" if details["valid"] else "✗ INVALID"
            report += f"\n{dataset_name.upper()}:\n"
            report += f"  Status: {status}\n"
            report += f"  Errors: {details['error_count']}\n"
            report += f"  Warnings: {details['warning_count']}\n"
            
            if dataset_name in self.validation_results:
                results = self.validation_results[dataset_name]
                if results["errors"]:
                    report += f"  Error Details:\n"
                    for error in results["errors"]:
                        report += f"    - {error}\n"
                
                if results["warnings"]:
                    report += f"  Warning Details:\n"
                    for warning in results["warnings"]:
                        report += f"    - {warning}\n"
        
        return report
    
    def validate_data_consistency(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate consistency across multiple datasets.
        
        Args:
            datasets: Dictionary of datasets to validate
            
        Returns:
            Consistency validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        try:
            # Check for common keys across datasets
            common_keys = set.intersection(*[set(df.columns) for df in datasets.values()])
            if common_keys:
                results["summary"]["common_columns"] = list(common_keys)
            
            # Check for referential integrity
            if "data_centers" in datasets and "workloads" in datasets:
                dc_names = set(datasets["data_centers"]["name"]) if "name" in datasets["data_centers"].columns else set()
                workload_dcs = set(datasets["workloads"]["data_center"]) if "data_center" in datasets["workloads"].columns else set()
                
                if workload_dcs and dc_names:
                    invalid_refs = workload_dcs - dc_names
                    if invalid_refs:
                        results["errors"].append(f"Invalid data center references: {invalid_refs}")
                        results["valid"] = False
            
            # Check for temporal consistency
            temporal_columns = ["timestamp", "date", "time"]
            for dataset_name, df in datasets.items():
                for col in temporal_columns:
                    if col in df.columns:
                        try:
                            pd.to_datetime(df[col])
                        except:
                            results["warnings"].append(f"Temporal column '{col}' in '{dataset_name}' may have invalid dates")
            
            results["summary"]["total_datasets"] = len(datasets)
            results["summary"]["total_records"] = sum(len(df) for df in datasets.values())
            
        except Exception as e:
            results["errors"].append(f"Consistency validation failed: {str(e)}")
            results["valid"] = False
        
        return results 