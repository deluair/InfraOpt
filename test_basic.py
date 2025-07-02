#!/usr/bin/env python3
"""
Basic test script for InfraOpt simulation platform.

This script tests the core functionality of the platform
to ensure everything is working correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from src.core.simulator import InfraOptSimulator
        from src.core.optimizer import CostOptimizer
        from src.core.scheduler import ResourceScheduler
        print("✓ Core modules imported successfully")
        
        from src.models.infrastructure import DataCenter, ComputingResource, GPUType
        from src.models.economics import EconomicEnvironment, EnergyMarket
        from src.models.workloads import WorkloadPattern, TrainingJob
        print("✓ Model modules imported successfully")
        
        from src.data.generator import SyntheticDataGenerator
        from src.data.loader import DataLoader
        from src.data.validator import DataValidator
        print("✓ Data modules imported successfully")
        
        from src.analytics.reporter import SimulationReporter
        from src.analytics.visualizer import SimulationVisualizer
        print("✓ Analytics modules imported successfully")
        
        from src.utils.config import SimulationConfig
        from src.utils.logger import get_logger
        from src.utils.helpers import calculate_tco, estimate_roi
        print("✓ Utility modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_generation():
    """Test synthetic data generation."""
    print("\nTesting data generation...")
    
    try:
        from src.utils.config import SimulationConfig
        from src.data.generator import SyntheticDataGenerator
        
        # Create basic configuration
        config = SimulationConfig({
            "num_data_centers": 5,
            "num_workloads": 10,
            "simulation_time_horizon": 30
        })
        
        # Generate synthetic data
        generator = SyntheticDataGenerator(config)
        
        data_centers = generator.generate_data_centers()
        economic_env = generator.generate_economic_environment()
        workloads = generator.generate_workload_patterns()
        
        print(f"✓ Generated {len(data_centers)} data centers")
        print(f"✓ Generated {len(workloads)} workload patterns")
        print(f"✓ Generated economic environment with {len(economic_env.energy_markets)} energy markets")
        
        return True
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False

def test_optimization():
    """Test optimization functionality."""
    print("\nTesting optimization...")
    
    try:
        from src.utils.config import SimulationConfig
        from src.core.optimizer import CostOptimizer
        from src.data.generator import SyntheticDataGenerator
        
        # Create configuration and generate data
        config = SimulationConfig({
            "num_data_centers": 3,
            "num_workloads": 5,
            "simulation_time_horizon": 30
        })
        
        generator = SyntheticDataGenerator(config)
        data_centers = generator.generate_data_centers()
        workloads = generator.generate_workload_patterns()
        
        # Test optimization
        optimizer = CostOptimizer(config)
        
        # Test cost optimization
        cost_result = optimizer.optimize_cost(data_centers, workloads)
        print(f"✓ Cost optimization completed: {cost_result.constraints_satisfied}")
        
        # Test energy optimization
        energy_result = optimizer.optimize_energy(data_centers, workloads)
        print(f"✓ Energy optimization completed: {energy_result.constraints_satisfied}")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        return False

def test_scheduling():
    """Test resource scheduling."""
    print("\nTesting resource scheduling...")
    
    try:
        from src.utils.config import SimulationConfig
        from src.core.scheduler import ResourceScheduler
        from src.data.generator import SyntheticDataGenerator
        
        # Create configuration and generate data
        config = SimulationConfig({
            "num_data_centers": 3,
            "num_workloads": 5,
            "simulation_time_horizon": 30
        })
        
        generator = SyntheticDataGenerator(config)
        data_centers = generator.generate_data_centers()
        workloads = generator.generate_workload_patterns()
        
        # Test scheduling
        scheduler = ResourceScheduler(config)
        
        # Mock optimization results
        optimization_results = {
            "multi_objective": {
                "allocation_matrix": None,
                "constraints_satisfied": True
            }
        }
        
        # Test basic scheduling
        result = scheduler.schedule_resources(data_centers, workloads, optimization_results)
        print(f"✓ Scheduling completed: {result.success_rate:.1%} success rate")
        print(f"✓ Scheduled {len(result.decisions)} workloads")
        
        return True
        
    except Exception as e:
        print(f"✗ Scheduling failed: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from src.utils.helpers import calculate_tco, estimate_roi, format_currency
        
        # Test TCO calculation
        tco = calculate_tco(capex=1000000, opex_annual=200000, lifetime_years=5)
        print(f"✓ TCO calculation: ${tco:,.2f}")
        
        # Test ROI estimation
        roi = estimate_roi(initial_investment=1000000, annual_revenue=300000, annual_costs=200000)
        print(f"✓ ROI calculation: {roi['simple_roi']:.1%}")
        
        # Test currency formatting
        formatted = format_currency(1234567.89)
        print(f"✓ Currency formatting: {formatted}")
        
        return True
        
    except Exception as e:
        print(f"✗ Utility functions failed: {e}")
        return False

def test_reporting():
    """Test reporting functionality."""
    print("\nTesting reporting...")
    
    try:
        from src.analytics.reporter import SimulationReporter
        
        # Create sample results
        sample_results = {
            "summary_stats": {
                "total_infrastructure_cost": 5000000,
                "total_power_capacity_mw": 100,
                "average_pue": 1.58,
                "data_centers_count": 5,
                "optimization_objectives": ["cost", "energy", "performance"],
                "risk_metrics": ["financial", "operational"]
            },
            "optimization": {
                "cost_optimization": {
                    "objective_value": 4000000,
                    "constraints_satisfied": True,
                    "optimization_time": 10.5
                }
            },
            "scheduling": {
                "success_rate": 0.95,
                "scheduling_time": 5.2,
                "resource_utilization": {
                    "compute": 0.8,
                    "memory": 0.7,
                    "storage": 0.6
                }
            },
            "risk_assessment": {
                "financial_risk": {
                    "var_95": 100000,
                    "expected_loss": 50000
                },
                "operational_risk": {
                    "avg_availability": 0.995,
                    "downtime_risk": 0.005
                }
            },
            "metadata": {
                "start_time": "2024-01-01T00:00:00",
                "end_time": "2024-01-01T01:00:00",
                "data_centers_count": 5,
                "workloads_count": 10
            }
        }
        
        # Generate report
        reporter = SimulationReporter()
        report_path = reporter.generate_report(sample_results, "test_report.html")
        
        print(f"✓ Report generated: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Reporting failed: {e}")
        return False

def main():
    """Run all tests."""
    print("InfraOpt Platform - Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Optimization", test_optimization),
        ("Scheduling", test_scheduling),
        ("Utilities", test_utilities),
        ("Reporting", test_reporting)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! InfraOpt platform is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 