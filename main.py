#!/usr/bin/env python3
"""
InfraOpt: AI Infrastructure Economics Simulation Platform

Main entry point for running simulations and generating reports.
"""

import argparse
import sys
import os
from pathlib import Path
import click
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.simulator import InfraOptSimulator
from src.utils.config import SimulationConfig, load_config, create_config_file
from src.utils.logger import setup_logging, get_logger
from src.analytics.dashboard import InfraOptDashboard, run_dashboard
from src.simulator.core import Simulator
from src.data.generator import DataGenerator

# Load environment variables from .env file
load_dotenv()

@click.group()
def cli():
    """InfraOpt: AI Infrastructure Economics Simulation Platform"""
    pass

@cli.command()
@click.option('--scenario', type=str, required=True, help='Name of the simulation scenario to run.')
@click.option('--duration', type=int, default=365, help='Simulation duration in days.')
@click.option('--output-dir', type=click.Path(file_okay=False, dir_okay=True), default='results', help='Directory to save simulation results.')
def simulate(scenario, duration, output_dir):
    """Run a simulation scenario."""
    log = setup_logger()
    log.info("Starting simulation...", scenario=scenario, duration=duration)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    sim = Simulator(output_dir=output_dir)
    results = sim.run_scenario(scenario_name=scenario, duration_days=duration)
    
    log.info("Simulation finished. Results saved.", output_dir=sim.results_path)
    print(f"Simulation for scenario '{scenario}' complete. Results are in {sim.results_path}")

@cli.command()
@click.option('--datacenters', type=int, default=50, help='Number of data centers to generate.')
@click.option('--gpus', type=int, default=10000, help='Number of GPU assets to generate.')
@click.option('--output-path', type=click.Path(dir_okay=False), default='data/generated/synthetic_data.json', help='Path to save generated data.')
def generate_data(datacenters, gpus, output_path):
    """Generate synthetic data for the simulation."""
    log = setup_logger()
    log.info("Generating synthetic data...", num_datacenters=datacenters, num_gpus=gpus)
    
    generator = DataGenerator()
    data = generator.generate_all(num_datacenters=datacenters, num_gpus=gpus)
    generator.save_data(data, output_path)
    
    log.info("Synthetic data generated successfully.", output_path=output_path)
    print(f"Synthetic data saved to {output_path}")

@cli.command()
@click.option('--port', type=int, default=8050, help='Port to run the dashboard on.')
@click.option('--debug', is_flag=True, help='Enable Dash debug mode.')
def dashboard(port, debug):
    """Launch the interactive analytics dashboard."""
    print(f"Starting dashboard on http://127.0.0.1:{port}")
    run_dashboard(port=port, debug=debug)

@cli.command()
@click.option('--constraints', type=str, default='cost,energy,performance', help='Optimization constraints (comma-separated).')
@click.option('--objective', type=str, default='tco', help='Primary optimization objective.')
def optimize(constraints, objective):
    """Run a resource optimization analysis."""
    log = setup_logger()
    log.info("Running optimization analysis...", constraints=constraints, objective=objective)
    
    # This is a placeholder for the optimization logic
    # In a real implementation, you would load data and run the optimizer
    print("Optimization analysis placeholder.")
    print(f"Constraints: {constraints.split(',')}")
    print(f"Objective: {objective}")
    log.info("Optimization analysis complete.")

def main():
    """Main entry point for InfraOpt simulation platform."""
    parser = argparse.ArgumentParser(
        description="InfraOpt: AI Infrastructure Economics Simulation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --scenario standard
  python main.py --config custom_config.yaml
  python main.py --dashboard
  python main.py --generate-data
        """
    )
    
    parser.add_argument(
        "--scenario",
        choices=["basic", "standard", "comprehensive"],
        default="standard",
        help="Simulation scenario to run"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch interactive dashboard"
    )
    
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic data only"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file or f"logs/infraopt_{args.scenario}.log"
    )
    
    logger = get_logger(__name__)
    logger.info("Starting InfraOpt simulation platform")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.dashboard:
            # Launch dashboard
            logger.info("Launching interactive dashboard...")
            dashboard = InfraOptDashboard()
            dashboard.run()
            return
        
        if args.generate_data:
            # Generate synthetic data only
            logger.info("Generating synthetic data...")
            config = load_config(args.scenario)
            if args.config:
                config = SimulationConfig.from_file(args.config)
            
            from src.data.generator import SyntheticDataGenerator
            generator = SyntheticDataGenerator(config)
            
            data_centers = generator.generate_data_centers()
            economic_env = generator.generate_economic_environment()
            workloads = generator.generate_workload_patterns()
            
            logger.info(f"Generated {len(data_centers)} data centers")
            logger.info(f"Generated {len(workloads)} workload patterns")
            return
        
        # Run simulation
        logger.info(f"Running {args.scenario} scenario...")
        
        # Load configuration
        if args.config:
            config = SimulationConfig.from_file(args.config)
        else:
            config = load_config(args.scenario)
        
        # Create and run simulator
        simulator = InfraOptSimulator(config.to_dict())
        results = simulator.run()
        
        # Generate reports
        logger.info("Generating reports...")
        report_path = simulator.generate_report(
            output_path=f"{args.output_dir}/infraopt_report.html"
        )
        
        # Print summary
        summary = simulator.get_summary_stats()
        logger.info("Simulation completed successfully!")
        logger.info(f"Total infrastructure cost: {summary.get('total_infrastructure_cost', 0):,.2f}")
        logger.info(f"Total power capacity: {summary.get('total_power_capacity_mw', 0):.1f} MW")
        logger.info(f"Average PUE: {summary.get('average_pue', 0):.2f}")
        logger.info(f"Report generated: {report_path}")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        sys.exit(1)


def create_sample_config():
    """Create a sample configuration file."""
    config_path = "config/sample_config.yaml"
    os.makedirs("config", exist_ok=True)
    create_config_file(config_path, "standard")
    print(f"Sample configuration created: {config_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print("InfraOpt: AI Infrastructure Economics Simulation Platform")
        print("=" * 60)
        print("Run 'python main.py --help' for usage information")
        print("Run 'python main.py --scenario basic' for a quick demo")
        print("Run 'python main.py --dashboard' to launch interactive dashboard")
        print()
        
        # Ask user what they want to do
        choice = input("What would you like to do?\n"
                      "1. Run basic simulation\n"
                      "2. Launch dashboard\n"
                      "3. Create sample config\n"
                      "4. Show help\n"
                      "Enter choice (1-4): ")
        
        if choice == "1":
            sys.argv = ["main.py", "--scenario", "basic"]
        elif choice == "2":
            sys.argv = ["main.py", "--dashboard"]
        elif choice == "3":
            create_sample_config()
            sys.exit(0)
        elif choice == "4":
            sys.argv = ["main.py", "--help"]
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
    
    cli() 