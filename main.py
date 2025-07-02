#!/usr/bin/env python3
"""
InfraOpt: AI Infrastructure Economics Simulation Platform

Main entry point for running simulations and generating reports.
"""

import click
from dotenv import load_dotenv
import os
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.simulator.core import Simulator
from src.analytics.dashboard import run_dashboard
from src.utils.logger import setup_logging

# Load environment variables from .env file
load_dotenv()

@click.group()
def cli():
    """InfraOpt: AI Infrastructure Economics Simulation Platform"""
    pass

@cli.command()
@click.option('--duration', type=int, default=72, help='Simulation duration in hours.')
@click.option('--output-dir', type=click.Path(file_okay=False, dir_okay=True), default='results', help='Directory to save simulation results.')
def simulate(duration, output_dir):
    """Run a detailed, time-step simulation."""
    log = setup_logging()
    log.info(f"Starting simulation... duration={duration} hours")

    sim = Simulator(output_dir=output_dir)
    sim.run_scenario(duration_hours=duration)

@cli.command()
@click.option('--port', type=int, default=8050, help='Port to run the dashboard on.')
@click.option('--debug', is_flag=True, help='Enable Dash debug mode.')
def dashboard(port, debug):
    """Launch the interactive dashboard."""
    log = setup_logging()
    log.info(f"Launching dashboard on port {port}...")
    run_dashboard(port=port, debug=debug)

if __name__ == '__main__':
    cli() 