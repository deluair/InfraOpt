import os
import pandas as pd
from datetime import datetime, timedelta
from collections import deque

from src.data.generator import DataGenerator
from src.optimizer.core import Optimizer
from src.analytics.visualizer import Visualizer
from src.analytics.reporter import Reporter
from src.models.workloads import Workload

class Simulator:
    """Orchestrates a detailed, time-step-based simulation."""

    def __init__(self, output_dir='results'):
        self.data_generator = DataGenerator()
        self.optimizer = Optimizer()
        self.visualizer = Visualizer(output_dir=f"{output_dir}/visualizations")
        self.reporter = Reporter(output_dir=f"{output_dir}/reports")
        
        self.datacenters = self.data_generator.generate_infrastructure()
        self.workload_queue = deque(self.data_generator.generate_workloads())
        self.running_workloads = []
        self.history = []

    def _assign_workloads(self):
        """Try to assign pending workloads to available GPUs."""
        assignable_workloads = list(self.workload_queue)
        for workload in assignable_workloads:
            best_gpu = self.optimizer.find_best_placement(workload, self.datacenters)
            if best_gpu:
                # Assign workload to GPU
                workload.status = "running"
                workload.assigned_gpu_id = best_gpu.id
                best_gpu.utilization = 1.0
                best_gpu.assigned_workload_id = workload.id
                
                self.running_workloads.append(workload)
                self.workload_queue.remove(workload)

    def _update_running_workloads(self, current_hour: int):
        """Update the status of running workloads and free up completed ones."""
        completed_workloads = []
        for workload in self.running_workloads:
            workload.duration_hours -= 1
            if workload.duration_hours <= 0:
                workload.status = "completed"
                # Find the GPU and free it
                for dc in self.datacenters:
                    for gpu in dc.resources:
                        if gpu.id == workload.assigned_gpu_id:
                            gpu.utilization = 0.0
                            gpu.assigned_workload_id = None
                            break
                completed_workloads.append(workload)
        
        # Remove completed workloads from the running list
        for wl in completed_workloads:
            self.running_workloads.remove(wl)
            
    def _record_metrics(self, hour: int):
        """Record a snapshot of the system's state at a given hour."""
        total_cost = sum(dc.operational_cost_per_hour for dc in self.datacenters)
        total_power = sum(dc.total_facility_power_kw for dc in self.datacenters)
        
        total_gpus = sum(len(dc.resources) for dc in self.datacenters)
        utilized_gpus = sum(1 for dc in self.datacenters for gpu in dc.resources if gpu.utilization > 0)
        gpu_utilization = utilized_gpus / total_gpus if total_gpus > 0 else 0

        self.history.append({
            "hour": hour,
            "total_cost_per_hour": total_cost,
            "total_power_kw": total_power,
            "gpu_utilization": gpu_utilization,
            "pending_workloads": len(self.workload_queue),
            "running_workloads": len(self.running_workloads),
        })

    def run_scenario(self, duration_hours: int = 72) -> str:
        """Runs the full simulation for a given duration."""
        print(f"Starting simulation for {duration_hours} hours...")
        
        for hour in range(duration_hours):
            # 1. Update running jobs and free up resources
            self._update_running_workloads(hour)
            
            # 2. Assign new jobs from the queue
            self._assign_workloads()

            # 3. Record metrics for the current hour
            self._record_metrics(hour)

            print(f"Hour {hour+1}/{duration_hours} | Running: {len(self.running_workloads)} | Pending: {len(self.workload_queue)}")

        # Create DataFrame from history
        results_df = pd.DataFrame(self.history)
        results_df.rename(columns={'hour': 'timestamp'}, inplace=True) # For compatibility with visualizer

        # Generate final report
        summary = {
            "total_operational_cost": f"${results_df['total_cost_per_hour'].sum():,.2f}",
            "total_energy_kwh": f"{results_df['total_power_kw'].sum():,.0f} kWh",
            "average_gpu_utilization": f"{results_df['gpu_utilization'].mean():.2%}",
            "workloads_processed": len(results_df) - len(self.workload_queue),
        }
        
        # Temp mapping for visualizer
        results_df.rename(columns={
            'total_cost_per_hour': 'total_cost', 
            'total_power_kw': 'energy_consumption_mwh', # Not really MWh but makes the chart work
            'gpu_utilization': 'gpu_utilization_avg'
            }, inplace=True)
            
        chart_path = self.visualizer.create_simulation_dashboard(results_df, filename="detailed_simulation_dashboard.html")
        report_path = self.reporter.generate_html_report(summary, chart_path, filename="detailed_simulation_report.html")
        
        print(f"Simulation finished. Report at: {report_path}")
        return report_path 