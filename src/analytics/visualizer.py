"""
Visualization module for InfraOpt simulation results.

This module provides comprehensive plotting and charting
capabilities for analyzing simulation results.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

class Visualizer:
    def __init__(self, output_dir='results/visualizations'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def create_simulation_dashboard(self, results_df: pd.DataFrame, filename="simulation_dashboard.html") -> str:
        """
        Creates a dashboard with plots for key simulation metrics.

        Args:
            results_df: DataFrame with simulation results.
            filename: The name of the output HTML file.

        Returns:
            The path to the generated HTML file.
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Total Cost Over Time", "Energy Consumption Over Time", "Average GPU Utilization Over Time")
        )

        # Plot Total Cost
        fig.add_trace(go.Scatter(x=results_df['timestamp'], y=results_df['total_cost'],
                                 mode='lines', name='Total Cost ($)'), row=1, col=1)

        # Plot Energy Consumption
        fig.add_trace(go.Scatter(x=results_df['timestamp'], y=results_df['energy_consumption_mwh'],
                                 mode='lines', name='Energy (MWh)', line=dict(color='orange')), row=2, col=1)

        # Plot GPU Utilization
        fig.add_trace(go.Scatter(x=results_df['timestamp'], y=results_df['gpu_utilization_avg'],
                                 mode='lines', name='GPU Utilization (%)', line=dict(color='green')), row=3, col=1)

        # Update layout
        fig.update_layout(
            title_text="InfraOpt Simulation Results Dashboard",
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
        fig.update_yaxes(title_text="Energy (MWh)", row=2, col=1)
        fig.update_yaxes(title_text="Utilization (%)", row=3, col=1)
        fig.update_yaxes(tickformat=".2%", row=3, col=1)
        
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        
        return output_path 