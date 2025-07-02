"""
Visualization module for InfraOpt simulation results.

This module provides comprehensive plotting and charting
capabilities for analyzing simulation results.
"""

import plotly.graph_objs as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SimulationVisualizer:
    """
    Comprehensive visualization engine for simulation results.
    
    This class provides various plotting functions for:
    - Cost analysis charts
    - Energy consumption visualizations
    - Performance metrics
    - Risk assessment plots
    - Geographic distributions
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.logger = logger
        self.color_scheme = {
            'primary': '#007acc',
            'secondary': '#ff6b35',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8'
        }
        
        self.logger.info("SimulationVisualizer initialized")
    
    def create_cost_analysis_chart(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create cost analysis visualization.
        
        Args:
            results: Simulation results
            
        Returns:
            Plotly figure object
        """
        try:
            optimization_results = results.get("optimization", {})
            
            # Extract cost data
            cost_data = []
            labels = []
            
            for opt_type, opt_result in optimization_results.items():
                if isinstance(opt_result, dict) and "objective_value" in opt_result:
                    cost_data.append(opt_result["objective_value"])
                    labels.append(opt_type.replace("_", " ").title())
            
            if not cost_data:
                return self._create_empty_figure("No cost data available")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=cost_data,
                    marker_color=self.color_scheme['primary'],
                    text=[f"${val:,.0f}" for val in cost_data],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Cost Optimization Results",
                xaxis_title="Optimization Type",
                yaxis_title="Cost ($)",
                template="plotly_white",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create cost analysis chart: {str(e)}")
            return self._create_empty_figure("Error creating cost chart")
    
    def create_energy_consumption_chart(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create energy consumption visualization.
        
        Args:
            results: Simulation results
            
        Returns:
            Plotly figure object
        """
        try:
            # Extract energy data from results
            summary_stats = results.get("summary_stats", {})
            total_power = summary_stats.get("total_power_capacity_mw", 0)
            avg_pue = summary_stats.get("average_pue", 0)
            
            # Create subplot
            fig = sp.make_subplots(
                rows=1, cols=2,
                subplot_titles=("Power Capacity Distribution", "PUE Analysis"),
                specs=[[{"type": "pie"}, {"type": "bar"}]]
            )
            
            # Power capacity pie chart (simulated data)
            power_distribution = {
                "Compute": total_power * 0.6,
                "Cooling": total_power * 0.25,
                "Networking": total_power * 0.1,
                "Other": total_power * 0.05
            }
            
            fig.add_trace(
                go.Pie(
                    labels=list(power_distribution.keys()),
                    values=list(power_distribution.values()),
                    name="Power Distribution"
                ),
                row=1, col=1
            )
            
            # PUE comparison bar chart
            pue_comparison = {
                "Current": avg_pue,
                "Industry Average": 1.58,
                "Best Practice": 1.2,
                "Target": 1.1
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(pue_comparison.keys()),
                    y=list(pue_comparison.values()),
                    marker_color=[self.color_scheme['primary'], self.color_scheme['warning'], 
                                self.color_scheme['success'], self.color_scheme['info']],
                    name="PUE Comparison"
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Energy Consumption Analysis",
                template="plotly_white",
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create energy consumption chart: {str(e)}")
            return self._create_empty_figure("Error creating energy chart")
    
    def create_performance_metrics_chart(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create performance metrics visualization.
        
        Args:
            results: Simulation results
            
        Returns:
            Plotly figure object
        """
        try:
            scheduling_results = results.get("scheduling", {})
            resource_utilization = scheduling_results.get("resource_utilization", {})
            
            if not resource_utilization:
                return self._create_empty_figure("No performance data available")
            
            # Create resource utilization chart
            resources = list(resource_utilization.keys())
            utilization_values = list(resource_utilization.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=resources,
                    y=utilization_values,
                    marker_color=[self.color_scheme['success'] if val > 0.8 else 
                                self.color_scheme['warning'] if val > 0.6 else 
                                self.color_scheme['danger'] for val in utilization_values],
                    text=[f"{val:.1%}" for val in utilization_values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Resource Utilization Analysis",
                xaxis_title="Resource Type",
                yaxis_title="Utilization Rate",
                yaxis_tickformat='.0%',
                yaxis_range=[0, 1],
                template="plotly_white",
                showlegend=False
            )
            
            # Add target line
            fig.add_hline(y=0.8, line_dash="dash", line_color=self.color_scheme['success'],
                         annotation_text="Target Utilization (80%)")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create performance metrics chart: {str(e)}")
            return self._create_empty_figure("Error creating performance chart")
    
    def create_risk_assessment_chart(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create risk assessment visualization.
        
        Args:
            results: Simulation results
            
        Returns:
            Plotly figure object
        """
        try:
            risk_results = results.get("risk_assessment", {})
            
            if not risk_results:
                return self._create_empty_figure("No risk assessment data available")
            
            # Create subplot for different risk types
            risk_types = list(risk_results.keys())
            fig = sp.make_subplots(
                rows=1, cols=len(risk_types),
                subplot_titles=[rt.replace("_", " ").title() for rt in risk_types],
                specs=[[{"type": "bar"} for _ in risk_types]]
            )
            
            for i, (risk_type, risk_data) in enumerate(risk_results.items()):
                if isinstance(risk_data, dict):
                    metrics = list(risk_data.keys())
                    values = list(risk_data.values())
                    
                    # Normalize values for visualization
                    if values:
                        max_val = max(abs(v) for v in values if isinstance(v, (int, float)))
                        if max_val > 0:
                            normalized_values = [v/max_val if isinstance(v, (int, float)) else 0 for v in values]
                        else:
                            normalized_values = values
                    else:
                        normalized_values = values
                    
                    fig.add_trace(
                        go.Bar(
                            x=metrics,
                            y=normalized_values,
                            name=risk_type.replace("_", " ").title(),
                            marker_color=self.color_scheme['danger']
                        ),
                        row=1, col=i+1
                    )
            
            fig.update_layout(
                title="Risk Assessment Analysis",
                template="plotly_white",
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create risk assessment chart: {str(e)}")
            return self._create_empty_figure("Error creating risk assessment chart")
    
    def create_geographic_distribution_chart(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create geographic distribution visualization.
        
        Args:
            results: Simulation results
            
        Returns:
            Plotly figure object
        """
        try:
            # Simulate geographic data (in real implementation, this would come from results)
            regions = ["US-East", "US-West", "Europe", "Asia-Pacific", "Other"]
            data_centers = [5, 4, 3, 2, 1]  # Simulated counts
            costs = [1000000, 1200000, 800000, 600000, 400000]  # Simulated costs
            
            fig = sp.make_subplots(
                rows=1, cols=2,
                subplot_titles=("Data Center Distribution", "Cost by Region"),
                specs=[[{"type": "pie"}, {"type": "bar"}]]
            )
            
            # Data center distribution pie chart
            fig.add_trace(
                go.Pie(
                    labels=regions,
                    values=data_centers,
                    name="Data Centers"
                ),
                row=1, col=1
            )
            
            # Cost by region bar chart
            fig.add_trace(
                go.Bar(
                    x=regions,
                    y=costs,
                    marker_color=self.color_scheme['primary'],
                    name="Costs"
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Geographic Distribution Analysis",
                template="plotly_white",
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create geographic distribution chart: {str(e)}")
            return self._create_empty_figure("Error creating geographic chart")
    
    def create_timeline_chart(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create timeline visualization for simulation results.
        
        Args:
            results: Simulation results
            
        Returns:
            Plotly figure object
        """
        try:
            # Generate sample timeline data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
            costs = np.random.uniform(800000, 1200000, len(dates))
            energy = np.random.uniform(5000, 8000, len(dates))
            
            fig = sp.make_subplots(
                rows=2, cols=1,
                subplot_titles=("Monthly Cost Trends", "Monthly Energy Consumption"),
                vertical_spacing=0.1
            )
            
            # Cost timeline
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=costs,
                    mode='lines+markers',
                    name='Costs',
                    line=dict(color=self.color_scheme['primary'])
                ),
                row=1, col=1
            )
            
            # Energy timeline
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=energy,
                    mode='lines+markers',
                    name='Energy',
                    line=dict(color=self.color_scheme['secondary'])
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Simulation Timeline Analysis",
                template="plotly_white",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create timeline chart: {str(e)}")
            return self._create_empty_figure("Error creating timeline chart")
    
    def create_comparison_chart(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create comparison visualization for different scenarios.
        
        Args:
            results: Simulation results
            
        Returns:
            Plotly figure object
        """
        try:
            # Simulate scenario comparison data
            scenarios = ["Baseline", "Optimized", "High-Performance", "Cost-Effective"]
            metrics = ["Cost ($M)", "Energy (MWh)", "Performance", "Risk Score"]
            
            # Simulated data matrix
            data_matrix = np.array([
                [10.0, 8.0, 0.7, 0.3],  # Baseline
                [8.5, 6.5, 0.8, 0.2],   # Optimized
                [12.0, 10.0, 0.9, 0.4], # High-Performance
                [7.0, 7.0, 0.6, 0.1]    # Cost-Effective
            ])
            
            # Normalize data for radar chart
            normalized_data = data_matrix / data_matrix.max(axis=0)
            
            fig = go.Figure()
            
            for i, scenario in enumerate(scenarios):
                fig.add_trace(go.Scatterpolar(
                    r=normalized_data[i],
                    theta=metrics,
                    fill='toself',
                    name=scenario
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Scenario Comparison Analysis"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison chart: {str(e)}")
            return self._create_empty_figure("Error creating comparison chart")
    
    def create_summary_dashboard(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create a comprehensive summary dashboard.
        
        Args:
            results: Simulation results
            
        Returns:
            Plotly figure object
        """
        try:
            # Create subplot grid
            fig = sp.make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Cost Analysis", "Energy Consumption",
                    "Performance Metrics", "Risk Assessment",
                    "Geographic Distribution", "Timeline"
                ),
                specs=[
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "pie"}, {"type": "scatter"}]
                ]
            )
            
            # Add various charts to subplots
            # This is a simplified version - in practice, you'd call the individual chart methods
            
            # Cost analysis (simplified)
            fig.add_trace(
                go.Bar(x=["Cost"], y=[1000000], name="Total Cost"),
                row=1, col=1
            )
            
            # Energy consumption (simplified)
            fig.add_trace(
                go.Pie(labels=["Compute", "Cooling"], values=[60, 40]),
                row=1, col=2
            )
            
            # Performance metrics (simplified)
            fig.add_trace(
                go.Bar(x=["CPU", "Memory", "Storage"], y=[0.8, 0.7, 0.6]),
                row=2, col=1
            )
            
            # Risk assessment (simplified)
            fig.add_trace(
                go.Bar(x=["Financial", "Operational"], y=[0.3, 0.2]),
                row=2, col=2
            )
            
            # Geographic distribution (simplified)
            fig.add_trace(
                go.Pie(labels=["US-East", "US-West", "Europe"], values=[40, 30, 30]),
                row=3, col=1
            )
            
            # Timeline (simplified)
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
            fig.add_trace(
                go.Scatter(x=dates, y=np.random.uniform(0.8, 1.2, len(dates))),
                row=3, col=2
            )
            
            fig.update_layout(
                title="InfraOpt Simulation Summary Dashboard",
                template="plotly_white",
                height=900,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create summary dashboard: {str(e)}")
            return self._create_empty_figure("Error creating summary dashboard")
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """
        Create an empty figure with error message.
        
        Args:
            message: Error message to display
            
        Returns:
            Empty Plotly figure
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_white"
        )
        return fig
    
    def save_chart(self, fig: go.Figure, filename: str, format: str = "html") -> str:
        """
        Save chart to file.
        
        Args:
            fig: Plotly figure object
            filename: Output filename
            format: Output format (html, png, jpg, svg, pdf)
            
        Returns:
            Path to saved file
        """
        try:
            if format == "html":
                fig.write_html(filename)
            elif format in ["png", "jpg", "svg", "pdf"]:
                fig.write_image(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Chart saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save chart: {str(e)}")
            raise 