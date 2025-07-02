"""
Interactive dashboard for InfraOpt simulation.

This module provides a web-based dashboard for visualizing
simulation results and running interactive analyses.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class InfraOptDashboard:
    """
    Interactive dashboard for InfraOpt simulation platform.
    
    This class provides a web-based interface for:
    - Visualizing simulation results
    - Running interactive analyses
    - Exploring optimization scenarios
    - Generating custom reports
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.logger = logger
        self.app = dash.Dash(__name__, title="InfraOpt Dashboard")
        self.setup_layout()
        self.setup_callbacks()
        
        self.logger.info("InfraOptDashboard initialized")
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.H1("InfraOpt: AI Infrastructure Economics Simulation", 
                   style={'textAlign': 'center', 'color': '#007acc'}),
            
            # Navigation tabs
            dcc.Tabs([
                # Overview tab
                dcc.Tab(label='Overview', children=[
                    html.Div([
                        html.H2("Simulation Overview"),
                        html.P("Welcome to the InfraOpt simulation platform. Use this dashboard to explore AI infrastructure economics and optimization scenarios."),
                        
                        # Key metrics cards
                        html.Div([
                            html.Div([
                                html.H3("Total Infrastructure Cost"),
                                html.H2(id="total-cost", children="$0"),
                                html.P("Annual infrastructure investment and operational costs")
                            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '10px'}),
                            
                            html.Div([
                                html.H3("Power Capacity"),
                                html.H2(id="power-capacity", children="0 MW"),
                                html.P("Total power capacity across all data centers")
                            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '10px'}),
                            
                            html.Div([
                                html.H3("Average PUE"),
                                html.H2(id="avg-pue", children="0.00"),
                                html.P("Power Usage Effectiveness across infrastructure")
                            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '10px'}),
                            
                            html.Div([
                                html.H3("Data Centers"),
                                html.H2(id="dc-count", children="0"),
                                html.P("Number of data centers in simulation")
                            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '10px'})
                        ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})
                    ])
                ]),
                
                # Optimization tab
                dcc.Tab(label='Optimization', children=[
                    html.Div([
                        html.H2("Optimization Results"),
                        html.P("Explore different optimization scenarios and their outcomes."),
                        
                        # Optimization controls
                        html.Div([
                            html.Label("Optimization Objective:"),
                            dcc.Dropdown(
                                id="optimization-objective",
                                options=[
                                    {'label': 'Cost Optimization', 'value': 'cost'},
                                    {'label': 'Energy Optimization', 'value': 'energy'},
                                    {'label': 'Performance Optimization', 'value': 'performance'},
                                    {'label': 'Multi-Objective', 'value': 'multi'}
                                ],
                                value='cost'
                            ),
                            
                            html.Label("Number of Data Centers:"),
                            dcc.Slider(
                                id="dc-slider",
                                min=5,
                                max=50,
                                step=5,
                                value=10,
                                marks={i: str(i) for i in range(5, 51, 5)}
                            ),
                            
                            html.Button("Run Optimization", id="run-optimization", n_clicks=0)
                        ], style={'margin': '20px'}),
                        
                        # Results visualization
                        html.Div([
                            dcc.Graph(id="optimization-results")
                        ])
                    ])
                ]),
                
                # Data Centers tab
                dcc.Tab(label='Data Centers', children=[
                    html.Div([
                        html.H2("Data Center Analysis"),
                        html.P("Explore data center specifications and performance metrics."),
                        
                        # Data center map
                        html.Div([
                            dcc.Graph(id="dc-map")
                        ]),
                        
                        # Data center table
                        html.Div([
                            html.H3("Data Center Specifications"),
                            html.Div(id="dc-table")
                        ])
                    ])
                ]),
                
                # Workloads tab
                dcc.Tab(label='Workloads', children=[
                    html.Div([
                        html.H2("Workload Analysis"),
                        html.P("Analyze workload patterns and resource requirements."),
                        
                        # Workload distribution
                        html.Div([
                            dcc.Graph(id="workload-distribution")
                        ]),
                        
                        # Resource requirements
                        html.Div([
                            dcc.Graph(id="resource-requirements")
                        ])
                    ])
                ]),
                
                # Risk Analysis tab
                dcc.Tab(label='Risk Analysis', children=[
                    html.Div([
                        html.H2("Risk Assessment"),
                        html.P("Evaluate financial, operational, and supply chain risks."),
                        
                        # Risk metrics
                        html.Div([
                            dcc.Graph(id="risk-metrics")
                        ]),
                        
                        # Monte Carlo simulation
                        html.Div([
                            dcc.Graph(id="monte-carlo-results")
                        ])
                    ])
                ])
            ]),
            
            # Footer
            html.Div([
                html.P("InfraOpt Simulation Platform - AI Infrastructure Economics Analysis",
                       style={'textAlign': 'center', 'color': '#666', 'marginTop': '50px'})
            ])
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output("optimization-results", "figure"),
            [Input("run-optimization", "n_clicks"),
             Input("optimization-objective", "value"),
             Input("dc-slider", "value")]
        )
        def update_optimization_results(n_clicks, objective, num_dcs):
            """Update optimization results visualization."""
            if n_clicks == 0:
                return go.Figure()
            
            # Generate sample optimization data
            data_centers = list(range(1, num_dcs + 1))
            costs = np.random.uniform(1000000, 10000000, num_dcs)
            energy = np.random.uniform(1000, 10000, num_dcs)
            performance = np.random.uniform(0.7, 0.95, num_dcs)
            
            fig = go.Figure()
            
            if objective == 'cost':
                fig.add_trace(go.Bar(x=data_centers, y=costs, name='Cost ($)'))
                fig.update_layout(title="Cost Optimization Results", yaxis_title="Cost ($)")
            elif objective == 'energy':
                fig.add_trace(go.Bar(x=data_centers, y=energy, name='Energy (MWh)'))
                fig.update_layout(title="Energy Optimization Results", yaxis_title="Energy (MWh)")
            elif objective == 'performance':
                fig.add_trace(go.Bar(x=data_centers, y=performance, name='Performance Score'))
                fig.update_layout(title="Performance Optimization Results", yaxis_title="Performance Score")
            else:  # multi-objective
                fig.add_trace(go.Scatter(x=costs, y=energy, mode='markers', name='Cost vs Energy'))
                fig.update_layout(title="Multi-Objective Optimization", xaxis_title="Cost ($)", yaxis_title="Energy (MWh)")
            
            return fig
        
        @self.app.callback(
            Output("dc-map", "figure"),
            [Input("dc-slider", "value")]
        )
        def update_dc_map(num_dcs):
            """Update data center map."""
            # Generate sample geographic data
            locations = ['US-East', 'US-West', 'Europe', 'Asia-Pacific']
            counts = np.random.multinomial(num_dcs, [0.3, 0.3, 0.2, 0.2])
            
            fig = go.Figure(data=go.Choropleth(
                locations=locations,
                z=counts,
                locationmode='country names',
                colorscale='Viridis',
                colorbar_title="Data Centers"
            ))
            
            fig.update_layout(
                title="Data Center Geographic Distribution",
                geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular')
            )
            
            return fig
        
        @self.app.callback(
            Output("workload-distribution", "figure"),
            [Input("dc-slider", "value")]
        )
        def update_workload_distribution(num_dcs):
            """Update workload distribution visualization."""
            workload_types = ['Training', 'Inference', 'Research', 'Commercial']
            counts = np.random.multinomial(num_dcs * 2, [0.4, 0.3, 0.2, 0.1])
            
            fig = go.Figure(data=[go.Pie(labels=workload_types, values=counts)])
            fig.update_layout(title="Workload Type Distribution")
            
            return fig
        
        @self.app.callback(
            Output("risk-metrics", "figure"),
            [Input("dc-slider", "value")]
        )
        def update_risk_metrics(num_dcs):
            """Update risk metrics visualization."""
            risk_types = ['Financial', 'Operational', 'Supply Chain', 'Geopolitical']
            risk_scores = np.random.uniform(0.1, 0.8, 4)
            
            fig = go.Figure(data=[go.Bar(x=risk_types, y=risk_scores)])
            fig.update_layout(
                title="Risk Assessment Metrics",
                yaxis_title="Risk Score",
                yaxis_range=[0, 1]
            )
            
            return fig
    
    def run(self, debug: bool = False, port: int = 8050):
        """
        Run the dashboard.
        
        Args:
            debug: Enable debug mode
            port: Port to run the dashboard on
        """
        self.logger.info(f"Starting InfraOpt dashboard on port {port}")
        
        try:
            self.app.run_server(
                debug=debug,
                port=port,
                host='0.0.0.0'
            )
        except Exception as e:
            self.logger.error(f"Dashboard failed to start: {str(e)}")
            raise


def create_sample_dashboard():
    """Create and run a sample dashboard."""
    dashboard = InfraOptDashboard()
    dashboard.run(debug=True)


def run_dashboard(port=8050, debug=False):
    print(f"Dashboard would run on http://127.0.0.1:{port} (debug={debug})")


if __name__ == "__main__":
    create_sample_dashboard() 