"""
Simulation reporter for generating comprehensive analysis reports.

This module creates detailed reports from simulation results including
financial analysis, performance metrics, and recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.helpers import format_currency, format_percentage

logger = get_logger(__name__)


class SimulationReporter:
    """
    Comprehensive simulation report generator.
    
    This class generates detailed reports including:
    - Executive summary
    - Financial analysis
    - Performance metrics
    - Risk assessment
    - Recommendations
    """
    
    def __init__(self):
        """Initialize the reporter."""
        self.logger = logger
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive simulation report.
        
        Args:
            results: Simulation results
            output_path: Optional output path for report
            
        Returns:
            Path to generated report
        """
        self.logger.info("Generating comprehensive simulation report...")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/infraopt_report_{timestamp}.html"
        
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate report content
        report_content = self._generate_html_report(results)
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Report generated: {output_path}")
        return output_path
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InfraOpt Simulation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #007acc;
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            color: #666;
            margin: 10px 0 0 0;
            font-size: 1.1em;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #fafafa;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007acc;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 1.1em;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #007acc;
            margin: 10px 0;
        }}
        .metric-description {{
            color: #666;
            font-size: 0.9em;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .table th, .table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        .table th {{
            background-color: #007acc;
            color: white;
            font-weight: 600;
        }}
        .table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .recommendation {{
            background: #e8f4fd;
            border-left: 4px solid #007acc;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .risk-high {{
            color: #dc3545;
            font-weight: bold;
        }}
        .risk-medium {{
            color: #ffc107;
            font-weight: bold;
        }}
        .risk-low {{
            color: #28a745;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_executive_summary(results)}
        {self._generate_financial_analysis(results)}
        {self._generate_performance_metrics(results)}
        {self._generate_risk_assessment(results)}
        {self._generate_recommendations(results)}
        {self._generate_footer()}
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""
        <div class="header">
            <h1>InfraOpt Simulation Report</h1>
            <p>AI Infrastructure Economics Optimization Analysis</p>
            <p>Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
        </div>
        """
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        metadata = results.get("metadata", {})
        
        summary = f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <p>This report presents the results of a comprehensive AI infrastructure economics simulation 
            conducted using the InfraOpt platform. The simulation analyzed {metadata.get('data_centers_count', 'N/A')} 
            data centers and {metadata.get('workloads_count', 'N/A')} workload patterns to optimize 
            infrastructure economics under realistic constraints.</p>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Total Infrastructure Cost</h3>
                    <div class="metric-value">{format_currency(results.get('summary_stats', {}).get('total_infrastructure_cost', 0))}</div>
                    <div class="metric-description">Annual infrastructure investment and operational costs</div>
                </div>
                <div class="metric-card">
                    <h3>Power Capacity</h3>
                    <div class="metric-value">{results.get('summary_stats', {}).get('total_power_capacity_mw', 0):.1f} MW</div>
                    <div class="metric-description">Total power capacity across all data centers</div>
                </div>
                <div class="metric-card">
                    <h3>Average PUE</h3>
                    <div class="metric-value">{results.get('summary_stats', {}).get('average_pue', 0):.2f}</div>
                    <div class="metric-description">Power Usage Effectiveness across infrastructure</div>
                </div>
                <div class="metric-card">
                    <h3>Optimization Objectives</h3>
                    <div class="metric-value">{len(results.get('summary_stats', {}).get('optimization_objectives', []))}</div>
                    <div class="metric-description">Number of optimization objectives analyzed</div>
                </div>
            </div>
        </div>
        """
        
        return summary
    
    def _generate_financial_analysis(self, results: Dict[str, Any]) -> str:
        """Generate financial analysis section."""
        optimization_results = results.get("optimization", {})
        
        financial_analysis = f"""
        <div class="section">
            <h2>Financial Analysis</h2>
            
            <h3>Cost Optimization Results</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Optimization Type</th>
                        <th>Objective Value</th>
                        <th>Status</th>
                        <th>Optimization Time</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for opt_type, opt_result in optimization_results.items():
            if isinstance(opt_result, dict):
                financial_analysis += f"""
                    <tr>
                        <td>{opt_type.replace('_', ' ').title()}</td>
                        <td>{format_currency(opt_result.get('objective_value', 0))}</td>
                        <td>{'✓ Optimal' if opt_result.get('constraints_satisfied', False) else '✗ Failed'}</td>
                        <td>{opt_result.get('optimization_time', 0):.2f}s</td>
                    </tr>
                """
        
        financial_analysis += """
                </tbody>
            </table>
            
            <h3>Cost Breakdown</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Capital Expenditure</h3>
                    <div class="metric-value">{format_currency(self._estimate_capex(results))}</div>
                    <div class="metric-description">Initial infrastructure investment</div>
                </div>
                <div class="metric-card">
                    <h3>Operational Expenditure</h3>
                    <div class="metric-value">{format_currency(self._estimate_opex(results))}</div>
                    <div class="metric-description">Annual operational costs</div>
                </div>
                <div class="metric-card">
                    <h3>Energy Costs</h3>
                    <div class="metric-value">{format_currency(self._estimate_energy_costs(results))}</div>
                    <div class="metric-description">Annual energy consumption costs</div>
                </div>
                <div class="metric-card">
                    <h3>Total Cost of Ownership</h3>
                    <div class="metric-value">{format_currency(self._calculate_tco(results))}</div>
                    <div class="metric-description">5-year TCO including all costs</div>
                </div>
            </div>
        </div>
        """
        
        return financial_analysis
    
    def _generate_performance_metrics(self, results: Dict[str, Any]) -> str:
        """Generate performance metrics section."""
        scheduling_results = results.get("scheduling", {})
        
        performance_metrics = f"""
        <div class="section">
            <h2>Performance Metrics</h2>
            
            <h3>Scheduling Performance</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Scheduling Success Rate</td>
                        <td>{format_percentage(scheduling_results.get('success_rate', 0))}</td>
                        <td>Percentage of workloads successfully scheduled</td>
                    </tr>
                    <tr>
                        <td>Total Scheduling Time</td>
                        <td>{scheduling_results.get('scheduling_time', 0):.2f}s</td>
                        <td>Time taken to complete scheduling</td>
                    </tr>
                    <tr>
                        <td>Resource Utilization</td>
                        <td>{format_percentage(np.mean(list(scheduling_results.get('resource_utilization', {}).values())))}</td>
                        <td>Average resource utilization across data centers</td>
                    </tr>
                </tbody>
            </table>
            
            <h3>Resource Utilization Breakdown</h3>
            <div class="metric-grid">
        """
        
        resource_utilization = scheduling_results.get('resource_utilization', {})
        for resource_type, utilization in resource_utilization.items():
            performance_metrics += f"""
                <div class="metric-card">
                    <h3>{resource_type.title()} Utilization</h3>
                    <div class="metric-value">{format_percentage(utilization)}</div>
                    <div class="metric-description">Average {resource_type} utilization across infrastructure</div>
                </div>
            """
        
        performance_metrics += """
            </div>
        </div>
        """
        
        return performance_metrics
    
    def _generate_risk_assessment(self, results: Dict[str, Any]) -> str:
        """Generate risk assessment section."""
        risk_results = results.get("risk_assessment", {})
        
        risk_assessment = f"""
        <div class="section">
            <h2>Risk Assessment</h2>
            
            <h3>Financial Risk Analysis</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Risk Metric</th>
                        <th>Value</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        financial_risk = risk_results.get("financial_risk", {})
        for metric, value in financial_risk.items():
            risk_level = self._assess_risk_level(metric, value)
            risk_class = f"risk-{risk_level}"
            
            risk_assessment += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{format_currency(value)}</td>
                    <td class="{risk_class}">{risk_level.title()}</td>
                </tr>
            """
        
        risk_assessment += """
                </tbody>
            </table>
            
            <h3>Operational Risk Analysis</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Risk Metric</th>
                        <th>Value</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        operational_risk = risk_results.get("operational_risk", {})
        for metric, value in operational_risk.items():
            risk_level = self._assess_operational_risk_level(metric, value)
            risk_class = f"risk-{risk_level}"
            
            risk_assessment += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{value:.4f}</td>
                    <td class="{risk_class}">{risk_level.title()}</td>
                </tr>
            """
        
        risk_assessment += """
                </tbody>
            </table>
        </div>
        """
        
        return risk_assessment
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate recommendations section."""
        recommendations = f"""
        <div class="section">
            <h2>Recommendations</h2>
            
            <div class="recommendation">
                <h3>Infrastructure Optimization</h3>
                <p>Based on the simulation results, consider implementing the following infrastructure optimizations:</p>
                <ul>
                    <li>Deploy energy-efficient cooling systems to improve PUE</li>
                    <li>Implement dynamic power management for non-critical workloads</li>
                    <li>Consider geographic distribution for renewable energy access</li>
                    <li>Optimize resource allocation based on workload patterns</li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h3>Cost Management</h3>
                <p>To reduce total cost of ownership:</p>
                <ul>
                    <li>Negotiate better energy rates with utility providers</li>
                    <li>Implement workload scheduling optimization</li>
                    <li>Consider hybrid cloud strategies for cost optimization</li>
                    <li>Invest in energy-efficient hardware with longer lifespans</li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h3>Risk Mitigation</h3>
                <p>To address identified risks:</p>
                <ul>
                    <li>Implement redundant systems for critical workloads</li>
                    <li>Develop disaster recovery plans</li>
                    <li>Monitor supply chain risks and diversify suppliers</li>
                    <li>Establish performance monitoring and alerting systems</li>
                </ul>
            </div>
        </div>
        """
        
        return recommendations
    
    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""
        <div class="footer">
            <p>Report generated by InfraOpt Simulation Platform</p>
            <p>For questions or additional analysis, please contact the simulation team</p>
        </div>
        """
    
    def _estimate_capex(self, results: Dict[str, Any]) -> float:
        """Estimate capital expenditure."""
        summary_stats = results.get("summary_stats", {})
        total_cost = summary_stats.get("total_infrastructure_cost", 0)
        return total_cost * 0.3  # Assume 30% is capex
    
    def _estimate_opex(self, results: Dict[str, Any]) -> float:
        """Estimate operational expenditure."""
        summary_stats = results.get("summary_stats", {})
        total_cost = summary_stats.get("total_infrastructure_cost", 0)
        return total_cost * 0.7  # Assume 70% is opex
    
    def _estimate_energy_costs(self, results: Dict[str, Any]) -> float:
        """Estimate energy costs."""
        summary_stats = results.get("summary_stats", {})
        total_cost = summary_stats.get("total_infrastructure_cost", 0)
        return total_cost * 0.4  # Assume 40% is energy costs
    
    def _calculate_tco(self, results: Dict[str, Any]) -> float:
        """Calculate total cost of ownership."""
        capex = self._estimate_capex(results)
        opex = self._estimate_opex(results)
        return capex + (opex * 5)  # 5-year TCO
    
    def _assess_risk_level(self, metric: str, value: float) -> str:
        """Assess financial risk level."""
        if "var" in metric.lower():
            if value > 1000000:  # $1M
                return "high"
            elif value > 100000:  # $100K
                return "medium"
            else:
                return "low"
        else:
            return "medium"
    
    def _assess_operational_risk_level(self, metric: str, value: float) -> str:
        """Assess operational risk level."""
        if "availability" in metric.lower():
            if value < 0.95:
                return "high"
            elif value < 0.99:
                return "medium"
            else:
                return "low"
        elif "downtime" in metric.lower():
            if value > 0.05:
                return "high"
            elif value > 0.01:
                return "medium"
            else:
                return "low"
        else:
            return "medium"
    
    def generate_json_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate JSON report for programmatic access.
        
        Args:
            results: Simulation results
            output_path: Optional output path
            
        Returns:
            Path to generated JSON report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/infraopt_results_{timestamp}.json"
        
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add metadata
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0",
                "platform": "InfraOpt"
            },
            "results": results
        }
        
        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"JSON report generated: {output_path}")
        return output_path 