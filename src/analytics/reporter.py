"""
Simulation reporter for generating comprehensive analysis reports.

This module creates detailed reports from simulation results including
financial analysis, performance metrics, and recommendations.
"""

import os
import json
from datetime import datetime

class Reporter:
    def __init__(self, output_dir='results/reports'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_html_report(self, summary: dict, chart_path: str, filename="simulation_report.html") -> str:
        """
        Generates a comprehensive HTML report for the simulation.

        Args:
            summary: A dictionary with summary statistics from the simulation.
            chart_path: The path to the interactive chart HTML file.
            filename: The name of the output HTML report file.

        Returns:
            The path to the generated HTML report.
        """
        
        # Read the content of the chart file
        try:
            with open(chart_path, 'r', encoding='utf-8') as f:
                chart_html = f.read()
        except FileNotFoundError:
            chart_html = "<p>Chart could not be loaded.</p>"

        # Basic styling
        html_style = """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f0f2f5; color: #333; }
            .header { background-color: #fff; padding: 20px 40px; border-bottom: 1px solid #ddd; }
            h1 { margin: 0; color: #1c2938; font-size: 28px; }
            .container { padding: 40px; }
            .card { background-color: #fff; padding: 20px 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px; }
            h2 { color: #333; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-top: 0; }
            .summary { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 20px; }
            .metric { background-color: #f9fafb; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb; text-align: center; }
            .metric h3 { margin: 0; color: #6b7280; font-size: 14px; font-weight: 500; }
            .metric p { font-size: 28px; font-weight: 600; margin: 8px 0 0; color: #111827; }
            .chart { margin-top: 20px; }
        </style>
        """

        # Build summary metrics HTML
        summary_html = ""
        for key, value in summary.items():
            summary_html += f"""
            <div class="metric">
                <h3>{key.replace('_', ' ').title()}</h3>
                <p>{value}</p>
            </div>
            """

        # Assemble the final HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>InfraOpt Simulation Report</title>
            {html_style}
        </head>
        <body>
            <div class="header">
                <h1>InfraOpt Simulation Report</h1>
                <p>Report generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
            </div>
            
            <div class="container">
                <div class="card">
                    <h2>Simulation Summary</h2>
                    <div class="summary">
                        {summary_html}
                    </div>
                </div>

                <div class="card">
                    <h2>Interactive Dashboard</h2>
                    <div class="chart">
                        {chart_html}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return output_path 