# InfraOpt: AI Infrastructure Economics Simulation Platform

## Overview

InfraOpt is a comprehensive simulation platform for modeling and optimizing AI infrastructure economics under realistic constraints and market dynamics. This project addresses the critical challenge of efficiently allocating computational resources while balancing cost, energy consumption, performance, and sustainability goals across geographically distributed data centers.

## Key Features

- **Multi-Objective Cost Optimization**: Real-time infrastructure cost modeling with CapEx/OpEx breakdown
- **Intelligent Resource Allocation**: Dynamic load balancing across distributed GPU clusters
- **Energy Management**: Sustainability-aware power optimization with PUE calculations
- **Financial Analytics**: Comprehensive cost-benefit analysis with scenario planning
- **Risk Assessment**: Monte Carlo simulations for financial and operational risk modeling
- **Interactive Dashboard**: Real-time monitoring and visualization capabilities

## Problem Statement

The global AI infrastructure investment is projected to reach $5.2-7.9 trillion by 2030, with companies facing an 89% increase in computing costs between 2023-2025. Data center electricity consumption is expected to more than double by 2030, potentially reaching 945 TWh globally.

This simulation addresses multi-dimensional optimization challenges including:
- Dynamic resource allocation across heterogeneous infrastructure
- Energy cost minimization while meeting sustainability commitments
- Load balancing optimization in distributed GPU clusters
- Financial risk assessment under supply chain uncertainties
- Geographic arbitrage opportunities for power and cooling costs

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 (optimized for Windows environment)
- Git

### Setup Instructions

1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd InfraOpt
   ```

2. **Create a virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```powershell
   copy .env.example .env
   # Edit .env with your configuration
   ```

## Quick Start

### Run a Basic Simulation

```powershell
python main.py --scenario basic --duration 30 --output results/
```

### Generate Synthetic Data

```powershell
python main.py --generate-data --datacenters 50 --gpus 10000
```

### Launch Interactive Dashboard

```powershell
python main.py --dashboard --port 8050
```

### Run Optimization Analysis

```powershell
python main.py --optimize --constraints energy,cost,performance
```

## Command Line Usage

InfraOpt provides a command-line interface (CLI) for running simulations, generating synthetic data, launching the dashboard, and running optimization analyses. Below are the available commands and their usage:

### Show Help

```powershell
python main.py --help
```

### Run a Simulation Scenario

```powershell
python main.py simulate --scenario <scenario_name> [--duration <days>] [--output-dir <directory>]
```
- `--scenario` (required): Name of the simulation scenario to run (e.g., basic, cost_optimization)
- `--duration`: Simulation duration in days (default: 365)
- `--output-dir`: Directory to save simulation results (default: results)

**Example:**
```powershell
python main.py simulate --scenario basic --duration 30 --output-dir results
```

### Generate Synthetic Data

```powershell
python main.py generate-data [--datacenters <num>] [--gpus <num>] [--output-path <path>]
```
- `--datacenters`: Number of data centers to generate (default: 50)
- `--gpus`: Number of GPU assets to generate (default: 10000)
- `--output-path`: Path to save generated data (default: data/generated/synthetic_data.json)

**Example:**
```powershell
python main.py generate-data --datacenters 10 --gpus 5000 --output-path data/generated/test_data.json
```

### Launch the Interactive Dashboard

```powershell
python main.py dashboard [--port <port>] [--debug]
```
- `--port`: Port to run the dashboard on (default: 8050)
- `--debug`: Enable Dash debug mode

**Example:**
```powershell
python main.py dashboard --port 8050 --debug
```

### Run Optimization Analysis

```powershell
python main.py optimize [--constraints <constraints>] [--objective <objective>]
```
- `--constraints`: Optimization constraints, comma-separated (default: cost,energy,performance)
- `--objective`: Primary optimization objective (default: tco)

**Example:**
```powershell
python main.py optimize --constraints cost,energy --objective roi
```

## Project Structure

```
InfraOpt/
├── src/
│   ├── __init__.py
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── core.py              # Main simulation orchestrator
│   │   ├── scenarios.py         # Predefined simulation scenarios
│   │   └── validation.py        # Simulation validation framework
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── cost_optimizer.py    # Multi-objective cost optimization
│   │   ├── resource_optimizer.py # Resource allocation optimization
│   │   └── energy_optimizer.py  # Energy efficiency optimization
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── workload_scheduler.py # Intelligent workload scheduling
│   │   ├── load_balancer.py     # Load balancing algorithms
│   │   └── priority_queue.py    # Priority-based scheduling
│   ├── models/
│   │   ├── __init__.py
│   │   ├── infrastructure.py    # Data center and hardware models
│   │   ├── economics.py         # Economic and financial models
│   │   └── workloads.py         # Workload and job models
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── dashboard.py         # Interactive web dashboard
│   │   ├── visualizer.py        # Data visualization components
│   │   └── reporter.py          # Report generation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py         # Synthetic data generation
│   │   ├── loader.py            # Data loading and preprocessing
│   │   └── validator.py         # Data validation and quality checks
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       ├── logger.py            # Logging utilities
│       └── helpers.py           # Helper functions and utilities
├── data/
│   ├── raw/                     # Raw input data
│   ├── processed/               # Processed data
│   └── generated/               # Generated synthetic data
├── results/
│   ├── reports/                 # Generated reports
│   ├── visualizations/          # Charts and graphs
│   └── logs/                    # Simulation logs
├── tests/
│   ├── __init__.py
│   ├── test_simulator.py
│   ├── test_optimizer.py
│   └── test_models.py
├── docs/
│   ├── api.md                   # API documentation
│   ├── scenarios.md             # Scenario descriptions
│   └── examples.md              # Usage examples
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Core Components

### 1. Economic Modeling Engine
- Real-time infrastructure cost modeling (CapEx/OpEx breakdown)
- Energy pricing dynamics with time-of-use variability
- Supply chain constraint modeling
- Financial risk assessment using Monte Carlo simulations
- ROI projections under different scaling scenarios

### 2. Resource Allocation Simulator
- Dynamic load balancing across geographically distributed clusters
- Heterogeneous GPU allocation optimization
- Multi-tenancy resource sharing with isolation guarantees
- Predictive scaling based on demand forecasting
- Fault tolerance and redundancy planning

### 3. Energy Management System
- Power consumption modeling with PUE efficiency calculations
- Renewable energy integration planning
- Cooling system efficiency modeling
- Carbon accounting with real-time emissions tracking
- Grid stability impact assessment

### 4. Financial Analytics Dashboard
- Infrastructure investment scenario planning
- Break-even analysis for on-premises vs. cloud deployment
- Sensitivity analysis for key economic variables
- Currency hedging strategies for international operations
- Tax optimization across multiple jurisdictions

## Usage Examples

### Scenario 1: Cost Optimization Analysis

```python
from src.simulator import InfrastructureSimulator
from src.optimizer import CostOptimizer

# Initialize simulator
sim = InfrastructureSimulator()

# Run cost optimization scenario
results = sim.run_scenario(
    scenario_type="cost_optimization",
    duration_days=30,
    constraints=["energy", "performance", "budget"]
)

# Generate report
sim.generate_report(results, "cost_optimization_report.html")
```

### Scenario 2: Energy Efficiency Study

```python
from src.models.infrastructure import DataCenter
from src.optimizer import EnergyOptimizer

# Create data center model
dc = DataCenter(
    power_capacity=100,  # MW
    pue_target=1.2,
    location="Northern Europe"
)

# Optimize energy efficiency
optimizer = EnergyOptimizer()
efficiency_plan = optimizer.optimize(dc, target_pue=1.15)
```

### Scenario 3: Risk Assessment

```python
from src.analytics import RiskAssessor
from src.models.economics import MarketModel

# Initialize risk assessment
risk_assessor = RiskAssessor()

# Run Monte Carlo simulation
risk_results = risk_assessor.run_monte_carlo(
    iterations=10000,
    scenarios=["gpu_shortage", "energy_price_spike", "regulatory_change"]
)
```

## Advanced Features

### Dynamic Optimization Algorithms
- Multi-Armed Bandit Resource Allocation
- Game-Theoretic Load Balancing
- Stochastic Programming
- Reinforcement Learning Schedulers

### Economic Scenario Modeling
- Market Shock Simulations
- Regulatory Impact Analysis
- Technology Disruption Modeling
- Competitive Intelligence

### Risk Assessment Framework
- Value-at-Risk Calculations
- Stress Testing
- Scenario Planning
- Black Swan Event Modeling

## Configuration

The platform can be configured through environment variables or configuration files:

```yaml
# config.yaml
simulation:
  default_duration: 30
  time_step: 3600  # seconds
  
optimization:
  max_iterations: 1000
  tolerance: 1e-6
  
data_centers:
  default_pue: 1.58
  power_capacity_range: [5, 500]  # MW
  
energy:
  carbon_intensity: 0.5  # kg CO2/kWh
  renewable_target: 0.8
```

## Performance Considerations

- Parallel processing capability for large-scale simulations
- Memory-efficient data structures for handling large datasets
- Caching mechanisms for repeated calculations
- Scalable architecture supporting cloud deployment

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

Run the test suite:

```powershell
pytest tests/ -v --cov=src
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the `docs/` folder
- Review the examples in `docs/examples.md`

## Acknowledgments

- Research on AI infrastructure economics
- Industry data and benchmarks
- Academic contributions to optimization algorithms
- Open source community for supporting libraries 