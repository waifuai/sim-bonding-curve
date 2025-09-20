# Token Economy Simulation with Bonding Curves

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT-0](https://img.shields.io/badge/License-MIT--0-0080ff.svg)](https://github.com/aws/mit-0)


## Overview

This project simulates a token-based economy where autonomous agents interact with a bonding curve mechanism. The simulation explores how different bonding curve parameters affect market stability, token supply, and price dynamics. An optimization component automatically tunes bonding curve parameters to minimize price volatility while maintaining healthy market activity.

The simulation provides insights into:
- Market dynamics under different bonding curve configurations
- Agent behavior patterns and wealth distribution
- Price stability and volatility characteristics
- Optimal parameter selection for stable markets

## Key Features

- **🎯 Multiple Bonding Curve Models**: Supports various mathematical models for price calculation:
  - Linear: Simple proportional pricing
  - Exponential: Rapid price growth with supply
  - Sigmoid: S-shaped curve with saturation point
  - Multi-segment: Hybrid approach combining linear and exponential phases

- **🤖 Autonomous Agents**: Intelligent trading agents with:
  - Price memory tracking and trend analysis
  - Adaptive trading strategies based on market conditions
  - Randomized trade frequency and size for realistic behavior
  - Wealth preservation and risk management

- **⚡ Parameter Optimization**: Advanced optimization using random search to:
  - Minimize price volatility
  - Maintain healthy market activity
  - Find optimal bonding curve parameters
  - Support multiple evaluation metrics

- **📊 Comprehensive Analytics**: Detailed tracking of:
  - Token supply dynamics over time
  - Price history and volatility metrics
  - Agent wealth and token distribution
  - Market efficiency and stability indicators

## Project Structure

```
.
├── src/                          # Source code directory
│   ├── agent.py                 # Agent class and trading logic
│   ├── bonding_curves.py        # Bonding curve price calculations
│   ├── config.py               # Simulation parameters and constants
│   ├── main.py                  # Main execution and visualization
│   ├── optimization.py          # Parameter optimization logic
│   ├── simulation.py            # Simulation state management
│   └── tests/                   # Test suite
│       ├── test_agent.py
│       ├── test_bonding_curves.py
│       ├── test_optimization.py
│       └── test_simulation.py
├── .venv/                       # Virtual environment (created automatically)
├── .gitignore                   # Git ignore patterns
├── pytest.ini                   # Pytest configuration
├── README.md                    # Project documentation
└── LICENSE                      # MIT-0 License
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/waifuai/sim-bonding-curve.git
   cd sim-bonding-curve
   ```

2. **Create virtual environment**:
   ```bash
   python -m uv venv .venv
   source .venv/Scripts/activate  # On Windows
   # or
   source .venv/bin/activate     # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   python -m uv pip install -e .[test]
   ```

### Alternative Installation (without uv)
```bash
pip install numpy matplotlib pytest
```

## Quick Start

1. **Run a basic simulation**:
   ```bash
   python -m src.main
   ```

2. **Run with custom parameters** (edit `src/config.py` first):
   ```bash
   python -m src.main
   ```

3. **Run tests**:
   ```bash
   python -m pytest src/tests/
   ```

## Configuration

Key parameters in `src/config.py`:

```python
# Simulation Scale
NUM_AGENTS = 100                    # Number of trading agents
SIMULATION_STEPS = 500             # Duration of simulation

# Initial Conditions
INITIAL_TOKEN_SUPPLY = 100.0       # Starting token supply
INITIAL_AGENT_CAPITAL = 100.0      # Starting capital per agent
INITIAL_TOKEN_PRICE = 1.0          # Starting token price

# Market Parameters
TRADING_FEE = 0.001               # Transaction fee (0.1%)
BONDING_CURVE_TYPE = 'sigmoid'    # Default curve type

# Agent Behavior
AGENT_TRADE_FREQUENCY = 0.1       # Probability of trading each step
AGENT_TRADE_SIZE_RANGE = [0.01, 0.1]  # Min/max trade size as fraction of holdings
AGENT_MEMORY_SIZE = 10            # Price history window
AGENT_TREND_THRESHOLD = 0.01      # Minimum trend for action
AGENT_TREND_DELAY = 2             # Steps between trades
```

### Bonding Curve Types

1. **Linear**: `price = m × supply + b`
   - Simple proportional pricing
   - Parameters: `m` (slope), `b` (intercept)

2. **Exponential**: `price = a × exp(k × supply)`
   - Rapid price growth
   - Parameters: `a` (scale), `k` (growth rate)

3. **Sigmoid**: `price = k_max / (1 + exp(-k × (supply - s0)))`
   - S-shaped curve with saturation
   - Parameters: `k` (steepness), `s0` (center), `k_max` (maximum)

4. **Multi-segment**: Combines linear and exponential phases
   - Smooth transition at breakpoint
   - Parameters: `breakpoint`, `m`, `a`, `k`

## Agent Behavior

Agents use sophisticated trading strategies based on:

- **Price Trend Analysis**: Compare current price to historical average
- **Momentum Trading**: Buy when prices are rising, sell when falling
- **Random Exploration**: Occasional random trades for market liquidity
- **Risk Management**: Maintain minimum time between trades

### Trading Decision Process
1. Track last `AGENT_MEMORY_SIZE` prices
2. Calculate price trend vs historical average
3. Execute trades when trend exceeds `AGENT_TREND_THRESHOLD`
4. Apply random trading with probability `AGENT_TRADE_FREQUENCY`
5. Enforce minimum `AGENT_TREND_DELAY` between trades

## Optimization

The optimization system finds optimal bonding curve parameters by:

1. **Parameter Sampling**: Random sampling within defined ranges
2. **Simulation Runs**: Multiple trials per parameter set
3. **Metric Evaluation**:
   - Price standard deviation (volatility)
   - Total price change
   - Supply stability
   - Market activity level
4. **Composite Scoring**: Weighted combination of metrics

### Optimization Configuration

```python
# Example for sigmoid curve optimization
optimization_ranges = {
    'k': (0.02, 0.05),      # Steepness parameter
    's0': (80, 120),        # Center point
    'k_max': (5, 10)        # Maximum price
}
```

## Results and Visualization

The simulation generates comprehensive outputs:

- **Time Series Plots**: Supply, price, and agent wealth over time
- **Distribution Analysis**: Wealth and token holding distributions
- **Volatility Metrics**: Price stability and market efficiency
- **Optimization Results**: Best parameter sets and performance

Example output includes:
- Supply dynamics showing market growth patterns
- Price evolution with volatility analysis
- Agent wealth distribution and Gini coefficient
- Performance comparison across parameter sets

## Development

### Running Tests
```bash
# Run all tests
python -m pytest src/tests/

# Run with coverage
python -m pytest --cov=src src/tests/

# Run specific test file
python -m pytest src/tests/test_bonding_curves.py
```

### Code Quality
- **Linting**: Use `flake8` or `black` for code formatting
- **Type Checking**: Add type hints for better code reliability
- **Documentation**: All functions include comprehensive docstrings

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## API Reference

### Core Classes

- **`Agent`**: Represents a trading agent with memory and decision logic
- **`SimulationState`**: Manages global simulation state and metrics
- **Bonding Curve Functions**: Price calculation for different curve types

### Key Functions

- `calculate_bonding_curve_price()`: Compute token price for given supply
- `simulation_step()`: Execute one simulation time step
- `optimize_bonding_curve()`: Find optimal bonding curve parameters
- `evaluate_parameters()`: Assess parameter set performance

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root
   ```bash
   python -m src.main  # Correct
   python src/main.py  # May cause import issues
   ```

2. **Slow Performance**: Reduce `NUM_AGENTS` or `SIMULATION_STEPS`

3. **Memory Issues**: Decrease `AGENT_MEMORY_SIZE` or simulation scale

4. **Convergence Problems**: Adjust optimization ranges in `optimization.py`

### Performance Tips
- Use smaller parameter ranges for faster optimization
- Reduce `num_runs` in evaluation for quicker feedback
- Consider parallel execution for multiple optimization trials

## License

This project is licensed under the MIT-0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this simulation in your research, please cite:

```
@software{bonding_curve_sim,
  title={Token Economy Simulation with Bonding Curves},
  author={Your Name},
  year={2024},
  url={https://github.com/waifuai/sim-bonding-curve}
}
```

## Acknowledgments

- Inspired by bonding curve implementations in DeFi protocols
- Built with modern Python scientific computing stack
- Designed for educational and research purposes