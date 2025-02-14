# Token Economy Simulation with Bonding Curves

## Overview
This project simulates a token-based economy where autonomous agents interact with a bonding curve mechanism. The simulation explores how different bonding curve parameters affect market stability, token supply, and price dynamics. An optimization component automatically tunes bonding curve parameters to minimize price volatility while maintaining healthy market activity.

## Key Features
- **Bonding Curve Models**: Supports multiple bonding curve types:
  - Linear
  - Exponential
  - Sigmoid
  - Multi-segment (combination of linear and exponential)
- **Autonomous Agents**: Agents with:
  - Price memory tracking
  - Trend-based trading strategies
  - Randomized trade frequency and size
- **Parameter Optimization**: Bayesian optimization for finding optimal bonding curve parameters
- **Simulation Metrics**: Tracks:
  - Token supply over time
  - Price history
  - Agent wealth distribution
  - Market volatility

## Project Structure
```bash
.
├── agent.py            # Agent class and trading logic
├── bonding_curves.py   # Bonding curve price calculations
├── config.py           # Simulation parameters and constants
├── main.py             # Main execution and visualization
├── optimization.py     # Parameter optimization logic
└── simulation.py       # Simulation state management
```

## Installation
1. Ensure Python 3.8+ is installed
2. Install required dependencies:
```bash
pip install numpy
```

## Usage
1. Configure simulation parameters in `config.py`
2. Run the simulation:
```bash
python main.py
```

### Configuration (config.py)
Key parameters:
```python
NUM_AGENTS = 100             # Number of trading agents
SIMULATION_STEPS = 500       # Simulation duration
BONDING_CURVE_TYPE = 'sigmoid'  # Default curve type
INITIAL_AGENT_CAPITAL = 100.0  # Starting capital per agent
TRADING_FEE = 0.001          # Transaction fee percentage
```

## Agent Behavior
Agents make trading decisions based on:
- Price trend analysis (window size = `AGENT_MEMORY_SIZE`)
- Randomized trade frequency (`AGENT_TRADE_FREQUENCY`)
- Minimum price movement threshold (`AGENT_TREND_THRESHOLD`)
- Trade cooldown period (`AGENT_TREND_DELAY`)

Trading logic:
1. Track historical prices in memory buffer
2. Calculate price trend relative to historical average
3. Execute buy/sell orders when trends exceed thresholds
4. Maintain minimum time between trades

## Bonding Curve Models
Implemented curve types:
1. **Linear**: `price = m*supply + b`
2. **Exponential**: `price = a*exp(k*supply)`
3. **Sigmoid**: `price = k_max / (1 + exp(-k*(supply - s0)))`
4. **Multi-segment**: Combination of linear and exponential phases

## Optimization Process
The optimization module:
1. Randomly samples parameter combinations
2. Runs multiple simulation trials
3. Evaluates based on:
   - Price standard deviation
   - Total price change
   - Supply stability
4. Selects parameters minimizing volatility while maintaining market activity

## Simulation Output
The main simulation tracks:
- Token supply over time
- Price history
- Agent capital distribution
- Token holdings distribution
- Wealth inequality metrics

Metrics are logged at regular intervals and visualized at simulation completion.