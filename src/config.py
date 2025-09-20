"""
Configuration file for the Token Economy Simulation with Bonding Curves.

This file contains all configurable parameters for the simulation, organized
into logical sections for easy modification and understanding.
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field


# =============================================================================
# SIMULATION SCALE PARAMETERS
# =============================================================================

@dataclass
class SimulationScale:
    """Parameters controlling the scale and duration of the simulation."""

    # Number of autonomous agents participating in the simulation
    NUM_AGENTS: int = 100

    # Total number of simulation steps to run
    SIMULATION_STEPS: int = 500

    # Random seed for reproducible results (set to None for random)
    RANDOM_SEED: int = 42


# =============================================================================
# INITIAL ECONOMIC CONDITIONS
# =============================================================================

@dataclass
class InitialConditions:
    """Initial economic parameters of the simulation."""

    # Starting supply of tokens in circulation
    INITIAL_TOKEN_SUPPLY: float = 100.0

    # Initial capital allocated to each agent
    INITIAL_AGENT_CAPITAL: float = 100.0

    # Starting price per token
    INITIAL_TOKEN_PRICE: float = 1.0

    # Transaction fee as a fraction (0.001 = 0.1%)
    TRADING_FEE: float = 0.001


# =============================================================================
# BONDING CURVE CONFIGURATION
# =============================================================================

@dataclass
class BondingCurveConfig:
    """Configuration for the bonding curve mechanism."""

    # Default bonding curve type
    DEFAULT_CURVE_TYPE: str = 'sigmoid'

    # Supported curve types
    SUPPORTED_CURVE_TYPES: List[str] = field(default_factory=lambda: [
        'linear', 'exponential', 'sigmoid', 'multi-segment'
    ])

    # Default parameters for each curve type
    DEFAULT_PARAMETERS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'linear': {'type': 'linear', 'm': 0.1, 'b': 1.0},
        'exponential': {'type': 'exponential', 'a': 0.1, 'k': 0.01},
        'sigmoid': {'type': 'sigmoid', 'k': 0.02, 's0': 100.0, 'k_max': 10.0},
        'multi-segment': {'type': 'multi-segment', 'breakpoint': 200.0, 'm': 0.05, 'a': 0.1, 'k': 0.02}
    })


# =============================================================================
# AGENT BEHAVIOR PARAMETERS
# =============================================================================

@dataclass
class AgentBehavior:
    """Parameters controlling agent trading behavior."""

    # Probability that an agent will consider trading each step (0-1)
    TRADE_FREQUENCY: float = 0.1

    # Range of trade sizes as fraction of available capital/tokens
    TRADE_SIZE_RANGE: Tuple[float, float] = (0.01, 0.1)

    # Number of historical prices each agent remembers
    MEMORY_SIZE: int = 10

    # Minimum price change threshold to trigger trades
    TREND_THRESHOLD: float = 0.01

    # Minimum number of steps between trades for each agent
    TREND_DELAY: int = 2

    # Risk tolerance (0 = conservative, 1 = aggressive)
    RISK_TOLERANCE: float = 0.5

    # Weight given to momentum vs mean reversion strategies
    MOMENTUM_WEIGHT: float = 0.7
    MEAN_REVERSION_WEIGHT: float = 0.3

    # Strategy diversity parameters
    STRATEGY_DIVERSITY: bool = True  # Enable different strategies per agent
    RANDOM_TRADING_PROBABILITY: float = 0.05  # Chance of random trades


# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================

@dataclass
class OptimizationConfig:
    """Parameters for the optimization process."""

    # Number of parameter combinations to test
    DEFAULT_TRIALS: int = 20

    # Number of simulation runs per parameter evaluation
    EVALUATION_RUNS: int = 2

    # Optimization algorithm settings
    ALGORITHM: str = 'random'  # Options: 'random', 'bayesian'

    # Parameter search ranges for each curve type
    PARAMETER_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        'linear': {
            'm': (0.01, 0.2),
            'b': (0.5, 2.0)
        },
        'exponential': {
            'a': (0.01, 0.2),
            'k': (0.005, 0.02)
        },
        'sigmoid': {
            'k': (0.01, 0.05),
            's0': (50.0, 150.0),
            'k_max': (2.0, 15.0)
        },
        'multi-segment': {
            'breakpoint': (100.0, 300.0),
            'm': (0.01, 0.1),
            'a': (0.01, 0.2),
            'k': (0.005, 0.03)
        }
    })


# =============================================================================
# VISUALIZATION AND OUTPUT
# =============================================================================

@dataclass
class VisualizationConfig:
    """Parameters for visualization and data output."""

    # Enable/disable plotting
    ENABLE_PLOTTING: bool = True

    # Save plots to files
    SAVE_PLOTS: bool = True

    # Plot file format
    PLOT_FORMAT: str = 'png'

    # Plot resolution
    PLOT_DPI: int = 300

    # Number of data points to show in wealth distribution plots
    WEALTH_PLOT_POINTS: int = 5

    # Enable real-time plotting during simulation
    REAL_TIME_PLOTTING: bool = False


# =============================================================================
# LOGGING AND DEBUGGING
# =============================================================================

@dataclass
class LoggingConfig:
    """Configuration for logging and debugging."""

    # Logging level
    LOG_LEVEL: str = 'INFO'

    # Enable detailed agent logging
    AGENT_LOGGING: bool = False

    # Enable trade logging
    TRADE_LOGGING: bool = False

    # Log file path
    LOG_FILE: str = 'simulation.log'

    # Maximum log file size (MB)
    MAX_LOG_SIZE: int = 10

    # Enable performance profiling
    PROFILING: bool = False


# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================

@dataclass
class PerformanceConfig:
    """Parameters for performance optimization."""

    # Use vectorized operations where possible
    VECTORIZE_OPERATIONS: bool = True

    # Batch size for bulk operations
    BATCH_SIZE: int = 100

    # Enable parallel processing for multiple optimization trials
    PARALLEL_OPTIMIZATION: bool = False

    # Number of worker processes
    NUM_WORKERS: int = 4

    # Memory limit (MB)
    MEMORY_LIMIT: int = 1000


# =============================================================================
# INSTANTIATE CONFIGURATION OBJECTS
# =============================================================================

# Create configuration instances
SIM_CONFIG = SimulationScale()
ECONOMIC_CONFIG = InitialConditions()
CURVE_CONFIG = BondingCurveConfig()
AGENT_CONFIG = AgentBehavior()
OPT_CONFIG = OptimizationConfig()
VIS_CONFIG = VisualizationConfig()
LOG_CONFIG = LoggingConfig()
PERF_CONFIG = PerformanceConfig()

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Maintain backward compatibility with existing code
NUM_AGENTS = SIM_CONFIG.NUM_AGENTS
SIMULATION_STEPS = SIM_CONFIG.SIMULATION_STEPS
INITIAL_TOKEN_SUPPLY = ECONOMIC_CONFIG.INITIAL_TOKEN_SUPPLY
INITIAL_AGENT_CAPITAL = ECONOMIC_CONFIG.INITIAL_AGENT_CAPITAL
INITIAL_TOKEN_PRICE = ECONOMIC_CONFIG.INITIAL_TOKEN_PRICE
TRADING_FEE = ECONOMIC_CONFIG.TRADING_FEE
BONDING_CURVE_TYPE = CURVE_CONFIG.DEFAULT_CURVE_TYPE

# Agent parameters
AGENT_TRADE_FREQUENCY = AGENT_CONFIG.TRADE_FREQUENCY
AGENT_TRADE_SIZE_RANGE = list(AGENT_CONFIG.TRADE_SIZE_RANGE)
AGENT_MEMORY_SIZE = AGENT_CONFIG.MEMORY_SIZE
AGENT_TREND_THRESHOLD = AGENT_CONFIG.TREND_THRESHOLD
AGENT_TREND_DELAY = AGENT_CONFIG.TREND_DELAY

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config_summary() -> Dict[str, Any]:
    """Get a summary of all configuration parameters."""
    return {
        'simulation_scale': {
            'num_agents': SIM_CONFIG.NUM_AGENTS,
            'simulation_steps': SIM_CONFIG.SIMULATION_STEPS,
            'random_seed': SIM_CONFIG.RANDOM_SEED
        },
        'economic_conditions': {
            'initial_supply': ECONOMIC_CONFIG.INITIAL_TOKEN_SUPPLY,
            'initial_capital': ECONOMIC_CONFIG.INITIAL_AGENT_CAPITAL,
            'initial_price': ECONOMIC_CONFIG.INITIAL_TOKEN_PRICE,
            'trading_fee': ECONOMIC_CONFIG.TRADING_FEE
        },
        'bonding_curve': {
            'default_type': CURVE_CONFIG.DEFAULT_CURVE_TYPE,
            'supported_types': CURVE_CONFIG.SUPPORTED_CURVE_TYPES
        },
        'agent_behavior': {
            'trade_frequency': AGENT_CONFIG.TRADE_FREQUENCY,
            'memory_size': AGENT_CONFIG.MEMORY_SIZE,
            'trend_threshold': AGENT_CONFIG.TREND_THRESHOLD
        }
    }


def validate_configuration() -> List[str]:
    """Validate the current configuration and return any issues."""
    issues = []

    # Validate simulation parameters
    if SIM_CONFIG.NUM_AGENTS <= 0:
        issues.append("NUM_AGENTS must be positive")

    if SIM_CONFIG.SIMULATION_STEPS <= 0:
        issues.append("SIMULATION_STEPS must be positive")

    # Validate economic parameters
    if ECONOMIC_CONFIG.INITIAL_TOKEN_SUPPLY < 0:
        issues.append("INITIAL_TOKEN_SUPPLY must be non-negative")

    if ECONOMIC_CONFIG.INITIAL_AGENT_CAPITAL < 0:
        issues.append("INITIAL_AGENT_CAPITAL must be non-negative")

    if ECONOMIC_CONFIG.TRADING_FEE < 0 or ECONOMIC_CONFIG.TRADING_FEE > 1:
        issues.append("TRADING_FEE must be between 0 and 1")

    # Validate agent parameters
    if not 0 <= AGENT_CONFIG.TRADE_FREQUENCY <= 1:
        issues.append("TRADE_FREQUENCY must be between 0 and 1")

    if AGENT_CONFIG.TRADE_SIZE_RANGE[0] < 0 or AGENT_CONFIG.TRADE_SIZE_RANGE[1] > 1:
        issues.append("TRADE_SIZE_RANGE values must be between 0 and 1")

    if AGENT_CONFIG.TRADE_SIZE_RANGE[0] >= AGENT_CONFIG.TRADE_SIZE_RANGE[1]:
        issues.append("TRADE_SIZE_RANGE[0] must be less than TRADE_SIZE_RANGE[1]")

    return issues


def reset_to_defaults() -> None:
    """Reset all configuration to default values."""
    global SIM_CONFIG, ECONOMIC_CONFIG, CURVE_CONFIG, AGENT_CONFIG
    global OPT_CONFIG, VIS_CONFIG, LOG_CONFIG, PERF_CONFIG

    SIM_CONFIG = SimulationScale()
    ECONOMIC_CONFIG = InitialConditions()
    CURVE_CONFIG = BondingCurveConfig()
    AGENT_CONFIG = AgentBehavior()
    OPT_CONFIG = OptimizationConfig()
    VIS_CONFIG = VisualizationConfig()
    LOG_CONFIG = LoggingConfig()
    PERF_CONFIG = PerformanceConfig()