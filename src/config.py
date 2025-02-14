# --- Configuration ---
NUM_AGENTS: int = 100  # Reduced for faster optimization
SIMULATION_STEPS: int = 500  # Reduced for faster optimization
INITIAL_TOKEN_SUPPLY: float = 100.0
INITIAL_AGENT_CAPITAL: float = 100.0
INITIAL_TOKEN_PRICE: float = 1.0
TRADING_FEE: float = 0.001

BONDING_CURVE_TYPE: str = 'sigmoid'  # Default for initial setup

# Agent Trading Params (keeping these constant for now)
AGENT_TRADE_FREQUENCY: float = 0.1
AGENT_TRADE_SIZE_RANGE: list[float] = [0.01, 0.1]
AGENT_MEMORY_SIZE: int = 10
AGENT_TREND_THRESHOLD: float = 0.01
AGENT_TREND_DELAY: int = 2