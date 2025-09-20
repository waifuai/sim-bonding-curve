import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .config import INITIAL_TOKEN_SUPPLY, TRADING_FEE
from .bonding_curves import calculate_bonding_curve_price
from .agent import Agent


@dataclass
class SimulationState:
    """
    Represents the complete state of the simulation.

    Attributes:
        supply: Current total token supply
        agents: List of all agents in the simulation
        step_count: Current simulation step number
        total_volume: Total trading volume
        price_history: History of prices for analysis
        transaction_count: Number of transactions processed
    """
    supply: float = INITIAL_TOKEN_SUPPLY
    agents: List[Agent] = field(default_factory=list)
    step_count: int = 0
    total_volume: float = 0.0
    price_history: List[float] = field(default_factory=list)
    transaction_count: int = 0

    def initialize_agents(self, num_agents: int) -> None:
        """Initialize agents for the simulation."""
        self.agents = [Agent(i) for i in range(num_agents)]

    def reset(self, num_agents: int) -> None:
        """Reset the simulation state to initial conditions."""
        self.supply = INITIAL_TOKEN_SUPPLY
        self.initialize_agents(num_agents)
        self.step_count = 0
        self.total_volume = 0.0
        self.price_history = []
        self.transaction_count = 0


@dataclass
class Trade:
    """
    Represents a single trade transaction.

    Attributes:
        agent_id: ID of the trading agent
        trade_type: Type of trade ("buy" or "sell")
        amount: Amount of tokens to trade
        price: Price per token
        timestamp: Simulation step when trade occurred
    """
    agent_id: int
    trade_type: str
    amount: float
    price: float
    timestamp: int


class SimulationEngine:
    """
    Handles the core simulation logic without global state.
    """

    def __init__(self, trading_fee: float = TRADING_FEE):
        """
        Initialize the simulation engine.

        Args:
            trading_fee: Fee percentage for trades
        """
        self.trading_fee = trading_fee

    def process_trades(
        self,
        agents: List[Agent],
        supply: float,
        bonding_curve_params: Dict[str, Any],
        step: int
    ) -> tuple[List[Agent], float, List[Trade], float]:
        """
        Process all trades for the current step.

        Args:
            agents: List of agents
            supply: Current token supply
            bonding_curve_params: Bonding curve parameters
            step: Current simulation step

        Returns:
            Tuple of (updated_agents, new_supply, trades_executed, volume)
        """
        trades_to_execute = []
        trades_executed = []

        # Collect all proposed trades
        for agent in agents:
            trade_type, trade_amount = agent.trade(supply, step, bonding_curve_params)
            if trade_type is not None and trade_amount > 0:
                trades_to_execute.append((agent, trade_type, trade_amount))

        total_volume = 0.0

        # Execute trades in random order to prevent ordering bias
        np.random.shuffle(trades_to_execute)

        for agent, trade_type, trade_amount in trades_to_execute:
            if trade_type == "buy":
                price = calculate_bonding_curve_price(supply, bonding_curve_params) * (1 + self.trading_fee)
                cost = trade_amount * price

                if agent.capital >= cost and cost > 0:
                    agent.capital -= cost
                    agent.tokens += trade_amount
                    supply += trade_amount
                    total_volume += cost

                    trades_executed.append(Trade(
                        agent_id=agent.agent_id,
                        trade_type="buy",
                        amount=trade_amount,
                        price=price,
                        timestamp=step
                    ))

            elif trade_type == "sell":
                if agent.tokens >= trade_amount and trade_amount > 0:
                    price = calculate_bonding_curve_price(supply, bonding_curve_params) * (1 - self.trading_fee)
                    revenue = trade_amount * price

                    agent.capital += revenue
                    agent.tokens -= trade_amount
                    supply -= trade_amount
                    total_volume += revenue

                    trades_executed.append(Trade(
                        agent_id=agent.agent_id,
                        trade_type="sell",
                        amount=trade_amount,
                        price=price,
                        timestamp=step
                    ))

        return agents, supply, trades_executed, total_volume

    def update_agent_memories(self, agents: List[Agent], current_price: float) -> None:
        """
        Update price memory for all agents.

        Args:
            agents: List of agents
            current_price: Current token price
        """
        for agent in agents:
            agent.update_memory(current_price)


def simulation_step(
    step: int,
    agents: List[Agent],
    resources: List[Any],  # Kept for compatibility
    params: Dict[str, Any],
    supply: float
) -> Dict[str, Any]:
    """
    Execute a single simulation step.

    Args:
        step: Current simulation step number
        agents: List of agents (will be modified in-place)
        resources: List of resources (for compatibility)
        params: Bonding curve parameters
        supply: Current token supply

    Returns:
        Dictionary containing simulation metrics for this step
    """
    # Create simulation engine
    engine = SimulationEngine()

    # Calculate current price
    current_price = calculate_bonding_curve_price(supply, params)

    # Update agent memories
    engine.update_agent_memories(agents, current_price)

    # Process trades
    updated_agents, new_supply, trades_executed, volume = engine.process_trades(
        agents, supply, params, step
    )

    # Calculate metrics
    agent_capitals = [agent.capital for agent in agents]
    agent_tokens = [agent.tokens for agent in agents]

    return {
        "supply": new_supply,
        "agent_capitals": agent_capitals,
        "agent_tokens": agent_tokens,
        "current_price": current_price,
        "volume": volume,
        "trade_count": len(trades_executed),
        "avg_price": current_price,
        "market_cap": new_supply * current_price,
        "total_wealth": sum(agent_capitals) + sum(t * current_price for t in agent_tokens)
    }


def create_simulation_state(num_agents: int) -> SimulationState:
    """
    Create a new simulation state with specified number of agents.

    Args:
        num_agents: Number of agents to create

    Returns:
        Initialized simulation state
    """
    state = SimulationState()
    state.initialize_agents(num_agents)
    return state