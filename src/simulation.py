import numpy as np
from typing import List, Tuple, Dict, Any

from bonding_curve.src.config import INITIAL_TOKEN_SUPPLY, NUM_AGENTS, TRADING_FEE
from bonding_curve.src.bonding_curves import calculate_bonding_curve_price
from bonding_curve.src.agent import Agent

# --- Simulation State Class ---
class SimulationState:
    """
    Represents the global state of the simulation.
    """
    def __init__(self):
        self.supply: float = INITIAL_TOKEN_SUPPLY
        self.agents: List[Agent] = [Agent(i) for i in range(NUM_AGENTS)]

global_state = SimulationState()

def _process_trades(agents: List[Agent], supply: float, bonding_curve_params: Dict[str, Any]) -> Tuple[List[Agent], float]:
    """Processes the trades for all agents."""
    trades = []
    for agent in agents:
        trade_type, trade_amount = agent.trade(supply, 0, bonding_curve_params) # current_step not used
        if trade_type is not None:
            trades.append((agent, trade_type, trade_amount))

    for agent, trade_type, trade_amount in trades:
        if trade_type == "buy":
            price = calculate_bonding_curve_price(supply, bonding_curve_params) * (1 + TRADING_FEE)
            cost = trade_amount * price
            if agent.capital >= cost:
                agent.capital -= cost
                agent.tokens += trade_amount
                supply += trade_amount

        elif trade_type == "sell":
            price = calculate_bonding_curve_price(supply, bonding_curve_params) * (1 - TRADING_FEE)
            revenue = trade_amount * price
            agent.capital += revenue
            agent.tokens -= trade_amount
            supply -= trade_amount
    return agents, supply

def simulation_step(step: int, agents: List[Agent], resources: List[Resource], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates a single step in the bonding curve simulation.

    Args:
        step (int): The current simulation step.
        agents (List[Agent]): The list of agents in the simulation.
        resources (List[Resource]): The list of resources in the simulation.
        params (Dict[str, Any]): The parameters of the bonding curve.

    Returns:
        Dict[str, Any]: A dictionary containing the simulation metrics.
    """
    bonding_curve_params = params
    global global_state
    current_price = calculate_bonding_curve_price(global_state.supply, bonding_curve_params)

    for agent in global_state.agents:
        agent.update_memory(current_price)

    global_state.agents, global_state.supply = _process_trades(global_state.agents, global_state.supply, bonding_curve_params)

    return {
        "supply": global_state.supply,
        "agent_capitals": [agent.capital for agent in global_state.agents],
        "agent_tokens": [agent.tokens for agent in global_state.agents],
        "current_price": current_price
    }