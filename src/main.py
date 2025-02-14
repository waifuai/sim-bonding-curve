import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from bonding_curve.src.config import INITIAL_TOKEN_SUPPLY, NUM_AGENTS, SIMULATION_STEPS, BONDING_CURVE_TYPE
from bonding_curve.src.bonding_curves import calculate_bonding_curve_price
from bonding_curve.src.agent import Agent
from bonding_curve.src.simulation import simulation_step
from bonding_curve.src.optimization import optimize_bonding_curve

def main():
    """
    Main entry point for the bonding curve simulation.
    """
    # Choose the bonding curve type to optimize
    OPTIMIZE_CURVE_TYPE: str = 'sigmoid' # Example: Optimize the sigmoid curve

    optimal_params: Dict[str, Any] = optimize_bonding_curve(OPTIMIZE_CURVE_TYPE, n_trials=20)

    # --- Simulate with Optimal Parameters ---
    print(f"\nSimulating with optimal parameters for {OPTIMIZE_CURVE_TYPE}: {optimal_params}")

    # Initialize simulation's global state
    from bonding_curve.src.simulation import global_state
    supply: float = INITIAL_TOKEN_SUPPLY
    agents: List[Agent] = [Agent(i) for i in range(NUM_AGENTS)]

    BONDING_CURVE_TYPE = OPTIMIZE_CURVE_TYPE # Set the global variable for plotting
    supply_history: List[float] = []
    price_history: List[float] = []
    agent_capital_history: List[List[float]] = []
    agent_token_history: List[List[float]] = []
    all_agent_capital_history: List[np.ndarray] = []
    all_agent_token_history: List[np.ndarray] = []
    start_time: float = time.time()

    for step in range(SIMULATION_STEPS):
        step_results = simulation_step(step, agents, [], optimal_params)
        current_supply = step_results["supply"]
        agent_capitals = step_results["agent_capitals"]
        agent_tokens = step_results["agent_tokens"]
        current_price = step_results["current_price"]

        supply_history.append(current_supply)
        price_history.append(current_price)
        agent_capital_history.append(agent_capitals)
        agent_token_history.append(agent_tokens)
        all_agent_capital_history.append(np.array(agent_capitals))
        all_agent_token_history.append(np.array(agent_tokens))

        if (step + 1) % 500 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            print(
                f"Step {step + 1}/{SIMULATION_STEPS} | Elapsed: {elapsed_time:.2f}s | Supply: {current_supply:.2f} | Price: {current_price:.2f} | Avg. Capital: {np.mean(agent_capitals):.2f} | Avg. Tokens: {np.mean(agent_tokens):.2f}"
            )

    print("Simulation with optimal parameters complete.")
    end_time: float = time.time()
    total_time: float = end_time - start_time
    print(f"Total simulation time: {total_time:.2f} seconds")

    # --- Analysis and Visualization ---
    supply_history = np.array(supply_history)
    price_history = np.array(price_history)
    agent_capital_history = np.array(agent_capital_history)
    agent_token_history = np.array(agent_token_history)
    all_agent_capital_history = np.array(all_agent_capital_history)
    all_agent_token_history = np.array(all_agent_token_history)

    # Plot wealth distribution over time
    num_timepoints_to_plot = 5
    timepoints = np.linspace(0, len(all_agent_capital_history) - 1, num_timepoints_to_plot, dtype=int)

if __name__ == "__main__":
    main()
