import random
import numpy as np
from config import INITIAL_TOKEN_SUPPLY, NUM_AGENTS, SIMULATION_STEPS
from simulation import simulation_step
from agent import Agent
from typing import Dict, Any

# --- Objective Function ---
def evaluate_parameters(params: Dict[str, Any], num_runs: int=3) -> float:
    """
    Evaluates the given bonding curve parameters.

    Args:
        params (Dict[str, Any]): The parameters of the bonding curve.
        num_runs (int): The number of simulation runs to perform.

    Returns:
        float: The average score across all simulation runs.
    """
    all_metrics = []
    for _ in range(num_runs):
        # Reset simulation state for each run
        supply = INITIAL_TOKEN_SUPPLY
        agents = [Agent(i) for i in range(NUM_AGENTS)]

        price_history = []
        for step in range(SIMULATION_STEPS):
            _, _, _, current_price = simulation_step(step, params)
            price_history.append(current_price)

        metrics = {
            'price_std': np.std(price_history),
            'supply_change': abs(supply - INITIAL_TOKEN_SUPPLY),
            'price_change': np.ptp(price_history)
        }
        all_metrics.append(metrics)

    # Composite score (lower is better)
    scores = []
    for m in all_metrics:
        if m['supply_change'] < 10 or m['price_change'] < 0.5:
            scores.append(1000)  # Penalize stagnant simulations
        else:
            scores.append(m['price_std'] * 1.5 + m['price_change'] * 0.5)

    return np.mean(scores)

# --- Optimization Function ---
def optimize_bonding_curve(curve_type: str, n_trials: int=10) -> Dict[str, Any]:
    """
    Optimizes the bonding curve parameters using random search.

    Args:
        curve_type (str): The type of bonding curve to optimize.
        n_trials (int): The number of random parameter sets to test.

    Returns:
        Dict[str, Any]: The best set of parameters found.
    """
    best_params = None
    best_objective_value = float('inf') # Lower standard deviation is better

    print(f"Starting optimization for {curve_type} bonding curve...")

    for i in range(n_trials):
        print(f"Optimization Trial {i+1}/{n_trials}")
        if curve_type == 'linear':
            params = {
                'type': 'linear',
                'm': random.uniform(0.01, 0.2),
                'b': random.uniform(0.5, 2.0)
            }
        elif curve_type == 'exponential':
            params = {
                'type': 'exponential',
                'a': random.uniform(0.01, 0.2),
                'k': random.uniform(0.005, 0.02)
            }
        elif curve_type == 'sigmoid':
            params = {
                'type': 'sigmoid',
                'k': random.uniform(0.02, 0.05),  # Tighter range for steeper curve
                's0': random.uniform(80, 120),  # Center around initial supply
                'k_max': random.uniform(5, 10)
            }
        elif curve_type == 'multi-segment':
            params = {
                'type': 'multi-segment',
                'breakpoint': random.uniform(100, 300),
                'm': random.uniform(0.01, 0.1),
                'a': random.uniform(0.01, 0.2),
                'k': random.uniform(0.01, 0.03)
            }
        else:
            raise ValueError("Invalid bonding curve type for optimization")

        objective_value = evaluate_parameters(params, num_runs=1) # Consider increasing num_runs for more robust evaluation

        print(f"  Trial {i+1} - Parameters: {params}, Price Std Dev: {objective_value:.4f}")

        if objective_value < best_objective_value:
            best_objective_value = objective_value
            best_params = params
            print(f"  New best parameters found with Std Dev: {best_objective_value:.4f}")

    print(f"Optimization for {curve_type} complete.")
    print(f"Best Parameters: {best_params}")
    return best_params