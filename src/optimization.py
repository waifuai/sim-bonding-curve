"""
Optimization module for the Token Economy Simulation with Bonding Curves.

This module provides parameter optimization functionality for finding optimal bonding
curve parameters that minimize price volatility while maintaining healthy market
activity. It uses random search with comprehensive evaluation metrics to test multiple
parameter combinations across multiple simulation runs. The module includes functions
for parameter evaluation, random parameter generation, and optimization with detailed
progress reporting and result analysis.
"""

import random
import numpy as np
from typing import Dict, Any, List, Callable
from .config import INITIAL_TOKEN_SUPPLY, SIMULATION_STEPS
from .simulation import simulation_step, SimulationState, create_simulation_state
from .agent import Agent


def evaluate_parameters(
    params: Dict[str, Any],
    num_runs: int = 3,
    num_agents: int = 100,
    simulation_steps: int = SIMULATION_STEPS
) -> float:
    """
    Evaluates the given bonding curve parameters across multiple simulation runs.

    Args:
        params: The parameters of the bonding curve
        num_runs: The number of simulation runs to perform
        num_agents: Number of agents in each simulation
        simulation_steps: Number of steps per simulation

    Returns:
        Average score across all simulation runs (lower is better)
    """
    all_metrics = []

    for run in range(num_runs):
        # Create fresh simulation state for each run
        sim_state = create_simulation_state(num_agents)
        supply = sim_state.supply
        agents = sim_state.agents

        price_history = []
        volume_history = []

        for step in range(simulation_steps):
            step_results = simulation_step(step, agents, [], params, supply)
            supply = step_results["supply"]
            current_price = step_results["current_price"]
            volume = step_results["volume"]

            price_history.append(current_price)
            volume_history.append(volume)

        # Calculate comprehensive metrics
        price_history = np.array(price_history)
        volume_history = np.array(volume_history)

        metrics = {
            'price_std': np.std(price_history),
            'price_mean': np.mean(price_history),
            'price_volatility': np.std(price_history) / np.mean(price_history) if np.mean(price_history) > 0 else float('inf'),
            'supply_change': abs(supply - INITIAL_TOKEN_SUPPLY),
            'price_change': np.ptp(price_history),
            'total_volume': np.sum(volume_history),
            'avg_volume': np.mean(volume_history),
            'price_trend': np.polyfit(range(len(price_history)), price_history, 1)[0] if len(price_history) > 1 else 0,
            'market_activity': len([v for v in volume_history if v > 0]) / len(volume_history)
        }
        all_metrics.append(metrics)

    # Enhanced composite scoring function
    scores = []
    for m in all_metrics:
        # Base volatility score
        volatility_score = m['price_volatility'] * 2.0

        # Price stability bonus
        stability_bonus = 0
        if m['price_volatility'] < 0.1:  # Low volatility
            stability_bonus = -0.5
        elif m['price_volatility'] > 0.5:  # High volatility penalty
            stability_bonus = 1.0

        # Market activity score
        activity_score = 0
        if m['market_activity'] < 0.1:  # Low activity penalty
            activity_score = 2.0
        elif m['market_activity'] > 0.3:  # Good activity bonus
            activity_score = -0.3

        # Supply stability
        supply_score = m['supply_change'] / INITIAL_TOKEN_SUPPLY

        # Total score (lower is better)
        total_score = (
            volatility_score +
            stability_bonus +
            activity_score +
            supply_score
        )

        # Penalize completely stagnant simulations
        if m['supply_change'] < 1 and m['price_change'] < 0.01:
            total_score += 10.0

        scores.append(total_score)

    return np.mean(scores)

def get_parameter_ranges(curve_type: str) -> Dict[str, tuple]:
    """
    Get parameter ranges for different bonding curve types.

    Args:
        curve_type: Type of bonding curve

    Returns:
        Dictionary of parameter ranges
    """
    ranges = {
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
            's0': (50, 150),
            'k_max': (2, 15)
        },
        'multi-segment': {
            'breakpoint': (100, 300),
            'm': (0.01, 0.1),
            'a': (0.01, 0.2),
            'k': (0.005, 0.03)
        }
    }
    return ranges.get(curve_type, {})


def generate_random_parameters(curve_type: str) -> Dict[str, Any]:
    """
    Generate random parameters for a given curve type.

    Args:
        curve_type: Type of bonding curve

    Returns:
        Dictionary of random parameters
    """
    ranges = get_parameter_ranges(curve_type)
    if not ranges:
        raise ValueError(f"Unknown curve type: {curve_type}")

    params = {'type': curve_type}
    for param_name, (min_val, max_val) in ranges.items():
        params[param_name] = random.uniform(min_val, max_val)

    return params


def optimize_bonding_curve(
    curve_type: str,
    n_trials: int = 10,
    num_agents: int = 100,
    simulation_steps: int = SIMULATION_STEPS,
    evaluation_runs: int = 2
) -> Dict[str, Any]:
    """
    Optimizes the bonding curve parameters using random search with enhanced evaluation.

    Args:
        curve_type: The type of bonding curve to optimize
        n_trials: The number of random parameter sets to test
        num_agents: Number of agents in simulation
        simulation_steps: Number of simulation steps
        evaluation_runs: Number of runs per parameter evaluation

    Returns:
        Dictionary of the best parameters found
    """
    if curve_type not in ['linear', 'exponential', 'sigmoid', 'multi-segment']:
        raise ValueError(f"Unsupported curve type: {curve_type}")

    best_params = None
    best_objective_value = float('inf')
    trial_results = []

    print(f"\nStarting optimization for {curve_type} bonding curve...")
    print(f"Running {n_trials} trials with {evaluation_runs} evaluation runs each")

    for trial in range(n_trials):
        print(f"\nOptimization Trial {trial + 1}/{n_trials}")

        # Generate random parameters
        params = generate_random_parameters(curve_type)
        print(f"  Testing parameters: {params}")

        # Evaluate parameters
        try:
            objective_value = evaluate_parameters(
                params,
                num_runs=evaluation_runs,
                num_agents=num_agents,
                simulation_steps=simulation_steps
            )

            trial_results.append({
                'trial': trial + 1,
                'params': params,
                'score': objective_value
            })

            print(f"  Score: {objective_value:.4f}")

            # Update best parameters
            if objective_value < best_objective_value:
                best_objective_value = objective_value
                best_params = params.copy()
                print(f"  🏆 New best parameters found! Score: {best_objective_value:.4f}")

        except Exception as e:
            print(f"  ❌ Trial {trial + 1} failed: {e}")
            continue

    # Print optimization summary
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Curve Type: {curve_type}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_objective_value:.4f}")
    print(f"Total Trials: {n_trials}")

    # Show top 3 results
    sorted_results = sorted(trial_results, key=lambda x: x['score'])[:3]
    print(f"\nTop 3 Results:")
    for i, result in enumerate(sorted_results, 1):
        print(f"  {i}. Score: {result['score']:.4f} - {result['params']}")

    return best_params