import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from .config import INITIAL_TOKEN_SUPPLY, NUM_AGENTS, SIMULATION_STEPS
from .agent import Agent
from .simulation import SimulationState, simulation_step
from .optimization import optimize_bonding_curve


def create_visualizations(
    supply_history: np.ndarray,
    price_history: np.ndarray,
    all_agent_capital_history: np.ndarray,
    all_agent_token_history: np.ndarray,
    curve_type: str,
    save_plots: bool = True
) -> None:
    """
    Create comprehensive visualizations of the simulation results.

    Args:
        supply_history: Array of supply values over time
        price_history: Array of price values over time
        all_agent_capital_history: Array of agent capital distributions over time
        all_agent_token_history: Array of agent token distributions over time
        curve_type: Type of bonding curve used
        save_plots: Whether to save plots to files
    """
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Bonding Curve Simulation Results ({curve_type})', fontsize=16)

    # Plot 1: Supply and Price over time
    ax1 = axes[0, 0]
    ax1.plot(supply_history, label='Token Supply', color='blue')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Supply', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(price_history, label='Token Price', color='red', linestyle='--')
    ax1_twin.set_ylabel('Price', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.legend(loc='upper right')

    # Plot 2: Agent capital distribution over time
    ax2 = axes[0, 1]
    num_timepoints = 5
    timepoints = np.linspace(0, len(all_agent_capital_history) - 1, num_timepoints, dtype=int)

    for i, t in enumerate(timepoints):
        ax2.hist(all_agent_capital_history[t], alpha=0.5, label=f'Step {t}',
                bins=20, density=True)

    ax2.set_xlabel('Agent Capital')
    ax2.set_ylabel('Density')
    ax2.set_title('Agent Capital Distribution')
    ax2.legend()

    # Plot 3: Token distribution over time
    ax3 = axes[1, 0]
    for i, t in enumerate(timepoints):
        ax3.hist(all_agent_token_history[t], alpha=0.5, label=f'Step {t}',
                bins=20, density=True)

    ax3.set_xlabel('Agent Token Holdings')
    ax3.set_ylabel('Density')
    ax3.set_title('Agent Token Distribution')
    ax3.legend()

    # Plot 4: Price volatility and returns
    ax4 = axes[1, 1]
    price_returns = np.diff(price_history) / price_history[:-1]
    ax4.plot(price_returns, label='Price Returns')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Return')
    ax4.set_title(f'Price Returns (Volatility: {np.std(price_returns):.4f})')
    ax4.legend()

    plt.tight_layout()

    if save_plots:
        plt.savefig(f'simulation_results_{curve_type}.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved to simulation_results_{curve_type}.png")

    plt.show()


def run_optimization(optimize_curve_type: str, n_trials: int) -> Dict[str, Any]:
    """
    Run parameter optimization for the specified bonding curve type.

    Args:
        optimize_curve_type: Type of curve to optimize
        n_trials: Number of optimization trials

    Returns:
        Dictionary of optimal parameters
    """
    print(f"\nOptimizing {optimize_curve_type} bonding curve parameters...")
    optimal_params = optimize_bonding_curve(optimize_curve_type, n_trials=n_trials)
    return optimal_params


def run_simulation(
    optimal_params: Dict[str, Any],
    curve_type: str,
    num_agents: int,
    simulation_steps: int,
    enable_logging: bool = True
) -> Dict[str, Any]:
    """
    Run the main simulation with optimal parameters.

    Args:
        optimal_params: Optimal bonding curve parameters
        curve_type: Type of bonding curve
        num_agents: Number of agents in simulation
        simulation_steps: Number of simulation steps
        enable_logging: Whether to enable detailed logging

    Returns:
        Dictionary containing simulation results
    """
    print(f"\nSimulating with optimal parameters for {curve_type}: {optimal_params}")

    # Initialize simulation state
    simulation_state = SimulationState()
    simulation_state.supply = INITIAL_TOKEN_SUPPLY
    simulation_state.agents = [Agent(i) for i in range(num_agents)]

    # Initialize data collection
    supply_history = []
    price_history = []
    all_agent_capital_history = []
    all_agent_token_history = []

    start_time = time.time()
    current_supply = simulation_state.supply

    for step in range(simulation_steps):
        step_results = simulation_step(step, simulation_state.agents, [], optimal_params, current_supply)
        current_supply = step_results["supply"]
        agent_capitals = step_results["agent_capitals"]
        agent_tokens = step_results["agent_tokens"]
        current_price = step_results["current_price"]

        # Update simulation state
        simulation_state.supply = current_supply

        # Collect data
        supply_history.append(current_supply)
        price_history.append(current_price)
        all_agent_capital_history.append(np.array(agent_capitals))
        all_agent_token_history.append(np.array(agent_tokens))

        if enable_logging and (step + 1) % max(1, simulation_steps // 10) == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            print(
                f"Step {step + 1}/{simulation_steps} | "
                f"Elapsed: {elapsed_time:.2f}s | "
                f"Supply: {current_supply:.2f} | "
                f"Price: {current_price:.2f} | "
                f"Avg. Capital: {np.mean(agent_capitals):.2f} | "
                f"Avg. Tokens: {np.mean(agent_tokens):.2f}"
            )

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nSimulation completed in {total_time:.2f} seconds")

    return {
        'supply_history': np.array(supply_history),
        'price_history': np.array(price_history),
        'all_agent_capital_history': np.array(all_agent_capital_history),
        'all_agent_token_history': np.array(all_agent_token_history),
        'total_time': total_time,
        'final_supply': supply_history[-1],
        'final_price': price_history[-1],
        'price_volatility': np.std(price_history),
        'price_change': (price_history[-1] - price_history[0]) / price_history[0] if price_history[0] != 0 else 0
    }


def main():
    """
    Main entry point for the bonding curve simulation.
    """
    parser = argparse.ArgumentParser(description='Bonding Curve Simulation')
    parser.add_argument('--curve-type', default='sigmoid', choices=['linear', 'exponential', 'sigmoid', 'multi-segment'],
                       help='Type of bonding curve to optimize')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of optimization trials')
    parser.add_argument('--num-agents', type=int, default=None,
                       help='Number of agents (overrides config)')
    parser.add_argument('--simulation-steps', type=int, default=None,
                       help='Number of simulation steps (overrides config)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plotting')
    parser.add_argument('--no-logging', action='store_true',
                       help='Disable detailed logging')

    args = parser.parse_args()

    # Use config values if not specified in args
    num_agents = args.num_agents if args.num_agents else NUM_AGENTS
    simulation_steps = args.simulation_steps if args.simulation_steps else SIMULATION_STEPS

    # Step 1: Optimize parameters
    optimal_params = run_optimization(args.curve_type, args.n_trials)

    # Step 2: Run simulation
    results = run_simulation(
        optimal_params=optimal_params,
        curve_type=args.curve_type,
        num_agents=num_agents,
        simulation_steps=simulation_steps,
        enable_logging=not args.no_logging
    )

    # Step 3: Display summary
    print("\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(f"Curve Type: {args.curve_type}")
    print(f"Optimal Parameters: {optimal_params}")
    print(f"Final Supply: {results['final_supply']:.2f}")
    print(f"Final Price: {results['final_price']:.2f}")
    print(f"Price Volatility: {results['price_volatility']:.4f}")
    print(f"Total Price Change: {results['price_change']:.2%}")
    print(f"Simulation Time: {results['total_time']:.2f} seconds")
    print("="*50)

    # Step 4: Create visualizations
    if not args.no_plots:
        create_visualizations(
            supply_history=results['supply_history'],
            price_history=results['price_history'],
            all_agent_capital_history=results['all_agent_capital_history'],
            all_agent_token_history=results['all_agent_token_history'],
            curve_type=args.curve_type
        )


if __name__ == "__main__":
    main()
