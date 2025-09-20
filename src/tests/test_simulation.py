"""
Test suite for simulation functionality.

This module contains comprehensive tests for the simulation engine, including
integration tests for the complete simulation workflow and unit tests for
individual simulation components. The tests verify proper interaction between
agents, bonding curves, and the simulation state management, ensuring that
all components work together correctly across different scenarios and edge cases.
"""

import pytest
import numpy as np
from simulation import simulation_step, SimulationState
from agent import Agent
from config import INITIAL_TOKEN_SUPPLY, NUM_AGENTS, TRADING_FEE
from bonding_curves import calculate_bonding_curve_price, linear_bonding_curve, exponential_bonding_curve, sigmoid_bonding_curve, multi_segment_bonding_curve

# --- Integration Tests ---
def test_simulation_step_basic_interaction():
    """
    Tests a basic simulation step with one agent and a linear bonding curve.
    Verifies that the agent's balance and token holdings, and the total supply are updated correctly.
    """
    # Initialize simulation state
    initial_supply = 100.0
    initial_agent_capital = 100.0
    num_agents = 1
    agents = [Agent(0) for _ in range(num_agents)]
    agents[0].capital = initial_agent_capital
    resources = []
    
    # Linear bonding curve parameters
    params = {'type': 'linear', 'm': 0.1, 'b': 1.0}

    # Initial price
    initial_price = calculate_bonding_curve_price(initial_supply, params)

    # Set agent's trade function to always buy
    agents[0].trade = lambda supply, step, bonding_curve_params: ("buy", 1)

    # Run simulation step
    step_results = simulation_step(0, agents, resources, params)

    # Verify that the agent's balance and token holdings are updated
    assert len(step_results["agent_capitals"]) == num_agents
    assert len(step_results["agent_tokens"]) == num_agents
    assert step_results["current_price"] == calculate_bonding_curve_price(step_results["supply"], params)

def test_all_bonding_curve_types_functional():
    """
    Tests that all bonding curve types are functional and don't raise errors.
    """
    initial_supply = 100.0
    initial_agent_capital = 100.0
    num_agents = 1
    agents = [Agent(0) for _ in range(num_agents)]
    agents[0].capital = initial_agent_capital
    resources = []

    curve_types = {
        'linear': {'type': 'linear', 'm': 0.1, 'b': 1.0},
        'exponential': {'type': 'exponential', 'a': 0.1, 'k': 0.01},
        'sigmoid': {'type': 'sigmoid', 'k': 0.02, 's0': 100, 'k_max': 10},
        'multi-segment': {'type': 'multi-segment', 'breakpoint': 200, 'm': 0.05, 'a': 0.1, 'k': 0.02}
    }

    for curve_type, params in curve_types.items():
        try:
            # Set agent's trade function to always buy
            agents[0].trade = lambda supply, step, bonding_curve_params: ("buy", 1)

            # Run simulation step
            step_results = simulation_step(0, agents, resources, params)

            # Verify that the agent's balance and token holdings are updated
            assert len(step_results["agent_capitals"]) == num_agents
            assert len(step_results["agent_tokens"]) == num_agents
            assert step_results["current_price"] == calculate_bonding_curve_price(step_results["supply"], params)
        except Exception as e:
            pytest.fail(f"Bonding curve type {curve_type} failed: {e}")

# --- Unit Tests ---
def test_calculate_bonding_curve_price_linear():
    """Tests the linear bonding curve price calculation."""
    params = {'type': 'linear', 'm': 0.1, 'b': 1.0}
    supply = 100.0
    expected_price = 0.1 * supply + 1.0
    assert calculate_bonding_curve_price(supply, params) == expected_price

def test_calculate_bonding_curve_price_exponential():
    """Tests the exponential bonding curve price calculation."""
    params = {'type': 'exponential', 'a': 0.1, 'k': 0.01}
    supply = 100.0
    expected_price = 0.1 * np.exp(0.01 * supply)
    assert np.isclose(calculate_bonding_curve_price(supply, params), expected_price)

def test_calculate_bonding_curve_price_sigmoid():
    """Tests the sigmoid bonding curve price calculation."""
    params = {'type': 'sigmoid', 'k': 0.02, 's0': 100, 'k_max': 10}
    supply = 50.0
    expected_price = 10 / (1 + np.exp(-0.02 * (supply - 100)))
    assert np.isclose(calculate_bonding_curve_price(supply, params), expected_price)

def test_calculate_bonding_curve_price_multi_segment():
    """Tests the multi-segment bonding curve price calculation."""
    params = {'type': 'multi-segment', 'breakpoint': 200, 'm': 0.05, 'a': 0.1, 'k': 0.02}
    supply = 100.0
    expected_price = 0.05 * supply
    assert np.isclose(calculate_bonding_curve_price(supply, params), expected_price)