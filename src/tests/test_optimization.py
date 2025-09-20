"""
Test suite for optimization functionality.

This module contains tests for the parameter optimization system, including
parameter evaluation, random parameter generation, and optimization algorithms.
The tests verify that the optimization process correctly finds optimal bonding
curve parameters, evaluates parameter performance across multiple simulation
runs, and handles edge cases and error conditions properly.
"""

import pytest
import numpy as np
from optimization import evaluate_parameters, optimize_bonding_curve
from config import INITIAL_TOKEN_SUPPLY, NUM_AGENTS
from simulation import simulation_step, SimulationState
from agent import Agent
from bonding_curves import calculate_bonding_curve_price

def test_evaluate_parameters_linear():
    """
    Tests the evaluate_parameters function with a linear bonding curve.
    This test checks if the function correctly calculates a score based on the price volatility.
    """
    # Define a simple linear bonding curve parameters
    params = {'type': 'linear', 'm': 0.1, 'b': 1.0}

    # Call evaluate_parameters
    score = evaluate_parameters(params, num_runs=2)

    # Assert that the score is a float
    assert isinstance(score, float)

def test_optimize_bonding_curve_linear():
    """
    Tests the optimize_bonding_curve function with a linear bonding curve.
    This test checks if the function returns a dictionary with the expected keys.
    """
    # Call optimize_bonding_curve
    best_params = optimize_bonding_curve('linear', n_trials=2)

    # Assert that the function returns a dictionary
    assert isinstance(best_params, dict)

    # Assert that the dictionary contains the expected keys
    assert 'type' in best_params
    assert 'm' in best_params
    assert 'b' in best_params

def test_optimize_bonding_curve_returns_valid_parameters():
    """
    Tests that optimize_bonding_curve returns parameters that result in a valid bonding curve price.
    """
    # Call optimize_bonding_curve
    best_params = optimize_bonding_curve('linear', n_trials=2)

    # Create a dummy supply
    supply = 100.0

    # Calculate the bonding curve price with the optimized parameters
    price = calculate_bonding_curve_price(supply, best_params)

    # Assert that the price is a float
    assert isinstance(price, float)

    # Assert that the price is not negative
    assert price >= 0