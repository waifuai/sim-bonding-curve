import pytest
import numpy as np
from bonding_curve.src.bonding_curves import calculate_bonding_curve_price
from bonding_curve.src.config import INITIAL_TOKEN_SUPPLY

def test_linear_curve():
    params = {'type': 'linear', 'm': 0.1, 'b': 1.0}
    assert calculate_bonding_curve_price(INITIAL_TOKEN_SUPPLY / 10, params) == 0.1 * (INITIAL_TOKEN_SUPPLY / 10) + 1.0
    params = {'type': 'linear', 'm': 0.5, 'b': 2.5}
    assert calculate_bonding_curve_price(INITIAL_TOKEN_SUPPLY / 20, params) == 0.5 * (INITIAL_TOKEN_SUPPLY / 20) + 2.5

def test_exponential_curve():
    params = {'type': 'exponential', 'a': 0.1, 'k': 0.01}
    assert abs(calculate_bonding_curve_price(10, params) - 0.110517) < 1e-6
    params = {'type': 'exponential', 'a': 0.2, 'k': 0.02}
    assert abs(calculate_bonding_curve_price(5, params) - 0.221034) < 1e-6

def test_sigmoid_curve():
    params = {'type': 'sigmoid', 'k': 0.02, 's0': 100, 'k_max': 10}
    assert abs(calculate_bonding_curve_price(50, params) - 2.689414) < 1e-6
    params = {'type': 'sigmoid', 'k': 0.04, 's0': 50, 'k_max': 20}
    assert abs(calculate_bonding_curve_price(25, params) - 4.107579) < 1e-6

def test_multisegment_curve():
    params = {'type': 'multi-segment', 'breakpoint': 200, 'm': 0.05, 'a': 0.1, 'k': 0.02}
    assert abs(calculate_bonding_curve_price(100, params) - 5.0) < 1e-6  # Linear part
    assert abs(calculate_bonding_curve_price(300, params) - 10.0 + 0.1 * np.exp(0.02 * 100)) < 1e-6 # Linear + exponential

def test_invalid_curve_type():
    params = {'type': 'invalid'}
    with pytest.raises(ValueError):
        calculate_bonding_curve_price(10, params)

def test_missing_parameters():
    # Test default values for linear
    params = {'type': 'linear'}
    assert calculate_bonding_curve_price(10, params) == 0.1 * 10 + 1.0