import numpy as np
from typing import Dict, Any

# --- Bonding Curve Functions ---
def calculate_bonding_curve_price(supply: float, params: Dict[str, Any]) -> float:
    """
    Calculates the price of a token based on a bonding curve.

    Args:
        supply (float): The current supply of the token.
        params (Dict[str, Any]): A dictionary of parameters for the bonding curve.

    Returns:
        float: The calculated price of the token.
    """
    supply = float(supply)
    curve_type = params.get('type', 'linear')  # Default to linear if type is missing

    if curve_type == 'linear':
        m = params.get('m', 0.1)
        b = params.get('b', 1.0)
        return m * supply + b
    elif curve_type == 'exponential':
        a = params.get('a', 0.1)
        k = params.get('k', 0.01)
        return a * np.exp(k * supply)
    elif curve_type == 'sigmoid':
        k = params.get('k', 0.02)
        s0 = params.get('s0', 100)
        k_max = params.get('k_max', 10)
        return k_max / (1 + np.exp(-k * (supply - s0)))
    elif curve_type == 'multi-segment':
        breakpoint = params.get('breakpoint', 200)
        m = params.get('m', 0.05)
        a = params.get('a', 0.1)
        k = params.get('k', 0.02)
        lin = m * np.minimum(supply, breakpoint)
        exp = a * np.exp(k * np.maximum(supply - breakpoint, 0))
        return lin + exp
    else:
        raise ValueError("Invalid bonding curve type")