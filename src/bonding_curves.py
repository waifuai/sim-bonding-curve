"""
Bonding Curves module for the Token Economy Simulation.

This module implements various mathematical bonding curve models used to calculate token
prices based on supply. It includes linear, exponential, sigmoid, and multi-segment
bonding curves with comprehensive validation and error handling. The module provides
a unified interface for price calculation and supports parameter validation for each
curve type to ensure numerical stability and prevent invalid calculations.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Union
import logging
from abc import ABC, abstractmethod


# Set up logging
logger = logging.getLogger(__name__)


class BondingCurveError(Exception):
    """Base exception for bonding curve related errors."""
    pass


class InvalidParametersError(BondingCurveError):
    """Raised when bonding curve parameters are invalid."""
    pass


class UnsupportedCurveTypeError(BondingCurveError):
    """Raised when an unsupported curve type is requested."""
    pass


class BondingCurveValidator:
    """Validates bonding curve parameters and inputs."""

    @staticmethod
    def validate_supply(supply: Union[float, int]) -> float:
        """Validate and convert supply to float."""
        try:
            supply = float(supply)
            if supply < 0:
                raise InvalidParametersError(f"Supply must be non-negative, got {supply}")
            if not np.isfinite(supply):
                raise InvalidParametersError(f"Supply must be finite, got {supply}")
            return supply
        except (ValueError, TypeError) as e:
            raise InvalidParametersError(f"Invalid supply value: {e}")

    @staticmethod
    def validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate bonding curve parameters."""
        if not isinstance(params, dict):
            raise InvalidParametersError("Parameters must be a dictionary")

        if 'type' not in params:
            raise InvalidParametersError("Parameters must include 'type'")

        curve_type = params['type']
        if not isinstance(curve_type, str):
            raise InvalidParametersError("Curve type must be a string")

        return params

    @staticmethod
    def validate_linear_params(params: Dict[str, Any]) -> None:
        """Validate linear curve parameters."""
        required = ['m', 'b']
        for param in required:
            if param not in params:
                raise InvalidParametersError(f"Linear curve requires '{param}' parameter")

        m, b = params['m'], params['b']
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in [m, b]):
            raise InvalidParametersError("Linear parameters must be finite numbers")

    @staticmethod
    def validate_exponential_params(params: Dict[str, Any]) -> None:
        """Validate exponential curve parameters."""
        required = ['a', 'k']
        for param in required:
            if param not in params:
                raise InvalidParametersError(f"Exponential curve requires '{param}' parameter")

        a, k = params['a'], params['k']
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in [a, k]):
            raise InvalidParametersError("Exponential parameters must be finite numbers")

        if a <= 0:
            raise InvalidParametersError("Exponential parameter 'a' must be positive")

    @staticmethod
    def validate_sigmoid_params(params: Dict[str, Any]) -> None:
        """Validate sigmoid curve parameters."""
        required = ['k', 's0', 'k_max']
        for param in required:
            if param not in params:
                raise InvalidParametersError(f"Sigmoid curve requires '{param}' parameter")

        k, s0, k_max = params['k'], params['s0'], params['k_max']
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in [k, s0, k_max]):
            raise InvalidParametersError("Sigmoid parameters must be finite numbers")

        if k_max <= 0:
            raise InvalidParametersError("Sigmoid parameter 'k_max' must be positive")

    @staticmethod
    def validate_multi_segment_params(params: Dict[str, Any]) -> None:
        """Validate multi-segment curve parameters."""
        required = ['breakpoint', 'm', 'a', 'k']
        for param in required:
            if param not in params:
                raise InvalidParametersError(f"Multi-segment curve requires '{param}' parameter")

        breakpoint, m, a, k = params['breakpoint'], params['m'], params['a'], params['k']
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in [breakpoint, m, a, k]):
            raise InvalidParametersError("Multi-segment parameters must be finite numbers")

        if breakpoint <= 0:
            raise InvalidParametersError("Multi-segment parameter 'breakpoint' must be positive")
        if a <= 0:
            raise InvalidParametersError("Multi-segment parameter 'a' must be positive")


class BaseBondingCurve(ABC):
    """Abstract base class for bonding curves."""

    def __init__(self, params: Dict[str, Any]):
        self.params = BondingCurveValidator.validate_params(params)
        self.validate_params()

    @abstractmethod
    def validate_params(self) -> None:
        """Validate curve-specific parameters."""
        pass

    @abstractmethod
    def calculate_price(self, supply: float) -> float:
        """Calculate price for given supply."""
        pass

    def calculate_multiple_prices(self, supplies: List[float]) -> List[float]:
        """Calculate prices for multiple supply values."""
        return [self.calculate_price(s) for s in supplies]

    def get_curve_info(self) -> Dict[str, Any]:
        """Get information about the curve."""
        return {
            'type': self.params['type'],
            'parameters': {k: v for k, v in self.params.items() if k != 'type'}
        }


class LinearBondingCurve(BaseBondingCurve):
    """Linear bonding curve: price = m * supply + b"""

    def validate_params(self) -> None:
        BondingCurveValidator.validate_linear_params(self.params)

    def calculate_price(self, supply: float) -> float:
        supply = BondingCurveValidator.validate_supply(supply)
        m = self.params['m']
        b = self.params['b']
        price = m * supply + b

        if price < 0:
            logger.warning(f"Linear curve produced negative price: {price}")
            price = 0.0

        return price


class ExponentialBondingCurve(BaseBondingCurve):
    """Exponential bonding curve: price = a * exp(k * supply)"""

    def validate_params(self) -> None:
        BondingCurveValidator.validate_exponential_params(self.params)

    def calculate_price(self, supply: float) -> float:
        supply = BondingCurveValidator.validate_supply(supply)
        a = self.params['a']
        k = self.params['k']

        try:
            price = a * np.exp(k * supply)

            if not np.isfinite(price):
                raise InvalidParametersError(f"Exponential curve produced non-finite price: {price}")

            if price < 0:
                logger.warning(f"Exponential curve produced negative price: {price}")
                price = 0.0

            return price
        except OverflowError:
            raise InvalidParametersError("Exponential curve overflow - parameters too large")


class SigmoidBondingCurve(BaseBondingCurve):
    """Sigmoid bonding curve: price = k_max / (1 + exp(-k * (supply - s0)))"""

    def validate_params(self) -> None:
        BondingCurveValidator.validate_sigmoid_params(self.params)

    def calculate_price(self, supply: float) -> float:
        supply = BondingCurveValidator.validate_supply(supply)
        k = self.params['k']
        s0 = self.params['s0']
        k_max = self.params['k_max']

        try:
            exponent = -k * (supply - s0)
            if abs(exponent) > 700:  # Prevent overflow
                price = k_max if exponent > 0 else 0.0
            else:
                price = k_max / (1 + np.exp(exponent))

            if not np.isfinite(price):
                raise InvalidParametersError(f"Sigmoid curve produced non-finite price: {price}")

            return max(0, price)  # Ensure non-negative
        except Exception as e:
            raise InvalidParametersError(f"Sigmoid curve calculation error: {e}")


class MultiSegmentBondingCurve(BaseBondingCurve):
    """Multi-segment bonding curve with linear and exponential phases."""

    def validate_params(self) -> None:
        BondingCurveValidator.validate_multi_segment_params(self.params)

    def calculate_price(self, supply: float) -> float:
        supply = BondingCurveValidator.validate_supply(supply)
        breakpoint = self.params['breakpoint']
        m = self.params['m']
        a = self.params['a']
        k = self.params['k']

        try:
            # Linear part
            linear_supply = np.minimum(supply, breakpoint)
            linear_price = m * linear_supply

            # Exponential part
            if supply > breakpoint:
                exponential_supply = supply - breakpoint
                exponential_price = a * np.exp(k * exponential_supply)
            else:
                exponential_price = 0.0

            price = linear_price + exponential_price

            if not np.isfinite(price):
                raise InvalidParametersError(f"Multi-segment curve produced non-finite price: {price}")

            return max(0, price)  # Ensure non-negative
        except Exception as e:
            raise InvalidParametersError(f"Multi-segment curve calculation error: {e}")


# Registry of curve types
CURVE_REGISTRY = {
    'linear': LinearBondingCurve,
    'exponential': ExponentialBondingCurve,
    'sigmoid': SigmoidBondingCurve,
    'multi-segment': MultiSegmentBondingCurve
}


def calculate_bonding_curve_price(supply: float, params: Dict[str, Any]) -> float:
    """
    Calculate the price of a token based on a bonding curve.

    This is the main interface function that handles all bonding curve types
    with comprehensive validation and error handling.

    Args:
        supply: The current supply of the token (must be non-negative)
        params: Dictionary containing curve type and parameters

    Returns:
        The calculated price (guaranteed to be non-negative)

    Raises:
        InvalidParametersError: If parameters are invalid
        UnsupportedCurveTypeError: If curve type is not supported
    """
    try:
        # Validate inputs
        supply = BondingCurveValidator.validate_supply(supply)
        params = BondingCurveValidator.validate_params(params)

        curve_type = params['type']

        if curve_type not in CURVE_REGISTRY:
            raise UnsupportedCurveTypeError(f"Unsupported curve type: {curve_type}")

        # Create curve instance and calculate price
        curve_class = CURVE_REGISTRY[curve_type]
        curve = curve_class(params)
        price = curve.calculate_price(supply)

        logger.debug(f"Calculated {curve_type} price: {price} for supply: {supply}")
        return price

    except (InvalidParametersError, UnsupportedCurveTypeError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error in calculate_bonding_curve_price: {e}")
        raise BondingCurveError(f"Price calculation failed: {e}")


def get_supported_curve_types() -> List[str]:
    """Get list of supported bonding curve types."""
    return list(CURVE_REGISTRY.keys())


def create_bonding_curve(params: Dict[str, Any]) -> BaseBondingCurve:
    """
    Create a bonding curve instance.

    Args:
        params: Curve parameters including type

    Returns:
        Bonding curve instance

    Raises:
        UnsupportedCurveTypeError: If curve type is not supported
        InvalidParametersError: If parameters are invalid
    """
    params = BondingCurveValidator.validate_params(params)
    curve_type = params['type']

    if curve_type not in CURVE_REGISTRY:
        raise UnsupportedCurveTypeError(f"Unsupported curve type: {curve_type}")

    curve_class = CURVE_REGISTRY[curve_type]
    return curve_class(params)