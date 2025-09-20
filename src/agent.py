import numpy as np
import random
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
from .config import INITIAL_AGENT_CAPITAL, AGENT_MEMORY_SIZE, AGENT_TREND_THRESHOLD, AGENT_TREND_DELAY, AGENT_TRADE_FREQUENCY, AGENT_TRADE_SIZE_RANGE, TRADING_FEE, INITIAL_TOKEN_PRICE
from .bonding_curves import calculate_bonding_curve_price


@dataclass
class TradeRecord:
    """
    Records a single trade transaction.

    Attributes:
        step: Simulation step when trade occurred
        trade_type: Type of trade ("buy" or "sell")
        amount: Amount of tokens traded
        price: Price per token
        capital_before: Agent's capital before trade
        tokens_before: Agent's token balance before trade
    """
    step: int
    trade_type: str
    amount: float
    price: float
    capital_before: float
    tokens_before: float


@dataclass
class AgentStrategy:
    """
    Defines the trading strategy parameters for an agent.

    Attributes:
        trend_threshold: Minimum price trend for trading decisions
        trade_frequency: Probability of trading each step
        trade_size_range: Min/max fraction of holdings to trade
        memory_size: Number of price points to remember
        trend_delay: Minimum steps between trades
        risk_tolerance: Agent's risk tolerance (0-1)
        momentum_weight: Weight for momentum-based trading
        mean_reversion_weight: Weight for mean-reversion trading
    """
    trend_threshold: float = AGENT_TREND_THRESHOLD
    trade_frequency: float = AGENT_TRADE_FREQUENCY
    trade_size_range: Tuple[float, float] = field(default_factory=lambda: AGENT_TRADE_SIZE_RANGE)
    memory_size: int = AGENT_MEMORY_SIZE
    trend_delay: int = AGENT_TREND_DELAY
    risk_tolerance: float = 0.5
    momentum_weight: float = 0.7
    mean_reversion_weight: float = 0.3


class Agent:
    """
    Represents an autonomous trading agent in the bonding curve simulation.

    The agent uses multiple trading strategies including momentum trading,
    mean reversion, and random exploration to make trading decisions.
    """

    def __init__(self, agent_id: int, strategy: Optional[AgentStrategy] = None):
        """
        Initialize a trading agent.

        Args:
            agent_id: Unique identifier for the agent
            strategy: Trading strategy parameters (uses defaults if None)
        """
        self.agent_id: int = agent_id
        self.capital: float = INITIAL_AGENT_CAPITAL
        self.tokens: float = 0.0
        self.strategy: AgentStrategy = strategy or AgentStrategy()

        # Price memory initialized with initial token price
        self.price_memory: np.ndarray = np.full(self.strategy.memory_size, INITIAL_TOKEN_PRICE)
        self.last_trade_step: int = -self.strategy.trend_delay

        # Trading history for analysis
        self.trade_history: List[TradeRecord] = []
        self.total_trades: int = 0
        self.profitable_trades: int = 0

    @property
    def total_wealth(self) -> float:
        """Calculate total wealth (capital + token value)."""
        return self.capital + self.tokens

    @property
    def wealth_change(self) -> float:
        """Calculate change in total wealth from initial capital."""
        return self.total_wealth - INITIAL_AGENT_CAPITAL

    def update_memory(self, current_price: float) -> None:
        """
        Update the agent's price memory with the latest price.

        Args:
            current_price: The current token price
        """
        if current_price is not None and current_price > 0:
            self.price_memory = np.roll(self.price_memory, -1)
            self.price_memory[-1] = current_price

    def calculate_trend_indicators(self) -> Dict[str, float]:
        """
        Calculate various trend indicators from price memory.

        Returns:
            Dictionary of trend indicators
        """
        if len(self.price_memory) < 2:
            return {
                'momentum': 0.0,
                'mean_reversion': 0.0,
                'volatility': 0.0,
                'trend_strength': 0.0
            }

        # Momentum indicator (recent price vs older price)
        recent_avg = np.mean(self.price_memory[-5:]) if len(self.price_memory) >= 5 else self.price_memory[-1]
        older_avg = np.mean(self.price_memory[:-5]) if len(self.price_memory) >= 10 else np.mean(self.price_memory)
        momentum = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

        # Mean reversion indicator (current price vs historical mean)
        historical_mean = np.mean(self.price_memory)
        current_price = self.price_memory[-1]
        mean_reversion = (historical_mean - current_price) / historical_mean if historical_mean > 0 else 0

        # Volatility indicator
        volatility = np.std(self.price_memory) / np.mean(self.price_memory) if np.mean(self.price_memory) > 0 else 0

        # Trend strength (slope of linear regression)
        if len(self.price_memory) >= 2:
            x = np.arange(len(self.price_memory))
            slope, _ = np.polyfit(x, self.price_memory, 1)
            trend_strength = slope / np.mean(self.price_memory) if np.mean(self.price_memory) > 0 else 0
        else:
            trend_strength = 0.0

        return {
            'momentum': momentum,
            'mean_reversion': mean_reversion,
            'volatility': volatility,
            'trend_strength': trend_strength
        }

    def should_trade(self, current_supply: float, current_step: int, bonding_curve_params: Dict[str, Any]) -> bool:
        """
        Determine if the agent should consider trading.

        Args:
            current_supply: Current token supply
            current_step: Current simulation step
            bonding_curve_params: Bonding curve parameters

        Returns:
            True if the agent should trade
        """
        # Check trade frequency
        if random.random() > self.strategy.trade_frequency:
            return False

        # Check minimum time between trades
        if current_step - self.last_trade_step < self.strategy.trend_delay:
            return False

        return True

    def update_memory(self, current_price: float) -> None:
        """
        Updates the agent's price memory.

        Args:
            current_price (float): The current price of the token.
        """
        if current_price is not None:
            self.price_memory = np.concatenate((self.price_memory[1:], [current_price]))

    def decide_trade_action(self, current_supply: float, bonding_curve_params: Dict[str, Any]) -> Optional[str]:
        """
        Decide whether to buy, sell, or hold based on market conditions.

        Args:
            current_supply: Current token supply
            bonding_curve_params: Bonding curve parameters

        Returns:
            "buy", "sell", or None
        """
        indicators = self.calculate_trend_indicators()
        current_price = calculate_bonding_curve_price(current_supply, bonding_curve_params)

        # Combined trading signal
        momentum_signal = indicators['momentum'] * self.strategy.momentum_weight
        mean_reversion_signal = indicators['mean_reversion'] * self.strategy.mean_reversion_weight
        combined_signal = momentum_signal + mean_reversion_signal

        # Risk-adjusted decision making
        volatility_adjustment = 1.0 - (indicators['volatility'] / (1 + indicators['volatility']))
        risk_adjusted_signal = combined_signal * volatility_adjustment

        # Decision thresholds based on risk tolerance
        buy_threshold = self.strategy.trend_threshold * (1 + self.strategy.risk_tolerance)
        sell_threshold = -self.strategy.trend_threshold * (1 + self.strategy.risk_tolerance)

        if risk_adjusted_signal > buy_threshold:
            return "buy"
        elif risk_adjusted_signal < sell_threshold:
            return "sell"

        # Random exploration for market liquidity
        if random.random() < 0.1:  # 10% chance of random trade
            if self.capital > 0 and random.random() < 0.6:  # Prefer buying
                return "buy"
            elif self.tokens > 0:
                return "sell"

        return None

    def calculate_trade_size(self, trade_action: str, current_supply: float, bonding_curve_params: Dict[str, Any]) -> float:
        """
        Calculate the appropriate trade size based on action and market conditions.

        Args:
            trade_action: "buy" or "sell"
            current_supply: Current token supply
            bonding_curve_params: Bonding curve parameters

        Returns:
            Amount of tokens to trade
        """
        base_size = random.uniform(self.strategy.trade_size_range[0], self.strategy.trade_size_range[1])
        current_price = calculate_bonding_curve_price(current_supply, bonding_curve_params)

        if trade_action == "buy":
            max_affordable = self.capital / (current_price * (1 + TRADING_FEE))
            trade_size = min(base_size * max_affordable, max_affordable * 0.9)  # Leave some buffer
            return max(trade_size, 0.01)  # Minimum trade size

        elif trade_action == "sell":
            trade_size = base_size * self.tokens
            return max(trade_size, 0.01)  # Minimum trade size

        return 0.0

    def execute_trade(self, trade_action: str, trade_size: float, current_supply: float,
                     bonding_curve_params: Dict[str, Any], current_step: int) -> bool:
        """
        Execute a trade and update agent state.

        Args:
            trade_action: "buy" or "sell"
            trade_size: Amount of tokens to trade
            current_supply: Current token supply
            bonding_curve_params: Bonding curve parameters
            current_step: Current simulation step

        Returns:
            True if trade was executed successfully
        """
        current_price = calculate_bonding_curve_price(current_supply, bonding_curve_params)

        # Record trade attempt
        trade_record = TradeRecord(
            step=current_step,
            trade_type=trade_action,
            amount=trade_size,
            price=current_price,
            capital_before=self.capital,
            tokens_before=self.tokens
        )

        success = False

        if trade_action == "buy":
            cost = trade_size * current_price * (1 + TRADING_FEE)
            if self.capital >= cost and trade_size > 0:
                self.capital -= cost
                self.tokens += trade_size
                success = True

        elif trade_action == "sell":
            if self.tokens >= trade_size and trade_size > 0:
                revenue = trade_size * current_price * (1 - TRADING_FEE)
                self.capital += revenue
                self.tokens -= trade_size
                success = True

        if success:
            self.last_trade_step = current_step
            self.total_trades += 1
            self.trade_history.append(trade_record)

            # Check if trade was profitable (simplified check)
            if trade_action == "buy":
                # Assume profitable if we bought at a reasonable price
                if current_price < np.mean(self.price_memory):
                    self.profitable_trades += 1

        return success

    def trade(self, current_supply: float, current_step: int, bonding_curve_params: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """
        Main trading interface for the simulation engine.

        Args:
            current_supply: Current token supply
            current_step: Current simulation step
            bonding_curve_params: Bonding curve parameters

        Returns:
            Tuple of (trade_action, trade_size) or (None, 0) if no trade
        """
        if not self.should_trade(current_supply, current_step, bonding_curve_params):
            return None, 0.0

        trade_action = self.decide_trade_action(current_supply, bonding_curve_params)

        if trade_action is None:
            return None, 0.0

        # Check if we have sufficient funds/tokens for the trade
        if trade_action == "buy" and self.capital <= 0:
            return None, 0.0
        elif trade_action == "sell" and self.tokens <= 0:
            return None, 0.0

        trade_size = self.calculate_trade_size(trade_action, current_supply, bonding_curve_params)

        if trade_size <= 0:
            return None, 0.0

        return trade_action, trade_size

    def get_trading_stats(self) -> Dict[str, Any]:
        """
        Get trading statistics for analysis.

        Returns:
            Dictionary of trading statistics
        """
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0.0,
                'total_wealth_change': self.wealth_change,
                'avg_trade_size': 0.0
            }

        trade_sizes = [trade.amount for trade in self.trade_history] if self.trade_history else []

        return {
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': self.profitable_trades / self.total_trades,
            'total_wealth_change': self.wealth_change,
            'avg_trade_size': np.mean(trade_sizes) if trade_sizes else 0.0
        }