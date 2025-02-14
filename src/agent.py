import numpy as np
import random
from config import INITIAL_AGENT_CAPITAL, AGENT_MEMORY_SIZE, AGENT_TREND_THRESHOLD, AGENT_TREND_DELAY, AGENT_TRADE_FREQUENCY, AGENT_TRADE_SIZE_RANGE, TRADING_FEE, INITIAL_TOKEN_PRICE
from bonding_curves import calculate_bonding_curve_price
from typing import Tuple, Optional, Dict, Any

# --- Agent State ---
class Agent:
    """
    Represents an agent trading on the bonding curve.
    """
    def __init__(self, agent_id: int):
        """
        Initializes an agent.

        Args:
            agent_id (int): The ID of the agent.
        """
        self.agent_id: int = agent_id
        self.capital: float = INITIAL_AGENT_CAPITAL
        self.tokens: float = 0.0
        # Initialize with first price instead of zeros
        self.price_memory: np.ndarray = np.full(AGENT_MEMORY_SIZE, INITIAL_TOKEN_PRICE)
        self.last_trade_step: int = -AGENT_TREND_DELAY

    def update_memory(self, current_price: float) -> None:
        """
        Updates the agent's price memory.

        Args:
            current_price (float): The current price of the token.
        """
        if current_price is not None:
            self.price_memory = np.concatenate((self.price_memory[1:], [current_price]))

    def trade(self, current_supply: float, current_step: int, bonding_curve_params: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """
        Determines whether the agent should trade and in which direction.

        Args:
            current_supply (float): The current supply of the token.
            current_step (int): The current simulation step.
            bonding_curve_params (Dict[str, Any]): The parameters of the bonding curve.

        Returns:
            Tuple[Optional[str], float]: A tuple containing the trade direction ("buy" or "sell") and the amount to trade, or (None, 0) if no trade.
        """
        if random.random() < AGENT_TRADE_FREQUENCY and current_step > (self.last_trade_step + AGENT_TREND_DELAY):
            trade_size = random.uniform(
                AGENT_TRADE_SIZE_RANGE[0], AGENT_TRADE_SIZE_RANGE[1]
            )

            current_price = calculate_bonding_curve_price(current_supply, bonding_curve_params)

            if np.sum(self.price_memory) != 0.0:
                average_price = np.mean(self.price_memory[:-1])
                price_diff = (self.price_memory[-1] - average_price) / average_price
                if price_diff > AGENT_TREND_THRESHOLD:
                    max_buy_tokens = self.capital / (current_price * (1 + TRADING_FEE))
                    tokens_to_buy = max_buy_tokens * trade_size
                    self.last_trade_step = current_step
                    return "buy", tokens_to_buy
                elif price_diff < -AGENT_TREND_THRESHOLD and self.tokens > 0:
                    tokens_to_sell = self.tokens * trade_size
                    self.last_trade_step = current_step
                    return "sell", tokens_to_sell

            if self.capital > 0:
                max_buy_tokens = self.capital / (current_price * (1 + TRADING_FEE))
                tokens_to_buy = max_buy_tokens * trade_size
                self.last_trade_step = current_step
                return "buy", tokens_to_buy

            elif self.tokens > 0:
                tokens_to_sell = self.tokens * trade_size
                self.last_trade_step = current_step
                return "sell", tokens_to_sell

        return None, 0