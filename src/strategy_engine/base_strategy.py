"""
Base Strategy Interface

Defines the abstract interface that all trading strategies must implement.
Provides common functionality for performance tracking, configuration management,
and signal generation standardization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


@dataclass
class StrategySignal:
    """
    Standardized signal output from trading strategies

    Represents a trading signal with all necessary information for position management.
    All strategies must output signals in this format for consistency.
    """

    symbol: str  # Trading pair (e.g., "BTCUSDT")
    action: str  # Action: "BUY", "SELL", "HOLD", "CLOSE"
    strength: float  # Signal strength [0, 1]
    confidence: float  # Confidence in signal [0, 1]
    stop_loss: Optional[float] = None  # Stop loss price (optional)
    take_profit: Optional[float] = None  # Take profit price (optional)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional strategy data

    def __post_init__(self):
        """Validate signal parameters after initialization"""
        if self.action not in ["BUY", "SELL", "HOLD", "CLOSE"]:
            raise ValueError(f"Invalid action: {self.action}")

        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"Strength must be between 0 and 1: {self.strength}")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0 and 1: {self.confidence}")


@dataclass
class StrategyConfig:
    """
    Configuration for trading strategies

    Contains all parameters needed to initialize and configure a trading strategy.
    """

    name: str  # Strategy name
    enabled: bool = True  # Whether strategy is active
    weight: float = 1.0  # Strategy weight in portfolio [0, 1]
    parameters: Dict[str, Any] = field(default_factory=dict)  # Strategy-specific parameters


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies

    Provides common functionality and defines the interface that all strategies must implement.
    Each concrete strategy must implement the generate_signal method.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize base strategy with configuration

        Args:
            config: Strategy configuration including name, parameters, etc.
        """
        self.name = config.name
        self.enabled = config.enabled
        self.weight = config.weight
        self.parameters = config.parameters

        # Performance tracking
        self.total_signals = 0
        self.winning_signals = 0
        self.total_pnl = 0.0

        # Signal history (for analysis)
        self.signal_history: List[StrategySignal] = []

    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> StrategySignal:
        """
        Generate trading signal based on market data

        Args:
            market_data: Dictionary containing market information including:
                - symbol: Trading pair
                - price/close: Current price
                - volume: Trading volume
                - ohlcv_data: Historical OHLCV DataFrame (if needed)
                - Additional market state information

        Returns:
            StrategySignal: Trading signal with action, strength, confidence

        Note:
            This method must be implemented by each concrete strategy.
            Should return HOLD signal when no clear trading opportunity exists.
        """
        pass

    def update_performance(self, pnl: float, winning: bool) -> None:
        """
        Update strategy performance metrics

        Args:
            pnl: Profit/loss from the trade
            winning: Whether the trade was profitable
        """
        self.total_signals += 1
        self.total_pnl += pnl

        if winning:
            self.winning_signals += 1

    def get_win_rate(self) -> float:
        """
        Calculate current win rate

        Returns:
            float: Win rate as percentage [0, 1]
        """
        if self.total_signals == 0:
            return 0.0

        return self.winning_signals / self.total_signals

    def get_average_pnl(self) -> float:
        """
        Calculate average PnL per trade

        Returns:
            float: Average profit/loss per trade
        """
        if self.total_signals == 0:
            return 0.0

        return self.total_pnl / self.total_signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information

        Returns:
            dict: Strategy statistics and configuration
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "weight": self.weight,
            "total_signals": self.total_signals,
            "winning_signals": self.winning_signals,
            "total_pnl": self.total_pnl,
            "win_rate": self.get_win_rate(),
            "average_pnl": self.get_average_pnl(),
            "parameters": self.parameters
        }

    def enable(self) -> None:
        """Enable strategy for signal generation"""
        self.enabled = True

    def disable(self) -> None:
        """Disable strategy (will not generate signals)"""
        self.enabled = False

    def update_weight(self, new_weight: float) -> None:
        """
        Update strategy weight with bounds checking

        Args:
            new_weight: New weight value [0, 1]
        """
        self.weight = max(0.0, min(1.0, new_weight))

    def reset_performance(self) -> None:
        """Reset all performance metrics to zero"""
        self.total_signals = 0
        self.winning_signals = 0
        self.total_pnl = 0.0
        self.signal_history.clear()

    def get_recent_signals(self, count: int = 10) -> List[StrategySignal]:
        """
        Get recent signals for analysis

        Args:
            count: Number of recent signals to return

        Returns:
            list: Recent StrategySignal objects
        """
        return self.signal_history[-count:] if self.signal_history else []

    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """
        Validate that market data contains required fields

        Args:
            market_data: Market data dictionary to validate

        Returns:
            bool: True if data is valid, False otherwise
        """
        required_fields = ["symbol"]

        for field in required_fields:
            if field not in market_data:
                return False

        # Additional validation for specific data types
        if "close" in market_data:
            try:
                float(market_data["close"])
            except (ValueError, TypeError):
                return False

        return True

    def _log_signal(self, signal: StrategySignal) -> None:
        """
        Log signal to history for analysis

        Args:
            signal: Generated signal to log
        """
        self.signal_history.append(signal)

        # Keep only last 1000 signals to prevent memory issues
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]