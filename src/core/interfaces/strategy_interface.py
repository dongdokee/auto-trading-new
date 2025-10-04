# src/core/interfaces/strategy_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class IStrategySignal:
    """
    Interface for strategy signals.

    Standard signal format that all strategies must produce.
    """
    symbol: str
    action: str  # "BUY", "SELL", "HOLD", "CLOSE"
    strength: float  # [0, 1] - Signal strength
    confidence: float  # [0, 1] - Signal confidence
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IStrategy(ABC):
    """
    Abstract interface for trading strategies.

    Defines the contract that all strategy implementations must follow,
    enabling the system to work with multiple strategies in a unified way.
    """

    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any], current_index: int = -1) -> IStrategySignal:
        """
        Generate trading signal based on market data.

        Args:
            market_data: Market data dictionary
            current_index: Current data index for backtesting

        Returns:
            IStrategySignal: Generated trading signal
        """
        pass

    @abstractmethod
    def update_parameters(self, **kwargs) -> None:
        """
        Update strategy parameters.

        Args:
            **kwargs: Parameters to update
        """
        pass

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics.

        Returns:
            Dict containing performance statistics
        """
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset strategy internal state.
        """
        pass


class IStrategyManager(ABC):
    """
    Abstract interface for strategy managers.

    Defines the contract for coordinating multiple strategies and
    aggregating their signals.
    """

    @abstractmethod
    def add_strategy(self, name: str, strategy: IStrategy, weight: float = 1.0) -> None:
        """
        Add a strategy to the manager.

        Args:
            name: Strategy name
            strategy: Strategy instance
            weight: Strategy weight for signal aggregation
        """
        pass

    @abstractmethod
    def remove_strategy(self, name: str) -> None:
        """
        Remove a strategy from the manager.

        Args:
            name: Strategy name to remove
        """
        pass

    @abstractmethod
    async def generate_trading_signals(self, market_data: Dict[str, Any], current_index: int = -1) -> Dict[str, Any]:
        """
        Generate aggregated trading signals from all strategies.

        Args:
            market_data: Market data dictionary
            current_index: Current data index for backtesting

        Returns:
            Dict containing aggregated signals and metadata
        """
        pass

    @abstractmethod
    def update_strategy_performance(self, strategy_name: str, pnl: Decimal, winning: bool) -> None:
        """
        Update strategy performance metrics.

        Args:
            strategy_name: Name of strategy
            pnl: Profit/loss from signal
            winning: Whether the signal was profitable
        """
        pass

    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status and performance.

        Returns:
            Dict containing system status information
        """
        pass