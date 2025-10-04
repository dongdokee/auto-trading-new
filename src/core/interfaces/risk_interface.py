# src/core/interfaces/risk_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from decimal import Decimal

from src.core.models import Order


class IRiskController(ABC):
    """
    Abstract interface for risk controllers.

    Defines the contract for risk management systems, enabling
    different risk management implementations while maintaining
    consistent interfaces.
    """

    @abstractmethod
    def validate_order(self, order: Order, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an order against risk limits.

        Args:
            order: Order to validate
            market_state: Current market conditions

        Returns:
            Dict containing validation result and any modifications
        """
        pass

    @abstractmethod
    def update_position(self, symbol: str, side: str, size: Decimal, price: Decimal) -> None:
        """
        Update position information for risk tracking.

        Args:
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            size: Position size
            price: Entry/exit price
        """
        pass

    @abstractmethod
    def get_portfolio_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current portfolio risk metrics.

        Returns:
            Dict containing risk metrics (VaR, exposure, etc.)
        """
        pass

    @abstractmethod
    def check_risk_limits(self, symbol: str) -> Dict[str, Any]:
        """
        Check if current positions violate risk limits.

        Args:
            symbol: Symbol to check (or 'ALL' for portfolio-wide)

        Returns:
            Dict containing risk limit status
        """
        pass

    @abstractmethod
    def get_maximum_position_size(self, symbol: str, side: str) -> Decimal:
        """
        Get maximum allowable position size for a symbol.

        Args:
            symbol: Trading symbol
            side: Position side (BUY/SELL)

        Returns:
            Maximum position size allowed
        """
        pass


class IPositionSizer(ABC):
    """
    Abstract interface for position sizing.

    Defines the contract for calculating optimal position sizes
    based on risk parameters and market conditions.
    """

    @abstractmethod
    def calculate_position_size(
        self,
        signal: Dict[str, Any],
        market_state: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size for a trading signal.

        Args:
            signal: Trading signal information
            market_state: Current market conditions
            portfolio_state: Current portfolio state

        Returns:
            Dict containing position size and risk information
        """
        pass

    @abstractmethod
    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win_loss_ratio: float,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate Kelly Criterion position size.

        Args:
            win_rate: Historical win rate
            avg_win_loss_ratio: Average win/loss ratio
            confidence: Confidence in the signal (0-1)

        Returns:
            Kelly optimal position size fraction
        """
        pass

    @abstractmethod
    def adjust_for_volatility(
        self,
        base_size: Decimal,
        symbol: str,
        lookback_days: int = 20
    ) -> Decimal:
        """
        Adjust position size based on asset volatility.

        Args:
            base_size: Base position size
            symbol: Trading symbol
            lookback_days: Volatility calculation period

        Returns:
            Volatility-adjusted position size
        """
        pass

    @abstractmethod
    def get_sizing_parameters(self) -> Dict[str, Any]:
        """
        Get current position sizing parameters.

        Returns:
            Dict containing sizing configuration
        """
        pass