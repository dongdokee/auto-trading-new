# src/core/interfaces/portfolio_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from decimal import Decimal
import pandas as pd


class IPortfolioManager(ABC):
    """
    Abstract interface for portfolio management.

    Defines the contract for portfolio optimization, allocation,
    and rebalancing operations.
    """

    @abstractmethod
    def optimize_weights(
        self,
        returns_data: pd.DataFrame,
        constraints: Optional[Dict[str, Any]] = None,
        objective: str = "max_sharpe"
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights.

        Args:
            returns_data: Historical returns data
            constraints: Optimization constraints
            objective: Optimization objective

        Returns:
            Optimization result with weights and metrics
        """
        pass

    @abstractmethod
    def calculate_rebalancing_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: Decimal
    ) -> List[Dict[str, Any]]:
        """
        Calculate trades needed for rebalancing.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value

        Returns:
            List of required trades
        """
        pass

    @abstractmethod
    def calculate_risk_metrics(
        self,
        weights: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate portfolio risk metrics.

        Args:
            weights: Portfolio weights
            returns_data: Historical returns

        Returns:
            Risk metrics (VaR, volatility, etc.)
        """
        pass

    @abstractmethod
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get current portfolio summary.

        Returns:
            Portfolio status and metrics
        """
        pass


class IPerformanceTracker(ABC):
    """
    Abstract interface for performance tracking.

    Defines the contract for tracking and analyzing
    portfolio and strategy performance.
    """

    @abstractmethod
    def add_trade(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        timestamp: Any,
        strategy: Optional[str] = None
    ) -> None:
        """
        Record a trade for performance tracking.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Trade quantity
            price: Trade price
            timestamp: Trade timestamp
            strategy: Strategy that generated the trade
        """
        pass

    @abstractmethod
    def calculate_returns(
        self,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Calculate portfolio returns for a period.

        Args:
            start_date: Start date for calculation
            end_date: End date for calculation

        Returns:
            Returns analysis
        """
        pass

    @abstractmethod
    def calculate_attribution(
        self,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Calculate performance attribution.

        Args:
            benchmark_returns: Benchmark returns for comparison

        Returns:
            Attribution analysis
        """
        pass

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.

        Returns:
            Performance statistics (Sharpe, Sortino, etc.)
        """
        pass

    @abstractmethod
    def get_drawdown_analysis(self) -> Dict[str, Any]:
        """
        Get drawdown analysis.

        Returns:
            Drawdown metrics and periods
        """
        pass

    @abstractmethod
    def generate_performance_report(
        self,
        format_type: str = "dict"
    ) -> Any:
        """
        Generate comprehensive performance report.

        Args:
            format_type: Output format (dict, html, pdf)

        Returns:
            Performance report in specified format
        """
        pass