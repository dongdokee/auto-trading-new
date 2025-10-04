# src/core/interfaces/__init__.py
"""
Core interface abstractions for module boundaries.

This module provides abstract interfaces that define contracts between
different modules in the trading system, promoting loose coupling and
dependency inversion.
"""

from .exchange_interface import IExchangeClient, IWebSocketManager
from .strategy_interface import IStrategy, IStrategyManager
from .risk_interface import IRiskController, IPositionSizer
from .execution_interface import IOrderManager, IExecutionEngine
from .portfolio_interface import IPortfolioManager, IPerformanceTracker

__all__ = [
    # Exchange interfaces
    "IExchangeClient",
    "IWebSocketManager",

    # Strategy interfaces
    "IStrategy",
    "IStrategyManager",

    # Risk management interfaces
    "IRiskController",
    "IPositionSizer",

    # Execution interfaces
    "IOrderManager",
    "IExecutionEngine",

    # Portfolio interfaces
    "IPortfolioManager",
    "IPerformanceTracker"
]