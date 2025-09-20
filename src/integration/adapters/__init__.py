# src/integration/adapters/__init__.py
"""
Component Adapters Module

Provides adapters that bridge existing trading components with the event-driven system.
"""

from .strategy_adapter import StrategyAdapter
from .risk_adapter import RiskAdapter
from .execution_adapter import ExecutionAdapter
from .portfolio_adapter import PortfolioAdapter

__all__ = [
    'StrategyAdapter',
    'RiskAdapter',
    'ExecutionAdapter',
    'PortfolioAdapter'
]