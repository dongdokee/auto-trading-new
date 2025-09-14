"""
Database module for the AutoTrading system.
Provides database schemas, models, and data access layer.
"""

from .schemas.base import BaseModel, TimestampMixin
from .schemas.trading_schemas import (
    Position,
    Trade,
    Order,
    MarketData,
    Portfolio,
    RiskMetrics,
    StrategyPerformance
)

__all__ = [
    'BaseModel',
    'TimestampMixin',
    'Position',
    'Trade',
    'Order',
    'MarketData',
    'Portfolio',
    'RiskMetrics',
    'StrategyPerformance'
]