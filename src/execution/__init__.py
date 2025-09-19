# src/execution/__init__.py
"""
Order execution engine module

This module handles smart order routing, execution algorithms, and slippage control.
"""

from .models import (
    Order,
    ExecutionResult,
    OrderSide,
    OrderUrgency,
    OrderStatus
)
from .market_analyzer import MarketConditionAnalyzer

__all__ = [
    "Order",
    "ExecutionResult",
    "OrderSide",
    "OrderUrgency",
    "OrderStatus",
    "MarketConditionAnalyzer"
]