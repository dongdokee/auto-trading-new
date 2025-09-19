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
from .order_router import SmartOrderRouter
from .execution_algorithms import ExecutionAlgorithms
from .order_manager import OrderManager, OrderInfo
from .slippage_controller import SlippageController, SlippageMetrics, SlippageAlert

__all__ = [
    "Order",
    "ExecutionResult",
    "OrderSide",
    "OrderUrgency",
    "OrderStatus",
    "MarketConditionAnalyzer",
    "SmartOrderRouter",
    "ExecutionAlgorithms",
    "OrderManager",
    "OrderInfo",
    "SlippageController",
    "SlippageMetrics",
    "SlippageAlert"
]