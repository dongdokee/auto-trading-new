# src/integration/__init__.py
"""
System Integration Module

Provides event-driven architecture for connecting all trading system components.
Implements orchestration, state management, and monitoring for complete automation.
"""

from .events.event_bus import EventBus
from .events.models import (
    BaseEvent, MarketDataEvent, StrategySignalEvent, PortfolioEvent,
    OrderEvent, ExecutionEvent, RiskEvent, SystemEvent
)
from .trading_orchestrator import TradingOrchestrator
from .state.manager import StateManager

__all__ = [
    'EventBus',
    'BaseEvent', 'MarketDataEvent', 'StrategySignalEvent', 'PortfolioEvent',
    'OrderEvent', 'ExecutionEvent', 'RiskEvent', 'SystemEvent',
    'TradingOrchestrator',
    'StateManager'
]