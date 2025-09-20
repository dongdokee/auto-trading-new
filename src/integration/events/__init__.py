# src/integration/events/__init__.py
"""
Event System Module

Provides the foundational event-driven architecture for system integration.
"""

from .event_bus import EventBus
from .models import (
    BaseEvent, MarketDataEvent, StrategySignalEvent, PortfolioEvent,
    OrderEvent, ExecutionEvent, RiskEvent, SystemEvent
)
from .handlers import BaseEventHandler

__all__ = [
    'EventBus',
    'BaseEvent', 'MarketDataEvent', 'StrategySignalEvent', 'PortfolioEvent',
    'OrderEvent', 'ExecutionEvent', 'RiskEvent', 'SystemEvent',
    'BaseEventHandler'
]