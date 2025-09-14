"""
Repository pattern implementation for data access abstraction.
Provides clean separation between business logic and data access.
"""

from .base import BaseRepository, RepositoryError
from .trading_repository import (
    PositionRepository,
    TradeRepository,
    OrderRepository,
    MarketDataRepository
)

__all__ = [
    'BaseRepository',
    'RepositoryError',
    'PositionRepository',
    'TradeRepository',
    'OrderRepository',
    'MarketDataRepository'
]