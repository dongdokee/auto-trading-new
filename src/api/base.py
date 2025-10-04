# src/api/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import time
import warnings

from src.core.models import Order, OrderSide, OrderUrgency
from src.core.patterns import BaseConnectionManager, LoggerFactory

# Keep old ConnectionError for backward compatibility
class ConnectionError(Exception):
    """Exception raised when connection operations fail"""
    pass


@dataclass
class ExchangeConfig:
    """Configuration for exchange API connection"""
    api_key: str
    api_secret: str
    testnet: bool = True
    timeout: int = 30
    rate_limit_per_minute: int = 1200

    def __post_init__(self):
        """Validate configuration parameters"""
        if not self.api_key:
            raise ValueError("API key cannot be empty")
        if not self.api_secret:
            raise ValueError("API secret cannot be empty")
        if len(self.api_key) < 8:
            raise ValueError("API key must be at least 8 characters")
        if len(self.api_secret) < 8:
            raise ValueError("API secret must be at least 8 characters")


class BaseExchangeClient(BaseConnectionManager):
    """Abstract base class for exchange API clients"""

    def __init__(self, config: ExchangeConfig):
        super().__init__(name="ExchangeClient")
        self.config = config
        self.logger = LoggerFactory.get_api_logger("exchange")

    async def _create_connection(self) -> Any:
        """Create exchange connection - implemented by subclasses"""
        return await self._create_exchange_connection()

    async def _close_connection(self, connection: Any) -> None:
        """Close exchange connection - implemented by subclasses"""
        await self._close_exchange_connection(connection)

    @abstractmethod
    async def _create_exchange_connection(self) -> Any:
        """Create the actual exchange connection"""
        pass

    @abstractmethod
    async def _close_exchange_connection(self, connection: Any) -> None:
        """Close the actual exchange connection"""
        pass

    @abstractmethod
    async def submit_order(self, order: Order) -> Dict[str, Any]:
        """Submit an order to the exchange"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get the status of an order"""
        pass

    @abstractmethod
    async def get_account_balance(self) -> Dict[str, Decimal]:
        """Get account balance information"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        pass

    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data for a symbol"""
        pass

    @abstractmethod
    async def subscribe_to_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to orderbook updates"""
        pass

    @abstractmethod
    async def subscribe_to_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to trade updates"""
        pass


class OrderConverter:
    """Utility class for converting internal orders to exchange format"""

    def to_exchange_format(self, order: Order) -> Dict[str, Any]:
        """Convert internal Order to exchange-specific format"""
        exchange_order = {
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': str(order.size)
        }

        # Determine order type based on urgency and price
        if order.urgency == OrderUrgency.IMMEDIATE or order.price is None:
            exchange_order['type'] = 'MARKET'
        else:
            exchange_order['type'] = 'LIMIT'
            exchange_order['price'] = str(order.price)

        return exchange_order


class RateLimitManager:
    """Manages API rate limiting with token bucket algorithm"""

    def __init__(self, requests_per_minute: int, window_seconds: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.tokens = requests_per_minute
        self.last_refill = time.time()
        self.requests_made = 0
        self.logger = LoggerFactory.get_logger("rate_limiter")

    def can_make_request(self) -> bool:
        """Check if a request can be made within rate limits"""
        self._refill_tokens()
        return self.tokens > 0

    def record_request(self) -> None:
        """Record that a request was made"""
        if self.can_make_request():
            self.tokens -= 1
            self.requests_made += 1

    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current window"""
        self._refill_tokens()
        return int(self.tokens)

    async def wait_for_reset(self) -> None:
        """Wait for the rate limit window to reset"""
        current_time = time.time()
        time_since_refill = current_time - self.last_refill

        if time_since_refill < self.window_seconds:
            wait_time = self.window_seconds - time_since_refill
            await asyncio.sleep(wait_time)

        self._refill_tokens()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time"""
        current_time = time.time()
        time_passed = current_time - self.last_refill

        if time_passed >= self.window_seconds:
            self.tokens = self.requests_per_minute
            self.last_refill = current_time
            self.requests_made = 0


# Deprecated: Use BaseConnectionManager from src.core.patterns instead
class ConnectionManager(BaseConnectionManager):
    """
    Manages connection state and lifecycle.

    DEPRECATED: Use BaseConnectionManager from src.core.patterns instead.
    This class is kept for backward compatibility only.
    """

    def __init__(self):
        warnings.warn(
            "ConnectionManager is deprecated. Use BaseConnectionManager from src.core.patterns instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(name="DeprecatedConnectionManager")

    async def _create_connection(self) -> Any:
        """Default connection factory - override in subclasses"""
        # Simulate connection establishment
        await asyncio.sleep(0.001)
        return "mock_connection"

    async def _close_connection(self, connection: Any) -> None:
        """Default disconnection - override in subclasses"""
        pass