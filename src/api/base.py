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

# Import enhanced logging if available
try:
    from src.utils.trading_logger import TradingMode, LogCategory
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False

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

        # Enhanced logging setup
        self._setup_enhanced_logging()

        # Trading session tracking
        self.current_session_id = None
        self.current_correlation_id = None

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

    def _setup_enhanced_logging(self):
        """Setup enhanced logging for exchange client"""
        if ENHANCED_LOGGING_AVAILABLE:
            # Use enhanced logger factory for API integration
            self.logger = LoggerFactory.get_component_trading_logger(
                component="api_integration",
                strategy="exchange_client"
            )
        else:
            # Fallback to standard logging
            self.logger = LoggerFactory.get_api_logger("exchange")

        # Setup logging methods
        self._setup_logging_methods()

    def _setup_logging_methods(self):
        """Setup enhanced logging methods"""
        if hasattr(self.logger, 'log_api_request'):
            # Enhanced logger available
            self.log_api_request = self._enhanced_log_api_request
            self.log_api_response = self._enhanced_log_api_response
            self.log_api_error = self._enhanced_log_api_error
            self.log_connection_event = self._enhanced_log_connection_event
            self.log_market_data = self._enhanced_log_market_data
        else:
            # Standard logger - use basic methods
            self.log_api_request = self._basic_log_api_request
            self.log_api_response = self._basic_log_api_response
            self.log_api_error = self._basic_log_api_error
            self.log_connection_event = self._basic_log_connection_event
            self.log_market_data = self._basic_log_market_data

    def set_trading_session(self, session_id: str, correlation_id: str = None):
        """Set trading session context for logging"""
        self.current_session_id = session_id
        self.current_correlation_id = correlation_id

        # Update logger context if enhanced logging is available
        if hasattr(self.logger, 'base_logger') and hasattr(self.logger.base_logger, 'set_context'):
            self.logger.base_logger.set_context(
                session_id=session_id,
                correlation_id=correlation_id,
                component="api_integration"
            )

    # Enhanced Logging Methods

    def _enhanced_log_api_request(self, method: str, endpoint: str, params: dict = None, **context):
        """Log API request using enhanced logger"""
        try:
            # Sanitize sensitive parameters
            safe_params = self._sanitize_params(params or {})

            self.logger.log_api_request(
                message=f"API request: {method} {endpoint}",
                method=method,
                endpoint=endpoint,
                parameters=safe_params,
                testnet=self.config.testnet,
                session_id=self.current_session_id,
                correlation_id=self.current_correlation_id,
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced API request logging failed: {e}")
            self._basic_log_api_request(method, endpoint, params, **context)

    def _basic_log_api_request(self, method: str, endpoint: str, params: dict = None, **context):
        """Log API request using basic logger"""
        safe_params = self._sanitize_params(params or {})
        testnet_str = " [TESTNET]" if self.config.testnet else ""

        self.logger.info(
            f"[API] {method} {endpoint}{testnet_str} - {safe_params}",
            extra={
                'api_method': method,
                'api_endpoint': endpoint,
                'api_params': safe_params,
                'testnet': self.config.testnet,
                'session_id': self.current_session_id,
                **context
            }
        )

    def _enhanced_log_api_response(self, method: str, endpoint: str, response_data: dict = None,
                                  latency_ms: float = None, **context):
        """Log API response using enhanced logger"""
        try:
            # Sanitize sensitive response data
            safe_response = self._sanitize_response(response_data or {})

            self.logger.log_api_response(
                message=f"API response: {method} {endpoint}",
                method=method,
                endpoint=endpoint,
                response_data=safe_response,
                latency_ms=latency_ms,
                response_size=len(str(response_data)) if response_data else 0,
                session_id=self.current_session_id,
                correlation_id=self.current_correlation_id,
                success=True,
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced API response logging failed: {e}")
            self._basic_log_api_response(method, endpoint, response_data, latency_ms, **context)

    def _basic_log_api_response(self, method: str, endpoint: str, response_data: dict = None,
                               latency_ms: float = None, **context):
        """Log API response using basic logger"""
        safe_response = self._sanitize_response(response_data or {})
        latency_str = f" ({latency_ms:.1f}ms)" if latency_ms is not None else ""

        self.logger.info(
            f"[API] Response {method} {endpoint}{latency_str} - {len(str(response_data)) if response_data else 0} bytes",
            extra={
                'api_method': method,
                'api_endpoint': endpoint,
                'response_size': len(str(response_data)) if response_data else 0,
                'latency_ms': latency_ms,
                'session_id': self.current_session_id,
                **context
            }
        )

    def _enhanced_log_api_error(self, method: str, endpoint: str, error: Exception, **context):
        """Log API error using enhanced logger"""
        try:
            self.logger.log_api_error(
                message=f"API error: {method} {endpoint} - {str(error)}",
                method=method,
                endpoint=endpoint,
                error_type=type(error).__name__,
                error_message=str(error),
                error_code=getattr(error, 'code', None),
                session_id=self.current_session_id,
                correlation_id=self.current_correlation_id,
                requires_retry=self._should_retry_error(error),
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced API error logging failed: {e}")
            self._basic_log_api_error(method, endpoint, error, **context)

    def _basic_log_api_error(self, method: str, endpoint: str, error: Exception, **context):
        """Log API error using basic logger"""
        error_code = getattr(error, 'code', 'N/A')

        self.logger.error(
            f"[API] Error {method} {endpoint} - {type(error).__name__}: {str(error)} (code: {error_code})",
            extra={
                'api_method': method,
                'api_endpoint': endpoint,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_code': error_code,
                'session_id': self.current_session_id,
                **context
            }
        )

    def _enhanced_log_connection_event(self, event: str, **context):
        """Log connection event using enhanced logger"""
        try:
            self.logger.log_connection(
                message=f"Connection event: {event}",
                event_type=event,
                exchange="binance" if "binance" in self.__class__.__name__.lower() else "exchange",
                testnet=self.config.testnet,
                session_id=self.current_session_id,
                correlation_id=self.current_correlation_id,
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced connection logging failed: {e}")
            self._basic_log_connection_event(event, **context)

    def _basic_log_connection_event(self, event: str, **context):
        """Log connection event using basic logger"""
        testnet_str = " [TESTNET]" if self.config.testnet else ""

        self.logger.info(
            f"[Connection] {event}{testnet_str}",
            extra={
                'connection_event': event,
                'testnet': self.config.testnet,
                'session_id': self.current_session_id,
                **context
            }
        )

    def _enhanced_log_market_data(self, symbol: str, data_type: str, data: dict, **context):
        """Log market data using enhanced logger"""
        try:
            self.logger.log_market_data(
                message=f"Market data: {symbol} {data_type}",
                symbol=symbol,
                data_type=data_type,
                timestamp=data.get('timestamp'),
                price=data.get('price'),
                quantity=data.get('quantity'),
                session_id=self.current_session_id,
                correlation_id=self.current_correlation_id,
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced market data logging failed: {e}")
            self._basic_log_market_data(symbol, data_type, data, **context)

    def _basic_log_market_data(self, symbol: str, data_type: str, data: dict, **context):
        """Log market data using basic logger"""
        price_str = f" @ {data.get('price')}" if data.get('price') else ""
        qty_str = f" qty: {data.get('quantity')}" if data.get('quantity') else ""

        self.logger.debug(
            f"[MarketData] {symbol} {data_type}{price_str}{qty_str}",
            extra={
                'symbol': symbol,
                'data_type': data_type,
                'price': data.get('price'),
                'quantity': data.get('quantity'),
                'session_id': self.current_session_id,
                **context
            }
        )

    def _sanitize_params(self, params: dict) -> dict:
        """Remove sensitive information from parameters"""
        sensitive_keys = {'signature', 'api_key', 'secret', 'password', 'token'}
        return {
            k: '***MASKED***' if k.lower() in sensitive_keys else v
            for k, v in params.items()
        }

    def _sanitize_response(self, response: dict) -> dict:
        """Remove sensitive information from response data"""
        if not isinstance(response, dict):
            return response

        sensitive_keys = {'api_key', 'secret', 'password', 'token', 'private_key'}
        sanitized = {}

        for k, v in response.items():
            if k.lower() in sensitive_keys:
                sanitized[k] = '***MASKED***'
            elif isinstance(v, dict):
                sanitized[k] = self._sanitize_response(v)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                sanitized[k] = [self._sanitize_response(item) for item in v]
            else:
                sanitized[k] = v

        return sanitized

    def _should_retry_error(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry"""
        # Common retry conditions
        retry_error_codes = {
            -1001,  # Internal error
            -1021,  # Timestamp outside recvWindow
            503,    # Service unavailable
            429,    # Rate limit exceeded
        }

        error_code = getattr(error, 'code', None)
        return error_code in retry_error_codes


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