# src/api/binance/websocket.py
import asyncio
import json
import logging
import time
from typing import Dict, Any, Callable, Optional
from decimal import Decimal

import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI

from src.api.base import ExchangeConfig
from src.core.patterns import LoggerFactory
from .exceptions import BinanceConnectionError

# Import enhanced logging if available
try:
    from src.utils.trading_logger import TradingMode, LogCategory
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False


class BinanceWebSocket:
    """Binance WebSocket manager for real-time data streams"""

    def __init__(self, config: ExchangeConfig):
        self.config = config

        # WebSocket URLs
        self.testnet_ws_url = "wss://stream.binancefuture.com"
        self.mainnet_ws_url = "wss://fstream.binance.com"
        self.base_ws_url = self.testnet_ws_url if config.testnet else self.mainnet_ws_url

        # Connection state
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._connected = False
        self._auto_reconnect = True
        self._reconnect_delay = 5  # seconds

        # Subscriptions
        self.subscriptions: Dict[str, Callable] = {}
        self._subscription_id = 1

        # Background tasks
        self._listen_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

        # Enhanced logging setup
        self._setup_enhanced_logging()

        # Trading session tracking
        self.current_session_id = None
        self.current_correlation_id = None

        # Performance metrics
        self._message_count = 0
        self._connection_start_time = None
        self._last_message_time = None

    async def connect(self) -> None:
        """Establish WebSocket connection"""
        try:
            self._connection_start_time = time.time()
            self.log_connection_event("connecting", url=f"{self.base_ws_url}/ws")

            self.websocket = await websockets.connect(f"{self.base_ws_url}/ws")
            self._connected = True

            # Start background tasks
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._ping_task = asyncio.create_task(self._ping_loop())

            connection_time = time.time() - self._connection_start_time
            self.log_connection_event("connected", connection_time_ms=connection_time * 1000)

        except (ConnectionClosed, InvalidURI, OSError) as e:
            self.log_websocket_error("connection_failed", e)
            raise BinanceConnectionError(f"Failed to connect WebSocket: {e}")

    async def disconnect(self) -> None:
        """Close WebSocket connection"""
        self._connected = False

        # Cancel background tasks
        if self._listen_task:
            self._listen_task.cancel()
        if self._ping_task:
            self._ping_task.cancel()

        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        # Clear subscriptions
        self.subscriptions.clear()

        self.log_connection_event("disconnected")

    def set_trading_session(self, session_id: str, correlation_id: str = None):
        """Set trading session context for logging"""
        self.current_session_id = session_id
        self.current_correlation_id = correlation_id

        # Update logger context if enhanced logging is available
        if hasattr(self.logger, 'base_logger') and hasattr(self.logger.base_logger, 'set_context'):
            self.logger.base_logger.set_context(
                session_id=session_id,
                correlation_id=correlation_id,
                component="websocket_integration"
            )

    def _setup_enhanced_logging(self):
        """Setup enhanced logging for WebSocket client"""
        if ENHANCED_LOGGING_AVAILABLE:
            # Use enhanced logger factory for WebSocket integration
            self.logger = LoggerFactory.get_component_trading_logger(
                component="websocket_integration",
                strategy="binance_websocket"
            )
        else:
            # Fallback to standard logging
            self.logger = LoggerFactory.get_api_logger("binance_websocket")

        # Setup logging methods
        self._setup_logging_methods()

    def _setup_logging_methods(self):
        """Setup enhanced logging methods"""
        if hasattr(self.logger, 'log_connection'):
            # Enhanced logger available
            self.log_connection_event = self._enhanced_log_connection_event
            self.log_subscription_event = self._enhanced_log_subscription_event
            self.log_stream_data = self._enhanced_log_stream_data
            self.log_websocket_error = self._enhanced_log_websocket_error
            self.log_performance_metrics = self._enhanced_log_performance_metrics
        else:
            # Standard logger - use basic methods
            self.log_connection_event = self._basic_log_connection_event
            self.log_subscription_event = self._basic_log_subscription_event
            self.log_stream_data = self._basic_log_stream_data
            self.log_websocket_error = self._basic_log_websocket_error
            self.log_performance_metrics = self._basic_log_performance_metrics

    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self._connected and self.websocket is not None

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to orderbook depth stream"""
        if not self.is_connected():
            raise BinanceConnectionError("WebSocket not connected")

        stream_name = self._get_orderbook_stream(symbol)
        self.log_subscription_event("subscribing", stream_name, "orderbook", symbol)
        await self._subscribe_to_stream(stream_name, callback)

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to trade stream"""
        if not self.is_connected():
            raise BinanceConnectionError("WebSocket not connected")

        stream_name = self._get_trade_stream(symbol)
        self.log_subscription_event("subscribing", stream_name, "trades", symbol)
        await self._subscribe_to_stream(stream_name, callback)

    async def subscribe_markprice(self, symbol: str, callback: Callable) -> None:
        """Subscribe to mark price stream"""
        if not self.is_connected():
            raise BinanceConnectionError("WebSocket not connected")

        stream_name = self._get_markprice_stream(symbol)
        self.log_subscription_event("subscribing", stream_name, "markprice", symbol)
        await self._subscribe_to_stream(stream_name, callback)

    async def unsubscribe(self, stream_name: str) -> None:
        """Unsubscribe from a stream"""
        if not self.is_connected():
            raise BinanceConnectionError("WebSocket not connected")

        if stream_name in self.subscriptions:
            subscription_message = {
                "method": "UNSUBSCRIBE",
                "params": [stream_name],
                "id": self._subscription_id
            }
            self._subscription_id += 1

            try:
                await self.websocket.send(json.dumps(subscription_message))
                del self.subscriptions[stream_name]
                self.log_subscription_event("unsubscribed", stream_name)
            except ConnectionClosed as e:
                self.log_websocket_error("unsubscribe_failed", e, stream_name=stream_name)
                raise BinanceConnectionError(f"Failed to unsubscribe: {e}")

    # Private helper methods

    async def _subscribe_to_stream(self, stream_name: str, callback: Callable) -> None:
        """Subscribe to a specific stream"""
        subscription_message = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": self._subscription_id
        }
        self._subscription_id += 1

        try:
            await self.websocket.send(json.dumps(subscription_message))
            self.subscriptions[stream_name] = callback
            self.log_subscription_event("subscribed", stream_name)
        except ConnectionClosed as e:
            self.log_websocket_error("subscribe_failed", e, stream_name=stream_name)
            raise BinanceConnectionError(f"Failed to subscribe to {stream_name}: {e}")

    async def _listen_loop(self) -> None:
        """Main message listening loop"""
        while self._connected:
            try:
                message = await self.websocket.recv()
                self._message_count += 1
                self._last_message_time = time.time()
                await self._process_message(message)
            except ConnectionClosed:
                self.log_connection_event("connection_lost")
                if self._auto_reconnect:
                    await self._handle_connection_loss()
                break
            except Exception as e:
                self.log_websocket_error("listen_loop_error", e)

    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)

            # Handle subscription confirmations
            if "result" in data and "id" in data:
                self.log_subscription_event("confirmation_received", result=data.get("result"))
                return

            # Handle stream data
            if "stream" in data and "data" in data:
                stream_name = data["stream"]
                stream_data = data["data"]

                if stream_name in self.subscriptions:
                    callback = self.subscriptions[stream_name]
                    processed_data = self._process_stream_data(stream_name, stream_data)

                    # Log stream data
                    symbol = processed_data.get("symbol", "unknown")
                    data_type = self._get_data_type_from_stream(stream_name)
                    self.log_stream_data(symbol, data_type, processed_data)

                    await callback(processed_data)

        except json.JSONDecodeError:
            self.log_websocket_error("invalid_json", None, message=message[:100])
        except Exception as e:
            self.log_websocket_error("message_processing_error", e, message=message[:100])

    def _process_stream_data(self, stream_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process stream data based on stream type"""
        if "@depth" in stream_name:
            return self._process_orderbook_data(data)
        elif "@aggTrade" in stream_name:
            return self._process_trade_data(data)
        elif "@markPrice" in stream_name:
            return self._process_markprice_data(data)
        else:
            return data

    def _process_orderbook_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process orderbook depth data"""
        return {
            "symbol": data["s"],
            "event_time": data["E"],
            "first_update_id": data["U"],
            "final_update_id": data["u"],
            "bids": [[Decimal(price), Decimal(qty)] for price, qty in data["b"]],
            "asks": [[Decimal(price), Decimal(qty)] for price, qty in data["a"]]
        }

    def _process_trade_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process trade data"""
        return {
            "symbol": data["s"],
            "aggregate_trade_id": data["a"],
            "price": data["p"],
            "quantity": data["q"],
            "first_trade_id": data["f"],
            "last_trade_id": data["l"],
            "trade_time": data["T"],
            "is_buyer_maker": data["m"],
            "event_time": data["E"]
        }

    def _process_markprice_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process mark price data"""
        return {
            "symbol": data["s"],
            "mark_price": Decimal(data["p"]),
            "index_price": Decimal(data["i"]),
            "estimated_settle_price": Decimal(data["P"]),
            "funding_rate": Decimal(data["r"]),
            "next_funding_time": data["T"],
            "event_time": data["E"]
        }

    async def _ping_loop(self) -> None:
        """Send ping messages to keep connection alive"""
        while self._connected:
            try:
                await asyncio.sleep(20)  # Ping every 20 seconds
                if self.is_connected():
                    await self._send_ping()
            except Exception as e:
                self.log_websocket_error("ping_loop_error", e)

    async def _send_ping(self) -> None:
        """Send ping message"""
        if self.websocket:
            try:
                await self.websocket.ping()
            except ConnectionClosed:
                self.log_connection_event("ping_failed")

    async def _handle_connection_loss(self) -> None:
        """Handle connection loss and attempt reconnection"""
        self.log_connection_event("reconnecting", delay_seconds=self._reconnect_delay)

        # Wait before reconnecting
        await asyncio.sleep(self._reconnect_delay)

        try:
            # Re-establish connection
            await self.connect()

            # Re-subscribe to all streams
            if self.subscriptions:
                streams_to_resubscribe = list(self.subscriptions.keys())
                callbacks = {stream: self.subscriptions[stream] for stream in streams_to_resubscribe}

                # Clear current subscriptions
                self.subscriptions.clear()

                # Re-subscribe
                for stream, callback in callbacks.items():
                    await self._subscribe_to_stream(stream, callback)

                self.log_connection_event("reconnected_and_resubscribed", stream_count=len(callbacks))

        except Exception as e:
            self.log_websocket_error("reconnection_failed", e)
            # Exponential backoff for next attempt
            self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def _get_orderbook_stream(self, symbol: str) -> str:
        """Get orderbook stream name"""
        return f"{symbol.lower()}@depth20@100ms"

    def _get_trade_stream(self, symbol: str) -> str:
        """Get trade stream name"""
        return f"{symbol.lower()}@aggTrade"

    def _get_markprice_stream(self, symbol: str) -> str:
        """Get mark price stream name"""
        return f"{symbol.lower()}@markPrice@1s"

    def _get_data_type_from_stream(self, stream_name: str) -> str:
        """Get data type from stream name"""
        if "@depth" in stream_name:
            return "orderbook"
        elif "@aggTrade" in stream_name:
            return "trades"
        elif "@markPrice" in stream_name:
            return "markprice"
        else:
            return "unknown"

    # Enhanced Logging Methods

    def _enhanced_log_connection_event(self, event: str, **context):
        """Log connection event using enhanced logger"""
        try:
            self.logger.log_connection(
                message=f"WebSocket {event}",
                event_type=event,
                exchange="binance",
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
            f"[WebSocket] {event}{testnet_str}",
            extra={
                'connection_event': event,
                'testnet': self.config.testnet,
                'session_id': self.current_session_id,
                **context
            }
        )

    def _enhanced_log_subscription_event(self, action: str, stream_name: str = None, data_type: str = None, symbol: str = None, **context):
        """Log subscription event using enhanced logger"""
        try:
            self.logger.log_subscription(
                message=f"Stream {action}: {stream_name or 'unknown'}",
                action=action,
                stream_name=stream_name,
                data_type=data_type,
                symbol=symbol,
                session_id=self.current_session_id,
                correlation_id=self.current_correlation_id,
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced subscription logging failed: {e}")
            self._basic_log_subscription_event(action, stream_name, data_type, symbol, **context)

    def _basic_log_subscription_event(self, action: str, stream_name: str = None, data_type: str = None, symbol: str = None, **context):
        """Log subscription event using basic logger"""
        stream_info = f" ({stream_name})" if stream_name else ""

        self.logger.info(
            f"[Subscription] {action}{stream_info}",
            extra={
                'subscription_action': action,
                'stream_name': stream_name,
                'data_type': data_type,
                'symbol': symbol,
                'session_id': self.current_session_id,
                **context
            }
        )

    def _enhanced_log_stream_data(self, symbol: str, data_type: str, data: dict, **context):
        """Log stream data using enhanced logger"""
        try:
            self.logger.log_market_data(
                message=f"Stream data: {symbol} {data_type}",
                symbol=symbol,
                data_type=data_type,
                timestamp=data.get('event_time'),
                price=data.get('price') or data.get('mark_price'),
                quantity=data.get('quantity'),
                session_id=self.current_session_id,
                correlation_id=self.current_correlation_id,
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced stream data logging failed: {e}")
            self._basic_log_stream_data(symbol, data_type, data, **context)

    def _basic_log_stream_data(self, symbol: str, data_type: str, data: dict, **context):
        """Log stream data using basic logger"""
        price_str = ""
        if data.get('price'):
            price_str = f" @ {data.get('price')}"
        elif data.get('mark_price'):
            price_str = f" @ {data.get('mark_price')}"

        self.logger.debug(
            f"[StreamData] {symbol} {data_type}{price_str}",
            extra={
                'symbol': symbol,
                'data_type': data_type,
                'price': data.get('price') or data.get('mark_price'),
                'quantity': data.get('quantity'),
                'session_id': self.current_session_id,
                **context
            }
        )

    def _enhanced_log_websocket_error(self, error_type: str, error: Exception = None, **context):
        """Log WebSocket error using enhanced logger"""
        try:
            error_message = str(error) if error else "Unknown error"
            self.logger.log_websocket_error(
                message=f"WebSocket error: {error_type} - {error_message}",
                error_type=error_type,
                error_message=error_message,
                error_class=type(error).__name__ if error else "Unknown",
                session_id=self.current_session_id,
                correlation_id=self.current_correlation_id,
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced WebSocket error logging failed: {e}")
            self._basic_log_websocket_error(error_type, error, **context)

    def _basic_log_websocket_error(self, error_type: str, error: Exception = None, **context):
        """Log WebSocket error using basic logger"""
        error_message = str(error) if error else "Unknown error"

        self.logger.error(
            f"[WebSocket] {error_type}: {error_message}",
            extra={
                'error_type': error_type,
                'error_message': error_message,
                'error_class': type(error).__name__ if error else "Unknown",
                'session_id': self.current_session_id,
                **context
            }
        )

    def _enhanced_log_performance_metrics(self, **metrics):
        """Log performance metrics using enhanced logger"""
        try:
            uptime = time.time() - self._connection_start_time if self._connection_start_time else 0

            self.logger.log_performance(
                message="WebSocket performance metrics",
                component="websocket",
                uptime_seconds=uptime,
                message_count=self._message_count,
                active_subscriptions=len(self.subscriptions),
                last_message_age=time.time() - self._last_message_time if self._last_message_time else None,
                session_id=self.current_session_id,
                correlation_id=self.current_correlation_id,
                **metrics
            )
        except Exception as e:
            self.logger.error(f"Enhanced performance logging failed: {e}")
            self._basic_log_performance_metrics(**metrics)

    def _basic_log_performance_metrics(self, **metrics):
        """Log performance metrics using basic logger"""
        uptime = time.time() - self._connection_start_time if self._connection_start_time else 0

        self.logger.info(
            f"[Performance] WebSocket uptime: {uptime:.1f}s, messages: {self._message_count}, subscriptions: {len(self.subscriptions)}",
            extra={
                'uptime_seconds': uptime,
                'message_count': self._message_count,
                'active_subscriptions': len(self.subscriptions),
                'session_id': self.current_session_id,
                **metrics
            }
        )

    def get_performance_metrics(self) -> dict:
        """Get current performance metrics"""
        uptime = time.time() - self._connection_start_time if self._connection_start_time else 0
        return {
            'uptime_seconds': uptime,
            'message_count': self._message_count,
            'active_subscriptions': len(self.subscriptions),
            'last_message_age': time.time() - self._last_message_time if self._last_message_time else None,
            'reconnect_delay': self._reconnect_delay,
            'is_connected': self.is_connected()
        }