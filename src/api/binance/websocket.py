# src/api/binance/websocket.py
import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional
from decimal import Decimal

import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI

from src.api.base import ExchangeConfig
from .exceptions import BinanceConnectionError


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

        # Logger
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> None:
        """Establish WebSocket connection"""
        try:
            self.websocket = await websockets.connect(f"{self.base_ws_url}/ws")
            self._connected = True

            # Start background tasks
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._ping_task = asyncio.create_task(self._ping_loop())

            self.logger.info("WebSocket connected successfully")

        except (ConnectionClosed, InvalidURI, OSError) as e:
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

        self.logger.info("WebSocket disconnected")

    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self._connected and self.websocket is not None

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to orderbook depth stream"""
        if not self.is_connected():
            raise BinanceConnectionError("WebSocket not connected")

        stream_name = self._get_orderbook_stream(symbol)
        await self._subscribe_to_stream(stream_name, callback)

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to trade stream"""
        if not self.is_connected():
            raise BinanceConnectionError("WebSocket not connected")

        stream_name = self._get_trade_stream(symbol)
        await self._subscribe_to_stream(stream_name, callback)

    async def subscribe_markprice(self, symbol: str, callback: Callable) -> None:
        """Subscribe to mark price stream"""
        if not self.is_connected():
            raise BinanceConnectionError("WebSocket not connected")

        stream_name = self._get_markprice_stream(symbol)
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
                self.logger.info(f"Unsubscribed from {stream_name}")
            except ConnectionClosed as e:
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
            self.logger.info(f"Subscribed to {stream_name}")
        except ConnectionClosed as e:
            raise BinanceConnectionError(f"Failed to subscribe to {stream_name}: {e}")

    async def _listen_loop(self) -> None:
        """Main message listening loop"""
        while self._connected:
            try:
                message = await self.websocket.recv()
                await self._process_message(message)
            except ConnectionClosed:
                self.logger.warning("WebSocket connection closed")
                if self._auto_reconnect:
                    await self._handle_connection_loss()
                break
            except Exception as e:
                self.logger.error(f"Error in listen loop: {e}")

    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)

            # Handle subscription confirmations
            if "result" in data and "id" in data:
                self.logger.debug(f"Subscription confirmation: {data}")
                return

            # Handle stream data
            if "stream" in data and "data" in data:
                stream_name = data["stream"]
                stream_data = data["data"]

                if stream_name in self.subscriptions:
                    callback = self.subscriptions[stream_name]
                    processed_data = self._process_stream_data(stream_name, stream_data)
                    await callback(processed_data)

        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON message: {message}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

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
                self.logger.error(f"Error in ping loop: {e}")

    async def _send_ping(self) -> None:
        """Send ping message"""
        if self.websocket:
            try:
                await self.websocket.ping()
            except ConnectionClosed:
                self.logger.warning("Failed to send ping - connection closed")

    async def _handle_connection_loss(self) -> None:
        """Handle connection loss and attempt reconnection"""
        self.logger.info("Attempting to reconnect WebSocket...")

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

                self.logger.info("Reconnected and resubscribed to all streams")

        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
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