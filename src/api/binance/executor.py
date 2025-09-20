# src/api/binance/executor.py
"""
Binance API executor that bridges the execution engine with the Binance API.
Provides order execution, market data integration, and real-time updates.
"""

from decimal import Decimal
from typing import Dict, Any, List, Optional, Callable
import asyncio
import logging

from src.execution.models import Order, OrderSide, OrderUrgency, ExecutionResult
from src.execution.order_manager import OrderManager
from src.execution.market_analyzer import MarketConditionAnalyzer
from src.api.base import ExchangeConfig
from .client import BinanceClient
from .websocket import BinanceWebSocket
from .exceptions import BinanceAPIError, BinanceOrderError


class BinanceExecutor:
    """
    Bridges the execution engine with Binance API for live trading.
    Provides order execution, market data, and real-time updates.
    """

    def __init__(self, config: ExchangeConfig):
        self.config = config

        # API clients
        self.client = BinanceClient(config)
        self.websocket = BinanceWebSocket(config)

        # Execution components
        self.order_manager = OrderManager()
        self.market_analyzer = MarketConditionAnalyzer()

        # State
        self._connected = False
        self._subscribed_symbols = set()

        # Callbacks
        self.orderbook_callbacks: Dict[str, List[Callable]] = {}
        self.trade_callbacks: Dict[str, List[Callable]] = {}

        # Logger
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> None:
        """Connect to Binance API and WebSocket streams"""
        try:
            # Connect REST API
            await self.client.connect()
            self.logger.info("Connected to Binance REST API")

            # Connect WebSocket
            await self.websocket.connect()
            self.logger.info("Connected to Binance WebSocket")

            self._connected = True
            self.logger.info("BinanceExecutor connected successfully")

        except Exception as e:
            self.logger.error(f"Failed to connect BinanceExecutor: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Binance API and WebSocket streams"""
        try:
            if self.websocket:
                await self.websocket.disconnect()

            if self.client:
                await self.client.disconnect()

            self._connected = False
            self._subscribed_symbols.clear()
            self.orderbook_callbacks.clear()
            self.trade_callbacks.clear()

            self.logger.info("BinanceExecutor disconnected")

        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

    def is_connected(self) -> bool:
        """Check if executor is connected to Binance"""
        return self._connected and self.client.is_connected() and self.websocket.is_connected()

    async def submit_order(self, order: Order) -> ExecutionResult:
        """
        Submit an order through the execution engine to Binance.

        Args:
            order: Order to submit

        Returns:
            ExecutionResult with execution details
        """
        if not self.is_connected():
            raise BinanceOrderError("Not connected to Binance")

        try:
            # Submit order to OrderManager first for tracking
            order_id = await self.order_manager.submit_order(order)

            # Submit to Binance
            binance_result = await self.client.submit_order(order)

            # Update order manager with result
            binance_order_id = str(binance_result.get("orderId"))
            filled_qty = Decimal(binance_result.get("executedQty", "0"))
            avg_price = Decimal(binance_result.get("avgPrice", "0"))

            await self.order_manager.update_order_status(
                order_id, filled_qty, avg_price
            )

            # Create execution result
            execution_result = ExecutionResult(
                order_id=binance_order_id,
                strategy="BINANCE_API",
                total_filled=filled_qty,
                avg_price=avg_price,
                total_cost=filled_qty * avg_price,
                original_size=order.size
            )

            self.logger.info(f"Order submitted successfully: {binance_order_id}")
            return execution_result

        except BinanceAPIError as e:
            self.logger.error(f"Binance API error submitting order: {e}")
            raise BinanceOrderError(f"Failed to submit order: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error submitting order: {e}")
            raise BinanceOrderError(f"Failed to submit order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order on Binance.

        Args:
            order_id: Binance order ID to cancel

        Returns:
            True if cancellation successful
        """
        if not self.is_connected():
            raise BinanceOrderError("Not connected to Binance")

        try:
            result = await self.client.cancel_order(order_id)

            # Update order manager
            await self.order_manager.cancel_order(order_id)

            self.logger.info(f"Order cancelled: {order_id}")
            return result

        except BinanceAPIError as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            raise BinanceOrderError(f"Failed to cancel order: {e}")

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status from Binance.

        Args:
            order_id: Binance order ID

        Returns:
            Order status information
        """
        if not self.is_connected():
            raise BinanceOrderError("Not connected to Binance")

        try:
            return await self.client.get_order_status(order_id)
        except BinanceAPIError as e:
            self.logger.error(f"Failed to get order status {order_id}: {e}")
            raise

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from Binance.

        Returns:
            Account balance and position information
        """
        if not self.is_connected():
            raise BinanceOrderError("Not connected to Binance")

        try:
            balance = await self.client.get_account_balance()
            positions = await self.client.get_positions()

            return {
                "balance": balance,
                "positions": positions
            }
        except BinanceAPIError as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise

    async def subscribe_market_data(self, symbol: str) -> None:
        """
        Subscribe to real-time market data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
        """
        if not self.is_connected():
            raise BinanceOrderError("Not connected to Binance")

        try:
            # Subscribe to orderbook
            await self.websocket.subscribe_orderbook(
                symbol,
                self._handle_orderbook_update
            )

            # Subscribe to trades
            await self.websocket.subscribe_trades(
                symbol,
                self._handle_trade_update
            )

            self._subscribed_symbols.add(symbol)
            self.logger.info(f"Subscribed to market data for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to {symbol}: {e}")
            raise

    async def get_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market conditions for analysis.

        Args:
            symbol: Trading symbol

        Returns:
            Market condition analysis
        """
        if not self.is_connected():
            raise BinanceOrderError("Not connected to Binance")

        try:
            # Get market data
            market_data = await self.client.get_market_data(symbol)

            # Analyze market conditions
            conditions = await self.market_analyzer.analyze_market_conditions(symbol)

            return {
                "market_data": market_data,
                "conditions": conditions
            }
        except Exception as e:
            self.logger.error(f"Failed to get market conditions for {symbol}: {e}")
            raise

    def add_orderbook_callback(self, symbol: str, callback: Callable) -> None:
        """Add callback for orderbook updates"""
        if symbol not in self.orderbook_callbacks:
            self.orderbook_callbacks[symbol] = []
        self.orderbook_callbacks[symbol].append(callback)

    def add_trade_callback(self, symbol: str, callback: Callable) -> None:
        """Add callback for trade updates"""
        if symbol not in self.trade_callbacks:
            self.trade_callbacks[symbol] = []
        self.trade_callbacks[symbol].append(callback)

    async def _handle_orderbook_update(self, data: Dict[str, Any]) -> None:
        """Handle orderbook update from WebSocket"""
        symbol = data.get("symbol")
        if symbol and symbol in self.orderbook_callbacks:
            # Update market analyzer
            await self.market_analyzer.update_orderbook(symbol, data)

            # Call registered callbacks
            for callback in self.orderbook_callbacks[symbol]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Error in orderbook callback: {e}")

    async def _handle_trade_update(self, data: Dict[str, Any]) -> None:
        """Handle trade update from WebSocket"""
        symbol = data.get("symbol")
        if symbol and symbol in self.trade_callbacks:
            # Update market analyzer
            await self.market_analyzer.update_trades(symbol, data)

            # Call registered callbacks
            for callback in self.trade_callbacks[symbol]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Error in trade callback: {e}")

    async def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics from OrderManager"""
        return self.order_manager.get_order_statistics()