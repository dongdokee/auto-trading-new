# src/market_data/websocket_bridge.py

import asyncio
import logging
from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import datetime

from ..api.binance.websocket import BinanceWebSocket
from .models import OrderBookSnapshot, TickData, TickType, OrderSide, OrderLevel
from .data_aggregator import DataAggregator


class MarketDataWebSocketBridge:
    """
    Bridge between WebSocket streams and market data processing pipeline
    """

    def __init__(self, data_aggregator: DataAggregator):
        self.logger = logging.getLogger(__name__)
        self.data_aggregator = data_aggregator

        # WebSocket connections per symbol
        self.websocket_connections: Dict[str, BinanceWebSocket] = {}
        self.subscribed_symbols: set = set()

        # Processing state
        self._running = False
        self.processing_stats = {
            'orderbooks_processed': 0,
            'trades_processed': 0,
            'errors': 0,
            'last_update': datetime.utcnow()
        }

    async def start(self) -> None:
        """Start the WebSocket bridge"""
        if self._running:
            return

        self._running = True
        await self.data_aggregator.start()
        self.logger.info("MarketDataWebSocketBridge started")

    async def stop(self) -> None:
        """Stop the WebSocket bridge"""
        if not self._running:
            return

        self._running = False

        # Disconnect all WebSocket connections
        for symbol, websocket in self.websocket_connections.items():
            try:
                await websocket.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting WebSocket for {symbol}: {e}")

        self.websocket_connections.clear()
        self.subscribed_symbols.clear()

        await self.data_aggregator.stop()
        self.logger.info("MarketDataWebSocketBridge stopped")

    async def subscribe_symbol(self, symbol: str, config: Any) -> None:
        """
        Subscribe to market data for a symbol

        Args:
            symbol: Trading symbol
            config: WebSocket configuration
        """
        if symbol in self.subscribed_symbols:
            self.logger.warning(f"Already subscribed to {symbol}")
            return

        try:
            # Subscribe to data aggregator
            self.data_aggregator.subscribe_symbol(symbol)

            # Create WebSocket connection
            websocket = BinanceWebSocket(config)
            await websocket.connect()

            # Subscribe to orderbook and trade streams
            await websocket.subscribe_orderbook(symbol, self._handle_orderbook_update)
            await websocket.subscribe_trades(symbol, self._handle_trade_update)

            # Store connection
            self.websocket_connections[symbol] = websocket
            self.subscribed_symbols.add(symbol)

            self.logger.info(f"Successfully subscribed to market data for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to {symbol}: {e}")
            # Cleanup on error
            if symbol in self.websocket_connections:
                del self.websocket_connections[symbol]
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
            raise

    async def unsubscribe_symbol(self, symbol: str) -> None:
        """
        Unsubscribe from market data for a symbol

        Args:
            symbol: Trading symbol
        """
        if symbol not in self.subscribed_symbols:
            return

        try:
            # Disconnect WebSocket
            if symbol in self.websocket_connections:
                websocket = self.websocket_connections[symbol]
                await websocket.disconnect()
                del self.websocket_connections[symbol]

            # Unsubscribe from data aggregator
            self.data_aggregator.unsubscribe_symbol(symbol)
            self.subscribed_symbols.remove(symbol)

            self.logger.info(f"Unsubscribed from market data for {symbol}")

        except Exception as e:
            self.logger.error(f"Error unsubscribing from {symbol}: {e}")

    async def _handle_orderbook_update(self, data: Dict[str, Any]) -> None:
        """
        Handle orderbook update from WebSocket

        Args:
            data: Orderbook data from WebSocket
        """
        try:
            # Convert WebSocket data to OrderBookSnapshot
            orderbook = self._convert_websocket_orderbook(data)

            # Process through data aggregator
            await self.data_aggregator.process_orderbook_update(orderbook)

            # Update stats
            self.processing_stats['orderbooks_processed'] += 1
            self.processing_stats['last_update'] = datetime.utcnow()

        except Exception as e:
            self.logger.error(f"Error processing orderbook update: {e}")
            self.processing_stats['errors'] += 1

    async def _handle_trade_update(self, data: Dict[str, Any]) -> None:
        """
        Handle trade update from WebSocket

        Args:
            data: Trade data from WebSocket
        """
        try:
            # Convert WebSocket data to TickData
            tick = self._convert_websocket_trade(data)

            # Process through data aggregator
            await self.data_aggregator.process_tick_update(tick)

            # Update stats
            self.processing_stats['trades_processed'] += 1
            self.processing_stats['last_update'] = datetime.utcnow()

        except Exception as e:
            self.logger.error(f"Error processing trade update: {e}")
            self.processing_stats['errors'] += 1

    def _convert_websocket_orderbook(self, data: Dict[str, Any]) -> OrderBookSnapshot:
        """
        Convert WebSocket orderbook data to OrderBookSnapshot

        Args:
            data: Raw WebSocket orderbook data

        Returns:
            OrderBookSnapshot: Converted orderbook
        """
        # Convert bid/ask arrays to OrderLevel objects
        bids = [
            OrderLevel(price=Decimal(str(level[0])), size=Decimal(str(level[1])))
            for level in data.get('bids', [])
        ]
        asks = [
            OrderLevel(price=Decimal(str(level[0])), size=Decimal(str(level[1])))
            for level in data.get('asks', [])
        ]

        return OrderBookSnapshot(
            symbol=data['symbol'],
            timestamp=datetime.utcnow(),  # Use current time if not provided
            event_time=data.get('event_time', 0),
            first_update_id=data.get('first_update_id', 0),
            final_update_id=data.get('final_update_id', 0),
            bids=bids,
            asks=asks
        )

    def _convert_websocket_trade(self, data: Dict[str, Any]) -> TickData:
        """
        Convert WebSocket trade data to TickData

        Args:
            data: Raw WebSocket trade data

        Returns:
            TickData: Converted tick data
        """
        # Determine trade side from buyer maker flag
        side = None
        if 'is_buyer_maker' in data:
            side = OrderSide.SELL if data['is_buyer_maker'] else OrderSide.BUY

        return TickData(
            symbol=data['symbol'],
            timestamp=datetime.utcnow(),
            tick_type=TickType.TRADE,
            price=Decimal(str(data['price'])) if 'price' in data else None,
            size=Decimal(str(data['quantity'])) if 'quantity' in data else None,
            side=side,
            trade_id=data.get('aggregate_trade_id'),
            event_time=data.get('event_time'),
            is_buyer_maker=data.get('is_buyer_maker')
        )

    def add_orderbook_callback(self, symbol: str, callback: callable) -> None:
        """
        Add callback for orderbook updates

        Args:
            symbol: Symbol to monitor
            callback: Callback function
        """
        self.data_aggregator.add_update_callback(symbol, callback)

    def add_pattern_callback(self, callback: callable) -> None:
        """
        Add callback for pattern detection

        Args:
            callback: Callback function
        """
        self.data_aggregator.add_pattern_callback(callback)

    async def get_market_data(self, symbol: str) -> Optional[Any]:
        """
        Get aggregated market data for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Aggregated market data or None
        """
        return await self.data_aggregator.get_market_data(symbol)

    async def estimate_market_impact(self, symbol: str, order_size: Decimal) -> Optional[Any]:
        """
        Estimate market impact for an order

        Args:
            symbol: Trading symbol
            order_size: Order size

        Returns:
            Market impact estimate or None
        """
        return await self.data_aggregator.estimate_market_impact(symbol, order_size)

    async def get_optimal_execution_windows(self, symbol: str, order_size: Decimal,
                                          hours_ahead: int = 24) -> List[Any]:
        """
        Get optimal execution windows

        Args:
            symbol: Trading symbol
            order_size: Order size
            hours_ahead: Hours to look ahead

        Returns:
            List of optimal execution windows
        """
        return await self.data_aggregator.get_optimal_execution_windows(
            symbol, order_size, hours_ahead
        )

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        bridge_stats = {
            'subscribed_symbols': len(self.subscribed_symbols),
            'websocket_connections': len(self.websocket_connections),
            'running': self._running,
            'bridge_stats': self.processing_stats.copy()
        }

        # Add aggregator stats
        aggregator_stats = self.data_aggregator.get_performance_metrics()

        return {
            'bridge': bridge_stats,
            'aggregator': aggregator_stats
        }

    def get_symbol_summary(self, symbol: str) -> Optional[Dict]:
        """Get summary for a specific symbol"""
        if symbol not in self.subscribed_symbols:
            return None

        return self.data_aggregator.get_symbol_summary(symbol)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all connections"""
        health_status = {
            'bridge_running': self._running,
            'subscribed_symbols_count': len(self.subscribed_symbols),
            'websocket_health': {},
            'aggregator_health': True
        }

        # Check WebSocket connections
        for symbol, websocket in self.websocket_connections.items():
            try:
                is_connected = websocket.is_connected()
                health_status['websocket_health'][symbol] = {
                    'connected': is_connected,
                    'subscriptions': len(websocket.subscriptions)
                }
            except Exception as e:
                health_status['websocket_health'][symbol] = {
                    'connected': False,
                    'error': str(e)
                }

        return health_status

    async def reconnect_symbol(self, symbol: str, config: Any) -> bool:
        """
        Reconnect WebSocket for a specific symbol

        Args:
            symbol: Trading symbol
            config: WebSocket configuration

        Returns:
            bool: True if reconnection successful
        """
        if symbol not in self.subscribed_symbols:
            return False

        try:
            # Disconnect existing connection
            if symbol in self.websocket_connections:
                await self.websocket_connections[symbol].disconnect()
                del self.websocket_connections[symbol]

            # Create new connection
            websocket = BinanceWebSocket(config)
            await websocket.connect()

            # Re-subscribe to streams
            await websocket.subscribe_orderbook(symbol, self._handle_orderbook_update)
            await websocket.subscribe_trades(symbol, self._handle_trade_update)

            # Store new connection
            self.websocket_connections[symbol] = websocket

            self.logger.info(f"Successfully reconnected WebSocket for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to reconnect WebSocket for {symbol}: {e}")
            return False