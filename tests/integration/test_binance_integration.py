# tests/integration/test_binance_integration.py
"""
Integration tests for Binance API and execution engine.
Tests the complete flow from order creation to execution.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.api.base import ExchangeConfig
from src.execution.models import Order, OrderSide, OrderUrgency
from src.api.binance.executor import BinanceExecutor


class TestBinanceIntegration:
    """Integration tests for Binance API and execution engine"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        return ExchangeConfig(
            api_key="test_api_key_123456",
            api_secret="test_api_secret_123456",
            testnet=True,
            timeout=30,
            rate_limit_per_minute=1200
        )

    @pytest.fixture
    def executor(self, config):
        """Test executor instance"""
        return BinanceExecutor(config)

    @pytest.mark.asyncio
    async def test_should_connect_all_components(self, executor):
        """Should connect REST API, WebSocket, and execution components"""
        with patch.object(executor.client, 'connect') as mock_client_connect, \
             patch.object(executor.websocket, 'connect') as mock_ws_connect, \
             patch.object(executor.client, 'is_connected', return_value=True), \
             patch.object(executor.websocket, 'is_connected', return_value=True):

            mock_client_connect.return_value = None
            mock_ws_connect.return_value = None

            await executor.connect()

            assert executor.is_connected() is True
            mock_client_connect.assert_called_once()
            mock_ws_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_submit_order_through_complete_flow(self, executor):
        """Should submit order through complete execution flow"""
        # Mock all dependencies
        executor._connected = True
        executor.client._connected = True
        executor.websocket._connected = True

        # Mock connection check methods
        executor.client.is_connected = MagicMock(return_value=True)
        executor.websocket.is_connected = MagicMock(return_value=True)

        # Mock order manager
        mock_order_id = "internal_order_123"
        executor.order_manager.submit_order = AsyncMock(return_value=mock_order_id)
        executor.order_manager.update_order_status = AsyncMock()

        # Mock Binance client response
        binance_response = {
            "orderId": 987654321,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "executedQty": "1.5",
            "avgPrice": "50000.0"
        }
        executor.client.submit_order = AsyncMock(return_value=binance_response)

        # Create test order
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("1.5"),
            urgency=OrderUrgency.MEDIUM,
            price=Decimal("50000.0")
        )

        # Execute
        result = await executor.submit_order(order)

        # Verify execution result
        assert result.order_id == "987654321"
        assert result.strategy == "BINANCE_API"
        assert result.total_filled == Decimal("1.5")
        assert result.avg_price == Decimal("50000.0")
        assert result.total_cost == Decimal("75000.0")

        # Verify order manager interactions
        executor.order_manager.submit_order.assert_called_once_with(order)
        executor.order_manager.update_order_status.assert_called_once_with(
            mock_order_id, Decimal("1.5"), Decimal("50000.0")
        )

        # Verify Binance client interaction
        executor.client.submit_order.assert_called_once_with(order)

    @pytest.mark.asyncio
    async def test_should_handle_partial_fill_correctly(self, executor):
        """Should handle partial fills correctly"""
        # Setup
        executor._connected = True
        executor.client._connected = True
        executor.websocket._connected = True

        # Mock connection check methods
        executor.client.is_connected = MagicMock(return_value=True)
        executor.websocket.is_connected = MagicMock(return_value=True)

        executor.order_manager.submit_order = AsyncMock(return_value="order_123")
        executor.order_manager.update_order_status = AsyncMock()

        # Partial fill response
        binance_response = {
            "orderId": 987654321,
            "symbol": "ETHUSDT",
            "status": "PARTIALLY_FILLED",
            "executedQty": "5.0",
            "avgPrice": "3000.0"
        }
        executor.client.submit_order = AsyncMock(return_value=binance_response)

        order = Order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            size=Decimal("10.0"),
            price=Decimal("3000.0")
        )

        result = await executor.submit_order(order)

        # Should reflect partial fill
        assert result.total_filled == Decimal("5.0")
        assert result.original_size == Decimal("10.0")
        assert result.fill_rate == Decimal("0.5")  # 50% filled

    @pytest.mark.asyncio
    async def test_should_cancel_order_through_both_systems(self, executor):
        """Should cancel order through both Binance and order manager"""
        # Setup
        executor._connected = True
        executor.client._connected = True
        executor.websocket._connected = True

        # Mock connection check methods
        executor.client.is_connected = MagicMock(return_value=True)
        executor.websocket.is_connected = MagicMock(return_value=True)

        executor.client.cancel_order = AsyncMock(return_value=True)
        executor.order_manager.cancel_order = AsyncMock()

        order_id = "987654321"
        result = await executor.cancel_order(order_id)

        assert result is True
        executor.client.cancel_order.assert_called_once_with(order_id)
        executor.order_manager.cancel_order.assert_called_once_with(order_id)

    @pytest.mark.asyncio
    async def test_should_subscribe_to_market_data(self, executor):
        """Should subscribe to market data streams"""
        # Setup
        executor._connected = True
        executor.client._connected = True
        executor.websocket._connected = True

        # Mock connection check methods
        executor.client.is_connected = MagicMock(return_value=True)
        executor.websocket.is_connected = MagicMock(return_value=True)

        executor.websocket.subscribe_orderbook = AsyncMock()
        executor.websocket.subscribe_trades = AsyncMock()

        symbol = "BTCUSDT"
        await executor.subscribe_market_data(symbol)

        # Should subscribe to both orderbook and trades
        executor.websocket.subscribe_orderbook.assert_called_once()
        executor.websocket.subscribe_trades.assert_called_once()

        # Should track subscribed symbols
        assert symbol in executor._subscribed_symbols

    @pytest.mark.asyncio
    async def test_should_handle_orderbook_updates(self, executor):
        """Should handle orderbook updates and call callbacks"""
        # Setup callback
        callback_called = False
        callback_data = None

        async def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data

        executor.add_orderbook_callback("BTCUSDT", test_callback)
        executor.market_analyzer.update_orderbook = AsyncMock()

        # Simulate orderbook update
        orderbook_data = {
            "symbol": "BTCUSDT",
            "bids": [[Decimal("50000"), Decimal("1.0")]],
            "asks": [[Decimal("50001"), Decimal("0.5")]]
        }

        await executor._handle_orderbook_update(orderbook_data)

        # Should update market analyzer
        executor.market_analyzer.update_orderbook.assert_called_once_with(
            "BTCUSDT", orderbook_data
        )

        # Should call registered callback
        assert callback_called is True
        assert callback_data == orderbook_data

    @pytest.mark.asyncio
    async def test_should_handle_trade_updates(self, executor):
        """Should handle trade updates and call callbacks"""
        # Setup callback
        trade_received = None

        def trade_callback(data):
            nonlocal trade_received
            trade_received = data

        executor.add_trade_callback("ETHUSDT", trade_callback)
        executor.market_analyzer.update_trades = AsyncMock()

        # Simulate trade update
        trade_data = {
            "symbol": "ETHUSDT",
            "price": "3000.0",
            "quantity": "2.5",
            "trade_time": 1634567890000
        }

        await executor._handle_trade_update(trade_data)

        # Should update market analyzer
        executor.market_analyzer.update_trades.assert_called_once_with(
            "ETHUSDT", trade_data
        )

        # Should call registered callback
        assert trade_received == trade_data

    @pytest.mark.asyncio
    async def test_should_get_account_information(self, executor):
        """Should retrieve account balance and positions"""
        # Setup
        executor._connected = True
        executor.client._connected = True
        executor.websocket._connected = True

        # Mock connection check methods
        executor.client.is_connected = MagicMock(return_value=True)
        executor.websocket.is_connected = MagicMock(return_value=True)

        mock_balance = {"USDT": Decimal("10000.0")}
        mock_positions = [{"symbol": "BTCUSDT", "size": "1.5"}]

        executor.client.get_account_balance = AsyncMock(return_value=mock_balance)
        executor.client.get_positions = AsyncMock(return_value=mock_positions)

        result = await executor.get_account_info()

        assert result["balance"] == mock_balance
        assert result["positions"] == mock_positions

    @pytest.mark.asyncio
    async def test_should_get_market_conditions(self, executor):
        """Should get and analyze market conditions"""
        # Setup
        executor._connected = True
        executor.client._connected = True
        executor.websocket._connected = True

        # Mock connection check methods
        executor.client.is_connected = MagicMock(return_value=True)
        executor.websocket.is_connected = MagicMock(return_value=True)

        mock_market_data = {"symbol": "BTCUSDT", "price": "50000.0"}
        mock_conditions = {"spread": 0.1, "liquidity": 0.8}

        executor.client.get_market_data = AsyncMock(return_value=mock_market_data)
        executor.market_analyzer.analyze_market_conditions = AsyncMock(return_value=mock_conditions)

        result = await executor.get_market_conditions("BTCUSDT")

        assert result["market_data"] == mock_market_data
        assert result["conditions"] == mock_conditions

    @pytest.mark.asyncio
    async def test_should_handle_api_errors_gracefully(self, executor):
        """Should handle API errors gracefully"""
        from src.api.binance.exceptions import BinanceAPIError, BinanceOrderError

        # Setup
        executor._connected = True
        executor.client._connected = True
        executor.websocket._connected = True

        # Mock connection check methods
        executor.client.is_connected = MagicMock(return_value=True)
        executor.websocket.is_connected = MagicMock(return_value=True)

        executor.order_manager.submit_order = AsyncMock(return_value="order_123")
        executor.client.submit_order = AsyncMock(
            side_effect=BinanceAPIError("Insufficient balance", -2019)
        )

        order = Order("BTCUSDT", OrderSide.BUY, Decimal("100.0"))

        with pytest.raises(BinanceOrderError) as exc_info:
            await executor.submit_order(order)

        assert "Insufficient balance" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_should_disconnect_all_components(self, executor):
        """Should disconnect all components properly"""
        # Setup connected state
        executor._connected = True
        executor._subscribed_symbols.add("BTCUSDT")
        executor.orderbook_callbacks["BTCUSDT"] = [lambda x: None]

        executor.websocket.disconnect = AsyncMock()
        executor.client.disconnect = AsyncMock()

        await executor.disconnect()

        # Should disconnect all components
        executor.websocket.disconnect.assert_called_once()
        executor.client.disconnect.assert_called_once()

        # Should reset state
        assert executor._connected is False
        assert len(executor._subscribed_symbols) == 0
        assert len(executor.orderbook_callbacks) == 0

    @pytest.mark.asyncio
    async def test_should_get_execution_statistics(self, executor):
        """Should get execution statistics from order manager"""
        mock_stats = {
            "total_orders": 10,
            "successful_orders": 8,
            "failed_orders": 2
        }

        executor.order_manager.get_order_statistics = MagicMock(return_value=mock_stats)

        result = await executor.get_execution_statistics()

        assert result == mock_stats
        executor.order_manager.get_order_statistics.assert_called_once()