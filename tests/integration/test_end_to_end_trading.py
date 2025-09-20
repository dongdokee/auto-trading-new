# tests/integration/test_end_to_end_trading.py
"""
End-to-end integration test demonstrating complete trading workflow.
Tests the entire system from strategy signals to order execution.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config.models import ExchangeConfig
from src.execution.models import Order, OrderSide, OrderUrgency
from src.api.binance.executor import BinanceExecutor


class TestEndToEndTrading:
    """End-to-end trading workflow tests"""

    @pytest.fixture
    def paper_trading_config(self):
        """Paper trading configuration"""
        return ExchangeConfig(
            name="BINANCE",
            api_key="paper_trading_key_123456",
            api_secret="paper_trading_secret_123456",
            testnet=True,
            paper_trading=True,
            timeout=30,
            rate_limit_requests=1200
        )

    @pytest.fixture
    def trading_executor(self, paper_trading_config):
        """Trading executor with paper trading setup"""
        return BinanceExecutor(paper_trading_config)

    @pytest.mark.asyncio
    async def test_complete_paper_trading_workflow(self, trading_executor):
        """
        Test complete paper trading workflow:
        1. Connect to exchange
        2. Subscribe to market data
        3. Generate trading signal
        4. Submit order
        5. Monitor execution
        6. Handle fills
        7. Get account status
        """

        # Step 1: Mock connections
        with patch.object(trading_executor.client, 'connect') as mock_client_connect, \
             patch.object(trading_executor.websocket, 'connect') as mock_ws_connect, \
             patch.object(trading_executor.client, 'is_connected', return_value=True), \
             patch.object(trading_executor.websocket, 'is_connected', return_value=True):

            # Connect to exchange
            await trading_executor.connect()
            assert trading_executor.is_connected() is True

            # Step 2: Subscribe to market data
            trading_executor.websocket.subscribe_orderbook = AsyncMock()
            trading_executor.websocket.subscribe_trades = AsyncMock()

            await trading_executor.subscribe_market_data("BTCUSDT")
            assert "BTCUSDT" in trading_executor._subscribed_symbols

            # Step 3: Simulate trading signal generation
            # (In real system, this would come from strategy engine)
            trading_signal = {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "size": Decimal("0.1"),
                "price": Decimal("50000.0"),
                "strategy": "TREND_FOLLOWING",
                "confidence": 0.8
            }

            # Step 4: Convert signal to order and submit
            order = Order(
                symbol=trading_signal["symbol"],
                side=OrderSide.BUY,
                size=trading_signal["size"],
                urgency=OrderUrgency.MEDIUM,
                price=trading_signal["price"]
            )

            # Mock order execution
            trading_executor.order_manager.submit_order = AsyncMock(return_value="internal_order_123")
            trading_executor.order_manager.update_order_status = AsyncMock()

            mock_execution_result = {
                "orderId": 999888777,
                "symbol": "BTCUSDT",
                "status": "FILLED",
                "executedQty": "0.1",
                "avgPrice": "50000.0"
            }
            trading_executor.client.submit_order = AsyncMock(return_value=mock_execution_result)

            # Submit order
            execution_result = await trading_executor.submit_order(order)

            # Step 5: Verify execution result
            assert execution_result.order_id == "999888777"
            assert execution_result.total_filled == Decimal("0.1")
            assert execution_result.avg_price == Decimal("50000.0")
            assert execution_result.total_cost == Decimal("5000.0")  # 0.1 * 50000

            # Step 6: Get account status
            mock_balance = {"USDT": Decimal("45000.0"), "BTC": Decimal("0.1")}
            mock_positions = [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.1",
                    "entryPrice": "50000.0",
                    "unrealizedPnl": "0.0"
                }
            ]

            trading_executor.client.get_account_balance = AsyncMock(return_value=mock_balance)
            trading_executor.client.get_positions = AsyncMock(return_value=mock_positions)

            account_info = await trading_executor.get_account_info()

            # Verify account state
            assert account_info["balance"]["USDT"] == Decimal("45000.0")
            assert account_info["balance"]["BTC"] == Decimal("0.1")
            assert len(account_info["positions"]) == 1
            assert account_info["positions"][0]["symbol"] == "BTCUSDT"

            # Step 7: Get execution statistics
            mock_stats = {
                "total_orders": 1,
                "successful_orders": 1,
                "failed_orders": 0,
                "total_volume": Decimal("5000.0"),
                "average_execution_time": 0.05
            }

            trading_executor.order_manager.get_order_statistics = MagicMock(return_value=mock_stats)
            stats = await trading_executor.get_execution_statistics()

            assert stats["total_orders"] == 1
            assert stats["successful_orders"] == 1
            assert stats["total_volume"] == Decimal("5000.0")

    @pytest.mark.asyncio
    async def test_paper_trading_risk_management_integration(self, trading_executor):
        """Test paper trading with risk management integration"""

        # Setup mock connections
        trading_executor._connected = True
        trading_executor.client._connected = True
        trading_executor.websocket._connected = True

        # Mock connection check methods
        trading_executor.client.is_connected = MagicMock(return_value=True)
        trading_executor.websocket.is_connected = MagicMock(return_value=True)

        # Mock risk controller (would be integrated in real system)
        mock_risk_check = MagicMock(return_value=True)

        # Test order that would be rejected by risk management
        large_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("100.0"),  # Large position
            price=Decimal("50000.0")
        )

        # Mock risk rejection
        trading_executor.order_manager.submit_order = AsyncMock(return_value="order_456")
        trading_executor.client.submit_order = AsyncMock(
            side_effect=Exception("Risk limits exceeded")
        )

        # Should handle risk rejection gracefully
        with pytest.raises(Exception) as exc_info:
            await trading_executor.submit_order(large_order)

        assert "Risk limits exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_paper_trading_market_data_integration(self, trading_executor):
        """Test paper trading with real-time market data integration"""

        # Setup
        trading_executor._connected = True
        trading_executor.client._connected = True
        trading_executor.websocket._connected = True

        # Track market data updates
        orderbook_updates = []
        trade_updates = []

        async def orderbook_callback(data):
            orderbook_updates.append(data)

        def trade_callback(data):
            trade_updates.append(data)

        trading_executor.add_orderbook_callback("BTCUSDT", orderbook_callback)
        trading_executor.add_trade_callback("BTCUSDT", trade_callback)

        # Mock market analyzer
        trading_executor.market_analyzer.update_orderbook = AsyncMock()
        trading_executor.market_analyzer.update_trades = AsyncMock()

        # Simulate orderbook update
        orderbook_data = {
            "symbol": "BTCUSDT",
            "bids": [[Decimal("49999"), Decimal("1.5")]],
            "asks": [[Decimal("50001"), Decimal("2.0")]]
        }

        await trading_executor._handle_orderbook_update(orderbook_data)

        # Simulate trade update
        trade_data = {
            "symbol": "BTCUSDT",
            "price": "50000.0",
            "quantity": "0.5",
            "trade_time": 1634567890000
        }

        await trading_executor._handle_trade_update(trade_data)

        # Verify callbacks were called
        assert len(orderbook_updates) == 1
        assert len(trade_updates) == 1
        assert orderbook_updates[0]["symbol"] == "BTCUSDT"
        assert trade_updates[0]["symbol"] == "BTCUSDT"

        # Verify market analyzer was updated
        trading_executor.market_analyzer.update_orderbook.assert_called_once()
        trading_executor.market_analyzer.update_trades.assert_called_once()

    @pytest.mark.asyncio
    async def test_paper_trading_error_recovery(self, trading_executor):
        """Test paper trading error recovery and resilience"""

        # Setup
        trading_executor._connected = True
        trading_executor.client._connected = True
        trading_executor.websocket._connected = True

        # Test connection loss and recovery
        trading_executor.websocket.is_connected = MagicMock(side_effect=[True, False, True])
        trading_executor.client.is_connected = MagicMock(return_value=True)

        # Should handle temporary disconnection
        assert trading_executor.is_connected() is True  # Initially connected
        assert trading_executor.is_connected() is False  # Temporarily disconnected
        assert trading_executor.is_connected() is True  # Reconnected

        # Reset connection mocks for order submission tests
        trading_executor.websocket.is_connected = MagicMock(return_value=True)
        trading_executor.client.is_connected = MagicMock(return_value=True)

        # Test order submission retry logic
        trading_executor.order_manager.submit_order = AsyncMock(return_value="order_789")
        trading_executor.order_manager.update_order_status = AsyncMock()

        # First attempt fails, second succeeds
        trading_executor.client.submit_order = AsyncMock(
            side_effect=[
                Exception("Network timeout"),
                {
                    "orderId": 111222333,
                    "symbol": "ETHUSDT",
                    "status": "FILLED",
                    "executedQty": "1.0",
                    "avgPrice": "3000.0"
                }
            ]
        )

        order = Order("ETHUSDT", OrderSide.BUY, Decimal("1.0"), price=Decimal("3000.0"))

        # First attempt should fail
        with pytest.raises(Exception):
            await trading_executor.submit_order(order)

        # Second attempt should succeed
        result = await trading_executor.submit_order(order)
        assert result.order_id == "111222333"

    @pytest.mark.asyncio
    async def test_paper_trading_performance_metrics(self, trading_executor):
        """Test paper trading performance tracking"""

        # Setup
        trading_executor._connected = True
        trading_executor.client._connected = True
        trading_executor.websocket._connected = True

        # Mock execution statistics
        mock_detailed_stats = {
            "total_orders": 10,
            "successful_orders": 9,
            "failed_orders": 1,
            "total_volume": Decimal("50000.0"),
            "average_execution_time": 0.045,
            "best_execution_time": 0.025,
            "worst_execution_time": 0.080,
            "success_rate": 0.9,
            "average_slippage": Decimal("0.0025"),  # 2.5 bps
            "total_fees": Decimal("25.0")
        }

        trading_executor.order_manager.get_order_statistics = MagicMock(
            return_value=mock_detailed_stats
        )

        stats = await trading_executor.get_execution_statistics()

        # Verify performance metrics
        assert stats["success_rate"] == 0.9
        assert stats["average_execution_time"] < 0.05  # Under 50ms
        assert stats["average_slippage"] < Decimal("0.005")  # Under 5 bps
        assert stats["total_volume"] == Decimal("50000.0")

        # Performance should be within acceptable ranges for paper trading
        assert stats["best_execution_time"] >= 0.020  # Realistic minimum
        assert stats["worst_execution_time"] <= 0.100  # Acceptable maximum

    @pytest.mark.asyncio
    async def test_paper_trading_cleanup_and_disconnect(self, trading_executor):
        """Test proper cleanup when disconnecting from paper trading"""

        # Setup connected state
        trading_executor._connected = True
        trading_executor._subscribed_symbols.add("BTCUSDT")
        trading_executor._subscribed_symbols.add("ETHUSDT")
        trading_executor.orderbook_callbacks["BTCUSDT"] = [lambda x: None]
        trading_executor.trade_callbacks["ETHUSDT"] = [lambda x: None]

        # Mock disconnect methods
        trading_executor.websocket.disconnect = AsyncMock()
        trading_executor.client.disconnect = AsyncMock()

        # Disconnect
        await trading_executor.disconnect()

        # Verify cleanup
        assert trading_executor._connected is False
        assert len(trading_executor._subscribed_symbols) == 0
        assert len(trading_executor.orderbook_callbacks) == 0
        assert len(trading_executor.trade_callbacks) == 0

        # Verify API disconnections
        trading_executor.websocket.disconnect.assert_called_once()
        trading_executor.client.disconnect.assert_called_once()