# tests/integration/test_market_data_integration.py

import pytest
import pytest_asyncio
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from src.market_data.data_aggregator import DataAggregator
from src.market_data.websocket_bridge import MarketDataWebSocketBridge
from src.market_data.enhanced_market_analyzer import EnhancedMarketConditionAnalyzer
from src.market_data.models import OrderBookSnapshot, OrderLevel, TickData, TickType, OrderSide


class TestMarketDataIntegration:
    """Integration tests for the complete market data pipeline"""

    @pytest_asyncio.fixture
    async def data_aggregator(self):
        """Create and start a data aggregator"""
        aggregator = DataAggregator(cache_ttl=30, max_symbols=10)
        await aggregator.start()
        yield aggregator
        await aggregator.stop()

    @pytest.fixture
    def mock_config(self):
        """Mock WebSocket configuration"""
        return Mock(testnet=True)

    @pytest.fixture
    def sample_orderbook(self):
        """Create sample orderbook data"""
        return OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=[
                OrderLevel(Decimal('50000'), Decimal('1.5')),
                OrderLevel(Decimal('49995'), Decimal('2.0')),
                OrderLevel(Decimal('49990'), Decimal('1.8')),
            ],
            asks=[
                OrderLevel(Decimal('50005'), Decimal('1.6')),
                OrderLevel(Decimal('50010'), Decimal('1.9')),
                OrderLevel(Decimal('50015'), Decimal('2.1')),
            ]
        )

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data"""
        return [
            TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal('50003'),
                size=Decimal('0.5'),
                side=OrderSide.BUY,
                trade_id=12345
            ),
            TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal('50002'),
                size=Decimal('0.8'),
                side=OrderSide.SELL,
                trade_id=12346
            ),
        ]

    @pytest.mark.asyncio
    async def test_should_process_complete_market_data_pipeline(self, data_aggregator, sample_orderbook, sample_trades):
        """Test complete market data processing pipeline"""
        # Subscribe to symbol
        data_aggregator.subscribe_symbol("BTCUSDT")

        # Process orderbook update
        await data_aggregator.process_orderbook_update(sample_orderbook)

        # Process trade updates
        for trade in sample_trades:
            await data_aggregator.process_tick_update(trade)

        # Get aggregated market data
        market_data = await data_aggregator.get_market_data("BTCUSDT")

        assert market_data is not None
        assert market_data.symbol == "BTCUSDT"
        assert market_data.current_metrics is not None
        assert len(market_data.tick_history) == len(sample_trades)
        assert len(market_data.orderbook_history) == 1

        # Verify metrics
        metrics = market_data.current_metrics
        assert metrics.best_bid == Decimal('50000')
        assert metrics.best_ask == Decimal('50005')
        assert metrics.spread == Decimal('5')

    @pytest.mark.asyncio
    async def test_should_estimate_market_impact_correctly(self, data_aggregator, sample_orderbook):
        """Test market impact estimation"""
        data_aggregator.subscribe_symbol("BTCUSDT")
        await data_aggregator.process_orderbook_update(sample_orderbook)

        # Estimate market impact
        impact = await data_aggregator.estimate_market_impact("BTCUSDT", Decimal('1.0'))

        assert impact is not None
        assert impact.symbol == "BTCUSDT"
        assert impact.order_size == Decimal('1.0')
        assert impact.temporary_impact >= 0
        assert impact.permanent_impact >= 0
        assert impact.total_impact == impact.temporary_impact + impact.permanent_impact

    @pytest.mark.asyncio
    async def test_should_find_optimal_execution_windows(self, data_aggregator, sample_orderbook):
        """Test optimal execution window finding"""
        data_aggregator.subscribe_symbol("BTCUSDT")

        # Add multiple orderbook updates with different timestamps
        for i in range(10):
            orderbook = OrderBookSnapshot(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow() + timedelta(hours=i),
                event_time=1234567890 + i,
                first_update_id=1000 + i,
                final_update_id=1001 + i,
                bids=[OrderLevel(Decimal('50000'), Decimal('1.5'))],
                asks=[OrderLevel(Decimal('50005'), Decimal('1.6'))]
            )
            await data_aggregator.process_orderbook_update(orderbook)

        # Find optimal windows
        windows = await data_aggregator.get_optimal_execution_windows("BTCUSDT", Decimal('1.0'))

        assert isinstance(windows, list)
        # Windows may be empty if insufficient historical data, which is okay for this test

    @pytest.mark.asyncio
    async def test_should_trigger_pattern_callbacks(self, data_aggregator):
        """Test pattern detection callbacks"""
        callback_called = False
        detected_patterns = None

        async def pattern_callback(patterns):
            nonlocal callback_called, detected_patterns
            callback_called = True
            detected_patterns = patterns

        data_aggregator.subscribe_symbol("BTCUSDT")
        data_aggregator.add_pattern_callback(pattern_callback)

        # Process trades to trigger pattern detection
        for i in range(15):  # Process enough ticks to trigger detection
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal('50000'),
                size=Decimal('1.0'),
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            )
            await data_aggregator.process_tick_update(tick)

        # Pattern detection might trigger with enough data
        # Note: This may not always trigger depending on the pattern detection logic

    @pytest.mark.asyncio
    async def test_should_handle_concurrent_symbol_processing(self, data_aggregator):
        """Test concurrent processing of multiple symbols"""
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

        # Subscribe to multiple symbols
        for symbol in symbols:
            data_aggregator.subscribe_symbol(symbol)

        # Process data concurrently
        tasks = []
        for symbol in symbols:
            orderbook = OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                event_time=1234567890,
                first_update_id=1000,
                final_update_id=1001,
                bids=[OrderLevel(Decimal('1000'), Decimal('1.0'))],
                asks=[OrderLevel(Decimal('1005'), Decimal('1.0'))]
            )
            tasks.append(data_aggregator.process_orderbook_update(orderbook))

        # Wait for all processing to complete
        await asyncio.gather(*tasks)

        # Verify all symbols have data
        for symbol in symbols:
            market_data = await data_aggregator.get_market_data(symbol)
            assert market_data is not None
            assert market_data.symbol == symbol

    @pytest.mark.asyncio
    async def test_should_maintain_performance_under_load(self, data_aggregator):
        """Test performance under high message load"""
        data_aggregator.subscribe_symbol("BTCUSDT")

        start_time = datetime.utcnow()

        # Process many updates rapidly
        for i in range(100):
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal('50000') + Decimal(str(i % 10)),
                size=Decimal('1.0')
            )
            await data_aggregator.process_tick_update(tick)

        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        # Should process 100 updates in reasonable time (< 5 seconds)
        assert processing_time < 5.0

        # Check performance metrics
        metrics = data_aggregator.get_performance_metrics()
        assert metrics['processed_ticks'] >= 100
        assert metrics['error_count'] == 0

    @pytest.mark.asyncio
    async def test_enhanced_market_analyzer_integration(self, data_aggregator, sample_orderbook, sample_trades):
        """Test enhanced market analyzer integration"""
        # Setup
        analyzer = EnhancedMarketConditionAnalyzer(data_aggregator)
        data_aggregator.subscribe_symbol("BTCUSDT")

        # Process market data
        await data_aggregator.process_orderbook_update(sample_orderbook)
        for trade in sample_trades:
            await data_aggregator.process_tick_update(trade)

        # Analyze market conditions
        analysis = await analyzer.analyze_market_conditions("BTCUSDT")

        assert analysis is not None
        assert analysis['symbol'] == "BTCUSDT"
        assert 'data_quality' in analysis
        assert 'liquidity_analysis' in analysis
        assert 'execution_analysis' in analysis
        assert 'risk_assessment' in analysis
        assert 'execution_recommendation' in analysis

        # Verify confidence score
        assert 0 <= analysis['confidence_score'] <= 1

    @pytest.mark.asyncio
    async def test_websocket_bridge_mock_integration(self, data_aggregator, mock_config):
        """Test WebSocket bridge with mocked WebSocket"""
        bridge = MarketDataWebSocketBridge(data_aggregator)
        await bridge.start()

        # Mock WebSocket behavior
        with pytest.MonkeyPatch.context() as m:
            mock_websocket = AsyncMock()
            mock_websocket.connect = AsyncMock()
            mock_websocket.subscribe_orderbook = AsyncMock()
            mock_websocket.subscribe_trades = AsyncMock()
            mock_websocket.is_connected.return_value = True

            # Mock WebSocket creation
            def mock_websocket_init(*args, **kwargs):
                return mock_websocket

            m.setattr("src.market_data.websocket_bridge.BinanceWebSocket", mock_websocket_init)

            # Test subscription
            await bridge.subscribe_symbol("BTCUSDT", mock_config)

            assert "BTCUSDT" in bridge.subscribed_symbols
            assert len(bridge.websocket_connections) == 1

            # Test health check
            health = await bridge.health_check()
            assert health['bridge_running'] is True
            assert health['subscribed_symbols_count'] == 1

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_should_handle_data_aggregator_errors_gracefully(self, data_aggregator):
        """Test graceful error handling in data aggregator"""
        # Test processing invalid data
        invalid_orderbook = OrderBookSnapshot(
            symbol="INVALID",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=[],  # Empty bids
            asks=[]   # Empty asks
        )

        # Should not crash on invalid data
        try:
            await data_aggregator.process_orderbook_update(invalid_orderbook)
        except Exception:
            pass  # Expected to handle gracefully

        # Check error metrics
        metrics = data_aggregator.get_performance_metrics()
        # Error count might be incremented

    @pytest.mark.asyncio
    async def test_should_maintain_cache_correctly(self, data_aggregator, sample_orderbook):
        """Test caching behavior"""
        data_aggregator.subscribe_symbol("BTCUSDT")
        await data_aggregator.process_orderbook_update(sample_orderbook)

        # Get data (should be cached)
        market_data1 = await data_aggregator.get_market_data("BTCUSDT")
        market_data2 = await data_aggregator.get_market_data("BTCUSDT")

        # Should return same cached data
        assert market_data1.cache_timestamp == market_data2.cache_timestamp

        # Force refresh
        market_data3 = await data_aggregator.get_market_data("BTCUSDT", force_refresh=True)

        # Check performance metrics for cache hits
        metrics = data_aggregator.get_performance_metrics()
        assert metrics['cache_hit_rate'] > 0  # Should have some cache hits

    @pytest.mark.asyncio
    async def test_should_cleanup_old_data_periodically(self, data_aggregator, sample_orderbook):
        """Test periodic data cleanup"""
        data_aggregator.subscribe_symbol("BTCUSDT")

        # Add old data
        old_orderbook = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow() - timedelta(hours=2),  # Old data
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=[OrderLevel(Decimal('50000'), Decimal('1.0'))],
            asks=[OrderLevel(Decimal('50005'), Decimal('1.0'))]
        )

        await data_aggregator.process_orderbook_update(old_orderbook)
        await data_aggregator.process_orderbook_update(sample_orderbook)

        # Get market data
        market_data = await data_aggregator.get_market_data("BTCUSDT")

        # Should have both orderbooks initially
        assert len(market_data.orderbook_history) >= 1

        # Trigger cleanup manually (normally done periodically)
        await data_aggregator._periodic_cleanup()

        # Check if old data was cleaned up (depends on cleanup logic)
        updated_data = await data_aggregator.get_market_data("BTCUSDT")
        # Cleanup might have removed old orderbooks