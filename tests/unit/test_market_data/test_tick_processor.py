# tests/unit/test_market_data/test_tick_processor.py

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from collections import deque

from src.market_data.models import TickData, TickType, OrderSide, MicrostructurePatterns
from src.market_data.tick_processor import TickDataAnalyzer


class TestTickDataAnalyzer:
    """Test suite for TickDataAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        return TickDataAnalyzer(buffer_size=100)

    @pytest.fixture
    def sample_trade_tick(self):
        """Create a sample trade tick"""
        return TickData(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            tick_type=TickType.TRADE,
            price=Decimal('50000'),
            size=Decimal('1.5'),
            side=OrderSide.BUY,
            trade_id=12345
        )

    @pytest.fixture
    def sample_quote_tick(self):
        """Create a sample quote tick"""
        return TickData(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            tick_type=TickType.QUOTE,
            price=Decimal('50005'),
            size=Decimal('2.0')
        )

    def test_should_initialize_with_correct_parameters(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.buffer_size == 100
        assert analyzer.vpin_window == 50
        assert analyzer.pattern_detection_window == 100
        assert len(analyzer.tick_buffer) == 0
        assert analyzer.trade_flow_imbalance == 0.0
        assert analyzer.last_vpin_score == 0.5

    def test_should_process_trade_tick_correctly(self, analyzer, sample_trade_tick):
        """Test trade tick processing"""
        patterns = analyzer.process_tick(sample_trade_tick)

        assert len(analyzer.tick_buffer) == 1
        assert len(analyzer.trade_buffer) == 1
        assert analyzer.tick_buffer[0] == sample_trade_tick

    def test_should_process_quote_tick_correctly(self, analyzer, sample_quote_tick):
        """Test quote tick processing"""
        patterns = analyzer.process_tick(sample_quote_tick)

        assert len(analyzer.tick_buffer) == 1
        assert len(analyzer.quote_buffer) == 1
        assert analyzer.tick_buffer[0] == sample_quote_tick

    def test_should_update_trade_flow_imbalance(self, analyzer):
        """Test trade flow imbalance calculation"""
        # Add first trade
        tick1 = TickData(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            tick_type=TickType.TRADE,
            price=Decimal('50000'),
            size=Decimal('1.0'),
            side=OrderSide.BUY
        )

        # Add second trade with higher price (buy pressure)
        tick2 = TickData(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            tick_type=TickType.TRADE,
            price=Decimal('50005'),  # Higher price
            size=Decimal('1.0'),
            side=OrderSide.BUY
        )

        analyzer.process_tick(tick1)
        analyzer.process_tick(tick2)

        # Should have positive imbalance (buy pressure)
        assert analyzer.trade_flow_imbalance > 0

    def test_should_calculate_vpin_correctly(self, analyzer):
        """Test VPIN calculation"""
        # Add trades with known sides
        for i in range(10):
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal('50000'),
                size=Decimal('1.0'),
                side=OrderSide.BUY if i < 7 else OrderSide.SELL  # 70% buy, 30% sell
            )
            analyzer.process_tick(tick)

        vpin = analyzer.calculate_vpin()

        # Should indicate imbalanced trading (high VPIN)
        assert 0 <= vpin <= 1
        assert vpin > 0.3  # Should be elevated due to imbalance

    def test_should_detect_quote_stuffing(self, analyzer):
        """Test quote stuffing detection"""
        # Add many quote ticks rapidly to trigger quote stuffing detection
        current_time = datetime.utcnow()

        # Add 150 quote ticks with minimal spacing (very high quote rate)
        for i in range(150):
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=current_time + timedelta(milliseconds=i*3),  # 3ms spacing = 333 quotes/sec
                tick_type=TickType.QUOTE,
                price=Decimal('50000'),
                size=Decimal('1.0')
            )
            analyzer.process_tick(tick)

        patterns = analyzer.detect_microstructure_patterns()
        assert patterns.quote_stuffing is True
        assert patterns.quote_rate > 100

    def test_should_detect_layering_pattern(self, analyzer):
        """Test layering pattern detection"""
        # First add enough ticks to satisfy pattern detection window
        for i in range(90):
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal('50000')
            )
            analyzer.process_tick(tick)

        # Then simulate order placement followed by rapid cancellations
        for i in range(15):
            if i < 5:
                tick_type = TickType.ORDER
            else:
                tick_type = TickType.CANCEL

            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=tick_type,
                price=Decimal('50000'),
                size=Decimal('1.0')
            )
            analyzer.process_tick(tick)

        patterns = analyzer.detect_microstructure_patterns()
        assert patterns.layering is True

    def test_should_detect_momentum_ignition(self, analyzer):
        """Test momentum ignition detection"""
        # First add enough ticks to satisfy pattern detection window
        for i in range(95):
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal('50000')
            )
            analyzer.process_tick(tick)

        base_price = 50000
        # Create series of trades in same direction with significant price movement
        for i in range(5):
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal(str(base_price + i * 50)),  # Increasing price
                size=Decimal('1.0'),
                side=OrderSide.BUY
            )
            analyzer.process_tick(tick)

        patterns = analyzer.detect_microstructure_patterns()
        assert patterns.momentum_ignition is True

    def test_should_detect_ping_pong_trading(self, analyzer):
        """Test ping-pong trading detection"""
        # First add enough ticks to satisfy pattern detection window
        for i in range(94):
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal('50000')
            )
            analyzer.process_tick(tick)

        # Create alternating buy/sell pattern
        for i in range(6):
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal('50000'),
                size=Decimal('1.0'),
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            )
            analyzer.process_tick(tick)

        patterns = analyzer.detect_microstructure_patterns()
        assert patterns.ping_pong is True

    def test_should_assess_pattern_confidence_correctly(self, analyzer):
        """Test pattern confidence assessment"""
        # Create pattern with multiple indicators
        patterns = MicrostructurePatterns(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            quote_stuffing=True,
            layering=True,
            momentum_ignition=False,
            ping_pong=False,
            vpin_score=0.8  # High VPIN
        )

        confidence, alert_level = analyzer._assess_pattern_confidence(patterns)

        assert 0 <= confidence <= 1
        assert alert_level in ["NONE", "LOW", "MEDIUM", "HIGH"]
        # With 2 patterns and high VPIN, should have elevated confidence
        assert confidence > 0.5

    def test_should_generate_meaningful_descriptions(self, analyzer):
        """Test pattern description generation"""
        patterns = MicrostructurePatterns(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            quote_stuffing=True,
            layering=False,
            vpin_score=0.7
        )

        description = analyzer._generate_pattern_description(patterns)

        assert "quote stuffing" in description
        assert "VPIN" in description
        assert isinstance(description, str)

    def test_should_maintain_buffer_limits(self, analyzer):
        """Test that buffers respect size limits"""
        # Add more ticks than buffer size
        for i in range(150):  # Exceed buffer size
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal('50000'),
                size=Decimal('1.0')
            )
            analyzer.process_tick(tick)

        assert len(analyzer.tick_buffer) <= analyzer.buffer_size
        assert len(analyzer.trade_buffer) <= analyzer.buffer_size

    def test_should_return_real_time_metrics(self, analyzer, sample_trade_tick):
        """Test real-time metrics retrieval"""
        analyzer.process_tick(sample_trade_tick)

        metrics = analyzer.get_real_time_metrics()

        assert 'trade_flow_imbalance' in metrics
        assert 'vpin_score' in metrics
        assert 'quote_rate' in metrics
        assert 'buffer_usage' in metrics
        assert 'processing_stats' in metrics

        # Check buffer usage
        buffer_usage = metrics['buffer_usage']
        assert buffer_usage['tick_buffer'] == 1
        assert buffer_usage['trade_buffer'] == 1

    def test_should_track_quote_rate_over_time(self, analyzer):
        """Test quote rate tracking with time decay"""
        current_time = datetime.utcnow()

        # Add recent quotes
        for i in range(10):
            analyzer.quote_rate_tracker.append(current_time + timedelta(seconds=i))

        # Add old quotes (should be cleaned up)
        old_time = current_time - timedelta(seconds=120)
        analyzer.quote_rate_tracker.appendleft(old_time)

        # Update quote rate (triggers cleanup)
        quote_tick = TickData(
            symbol="BTCUSDT",
            timestamp=current_time,
            tick_type=TickType.QUOTE,
            price=Decimal('50000')
        )
        analyzer._update_quote_rate(quote_tick)

        # Old quotes should be removed
        assert all(qt >= current_time - timedelta(seconds=60) for qt in analyzer.quote_rate_tracker)

    def test_should_calculate_order_cancel_ratio(self, analyzer):
        """Test order/cancel ratio calculation"""
        # Add orders and cancels
        for i in range(10):
            if i < 6:
                tick_type = TickType.ORDER
            else:
                tick_type = TickType.CANCEL

            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=tick_type
            )
            analyzer.process_tick(tick)

        ratio = analyzer._calculate_order_cancel_ratio()
        expected_ratio = 4 / 6  # 4 cancels, 6 orders
        assert abs(ratio - expected_ratio) < 0.01

    def test_should_calculate_price_momentum(self, analyzer):
        """Test price momentum calculation"""
        base_price = 50000

        # Create upward price trend
        for i in range(5):
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal(str(base_price + i * 20))  # +20 each time
            )
            analyzer.process_tick(tick)

        momentum = analyzer._calculate_price_momentum()
        assert momentum > 0  # Should be positive (upward trend)

    def test_should_handle_empty_buffers_gracefully(self, analyzer):
        """Test handling of empty buffers"""
        # Should not crash with empty buffers
        vpin = analyzer.calculate_vpin()
        assert vpin == 0.5  # Default value

        patterns = analyzer.detect_microstructure_patterns()
        assert isinstance(patterns, MicrostructurePatterns)

        metrics = analyzer.get_real_time_metrics()
        assert isinstance(metrics, dict)

    def test_should_reset_buffers_correctly(self, analyzer, sample_trade_tick):
        """Test buffer reset functionality"""
        # Add some data
        analyzer.process_tick(sample_trade_tick)
        analyzer.trade_flow_imbalance = 0.5
        analyzer.processed_ticks = 100

        # Reset
        analyzer.reset_buffers()

        assert len(analyzer.tick_buffer) == 0
        assert len(analyzer.trade_buffer) == 0
        assert len(analyzer.quote_buffer) == 0
        assert analyzer.trade_flow_imbalance == 0.0
        assert analyzer.processed_ticks == 0

    def test_should_return_pattern_history(self, analyzer):
        """Test pattern history retrieval"""
        # Process some ticks to generate patterns
        for i in range(15):
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE if i % 2 == 0 else TickType.QUOTE,
                price=Decimal('50000')
            )
            analyzer.process_tick(tick)

        history = analyzer.get_pattern_history()
        assert isinstance(history, list)
        assert len(history) <= 10  # Should respect limit

    def test_should_use_tick_rule_when_side_unavailable(self, analyzer):
        """Test tick rule application when trade side is not provided"""
        # First trade
        tick1 = TickData(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            tick_type=TickType.TRADE,
            price=Decimal('50000'),
            size=Decimal('1.0'),
            side=None  # No side information
        )

        # Second trade with higher price
        tick2 = TickData(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            tick_type=TickType.TRADE,
            price=Decimal('50005'),  # Higher price -> buy
            size=Decimal('1.0'),
            side=None  # No side information
        )

        analyzer.process_tick(tick1)
        analyzer.process_tick(tick2)

        # Should still calculate VPIN using tick rule
        vpin = analyzer.calculate_vpin()
        assert 0 <= vpin <= 1

    def test_should_handle_identical_prices_in_tick_rule(self, analyzer):
        """Test tick rule with identical consecutive prices"""
        # Two trades with same price
        for i in range(2):
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=Decimal('50000'),  # Same price
                size=Decimal('1.0'),
                side=None
            )
            analyzer.process_tick(tick)

        # Should handle gracefully without errors
        vpin = analyzer.calculate_vpin()
        assert 0 <= vpin <= 1

    def test_should_calculate_vpin_with_mixed_trade_types(self, analyzer):
        """Test VPIN calculation with mixed trade identification methods"""
        # Mix of trades with explicit sides and tick rule
        trades = [
            (Decimal('50000'), OrderSide.BUY),
            (Decimal('50005'), None),  # Higher price -> buy via tick rule
            (Decimal('50003'), OrderSide.SELL),
            (Decimal('49998'), None),  # Lower price -> sell via tick rule
        ]

        for price, side in trades:
            tick = TickData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                tick_type=TickType.TRADE,
                price=price,
                size=Decimal('1.0'),
                side=side
            )
            analyzer.process_tick(tick)

        vpin = analyzer.calculate_vpin()
        assert 0 <= vpin <= 1