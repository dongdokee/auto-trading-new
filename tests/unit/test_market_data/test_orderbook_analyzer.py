# tests/unit/test_market_data/test_orderbook_analyzer.py

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime

from src.market_data.models import OrderBookSnapshot, OrderLevel, BookShape
from src.market_data.orderbook_analyzer import OrderBookAnalyzer


class TestOrderBookAnalyzer:
    """Test suite for OrderBookAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        return OrderBookAnalyzer()

    @pytest.fixture
    def sample_orderbook(self):
        """Create a sample order book for testing"""
        bids = [
            OrderLevel(Decimal('50000'), Decimal('1.5')),
            OrderLevel(Decimal('49995'), Decimal('2.0')),
            OrderLevel(Decimal('49990'), Decimal('1.8')),
            OrderLevel(Decimal('49985'), Decimal('2.2')),
            OrderLevel(Decimal('49980'), Decimal('1.9'))
        ]
        asks = [
            OrderLevel(Decimal('50005'), Decimal('1.6')),
            OrderLevel(Decimal('50010'), Decimal('1.9')),
            OrderLevel(Decimal('50015'), Decimal('2.1')),
            OrderLevel(Decimal('50020'), Decimal('1.7')),
            OrderLevel(Decimal('50025'), Decimal('2.0'))
        ]

        return OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=bids,
            asks=asks
        )

    def test_should_analyze_basic_spread_metrics_correctly(self, analyzer, sample_orderbook):
        """Test basic spread calculation"""
        metrics = analyzer.analyze_orderbook(sample_orderbook)

        assert metrics.symbol == "BTCUSDT"
        assert metrics.best_bid == Decimal('50000')
        assert metrics.best_ask == Decimal('50005')
        assert metrics.mid_price == Decimal('50002.5')
        assert metrics.spread == Decimal('5')
        assert abs(metrics.spread_bps - 1.0) < 0.01  # ~1 bp

    def test_should_calculate_order_book_imbalance(self, analyzer, sample_orderbook):
        """Test order book imbalance calculation"""
        metrics = analyzer.analyze_orderbook(sample_orderbook)

        # Calculate expected imbalance
        bid_vol_5 = sum(level.size for level in sample_orderbook.bids[:5])
        ask_vol_5 = sum(level.size for level in sample_orderbook.asks[:5])
        total_vol = bid_vol_5 + ask_vol_5
        expected_imbalance = float((bid_vol_5 - ask_vol_5) / total_vol)

        assert metrics.bid_volume_5 == bid_vol_5
        assert metrics.ask_volume_5 == ask_vol_5
        assert abs(metrics.imbalance - expected_imbalance) < 0.001

    def test_should_calculate_liquidity_score(self, analyzer, sample_orderbook):
        """Test liquidity score calculation"""
        metrics = analyzer.analyze_orderbook(sample_orderbook)

        assert 0 <= metrics.liquidity_score <= 1
        assert metrics.liquidity_score > 0  # Should have some liquidity

    def test_should_detect_book_shape_correctly(self, analyzer):
        """Test book shape analysis"""
        # Create bid-heavy book
        bids = [OrderLevel(Decimal(f'{50000-i*5}'), Decimal(f'{1+i*0.5}')) for i in range(5)]
        asks = [OrderLevel(Decimal(f'{50005+i*5}'), Decimal('1.0')) for i in range(5)]

        orderbook = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=bids,
            asks=asks
        )

        metrics = analyzer.analyze_orderbook(orderbook)

        assert metrics.book_shape in [BookShape.BID_HEAVY, BookShape.ASK_HEAVY, BookShape.FLAT]
        assert metrics.bid_slope is not None
        assert metrics.ask_slope is not None

    def test_should_detect_large_orders(self, analyzer):
        """Test large order detection"""
        # Create orderbook with one large bid
        bids = [
            OrderLevel(Decimal('50000'), Decimal('10.0')),  # Large order
            OrderLevel(Decimal('49995'), Decimal('1.0')),
            OrderLevel(Decimal('49990'), Decimal('1.0')),
        ]
        asks = [
            OrderLevel(Decimal('50005'), Decimal('1.0')),
            OrderLevel(Decimal('50010'), Decimal('1.0')),
            OrderLevel(Decimal('50015'), Decimal('1.0')),
        ]

        orderbook = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=bids,
            asks=asks
        )

        metrics = analyzer.analyze_orderbook(orderbook)

        assert len(metrics.large_orders) > 0
        large_bid = next((order for order in metrics.large_orders if order['side'] == 'BID'), None)
        assert large_bid is not None
        assert large_bid['size'] == 10.0

    def test_should_create_price_impact_function(self, analyzer, sample_orderbook):
        """Test price impact function creation"""
        metrics = analyzer.analyze_orderbook(sample_orderbook)

        impact_function = metrics.price_impact_function
        assert callable(impact_function)

        # Test small order
        small_impact = impact_function(Decimal('0.1'), 'BUY')
        assert 0 <= small_impact <= 0.1

        # Test large order
        large_impact = impact_function(Decimal('10.0'), 'BUY')
        assert large_impact >= small_impact

    def test_should_handle_empty_orderbook_gracefully(self, analyzer):
        """Test handling of empty order book"""
        empty_orderbook = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=[],
            asks=[]
        )

        with pytest.raises(ValueError):
            analyzer.analyze_orderbook(empty_orderbook)

    def test_should_update_trade_history(self, analyzer):
        """Test trade history update"""
        trade_data = {
            'price': 50000,
            'size': 1.0,
            'side': 'BUY',
            'mid_price_at_execution': 50002.5
        }

        analyzer.update_trade_history(trade_data)
        assert len(analyzer._trade_history) == 1

    def test_should_calculate_bid_ask_pressure(self, analyzer, sample_orderbook):
        """Test bid/ask pressure calculation"""
        pressure = analyzer.get_bid_ask_pressure(sample_orderbook)

        assert 'bid_pressure' in pressure
        assert 'ask_pressure' in pressure
        assert 'pressure_ratio' in pressure
        assert 'total_volume' in pressure
        assert 'pressure_imbalance' in pressure

        assert 0 <= pressure['bid_pressure'] <= 1
        assert 0 <= pressure['ask_pressure'] <= 1
        assert abs(pressure['bid_pressure'] + pressure['ask_pressure'] - 1.0) < 0.001

    def test_should_calculate_book_stability(self, analyzer):
        """Test book stability calculation"""
        # Create multiple orderbooks with stable spreads
        orderbooks = []
        for i in range(5):
            bids = [OrderLevel(Decimal(f'{50000-j*5}'), Decimal('1.0')) for j in range(3)]
            asks = [OrderLevel(Decimal(f'{50005+j*5}'), Decimal('1.0')) for j in range(3)]

            orderbook = OrderBookSnapshot(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                event_time=1234567890 + i,
                first_update_id=1000 + i,
                final_update_id=1001 + i,
                bids=bids,
                asks=asks
            )
            orderbooks.append(orderbook)

        stability = analyzer.calculate_book_stability(orderbooks)
        assert 0 <= stability <= 1

    def test_should_handle_price_impact_for_insufficient_liquidity(self, analyzer):
        """Test price impact when order size exceeds available liquidity"""
        bids = [OrderLevel(Decimal('50000'), Decimal('0.1'))]
        asks = [OrderLevel(Decimal('50005'), Decimal('0.1'))]

        orderbook = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=bids,
            asks=asks
        )

        metrics = analyzer.analyze_orderbook(orderbook)
        impact_function = metrics.price_impact_function

        # Order larger than available liquidity
        large_order_impact = impact_function(Decimal('1.0'), 'BUY')
        assert large_order_impact == 0.05  # Should return penalty

    def test_should_handle_identical_price_levels_in_slope_calculation(self, analyzer):
        """Test slope calculation with identical prices (edge case)"""
        # Create orderbook with identical prices (shouldn't happen in reality)
        bids = [OrderLevel(Decimal('50000'), Decimal(f'{1+i}')) for i in range(5)]
        asks = [OrderLevel(Decimal('50005'), Decimal(f'{1+i}')) for i in range(5)]

        orderbook = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=bids,
            asks=asks
        )

        metrics = analyzer.analyze_orderbook(orderbook)

        # Should handle gracefully without crashing
        assert metrics.book_shape == BookShape.FLAT

    def test_should_calculate_effective_spread_with_trade_history(self, analyzer, sample_orderbook):
        """Test effective spread calculation with trade history"""
        # Add some trade history
        trades = [
            {'price': 50003, 'side': 'BUY', 'mid_price_at_execution': 50002.5},
            {'price': 50002, 'side': 'SELL', 'mid_price_at_execution': 50002.5},
            {'price': 50004, 'side': 'BUY', 'mid_price_at_execution': 50002.5},
        ]

        for trade in trades:
            analyzer.update_trade_history(trade)

        metrics = analyzer.analyze_orderbook(sample_orderbook)
        assert metrics.effective_spread is not None
        assert isinstance(metrics.effective_spread, float)

    def test_should_maintain_trade_history_limit(self, analyzer):
        """Test that trade history respects size limit"""
        # Add more trades than the limit
        for i in range(1200):  # Exceed the 1000 limit
            trade_data = {
                'price': 50000 + i,
                'size': 1.0,
                'side': 'BUY',
                'mid_price_at_execution': 50000 + i
            }
            analyzer.update_trade_history(trade_data)

        assert len(analyzer._trade_history) <= analyzer._trade_history_limit

    def test_should_handle_very_small_spreads(self, analyzer):
        """Test handling of very small spreads"""
        bids = [OrderLevel(Decimal('50000.00'), Decimal('1.0'))]
        asks = [OrderLevel(Decimal('50000.01'), Decimal('1.0'))]  # 1 cent spread

        orderbook = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=bids,
            asks=asks
        )

        metrics = analyzer.analyze_orderbook(orderbook)
        assert metrics.spread == Decimal('0.01')
        assert metrics.spread_bps < 1.0  # Should be very small in bps

    def test_should_handle_very_large_spreads(self, analyzer):
        """Test handling of very large spreads"""
        bids = [OrderLevel(Decimal('49000'), Decimal('1.0'))]
        asks = [OrderLevel(Decimal('51000'), Decimal('1.0'))]  # $2000 spread

        orderbook = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=bids,
            asks=asks
        )

        metrics = analyzer.analyze_orderbook(orderbook)
        assert metrics.spread == Decimal('2000')
        assert metrics.spread_bps > 100  # Should be large in bps

    def test_should_handle_orderbook_with_minimal_levels(self, analyzer):
        """Test orderbook with only one level on each side"""
        bids = [OrderLevel(Decimal('50000'), Decimal('1.0'))]
        asks = [OrderLevel(Decimal('50005'), Decimal('1.0'))]

        orderbook = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=bids,
            asks=asks
        )

        metrics = analyzer.analyze_orderbook(orderbook)

        # Should still produce valid metrics
        assert metrics.spread == Decimal('5')
        assert metrics.bid_volume_5 == Decimal('1.0')
        assert metrics.ask_volume_5 == Decimal('1.0')
        assert metrics.liquidity_score >= 0

    def test_should_convert_raw_price_size_tuples(self, analyzer):
        """Test conversion of raw [price, size] tuples to OrderLevel objects"""
        # Simulate raw WebSocket data format
        raw_orderbook = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            event_time=1234567890,
            first_update_id=1000,
            final_update_id=1001,
            bids=[['50000', '1.5'], ['49995', '2.0']],  # Raw format
            asks=[['50005', '1.6'], ['50010', '1.9']]   # Raw format
        )

        # Should automatically convert to OrderLevel objects
        assert isinstance(raw_orderbook.bids[0], OrderLevel)
        assert isinstance(raw_orderbook.asks[0], OrderLevel)

        metrics = analyzer.analyze_orderbook(raw_orderbook)
        assert metrics.best_bid == Decimal('50000')
        assert metrics.best_ask == Decimal('50005')