# tests/unit/test_execution/test_market_analyzer.py
import pytest
from decimal import Decimal
from typing import Dict, List
from src.execution.market_analyzer import MarketConditionAnalyzer


class TestMarketConditionAnalyzer:
    """MarketConditionAnalyzer 클래스에 대한 TDD 테스트"""

    @pytest.fixture
    def sample_orderbook(self) -> Dict:
        """테스트용 주문북 데이터"""
        return {
            'symbol': 'BTCUSDT',
            'bids': [
                {'price': Decimal('50000.0'), 'size': Decimal('1.5')},
                {'price': Decimal('49999.0'), 'size': Decimal('2.0')},
                {'price': Decimal('49998.0'), 'size': Decimal('1.0')},
                {'price': Decimal('49997.0'), 'size': Decimal('3.0')},
                {'price': Decimal('49996.0'), 'size': Decimal('2.5')},
            ],
            'asks': [
                {'price': Decimal('50001.0'), 'size': Decimal('1.2')},
                {'price': Decimal('50002.0'), 'size': Decimal('1.8')},
                {'price': Decimal('50003.0'), 'size': Decimal('0.8')},
                {'price': Decimal('50004.0'), 'size': Decimal('2.2')},
                {'price': Decimal('50005.0'), 'size': Decimal('1.5')},
            ],
            'timestamp': 1640995200000
        }

    @pytest.fixture
    def analyzer(self) -> MarketConditionAnalyzer:
        """테스트용 MarketConditionAnalyzer 인스턴스"""
        return MarketConditionAnalyzer()

    def test_should_calculate_basic_spread_metrics_correctly(self, analyzer, sample_orderbook):
        """기본 스프레드 지표를 올바르게 계산해야 함"""
        # When
        analysis = analyzer.analyze_orderbook(sample_orderbook)

        # Then
        assert analysis['best_bid'] == Decimal('50000.0')
        assert analysis['best_ask'] == Decimal('50001.0')
        assert analysis['mid_price'] == Decimal('50000.5')
        assert analysis['spread'] == Decimal('1.0')
        assert analysis['spread_bps'] == pytest.approx(0.2, rel=1e-3)  # 1/50000.5 * 10000

    def test_should_calculate_orderbook_imbalance_correctly(self, analyzer, sample_orderbook):
        """주문북 불균형을 올바르게 계산해야 함"""
        # When
        analysis = analyzer.analyze_orderbook(sample_orderbook)

        # Then
        expected_bid_volume_5 = Decimal('10.0')  # 1.5+2.0+1.0+3.0+2.5
        expected_ask_volume_5 = Decimal('7.5')   # 1.2+1.8+0.8+2.2+1.5
        expected_total_volume = expected_bid_volume_5 + expected_ask_volume_5
        expected_imbalance = (expected_bid_volume_5 - expected_ask_volume_5) / expected_total_volume

        assert analysis['bid_volume_5'] == expected_bid_volume_5
        assert analysis['ask_volume_5'] == expected_ask_volume_5
        assert analysis['top_5_liquidity'] == expected_total_volume
        assert abs(analysis['imbalance'] - expected_imbalance) < Decimal('0.0001')

    def test_should_calculate_liquidity_score_between_zero_and_one(self, analyzer, sample_orderbook):
        """유동성 점수가 0과 1 사이에 있어야 함"""
        # When
        analysis = analyzer.analyze_orderbook(sample_orderbook)

        # Then
        liquidity_score = analysis['liquidity_score']
        assert Decimal('0') <= liquidity_score <= Decimal('1')
        assert isinstance(liquidity_score, (float, Decimal))

    def test_should_estimate_price_impact_function(self, analyzer, sample_orderbook):
        """가격 충격 함수를 추정해야 함"""
        # When
        analysis = analyzer.analyze_orderbook(sample_orderbook)

        # Then
        assert 'price_impact' in analysis
        impact_function = analysis['price_impact']

        # 작은 주문의 경우 충격이 적어야 함
        small_impact = impact_function(Decimal('0.1'), 'BUY')
        large_impact = impact_function(Decimal('5.0'), 'BUY')

        assert isinstance(small_impact, (float, Decimal))
        assert isinstance(large_impact, (float, Decimal))
        assert large_impact > small_impact

    def test_should_detect_large_orders_in_orderbook(self, analyzer, sample_orderbook):
        """주문북에서 대량 주문을 감지해야 함"""
        # When
        analysis = analyzer.analyze_orderbook(sample_orderbook)

        # Then
        assert 'large_orders' in analysis
        large_orders = analysis['large_orders']
        assert isinstance(large_orders, dict)
        assert 'bid_side' in large_orders
        assert 'ask_side' in large_orders

    def test_should_handle_empty_orderbook_gracefully(self, analyzer):
        """빈 주문북을 우아하게 처리해야 함"""
        # Given
        empty_orderbook = {
            'symbol': 'BTCUSDT',
            'bids': [],
            'asks': [],
            'timestamp': 1640995200000
        }

        # When
        analysis = analyzer.analyze_orderbook(empty_orderbook)

        # Then
        assert analysis['best_bid'] is None
        assert analysis['best_ask'] is None
        assert analysis['mid_price'] is None
        assert analysis['spread'] is None
        assert analysis['spread_bps'] is None
        assert analysis['imbalance'] == Decimal('0')

    def test_should_handle_one_sided_orderbook(self, analyzer):
        """한쪽만 있는 주문북을 처리해야 함"""
        # Given
        one_sided_orderbook = {
            'symbol': 'BTCUSDT',
            'bids': [
                {'price': Decimal('50000.0'), 'size': Decimal('1.5')},
                {'price': Decimal('49999.0'), 'size': Decimal('2.0')},
            ],
            'asks': [],
            'timestamp': 1640995200000
        }

        # When
        analysis = analyzer.analyze_orderbook(one_sided_orderbook)

        # Then
        assert analysis['best_bid'] == Decimal('50000.0')
        assert analysis['best_ask'] is None
        assert analysis['mid_price'] is None
        assert analysis['spread'] is None
        assert analysis['bid_volume_5'] == Decimal('3.5')
        assert analysis['ask_volume_5'] == Decimal('0')

    def test_should_calculate_effective_spread(self, analyzer, sample_orderbook):
        """실효 스프레드를 계산해야 함"""
        # When
        analysis = analyzer.analyze_orderbook(sample_orderbook)

        # Then
        assert 'effective_spread' in analysis
        effective_spread = analysis['effective_spread']
        assert isinstance(effective_spread, (float, Decimal))
        assert effective_spread >= 0

    def test_should_analyze_book_shape_characteristics(self, analyzer, sample_orderbook):
        """주문북 형태 특성을 분석해야 함"""
        # When
        analysis = analyzer.analyze_orderbook(sample_orderbook)

        # Then
        assert 'book_shape' in analysis
        book_shape = analysis['book_shape']
        assert isinstance(book_shape, dict)
        assert 'depth_ratio' in book_shape
        assert 'concentration' in book_shape

    def test_should_handle_invalid_orderbook_data(self, analyzer):
        """잘못된 주문북 데이터를 처리해야 함"""
        # Given
        invalid_orderbook = {
            'symbol': 'BTCUSDT',
            'bids': [
                {'price': 'invalid', 'size': Decimal('1.5')},  # 잘못된 가격
            ],
            'asks': [
                {'price': Decimal('50001.0'), 'size': 'invalid'},  # 잘못된 크기
            ],
            'timestamp': 1640995200000
        }

        # When & Then
        with pytest.raises(ValueError, match="Invalid orderbook data"):
            analyzer.analyze_orderbook(invalid_orderbook)

    def test_should_validate_symbol_format(self, analyzer, sample_orderbook):
        """심볼 형식을 검증해야 함"""
        # Given
        sample_orderbook['symbol'] = 'btc-usdt'  # 잘못된 형식

        # When & Then
        with pytest.raises(ValueError, match="Invalid symbol format"):
            analyzer.analyze_orderbook(sample_orderbook)

    def test_should_require_timestamp_in_orderbook(self, analyzer, sample_orderbook):
        """주문북에 타임스탬프가 있어야 함"""
        # Given
        del sample_orderbook['timestamp']

        # When & Then
        with pytest.raises(ValueError, match="Timestamp is required"):
            analyzer.analyze_orderbook(sample_orderbook)

    def test_should_calculate_volume_weighted_average_prices(self, analyzer, sample_orderbook):
        """볼륨 가중 평균 가격을 계산해야 함"""
        # When
        analysis = analyzer.analyze_orderbook(sample_orderbook)

        # Then
        assert 'vwap_bid' in analysis
        assert 'vwap_ask' in analysis
        assert isinstance(analysis['vwap_bid'], (float, Decimal))
        assert isinstance(analysis['vwap_ask'], (float, Decimal))

    def test_should_estimate_market_volatility_indicators(self, analyzer, sample_orderbook):
        """시장 변동성 지표를 추정해야 함"""
        # When
        analysis = analyzer.analyze_orderbook(sample_orderbook)

        # Then
        assert 'volatility_estimate' in analysis
        volatility = analysis['volatility_estimate']
        assert isinstance(volatility, (float, Decimal))
        assert volatility >= 0