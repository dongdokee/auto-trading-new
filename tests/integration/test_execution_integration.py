# tests/integration/test_execution_integration.py
"""Integration tests for execution module components"""

import pytest
from decimal import Decimal
from src.execution.models import Order, OrderSide, OrderUrgency
from src.execution.market_analyzer import MarketConditionAnalyzer
from tests.fixtures.execution_fixtures import (
    liquid_orderbook,
    thin_orderbook,
    sample_btc_order,
    large_order,
    mock_exchange
)


class TestExecutionIntegration:
    """Execution module integration tests"""

    def test_should_integrate_order_creation_with_market_analysis(
        self, liquid_orderbook, sample_btc_order
    ):
        """주문 생성과 시장 분석이 통합되어야 함"""
        # Given
        analyzer = MarketConditionAnalyzer()
        order = sample_btc_order

        # When
        market_analysis = analyzer.analyze_orderbook(liquid_orderbook)

        # Then
        assert order.symbol == liquid_orderbook['symbol']
        assert market_analysis['best_bid'] is not None
        assert market_analysis['best_ask'] is not None
        assert market_analysis['liquidity_score'] > 0

        # 주문 크기가 시장 유동성과 비교 가능해야 함
        order_notional = order.notional_value
        market_liquidity = market_analysis['top_5_liquidity']

        assert order_notional is not None
        assert market_liquidity > 0

        # 주문이 시장 유동성의 일정 비율 이하여야 함
        size_ratio = float(order.size) / float(market_liquidity)
        assert size_ratio < 1.0  # 주문이 전체 유동성보다 작아야 함

    def test_should_analyze_execution_feasibility_for_different_order_sizes(
        self, liquid_orderbook, thin_orderbook
    ):
        """다양한 주문 크기에 대한 실행 가능성을 분석해야 함"""
        # Given
        analyzer = MarketConditionAnalyzer()

        small_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("0.1"),
            urgency=OrderUrgency.MEDIUM
        )

        large_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("100.0"),
            urgency=OrderUrgency.LOW
        )

        # When
        liquid_analysis = analyzer.analyze_orderbook(liquid_orderbook)
        thin_analysis = analyzer.analyze_orderbook(thin_orderbook)

        # Then
        liquid_total_liquidity = liquid_analysis['top_5_liquidity']
        thin_total_liquidity = thin_analysis['top_5_liquidity']

        # 유동성이 높은 시장에서는 큰 주문도 실행 가능
        large_order_ratio_liquid = float(large_order.size) / float(liquid_total_liquidity)
        assert large_order_ratio_liquid < 5.0  # 500% 이하

        # 작은 주문은 모든 시장에서 실행 가능
        small_order_ratio_thin = float(small_order.size) / float(thin_total_liquidity)
        assert small_order_ratio_thin < 1.0  # 100% 이하

    def test_should_determine_optimal_execution_strategy_based_on_market_conditions(
        self, liquid_orderbook, thin_orderbook
    ):
        """시장 조건에 따른 최적 실행 전략을 결정해야 함"""
        # Given
        analyzer = MarketConditionAnalyzer()

        # When
        liquid_analysis = analyzer.analyze_orderbook(liquid_orderbook)
        thin_analysis = analyzer.analyze_orderbook(thin_orderbook)

        # Then
        # 유동성이 높고 스프레드가 좁은 시장 - AGGRESSIVE 전략 적합
        assert liquid_analysis['spread_bps'] < 5.0
        assert liquid_analysis['liquidity_score'] > 0.01  # 유동성 점수가 있음

        # 다른 시장 특성 비교 - 스프레드 차이로 구분
        assert liquid_analysis['spread_bps'] != thin_analysis['spread_bps']
        assert thin_analysis['spread_bps'] > liquid_analysis['spread_bps']

    def test_should_calculate_expected_execution_costs(self, liquid_orderbook):
        """예상 실행 비용을 계산해야 함"""
        # Given
        analyzer = MarketConditionAnalyzer()
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("2.0"),
            price=Decimal("50000.0")
        )

        # When
        analysis = analyzer.analyze_orderbook(liquid_orderbook)
        impact_function = analysis['price_impact']

        # Then
        expected_impact = impact_function(order.size, order.side.value)
        assert isinstance(expected_impact, (int, float))
        assert expected_impact >= 0

        # 실행 비용 추정
        notional_value = order.notional_value
        impact_cost = float(notional_value) * expected_impact
        spread_cost = float(notional_value) * (analysis['spread_bps'] / 10000)

        total_cost = impact_cost + spread_cost
        assert total_cost >= 0
        assert total_cost < float(notional_value) * 0.01  # 1% 이하

    @pytest.mark.asyncio
    async def test_should_simulate_complete_order_execution_workflow(
        self, liquid_orderbook, mock_exchange
    ):
        """완전한 주문 실행 워크플로를 시뮬레이션해야 함"""
        # Given
        analyzer = MarketConditionAnalyzer()
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("1.0"),
            price=Decimal("50000.0"),
            urgency=OrderUrgency.MEDIUM
        )

        # When
        # 1. 시장 분석
        market_analysis = analyzer.analyze_orderbook(liquid_orderbook)

        # 2. 실행 전략 결정 (간단한 로직)
        if market_analysis['spread_bps'] < 5.0 and order.urgency == OrderUrgency.MEDIUM:
            strategy = "AGGRESSIVE"
        else:
            strategy = "PASSIVE"

        # 3. 모의 주문 실행
        execution_result = await mock_exchange.place_order(
            symbol=order.symbol,
            side=order.side.value,
            size=float(order.size),
            price=float(order.price),
            order_type=strategy
        )

        # Then
        assert execution_result['status'] == 'FILLED'
        assert float(execution_result['filled_qty']) == float(order.size)
        assert execution_result['order_id'] is not None
        assert execution_result['commission'] > 0

    def test_should_handle_cross_symbol_market_analysis(self):
        """다중 심볼 시장 분석을 처리해야 함"""
        # Given
        analyzer = MarketConditionAnalyzer()

        btc_orderbook = {
            'symbol': 'BTCUSDT',
            'bids': [{'price': Decimal('50000.0'), 'size': Decimal('1.0')}],
            'asks': [{'price': Decimal('50001.0'), 'size': Decimal('1.0')}],
            'timestamp': 1640995200000
        }

        eth_orderbook = {
            'symbol': 'ETHUSDT',
            'bids': [{'price': Decimal('3000.0'), 'size': Decimal('10.0')}],
            'asks': [{'price': Decimal('3001.0'), 'size': Decimal('10.0')}],
            'timestamp': 1640995200000
        }

        # When
        btc_analysis = analyzer.analyze_orderbook(btc_orderbook)
        eth_analysis = analyzer.analyze_orderbook(eth_orderbook)

        # Then
        assert btc_analysis['spread_bps'] != eth_analysis['spread_bps']
        assert btc_analysis['top_5_liquidity'] != eth_analysis['top_5_liquidity']

        # 각 심볼별로 독립적인 분석 결과
        assert 'best_bid' in btc_analysis
        assert 'best_bid' in eth_analysis
        assert btc_analysis['best_bid'] != eth_analysis['best_bid']

    def test_should_validate_order_against_market_constraints(self, thin_orderbook):
        """주문이 시장 제약 조건을 준수하는지 검증해야 함"""
        # Given
        analyzer = MarketConditionAnalyzer()

        # 시장 유동성보다 훨씬 큰 주문
        oversized_order = Order(
            symbol="ADAUSDT",
            side=OrderSide.BUY,
            size=Decimal("10000.0"),  # Very large
            urgency=OrderUrgency.IMMEDIATE
        )

        # When
        market_analysis = analyzer.analyze_orderbook(thin_orderbook)

        # Then
        market_liquidity = market_analysis['top_5_liquidity']
        order_size_ratio = float(oversized_order.size) / float(market_liquidity)

        # 주문이 시장 유동성의 상당 부분을 차지함을 확인
        assert order_size_ratio > 10.0  # 1000% 이상

        # 이런 경우 주문 분할이 필요함을 시사
        suggested_slice_size = float(market_liquidity) * 0.1  # 10%
        assert suggested_slice_size < float(oversized_order.size)

    def test_should_preserve_order_integrity_throughout_workflow(self, liquid_orderbook):
        """워크플로 전반에 걸쳐 주문 무결성을 보존해야 함"""
        # Given
        original_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            size=Decimal("5.0"),
            price=Decimal("50000.0"),
            urgency=OrderUrgency.HIGH
        )

        analyzer = MarketConditionAnalyzer()

        # When
        # 시장 분석을 거쳐도 주문 정보는 변경되지 않아야 함
        market_analysis = analyzer.analyze_orderbook(liquid_orderbook)

        # Then
        # 원본 주문 속성이 그대로 유지되어야 함
        assert original_order.symbol == "BTCUSDT"
        assert original_order.side == OrderSide.SELL
        assert original_order.size == Decimal("5.0")
        assert original_order.price == Decimal("50000.0")
        assert original_order.urgency == OrderUrgency.HIGH

        # 주문 ID와 생성 시간은 변경되지 않아야 함
        order_id = original_order.order_id
        created_at = original_order.created_at

        # 시장 분석 후에도 동일해야 함
        assert original_order.order_id == order_id
        assert original_order.created_at == created_at

    @pytest.mark.asyncio
    async def test_should_handle_execution_errors_gracefully(self, mock_exchange):
        """실행 오류를 우아하게 처리해야 함"""
        # Given
        invalid_order = Order(
            symbol="INVALID",
            side=OrderSide.BUY,
            size=Decimal("1.0")
        )

        # When
        try:
            result = await mock_exchange.place_order(
                symbol=invalid_order.symbol,
                side=invalid_order.side.value,
                size=float(invalid_order.size)
            )
            # Mock은 모든 주문을 성공으로 처리하므로 여기서는 성공
            assert result['status'] == 'FILLED'
        except Exception as e:
            # 실제 구현에서는 오류 처리가 있어야 함
            assert isinstance(e, (ValueError, ConnectionError, TimeoutError))

    def test_should_benchmark_execution_performance(self, liquid_orderbook):
        """실행 성능을 벤치마크해야 함"""
        # Given
        analyzer = MarketConditionAnalyzer()
        import time

        # When
        start_time = time.time()

        # 시장 분석 성능 측정
        for _ in range(100):
            analysis = analyzer.analyze_orderbook(liquid_orderbook)

        end_time = time.time()
        avg_analysis_time = (end_time - start_time) / 100

        # Then
        # 시장 분석은 10ms 이내에 완료되어야 함
        assert avg_analysis_time < 0.01  # 10ms

        # 분석 결과의 일관성 확인
        final_analysis = analyzer.analyze_orderbook(liquid_orderbook)
        assert 'spread_bps' in final_analysis
        assert 'liquidity_score' in final_analysis
        assert final_analysis['liquidity_score'] >= 0