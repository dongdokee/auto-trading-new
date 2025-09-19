# tests/unit/test_execution/test_order_router.py
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from src.execution.models import Order, ExecutionResult, OrderSide, OrderUrgency
from src.execution.order_router import SmartOrderRouter


class TestSmartOrderRouter:
    """SmartOrderRouter 클래스에 대한 TDD 테스트"""

    @pytest.fixture
    def router(self):
        """테스트용 SmartOrderRouter 인스턴스"""
        return SmartOrderRouter()

    @pytest.fixture
    def sample_order(self):
        """테스트용 기본 주문"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("1.0"),
            urgency=OrderUrgency.MEDIUM,
            price=Decimal("50000.0")
        )

    @pytest.fixture
    def high_liquidity_analysis(self):
        """높은 유동성 시장 분석 결과"""
        return {
            'spread_bps': 2.0,
            'liquidity_score': 0.8,
            'avg_volume_1min': 10000.0,
            'best_bid': Decimal('50000.0'),
            'best_ask': Decimal('50001.0'),
            'top_5_liquidity': 50000.0,
            'imbalance': 0.1
        }

    @pytest.fixture
    def low_liquidity_analysis(self):
        """낮은 유동성 시장 분석 결과"""
        return {
            'spread_bps': 15.0,
            'liquidity_score': 0.2,
            'avg_volume_1min': 1000.0,
            'best_bid': Decimal('50000.0'),
            'best_ask': Decimal('50075.0'),
            'top_5_liquidity': 5000.0,
            'imbalance': 0.3
        }

    def test_should_initialize_with_execution_strategies(self, router):
        """실행 전략들로 올바르게 초기화되어야 함"""
        # Then
        expected_strategies = ['AGGRESSIVE', 'PASSIVE', 'TWAP', 'ADAPTIVE']
        assert hasattr(router, 'execution_strategies')

        for strategy in expected_strategies:
            assert strategy in router.execution_strategies
            assert callable(router.execution_strategies[strategy])

    @pytest.mark.asyncio
    async def test_should_route_order_through_complete_workflow(self, router, sample_order):
        """주문을 완전한 워크플로를 통해 라우팅해야 함"""
        # Given
        mock_analysis = {
            'spread_bps': 2.0,
            'liquidity_score': 0.8,
            'avg_volume_1min': 10000.0,
            'best_bid': Decimal('50000.0'),
            'best_ask': Decimal('50001.0')
        }

        mock_result = {
            'strategy': 'AGGRESSIVE',
            'total_filled': Decimal('1.0'),
            'avg_price': Decimal('50000.0')
        }

        router.analyze_market_conditions = AsyncMock(return_value=mock_analysis)
        router._select_execution_strategy = MagicMock(return_value='AGGRESSIVE')

        # Mock the execution strategies dictionary to use our mock
        mock_execute_aggressive = AsyncMock(return_value=mock_result)
        router.execution_strategies['AGGRESSIVE'] = mock_execute_aggressive

        # When
        result = await router.route_order(sample_order)

        # Then
        assert result['strategy'] == 'AGGRESSIVE'
        assert result['total_filled'] == Decimal('1.0')
        router.analyze_market_conditions.assert_called_once_with(sample_order.symbol)
        router._select_execution_strategy.assert_called_once()
        mock_execute_aggressive.assert_called_once()

    def test_should_select_aggressive_strategy_for_immediate_urgency(self, router, high_liquidity_analysis):
        """즉시 긴급도에 대해 AGGRESSIVE 전략을 선택해야 함"""
        # Given
        immediate_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("1.0"),
            urgency=OrderUrgency.IMMEDIATE
        )

        # When
        strategy = router._select_execution_strategy(immediate_order, high_liquidity_analysis)

        # Then
        assert strategy == 'AGGRESSIVE'

    def test_should_select_passive_strategy_for_small_orders_wide_spread(self, router, low_liquidity_analysis):
        """작은 주문과 넓은 스프레드에 대해 PASSIVE 전략을 선택해야 함"""
        # Given
        small_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("0.1"),  # Small order
            urgency=OrderUrgency.LOW
        )

        # Order size percentage calculation
        low_liquidity_analysis['avg_volume_1min'] = 1000.0  # Small volume

        # When
        strategy = router._select_execution_strategy(small_order, low_liquidity_analysis)

        # Then
        assert strategy == 'PASSIVE'

    def test_should_select_twap_strategy_for_large_orders_high_liquidity(self, router, high_liquidity_analysis):
        """큰 주문과 높은 유동성에 대해 TWAP 전략을 선택해야 함"""
        # Given
        large_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("100.0"),  # Large order
            urgency=OrderUrgency.MEDIUM
        )

        # Set high liquidity
        high_liquidity_analysis['avg_volume_1min'] = 100.0  # Makes order 100% of volume

        # When
        strategy = router._select_execution_strategy(large_order, high_liquidity_analysis)

        # Then
        assert strategy == 'TWAP'

    def test_should_select_adaptive_strategy_for_large_orders_low_liquidity(self, router, low_liquidity_analysis):
        """큰 주문과 낮은 유동성에 대해 ADAPTIVE 전략을 선택해야 함"""
        # Given
        large_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("50.0"),  # Large order
            urgency=OrderUrgency.MEDIUM
        )

        # Set low liquidity
        low_liquidity_analysis['avg_volume_1min'] = 50.0  # Makes order 100% of volume
        low_liquidity_analysis['liquidity_score'] = 0.3  # Low liquidity

        # When
        strategy = router._select_execution_strategy(large_order, low_liquidity_analysis)

        # Then
        assert strategy == 'ADAPTIVE'

    def test_should_select_adaptive_strategy_for_medium_orders(self, router, high_liquidity_analysis):
        """중간 크기 주문에 대해 ADAPTIVE 전략을 선택해야 함"""
        # Given
        medium_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("5.0"),  # Medium order
            urgency=OrderUrgency.MEDIUM
        )

        # Set medium volume
        high_liquidity_analysis['avg_volume_1min'] = 20.0  # Makes order 25% of volume

        # When
        strategy = router._select_execution_strategy(medium_order, high_liquidity_analysis)

        # Then
        assert strategy == 'ADAPTIVE'

    @pytest.mark.asyncio
    async def test_should_execute_aggressive_strategy_correctly(self, router, sample_order, high_liquidity_analysis):
        """AGGRESSIVE 전략을 올바르게 실행해야 함"""
        # Given
        router.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('1.0'),
            'avg_price': Decimal('50001.0'),
            'commission': Decimal('20.0'),
            'order_id': 'test-123'
        })
        router._get_aggressive_price = MagicMock(return_value=Decimal('50001.0'))

        # When
        result = await router.execute_aggressive(sample_order, high_liquidity_analysis)

        # Then
        assert result['strategy'] == 'AGGRESSIVE'
        assert result['total_filled'] == Decimal('1.0')
        assert result['avg_price'] == Decimal('50001.0')
        assert result['total_cost'] == Decimal('20.0')
        assert len(result['slices']) == 1

        router.place_order.assert_called_once_with(
            symbol=sample_order.symbol,
            side=sample_order.side,
            size=sample_order.size,
            order_type='IOC',
            price=Decimal('50001.0')
        )

    @pytest.mark.asyncio
    async def test_should_execute_passive_strategy_with_post_only(self, router, sample_order, high_liquidity_analysis):
        """PASSIVE 전략을 Post-Only로 실행해야 함"""
        # Given
        router.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('1.0'),
            'avg_price': Decimal('50000.0'),
            'commission': Decimal('20.0'),
            'status': 'FILLED'
        })

        # When
        result = await router.execute_passive(sample_order, high_liquidity_analysis)

        # Then
        assert result['strategy'] == 'PASSIVE'
        assert result['total_filled'] == Decimal('1.0')
        assert result['avg_price'] == Decimal('50000.0')

        # Should use best_bid for BUY orders
        expected_price = high_liquidity_analysis['best_bid']
        router.place_order.assert_called_with(
            symbol=sample_order.symbol,
            side=sample_order.side,
            size=sample_order.size,
            order_type='POST_ONLY',
            price=expected_price
        )

    @pytest.mark.asyncio
    async def test_should_handle_partial_fills_in_passive_strategy(self, router, sample_order, high_liquidity_analysis):
        """PASSIVE 전략에서 부분 체결을 처리해야 함"""
        # Given
        router.place_order = AsyncMock(return_value={
            'filled_qty': Decimal('0.5'),
            'avg_price': Decimal('50000.0'),
            'commission': Decimal('10.0'),
            'status': 'PARTIALLY_FILLED'
        })
        router.execute_aggressive = AsyncMock(return_value={
            'slices': [{'filled_qty': Decimal('0.5'), 'avg_price': Decimal('50001.0')}],
            'total_filled': Decimal('0.5'),
            'avg_price': Decimal('50001.0')
        })

        # When
        result = await router.execute_passive(sample_order, high_liquidity_analysis)

        # Then
        # Should have attempted aggressive execution for remaining
        router.execute_aggressive.assert_called_once()
        assert len(result['slices']) >= 1

    def test_should_calculate_aggressive_price_correctly(self, router, high_liquidity_analysis):
        """공격적 가격을 올바르게 계산해야 함"""
        # Given
        buy_order = Order(symbol="BTCUSDT", side=OrderSide.BUY, size=Decimal("1.0"))
        sell_order = Order(symbol="BTCUSDT", side=OrderSide.SELL, size=Decimal("1.0"))

        # When
        buy_price = router._get_aggressive_price(buy_order, high_liquidity_analysis)
        sell_price = router._get_aggressive_price(sell_order, high_liquidity_analysis)

        # Then
        best_ask = high_liquidity_analysis['best_ask']
        best_bid = high_liquidity_analysis['best_bid']

        assert buy_price > best_ask  # BUY should pay above ask
        assert sell_price < best_bid  # SELL should receive below bid

    @pytest.mark.asyncio
    async def test_should_aggregate_results_correctly(self, router):
        """결과를 올바르게 집계해야 함"""
        # Given
        result = {
            'strategy': 'TEST',
            'slices': [
                {'filled_qty': Decimal('0.5'), 'avg_price': Decimal('50000.0'), 'commission': Decimal('10.0')},
                {'filled_qty': Decimal('0.3'), 'avg_price': Decimal('50100.0'), 'commission': Decimal('6.0')},
            ],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0')
        }

        # When
        router._aggregate_results(result)

        # Then
        assert result['total_filled'] == Decimal('0.8')

        # Weighted average price calculation
        expected_avg_price = (Decimal('0.5') * Decimal('50000.0') + Decimal('0.3') * Decimal('50100.0')) / Decimal('0.8')
        assert result['avg_price'] == expected_avg_price
        assert result['total_cost'] == Decimal('16.0')

    @pytest.mark.asyncio
    async def test_should_handle_zero_fills_in_aggregation(self, router):
        """집계에서 체결량 0을 처리해야 함"""
        # Given
        result = {
            'strategy': 'TEST',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0')
        }

        # When
        router._aggregate_results(result)

        # Then
        assert result['total_filled'] == Decimal('0')
        assert result['avg_price'] == Decimal('0')
        assert result['total_cost'] == Decimal('0')

    @pytest.mark.asyncio
    async def test_should_analyze_market_conditions_for_symbol(self, router):
        """심볼에 대한 시장 조건을 분석해야 함"""
        # When
        analysis = await router.analyze_market_conditions("BTCUSDT")

        # Then
        # This is a mock implementation, should return default values
        assert isinstance(analysis, dict)
        assert 'spread_bps' in analysis
        assert 'liquidity_score' in analysis
        assert 'avg_volume_1min' in analysis

    @pytest.mark.asyncio
    async def test_should_place_order_with_correct_parameters(self, router):
        """올바른 매개변수로 주문을 배치해야 함"""
        # When
        result = await router.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("1.0"),
            order_type="IOC",
            price=Decimal("50000.0")
        )

        # Then
        # This is a mock implementation
        assert isinstance(result, dict)
        assert 'filled_qty' in result
        assert 'avg_price' in result
        assert 'commission' in result

    def test_should_validate_order_before_routing(self, router):
        """라우팅 전에 주문을 검증해야 함"""
        # When & Then
        # Order validation should happen in Order.__post_init__

        # Test invalid symbol
        with pytest.raises(ValueError, match="Symbol must be uppercase"):
            Order(
                symbol="btcusdt",  # lowercase - invalid
                side=OrderSide.BUY,
                size=Decimal("1.0")
            )

        # Test negative size
        with pytest.raises(ValueError, match="Size must be positive"):
            Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                size=Decimal("-1.0")  # negative - invalid
            )

    def test_should_handle_different_order_sides_correctly(self, router, high_liquidity_analysis):
        """다양한 주문 방향을 올바르게 처리해야 함"""
        # Given
        buy_order = Order(symbol="BTCUSDT", side=OrderSide.BUY, size=Decimal("1.0"))
        sell_order = Order(symbol="BTCUSDT", side=OrderSide.SELL, size=Decimal("1.0"))

        # When
        buy_strategy = router._select_execution_strategy(buy_order, high_liquidity_analysis)
        sell_strategy = router._select_execution_strategy(sell_order, high_liquidity_analysis)

        # Then
        # Both should use same strategy selection logic regardless of side
        assert buy_strategy == sell_strategy

    def test_should_handle_edge_case_order_sizes(self, router, high_liquidity_analysis):
        """극단적인 주문 크기를 처리해야 함"""
        # Given
        tiny_order = Order(symbol="BTCUSDT", side=OrderSide.BUY, size=Decimal("0.00001"))
        huge_order = Order(symbol="BTCUSDT", side=OrderSide.BUY, size=Decimal("1000000.0"))

        # When
        tiny_strategy = router._select_execution_strategy(tiny_order, high_liquidity_analysis)
        huge_strategy = router._select_execution_strategy(huge_order, high_liquidity_analysis)

        # Then
        assert tiny_strategy in ['AGGRESSIVE', 'PASSIVE', 'TWAP', 'ADAPTIVE']
        assert huge_strategy in ['AGGRESSIVE', 'PASSIVE', 'TWAP', 'ADAPTIVE']