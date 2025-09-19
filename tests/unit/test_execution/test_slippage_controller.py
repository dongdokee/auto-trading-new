# tests/unit/test_execution/test_slippage_controller.py
import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from src.execution.slippage_controller import SlippageController, SlippageMetrics, SlippageAlert
from src.execution.models import Order, OrderSide, OrderUrgency


class TestSlippageController:
    """SlippageController 클래스에 대한 TDD 테스트"""

    @pytest.fixture
    def slippage_controller(self):
        """테스트용 SlippageController 인스턴스"""
        return SlippageController()

    @pytest.fixture
    def sample_order(self):
        """테스트용 기본 주문"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("2.0"),
            urgency=OrderUrgency.MEDIUM,
            price=Decimal("50000.0")
        )

    def test_should_initialize_with_default_parameters(self, slippage_controller):
        """SlippageController가 기본 매개변수로 초기화되어야 함"""
        # Then
        assert slippage_controller.max_slippage_bps == 50  # 0.5% default
        assert slippage_controller.alert_threshold_bps == 25  # 0.25% default
        assert slippage_controller.measurement_window == 300  # 5 minutes
        assert len(slippage_controller.slippage_history) == 0
        assert len(slippage_controller.active_alerts) == 0

    def test_should_calculate_slippage_correctly_for_buy_order(self, slippage_controller):
        """매수 주문에 대한 슬리피지를 올바르게 계산해야 함"""
        # Given
        benchmark_price = Decimal("50000.0")
        execution_price = Decimal("50100.0")  # Higher price = negative slippage for buy

        # When
        slippage_bps = slippage_controller.calculate_slippage(
            benchmark_price, execution_price, OrderSide.BUY
        )

        # Then
        expected_slippage = (execution_price - benchmark_price) / benchmark_price * 10000
        assert slippage_bps == expected_slippage
        assert slippage_bps == Decimal("20.0")  # 0.2% slippage

    def test_should_calculate_slippage_correctly_for_sell_order(self, slippage_controller):
        """매도 주문에 대한 슬리피지를 올바르게 계산해야 함"""
        # Given
        benchmark_price = Decimal("50000.0")
        execution_price = Decimal("49900.0")  # Lower price = negative slippage for sell

        # When
        slippage_bps = slippage_controller.calculate_slippage(
            benchmark_price, execution_price, OrderSide.SELL
        )

        # Then
        expected_slippage = (benchmark_price - execution_price) / benchmark_price * 10000
        assert slippage_bps == expected_slippage
        assert slippage_bps == Decimal("20.0")  # 0.2% slippage

    @pytest.mark.asyncio
    async def test_should_record_slippage_metrics(self, slippage_controller, sample_order):
        """슬리피지 지표를 기록해야 함"""
        # Given
        benchmark_price = Decimal("50000.0")
        execution_price = Decimal("50050.0")
        filled_qty = Decimal("2.0")

        # When
        await slippage_controller.record_slippage(
            order=sample_order,
            benchmark_price=benchmark_price,
            execution_price=execution_price,
            filled_qty=filled_qty
        )

        # Then
        assert len(slippage_controller.slippage_history) == 1
        metrics = slippage_controller.slippage_history[0]
        assert metrics.order_id == sample_order.order_id
        assert metrics.symbol == sample_order.symbol
        assert metrics.slippage_bps == Decimal("10.0")  # 0.1% slippage
        assert metrics.cost_impact == filled_qty * (execution_price - benchmark_price)
        assert isinstance(metrics.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_should_trigger_alert_when_slippage_exceeds_threshold(self, slippage_controller, sample_order):
        """슬리피지가 임계값을 초과할 때 알림을 트리거해야 함"""
        # Given
        slippage_controller.alert_threshold_bps = 20  # 0.2% threshold
        benchmark_price = Decimal("50000.0")
        execution_price = Decimal("50150.0")  # 0.3% slippage

        # When
        await slippage_controller.record_slippage(
            order=sample_order,
            benchmark_price=benchmark_price,
            execution_price=execution_price,
            filled_qty=Decimal("2.0")
        )

        # Then
        assert len(slippage_controller.active_alerts) == 1
        alert = slippage_controller.active_alerts[0]
        assert alert.symbol == sample_order.symbol
        assert alert.slippage_bps == Decimal("30.0")
        assert alert.severity == "HIGH"

    @pytest.mark.asyncio
    async def test_should_not_trigger_alert_when_slippage_below_threshold(self, slippage_controller, sample_order):
        """슬리피지가 임계값 아래일 때 알림을 트리거하지 않아야 함"""
        # Given
        slippage_controller.alert_threshold_bps = 20  # 0.2% threshold
        benchmark_price = Decimal("50000.0")
        execution_price = Decimal("50050.0")  # 0.1% slippage

        # When
        await slippage_controller.record_slippage(
            order=sample_order,
            benchmark_price=benchmark_price,
            execution_price=execution_price,
            filled_qty=Decimal("2.0")
        )

        # Then
        assert len(slippage_controller.active_alerts) == 0

    @pytest.mark.asyncio
    async def test_should_check_order_against_slippage_limits(self, slippage_controller, sample_order):
        """주문이 슬리피지 한도를 위반하는지 확인해야 함"""
        # Given
        benchmark_price = Decimal("50000.0")
        execution_price = Decimal("50300.0")  # 0.6% slippage, exceeds 0.5% default limit

        # When
        is_allowed = await slippage_controller.check_slippage_limit(
            order=sample_order,
            benchmark_price=benchmark_price,
            proposed_price=execution_price
        )

        # Then
        assert is_allowed is False

    @pytest.mark.asyncio
    async def test_should_allow_order_within_slippage_limits(self, slippage_controller, sample_order):
        """슬리피지 한도 내의 주문을 허용해야 함"""
        # Given
        benchmark_price = Decimal("50000.0")
        execution_price = Decimal("50200.0")  # 0.4% slippage, within 0.5% limit

        # When
        is_allowed = await slippage_controller.check_slippage_limit(
            order=sample_order,
            benchmark_price=benchmark_price,
            proposed_price=execution_price
        )

        # Then
        assert is_allowed is True

    def test_should_calculate_aggregate_slippage_statistics(self, slippage_controller):
        """집계 슬리피지 통계를 계산해야 함"""
        # Given - add some historical slippage data
        metrics = [
            SlippageMetrics("order1", "BTCUSDT", Decimal("10.0"), Decimal("50.0"), datetime.now()),
            SlippageMetrics("order2", "BTCUSDT", Decimal("20.0"), Decimal("100.0"), datetime.now()),
            SlippageMetrics("order3", "ETHUSDT", Decimal("15.0"), Decimal("30.0"), datetime.now()),
        ]
        slippage_controller.slippage_history.extend(metrics)

        # When
        stats = slippage_controller.get_slippage_statistics()

        # Then
        assert stats['total_orders'] == 3
        assert stats['avg_slippage_bps'] == Decimal("15.0")  # (10+20+15)/3
        assert stats['max_slippage_bps'] == Decimal("20.0")
        assert stats['total_cost_impact'] == Decimal("180.0")  # 50+100+30
        assert 'symbols' in stats
        assert len(stats['symbols']) == 2  # BTCUSDT and ETHUSDT

    def test_should_get_slippage_statistics_by_symbol(self, slippage_controller):
        """심볼별 슬리피지 통계를 반환해야 함"""
        # Given
        metrics = [
            SlippageMetrics("order1", "BTCUSDT", Decimal("10.0"), Decimal("50.0"), datetime.now()),
            SlippageMetrics("order2", "BTCUSDT", Decimal("20.0"), Decimal("100.0"), datetime.now()),
            SlippageMetrics("order3", "ETHUSDT", Decimal("15.0"), Decimal("30.0"), datetime.now()),
        ]
        slippage_controller.slippage_history.extend(metrics)

        # When
        btc_stats = slippage_controller.get_symbol_slippage_statistics("BTCUSDT")

        # Then
        assert btc_stats['symbol'] == "BTCUSDT"
        assert btc_stats['order_count'] == 2
        assert btc_stats['avg_slippage_bps'] == Decimal("15.0")  # (10+20)/2
        assert btc_stats['total_cost_impact'] == Decimal("150.0")  # 50+100

    @pytest.mark.asyncio
    async def test_should_implement_real_time_monitoring(self, slippage_controller):
        """실시간 모니터링을 구현해야 함"""
        # Given
        monitoring_started = False

        # Mock the monitoring callback
        async def mock_callback(metrics):
            nonlocal monitoring_started
            monitoring_started = True

        slippage_controller.monitoring_callback = mock_callback

        # When
        await slippage_controller.start_monitoring()

        # Simulate slippage event
        sample_order = Order("BTCUSDT", OrderSide.BUY, Decimal("1.0"))
        await slippage_controller.record_slippage(
            order=sample_order,
            benchmark_price=Decimal("50000.0"),
            execution_price=Decimal("50100.0"),
            filled_qty=Decimal("1.0")
        )

        # Then
        assert slippage_controller.is_monitoring is True

    @pytest.mark.asyncio
    async def test_should_stop_monitoring_when_requested(self, slippage_controller):
        """요청 시 모니터링을 중단해야 함"""
        # Given
        await slippage_controller.start_monitoring()

        # When
        await slippage_controller.stop_monitoring()

        # Then
        assert slippage_controller.is_monitoring is False

    def test_should_clear_old_slippage_history(self, slippage_controller):
        """오래된 슬리피지 이력을 정리해야 함"""
        # Given
        old_time = datetime.now() - timedelta(hours=2)
        recent_time = datetime.now()

        old_metrics = [
            SlippageMetrics("old1", "BTCUSDT", Decimal("10.0"), Decimal("50.0"), old_time),
            SlippageMetrics("old2", "BTCUSDT", Decimal("15.0"), Decimal("75.0"), old_time),
        ]
        recent_metrics = [
            SlippageMetrics("recent1", "BTCUSDT", Decimal("20.0"), Decimal("100.0"), recent_time),
        ]

        slippage_controller.slippage_history.extend(old_metrics + recent_metrics)

        # When
        slippage_controller.cleanup_old_history(max_age_hours=1)

        # Then
        assert len(slippage_controller.slippage_history) == 1
        assert slippage_controller.slippage_history[0].order_id == "recent1"

    @pytest.mark.asyncio
    async def test_should_handle_extreme_slippage_scenarios(self, slippage_controller, sample_order):
        """극단적인 슬리피지 시나리오를 처리해야 함"""
        # Given
        benchmark_price = Decimal("50000.0")
        extreme_price = Decimal("55000.0")  # 10% slippage

        # When
        await slippage_controller.record_slippage(
            order=sample_order,
            benchmark_price=benchmark_price,
            execution_price=extreme_price,
            filled_qty=Decimal("2.0")
        )

        # Then
        assert len(slippage_controller.active_alerts) == 1
        alert = slippage_controller.active_alerts[0]
        assert alert.severity == "CRITICAL"  # Should escalate to critical
        assert alert.slippage_bps == Decimal("1000.0")  # 10% in bps

    def test_should_calculate_implementation_shortfall(self, slippage_controller):
        """구현 부족(Implementation Shortfall)을 계산해야 함"""
        # Given
        decision_price = Decimal("50000.0")
        execution_price = Decimal("50150.0")
        order_size = Decimal("10.0")

        # When
        shortfall = slippage_controller.calculate_implementation_shortfall(
            decision_price=decision_price,
            execution_price=execution_price,
            order_size=order_size,
            side=OrderSide.BUY
        )

        # Then
        expected_shortfall = (execution_price - decision_price) * order_size
        assert shortfall == expected_shortfall
        assert shortfall == Decimal("1500.0")  # $150 per unit * 10 units

    def test_should_estimate_market_impact(self, slippage_controller):
        """시장 영향을 추정해야 함"""
        # Given
        order_size = Decimal("100.0")
        avg_daily_volume = 1000000
        volatility = Decimal("0.02")

        # When
        impact_bps = slippage_controller.estimate_market_impact(
            order_size=order_size,
            avg_daily_volume=avg_daily_volume,
            volatility=volatility
        )

        # Then
        assert isinstance(impact_bps, Decimal)
        assert impact_bps > 0
        # Market impact should increase with order size and volatility

    @pytest.mark.asyncio
    async def test_should_provide_slippage_attribution_analysis(self, slippage_controller):
        """슬리피지 귀인 분석을 제공해야 함"""
        # Given - simulate different types of slippage
        orders_data = [
            ("timing", Decimal("10.0")),
            ("market_impact", Decimal("15.0")),
            ("spread", Decimal("5.0")),
        ]

        for slippage_type, slippage_bps in orders_data:
            order = Order("BTCUSDT", OrderSide.BUY, Decimal("1.0"))
            metrics = SlippageMetrics(
                order.order_id, "BTCUSDT", slippage_bps, Decimal("100.0"), datetime.now()
            )
            metrics.slippage_type = slippage_type
            slippage_controller.slippage_history.append(metrics)

        # When
        attribution = slippage_controller.get_slippage_attribution()

        # Then
        assert 'timing' in attribution
        assert 'market_impact' in attribution
        assert 'spread' in attribution
        assert attribution['total_slippage_bps'] == Decimal("30.0")

    def test_should_validate_benchmark_price_quality(self, slippage_controller):
        """벤치마크 가격 품질을 검증해야 함"""
        # Given
        current_market_price = Decimal("50000.0")
        benchmark_price = Decimal("49000.0")  # 2% difference - suspicious

        # When
        is_valid = slippage_controller.validate_benchmark_price(
            benchmark_price=benchmark_price,
            current_market_price=current_market_price,
            max_deviation_bps=100  # 1% max deviation
        )

        # Then
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_should_handle_concurrent_slippage_recording(self, slippage_controller):
        """동시 슬리피지 기록을 처리해야 함"""
        # When - record multiple slippages concurrently
        tasks = []
        for i in range(5):
            order = Order("BTCUSDT", OrderSide.BUY, Decimal("1.0"))
            task = slippage_controller.record_slippage(
                order=order,
                benchmark_price=Decimal("50000.0"),
                execution_price=Decimal("50100.0"),
                filled_qty=Decimal("1.0")
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Then
        assert len(slippage_controller.slippage_history) == 5
        # All should have been recorded without data corruption

    def test_should_export_slippage_report(self, slippage_controller):
        """슬리피지 보고서를 내보내기해야 함"""
        # Given
        metrics = [
            SlippageMetrics("order1", "BTCUSDT", Decimal("10.0"), Decimal("50.0"), datetime.now()),
            SlippageMetrics("order2", "ETHUSDT", Decimal("15.0"), Decimal("75.0"), datetime.now()),
        ]
        slippage_controller.slippage_history.extend(metrics)

        # When
        report = slippage_controller.generate_slippage_report(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )

        # Then
        assert 'summary' in report
        assert 'by_symbol' in report
        assert 'alerts' in report
        assert report['summary']['total_orders'] == 2
        assert len(report['by_symbol']) == 2