# tests/unit/test_execution/test_order_manager.py
import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from src.execution.order_manager import OrderManager, OrderInfo
from src.execution.models import Order, OrderSide, OrderUrgency, OrderStatus


class TestOrderManager:
    """OrderManager 클래스에 대한 TDD 테스트"""

    @pytest.fixture
    def order_manager(self):
        """테스트용 OrderManager 인스턴스"""
        return OrderManager()

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

    @pytest.fixture
    def large_order(self):
        """테스트용 대량 주문"""
        return Order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            size=Decimal("100.0"),
            urgency=OrderUrgency.LOW,
            price=Decimal("3000.0")
        )

    def test_should_initialize_with_empty_state(self, order_manager):
        """OrderManager가 빈 상태로 초기화되어야 함"""
        # Then
        assert len(order_manager.active_orders) == 0
        assert len(order_manager.order_history) == 0
        assert order_manager.max_order_age == 300  # 5 minutes default
        assert order_manager.max_active_orders == 100  # Default limit

    @pytest.mark.asyncio
    async def test_should_submit_order_and_generate_unique_id(self, order_manager, sample_order):
        """주문을 제출하고 고유 ID를 생성해야 함"""
        # When
        order_id = await order_manager.submit_order(sample_order)

        # Then
        assert order_id is not None
        assert isinstance(order_id, str)
        assert len(order_id) > 0
        assert order_id in order_manager.active_orders

        order_info = order_manager.active_orders[order_id]
        assert order_info.order == sample_order
        assert order_info.status == OrderStatus.PENDING
        assert order_info.filled_qty == Decimal('0')
        assert order_info.avg_price == Decimal('0')
        assert order_info.attempts == 0
        assert isinstance(order_info.submitted_at, datetime)

    @pytest.mark.asyncio
    async def test_should_reject_order_when_max_active_limit_reached(self, order_manager, sample_order):
        """최대 활성 주문 수에 도달했을 때 주문을 거부해야 함"""
        # Given
        order_manager.max_active_orders = 2

        # Submit orders up to limit
        await order_manager.submit_order(sample_order)
        await order_manager.submit_order(sample_order)

        # When & Then
        with pytest.raises(ValueError, match="Maximum active orders limit reached"):
            await order_manager.submit_order(sample_order)

    @pytest.mark.asyncio
    async def test_should_cancel_active_order_successfully(self, order_manager, sample_order):
        """활성 주문을 성공적으로 취소해야 함"""
        # Given
        order_id = await order_manager.submit_order(sample_order)

        # When
        success = await order_manager.cancel_order(order_id)

        # Then
        assert success is True
        assert order_id not in order_manager.active_orders
        assert len(order_manager.order_history) == 1

        cancelled_order = order_manager.order_history[0]
        assert cancelled_order.status == OrderStatus.CANCELLED
        assert hasattr(cancelled_order, 'cancelled_at')

    @pytest.mark.asyncio
    async def test_should_fail_to_cancel_nonexistent_order(self, order_manager):
        """존재하지 않는 주문 취소 시 실패해야 함"""
        # When
        success = await order_manager.cancel_order("nonexistent-id")

        # Then
        assert success is False

    @pytest.mark.asyncio
    async def test_should_fail_to_cancel_already_filled_order(self, order_manager, sample_order):
        """이미 체결된 주문 취소 시 실패해야 함"""
        # Given
        order_id = await order_manager.submit_order(sample_order)
        await order_manager.update_order_status(
            order_id, Decimal('2.0'), Decimal('50000.0')
        )

        # When
        success = await order_manager.cancel_order(order_id)

        # Then
        assert success is False

    @pytest.mark.asyncio
    async def test_should_update_order_status_for_partial_fill(self, order_manager, sample_order):
        """부분 체결에 대한 주문 상태를 업데이트해야 함"""
        # Given
        order_id = await order_manager.submit_order(sample_order)

        # When
        await order_manager.update_order_status(
            order_id, Decimal('1.0'), Decimal('50000.0')
        )

        # Then
        assert order_id in order_manager.active_orders
        order_info = order_manager.active_orders[order_id]
        assert order_info.status == OrderStatus.PARTIALLY_FILLED
        assert order_info.filled_qty == Decimal('1.0')
        assert order_info.avg_price == Decimal('50000.0')

    @pytest.mark.asyncio
    async def test_should_complete_order_on_full_fill(self, order_manager, sample_order):
        """완전 체결 시 주문을 완료해야 함"""
        # Given
        order_id = await order_manager.submit_order(sample_order)

        # When
        await order_manager.update_order_status(
            order_id, Decimal('2.0'), Decimal('50001.0')
        )

        # Then
        assert order_id not in order_manager.active_orders
        assert len(order_manager.order_history) == 1

        completed_order = order_manager.order_history[0]
        assert completed_order.status == OrderStatus.FILLED
        assert completed_order.filled_qty == Decimal('2.0')
        assert completed_order.avg_price == Decimal('50001.0')
        assert hasattr(completed_order, 'filled_at')

    @pytest.mark.asyncio
    async def test_should_handle_overfill_gracefully(self, order_manager, sample_order):
        """초과 체결을 우아하게 처리해야 함"""
        # Given
        order_id = await order_manager.submit_order(sample_order)

        # When - report more filled than order size
        await order_manager.update_order_status(
            order_id, Decimal('2.5'), Decimal('50000.0')
        )

        # Then
        assert order_id not in order_manager.active_orders
        completed_order = order_manager.order_history[0]
        assert completed_order.status == OrderStatus.FILLED
        assert completed_order.filled_qty == Decimal('2.5')  # Record actual fill

    @pytest.mark.asyncio
    async def test_should_check_and_handle_stale_orders(self, order_manager, sample_order):
        """오래된 주문을 확인하고 처리해야 함"""
        # Given
        order_manager.max_order_age = 1  # 1 second for testing
        order_id = await order_manager.submit_order(sample_order)

        # Wait for order to become stale
        await asyncio.sleep(1.1)

        # When
        stale_count = await order_manager.check_stale_orders()

        # Then
        assert stale_count == 1
        assert order_id not in order_manager.active_orders
        assert len(order_manager.order_history) == 1

        stale_order = order_manager.order_history[0]
        assert stale_order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_should_track_order_attempts_and_retry_logic(self, order_manager, sample_order):
        """주문 시도 횟수를 추적하고 재시도 로직을 적용해야 함"""
        # Given
        order_id = await order_manager.submit_order(sample_order)

        # When
        await order_manager.increment_attempts(order_id)
        await order_manager.increment_attempts(order_id)

        # Then
        order_info = order_manager.active_orders[order_id]
        assert order_info.attempts == 2

    @pytest.mark.asyncio
    async def test_should_reject_order_after_max_attempts(self, order_manager, sample_order):
        """최대 시도 횟수 후 주문을 거부해야 함"""
        # Given
        order_manager.max_attempts = 3
        order_id = await order_manager.submit_order(sample_order)

        # When
        for _ in range(4):  # Exceed max attempts
            await order_manager.increment_attempts(order_id)

        # Then
        assert order_id not in order_manager.active_orders
        rejected_order = order_manager.order_history[0]
        assert rejected_order.status == OrderStatus.REJECTED

    def test_should_get_order_status_correctly(self, order_manager, sample_order):
        """주문 상태를 올바르게 반환해야 함"""
        # Given - submit order synchronously for this test
        order_info = OrderInfo(
            id="test-id",
            order=sample_order,
            status=OrderStatus.PENDING,
            submitted_at=datetime.now()
        )
        order_manager.active_orders["test-id"] = order_info

        # When
        status = order_manager.get_order_status("test-id")

        # Then
        assert status == OrderStatus.PENDING

        # Test nonexistent order
        assert order_manager.get_order_status("nonexistent") is None

    def test_should_get_order_info_correctly(self, order_manager, sample_order):
        """주문 정보를 올바르게 반환해야 함"""
        # Given
        order_info = OrderInfo(
            id="test-id",
            order=sample_order,
            status=OrderStatus.PARTIALLY_FILLED,
            submitted_at=datetime.now(),
            filled_qty=Decimal('1.0'),
            avg_price=Decimal('50000.0')
        )
        order_manager.active_orders["test-id"] = order_info

        # When
        retrieved_info = order_manager.get_order_info("test-id")

        # Then
        assert retrieved_info == order_info
        assert retrieved_info.filled_qty == Decimal('1.0')
        assert retrieved_info.avg_price == Decimal('50000.0')

    def test_should_calculate_order_statistics_correctly(self, order_manager):
        """주문 통계를 올바르게 계산해야 함"""
        # Given - setup some order history
        completed_orders = [
            OrderInfo("id1", None, OrderStatus.FILLED, datetime.now(),
                     filled_qty=Decimal('10.0'), avg_price=Decimal('50000.0')),
            OrderInfo("id2", None, OrderStatus.FILLED, datetime.now(),
                     filled_qty=Decimal('5.0'), avg_price=Decimal('51000.0')),
            OrderInfo("id3", None, OrderStatus.CANCELLED, datetime.now()),
        ]
        order_manager.order_history.extend(completed_orders)

        # When
        stats = order_manager.get_order_statistics()

        # Then
        assert stats['total_orders'] == 3
        assert stats['filled_orders'] == 2
        assert stats['cancelled_orders'] == 1
        assert stats['fill_rate'] == 2/3
        assert stats['total_volume'] == Decimal('15.0')
        assert 'average_price' in stats

    @pytest.mark.asyncio
    async def test_should_handle_concurrent_order_operations(self, order_manager, sample_order):
        """동시 주문 작업을 처리해야 함"""
        # When - submit multiple orders concurrently
        tasks = [
            order_manager.submit_order(sample_order) for _ in range(5)
        ]
        order_ids = await asyncio.gather(*tasks)

        # Then
        assert len(order_ids) == 5
        assert len(set(order_ids)) == 5  # All unique
        assert len(order_manager.active_orders) == 5

    @pytest.mark.asyncio
    async def test_should_maintain_order_priority_queue(self, order_manager):
        """주문 우선순위 큐를 유지해야 함"""
        # Given
        urgent_order = Order("BTCUSDT", OrderSide.BUY, Decimal("1.0"), OrderUrgency.IMMEDIATE)
        normal_order = Order("BTCUSDT", OrderSide.BUY, Decimal("1.0"), OrderUrgency.MEDIUM)
        low_order = Order("BTCUSDT", OrderSide.BUY, Decimal("1.0"), OrderUrgency.LOW)

        # When
        id1 = await order_manager.submit_order(normal_order)
        id2 = await order_manager.submit_order(urgent_order)
        id3 = await order_manager.submit_order(low_order)

        # Then
        priority_queue = order_manager.get_orders_by_priority()
        assert len(priority_queue) == 3
        assert priority_queue[0][1] == id2  # Urgent order first
        assert priority_queue[-1][1] == id3  # Low priority last

    @pytest.mark.asyncio
    async def test_should_provide_order_performance_metrics(self, order_manager, sample_order):
        """주문 성능 지표를 제공해야 함"""
        # Given
        order_id = await order_manager.submit_order(sample_order)

        # Simulate execution time
        start_time = datetime.now()
        await asyncio.sleep(0.1)
        await order_manager.update_order_status(order_id, Decimal('2.0'), Decimal('50000.0'))

        # When
        metrics = order_manager.get_performance_metrics()

        # Then
        assert 'average_execution_time' in metrics
        assert 'fill_rate' in metrics
        assert 'success_rate' in metrics
        assert metrics['total_processed'] >= 1

    @pytest.mark.asyncio
    async def test_should_handle_order_modifications(self, order_manager, sample_order):
        """주문 수정을 처리해야 함"""
        # Given
        order_id = await order_manager.submit_order(sample_order)

        # When
        success = await order_manager.modify_order(
            order_id,
            new_size=Decimal('3.0'),
            new_price=Decimal('49000.0')
        )

        # Then
        assert success is True
        order_info = order_manager.active_orders[order_id]
        assert order_info.order.size == Decimal('3.0')
        assert order_info.order.price == Decimal('49000.0')

    @pytest.mark.asyncio
    async def test_should_cleanup_completed_orders_automatically(self, order_manager, sample_order):
        """완료된 주문을 자동으로 정리해야 함"""
        # Given
        order_manager.max_history_size = 2

        # Submit and complete orders
        for i in range(3):
            order_id = await order_manager.submit_order(sample_order)
            await order_manager.update_order_status(order_id, Decimal('2.0'), Decimal('50000.0'))

        # When
        await order_manager.cleanup_old_orders()

        # Then
        assert len(order_manager.order_history) <= 2

    def test_should_validate_order_before_submission(self, order_manager):
        """제출 전 주문을 검증해야 함"""
        # When & Then - test empty symbol
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            Order(
                symbol="",  # Empty symbol
                side=OrderSide.BUY,
                size=Decimal('1.0')
            )

        # When & Then - test zero size
        with pytest.raises(ValueError, match="Size must be positive"):
            Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                size=Decimal('0')  # Zero size
            )

    @pytest.mark.asyncio
    async def test_should_handle_order_timeout_scenarios(self, order_manager, sample_order):
        """주문 타임아웃 시나리오를 처리해야 함"""
        # Given
        order_id = await order_manager.submit_order(sample_order)

        # When
        await order_manager.handle_order_timeout(order_id)

        # Then
        assert order_id not in order_manager.active_orders
        timeout_order = order_manager.order_history[0]
        assert timeout_order.status == OrderStatus.CANCELLED
        assert hasattr(timeout_order, 'timeout_reason')