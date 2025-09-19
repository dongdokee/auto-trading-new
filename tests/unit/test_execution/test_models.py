# tests/unit/test_execution/test_models.py
import pytest
from decimal import Decimal
from datetime import datetime
from typing import Optional
from src.execution.models import Order, ExecutionResult, OrderStatus, OrderSide, OrderUrgency


class TestOrder:
    """Order 데이터 클래스에 대한 TDD 테스트"""

    def test_should_create_valid_order_with_required_fields(self):
        """필수 필드로 유효한 주문이 생성되어야 함"""
        # Given
        symbol = "BTCUSDT"
        side = OrderSide.BUY
        size = Decimal("1.0")

        # When
        order = Order(symbol=symbol, side=side, size=size)

        # Then
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.size == Decimal("1.0")
        assert order.urgency == OrderUrgency.MEDIUM  # default
        assert order.price is None  # default
        assert order.order_id is not None  # auto-generated
        assert isinstance(order.created_at, datetime)

    def test_should_create_order_with_custom_urgency_and_price(self):
        """커스텀 긴급도와 가격으로 주문 생성이 가능해야 함"""
        # Given
        symbol = "ETHUSDT"
        side = OrderSide.SELL
        size = Decimal("2.5")
        urgency = OrderUrgency.HIGH
        price = Decimal("3000.50")

        # When
        order = Order(
            symbol=symbol,
            side=side,
            size=size,
            urgency=urgency,
            price=price
        )

        # Then
        assert order.urgency == OrderUrgency.HIGH
        assert order.price == Decimal("3000.50")

    def test_should_reject_invalid_symbol_format(self):
        """잘못된 심볼 형식을 거부해야 함"""
        # Given
        invalid_symbol = "btc-usdt"  # lowercase and dash

        # When & Then
        with pytest.raises(ValueError, match="Symbol must be uppercase"):
            Order(symbol=invalid_symbol, side=OrderSide.BUY, size=Decimal("1.0"))

    def test_should_reject_zero_or_negative_size(self):
        """0 또는 음수 크기를 거부해야 함"""
        # Given
        symbol = "BTCUSDT"
        side = OrderSide.BUY

        # When & Then
        with pytest.raises(ValueError, match="Size must be positive"):
            Order(symbol=symbol, side=side, size=Decimal("0"))

        with pytest.raises(ValueError, match="Size must be positive"):
            Order(symbol=symbol, side=side, size=Decimal("-1.0"))

    def test_should_reject_negative_price(self):
        """음수 가격을 거부해야 함"""
        # Given
        symbol = "BTCUSDT"
        side = OrderSide.BUY
        size = Decimal("1.0")
        price = Decimal("-100.0")

        # When & Then
        with pytest.raises(ValueError, match="Price must be positive"):
            Order(symbol=symbol, side=side, size=size, price=price)

    def test_should_generate_unique_order_ids(self):
        """고유한 주문 ID가 생성되어야 함"""
        # Given & When
        order1 = Order(symbol="BTCUSDT", side=OrderSide.BUY, size=Decimal("1.0"))
        order2 = Order(symbol="BTCUSDT", side=OrderSide.BUY, size=Decimal("1.0"))

        # Then
        assert order1.order_id != order2.order_id
        assert len(order1.order_id) > 0
        assert len(order2.order_id) > 0

    def test_should_calculate_notional_value_correctly(self):
        """명목 가치를 올바르게 계산해야 함"""
        # Given
        order_with_price = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("2.0"),
            price=Decimal("50000.0")
        )
        order_without_price = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal("2.0")
        )

        # When & Then
        assert order_with_price.notional_value == Decimal("100000.0")
        assert order_without_price.notional_value is None


class TestExecutionResult:
    """ExecutionResult 데이터 클래스에 대한 TDD 테스트"""

    def test_should_create_valid_execution_result(self):
        """유효한 실행 결과가 생성되어야 함"""
        # Given
        order_id = "test-order-123"
        strategy = "AGGRESSIVE"
        total_filled = Decimal("1.5")
        avg_price = Decimal("50000.0")

        # When
        result = ExecutionResult(
            order_id=order_id,
            strategy=strategy,
            total_filled=total_filled,
            avg_price=avg_price
        )

        # Then
        assert result.order_id == "test-order-123"
        assert result.strategy == "AGGRESSIVE"
        assert result.total_filled == Decimal("1.5")
        assert result.avg_price == Decimal("50000.0")
        assert result.total_cost == Decimal("0")  # default
        assert result.slices == []  # default
        assert isinstance(result.execution_time, datetime)

    def test_should_calculate_total_value_correctly(self):
        """총 거래 가치를 올바르게 계산해야 함"""
        # Given
        result = ExecutionResult(
            order_id="test-order-123",
            strategy="TWAP",
            total_filled=Decimal("2.0"),
            avg_price=Decimal("3000.0")
        )

        # When & Then
        assert result.total_value == Decimal("6000.0")

    def test_should_calculate_fill_rate_correctly(self):
        """체결률을 올바르게 계산해야 함"""
        # Given
        original_size = Decimal("10.0")
        result = ExecutionResult(
            order_id="test-order-123",
            strategy="PASSIVE",
            total_filled=Decimal("7.5"),
            avg_price=Decimal("1000.0"),
            original_size=original_size
        )

        # When & Then
        assert result.fill_rate == Decimal("0.75")  # 75%

    def test_should_handle_zero_fill_gracefully(self):
        """체결량이 0인 경우를 우아하게 처리해야 함"""
        # Given
        result = ExecutionResult(
            order_id="test-order-123",
            strategy="PASSIVE",
            total_filled=Decimal("0"),
            avg_price=Decimal("0"),
            original_size=Decimal("5.0")
        )

        # When & Then
        assert result.total_value == Decimal("0")
        assert result.fill_rate == Decimal("0")

    def test_should_validate_execution_result_fields(self):
        """실행 결과 필드 검증이 올바르게 작동해야 함"""
        # Given & When & Then
        with pytest.raises(ValueError, match="Total filled cannot be negative"):
            ExecutionResult(
                order_id="test",
                strategy="TEST",
                total_filled=Decimal("-1.0"),
                avg_price=Decimal("1000.0")
            )

        with pytest.raises(ValueError, match="Average price cannot be negative"):
            ExecutionResult(
                order_id="test",
                strategy="TEST",
                total_filled=Decimal("1.0"),
                avg_price=Decimal("-1000.0")
            )


class TestOrderEnums:
    """주문 관련 Enum 클래스 테스트"""

    def test_order_side_enum_values(self):
        """OrderSide enum 값들이 올바르게 정의되어야 함"""
        assert OrderSide.BUY == "BUY"
        assert OrderSide.SELL == "SELL"

    def test_order_urgency_enum_values(self):
        """OrderUrgency enum 값들이 올바르게 정의되어야 함"""
        assert OrderUrgency.LOW == "LOW"
        assert OrderUrgency.MEDIUM == "MEDIUM"
        assert OrderUrgency.HIGH == "HIGH"
        assert OrderUrgency.IMMEDIATE == "IMMEDIATE"

    def test_order_status_enum_values(self):
        """OrderStatus enum 값들이 올바르게 정의되어야 함"""
        assert OrderStatus.PENDING == "PENDING"
        assert OrderStatus.PARTIALLY_FILLED == "PARTIALLY_FILLED"
        assert OrderStatus.FILLED == "FILLED"
        assert OrderStatus.CANCELLED == "CANCELLED"
        assert OrderStatus.REJECTED == "REJECTED"