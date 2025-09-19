# tests/fixtures/execution_fixtures.py
"""Test fixtures for execution module"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List
from src.execution.models import Order, ExecutionResult, OrderSide, OrderUrgency


@pytest.fixture
def sample_btc_order() -> Order:
    """Sample BTC buy order for testing"""
    return Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        size=Decimal("1.5"),
        urgency=OrderUrgency.MEDIUM,
        price=Decimal("50000.0")
    )


@pytest.fixture
def sample_eth_order() -> Order:
    """Sample ETH sell order for testing"""
    return Order(
        symbol="ETHUSDT",
        side=OrderSide.SELL,
        size=Decimal("10.0"),
        urgency=OrderUrgency.HIGH,
        price=Decimal("3000.0")
    )


@pytest.fixture
def urgent_order() -> Order:
    """Urgent order for immediate execution testing"""
    return Order(
        symbol="ADAUSDT",
        side=OrderSide.BUY,
        size=Decimal("1000.0"),
        urgency=OrderUrgency.IMMEDIATE
    )


@pytest.fixture
def large_order() -> Order:
    """Large order for testing order splitting"""
    return Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        size=Decimal("50.0"),  # Large size
        urgency=OrderUrgency.LOW,
        price=Decimal("50000.0")
    )


@pytest.fixture
def market_order() -> Order:
    """Market order without price"""
    return Order(
        symbol="ETHUSDT",
        side=OrderSide.SELL,
        size=Decimal("5.0"),
        urgency=OrderUrgency.IMMEDIATE
    )


@pytest.fixture
def sample_execution_result() -> ExecutionResult:
    """Sample execution result for testing"""
    return ExecutionResult(
        order_id="test-order-123",
        strategy="AGGRESSIVE",
        total_filled=Decimal("1.5"),
        avg_price=Decimal("50000.0"),
        total_cost=Decimal("30.0"),
        original_size=Decimal("1.5")
    )


@pytest.fixture
def partial_execution_result() -> ExecutionResult:
    """Partially filled execution result for testing"""
    return ExecutionResult(
        order_id="test-order-456",
        strategy="PASSIVE",
        total_filled=Decimal("7.5"),
        avg_price=Decimal("3000.0"),
        total_cost=Decimal("15.0"),
        original_size=Decimal("10.0")
    )


@pytest.fixture
def liquid_orderbook() -> Dict:
    """High liquidity orderbook for testing"""
    return {
        'symbol': 'BTCUSDT',
        'bids': [
            {'price': Decimal('50000.0'), 'size': Decimal('5.0')},
            {'price': Decimal('49999.0'), 'size': Decimal('8.0')},
            {'price': Decimal('49998.0'), 'size': Decimal('3.0')},
            {'price': Decimal('49997.0'), 'size': Decimal('12.0')},
            {'price': Decimal('49996.0'), 'size': Decimal('6.0')},
            {'price': Decimal('49995.0'), 'size': Decimal('4.0')},
            {'price': Decimal('49994.0'), 'size': Decimal('9.0')},
            {'price': Decimal('49993.0'), 'size': Decimal('2.0')},
            {'price': Decimal('49992.0'), 'size': Decimal('7.0')},
            {'price': Decimal('49991.0'), 'size': Decimal('5.0')},
        ],
        'asks': [
            {'price': Decimal('50001.0'), 'size': Decimal('4.0')},
            {'price': Decimal('50002.0'), 'size': Decimal('7.0')},
            {'price': Decimal('50003.0'), 'size': Decimal('2.0')},
            {'price': Decimal('50004.0'), 'size': Decimal('10.0')},
            {'price': Decimal('50005.0'), 'size': Decimal('6.0')},
            {'price': Decimal('50006.0'), 'size': Decimal('3.0')},
            {'price': Decimal('50007.0'), 'size': Decimal('8.0')},
            {'price': Decimal('50008.0'), 'size': Decimal('5.0')},
            {'price': Decimal('50009.0'), 'size': Decimal('9.0')},
            {'price': Decimal('50010.0'), 'size': Decimal('4.0')},
        ],
        'timestamp': 1640995200000
    }


@pytest.fixture
def thin_orderbook() -> Dict:
    """Thin liquidity orderbook for testing"""
    return {
        'symbol': 'ADAUSDT',
        'bids': [
            {'price': Decimal('1.2000'), 'size': Decimal('100.0')},
            {'price': Decimal('1.1950'), 'size': Decimal('50.0')},
            {'price': Decimal('1.1900'), 'size': Decimal('75.0')},
        ],
        'asks': [
            {'price': Decimal('1.2050'), 'size': Decimal('80.0')},
            {'price': Decimal('1.2100'), 'size': Decimal('60.0')},
            {'price': Decimal('1.2150'), 'size': Decimal('40.0')},
        ],
        'timestamp': 1640995300000
    }


@pytest.fixture
def wide_spread_orderbook() -> Dict:
    """Wide spread orderbook for testing"""
    return {
        'symbol': 'DOGEUSDT',
        'bids': [
            {'price': Decimal('0.1000'), 'size': Decimal('10000.0')},
            {'price': Decimal('0.0990'), 'size': Decimal('15000.0')},
            {'price': Decimal('0.0980'), 'size': Decimal('12000.0')},
        ],
        'asks': [
            {'price': Decimal('0.1100'), 'size': Decimal('8000.0')},
            {'price': Decimal('0.1110'), 'size': Decimal('12000.0')},
            {'price': Decimal('0.1120'), 'size': Decimal('9000.0')},
        ],
        'timestamp': 1640995400000
    }


@pytest.fixture
def imbalanced_orderbook() -> Dict:
    """Imbalanced orderbook (more bids than asks) for testing"""
    return {
        'symbol': 'ETHUSDT',
        'bids': [
            {'price': Decimal('3000.0'), 'size': Decimal('20.0')},
            {'price': Decimal('2999.0'), 'size': Decimal('25.0')},
            {'price': Decimal('2998.0'), 'size': Decimal('30.0')},
            {'price': Decimal('2997.0'), 'size': Decimal('15.0')},
            {'price': Decimal('2996.0'), 'size': Decimal('35.0')},
        ],
        'asks': [
            {'price': Decimal('3001.0'), 'size': Decimal('5.0')},
            {'price': Decimal('3002.0'), 'size': Decimal('8.0')},
            {'price': Decimal('3003.0'), 'size': Decimal('3.0')},
            {'price': Decimal('3004.0'), 'size': Decimal('6.0')},
            {'price': Decimal('3005.0'), 'size': Decimal('4.0')},
        ],
        'timestamp': 1640995500000
    }


@pytest.fixture
def market_conditions_high_volatility() -> Dict:
    """High volatility market conditions for testing"""
    return {
        'spread_bps': 15.0,
        'liquidity_score': 0.3,
        'avg_volume_1min': 1000.0,
        'best_bid': Decimal('50000.0'),
        'best_ask': Decimal('50075.0'),  # Wide spread
        'top_5_liquidity': 5000.0,
        'daily_volume': 50000.0,
        'volatility': 0.05,  # High volatility
        'imbalance': 0.1
    }


@pytest.fixture
def market_conditions_low_volatility() -> Dict:
    """Low volatility market conditions for testing"""
    return {
        'spread_bps': 2.0,
        'liquidity_score': 0.9,
        'avg_volume_1min': 10000.0,
        'best_bid': Decimal('50000.0'),
        'best_ask': Decimal('50001.0'),  # Tight spread
        'top_5_liquidity': 50000.0,
        'daily_volume': 500000.0,
        'volatility': 0.01,  # Low volatility
        'imbalance': 0.02
    }


@pytest.fixture
def market_conditions_illiquid() -> Dict:
    """Illiquid market conditions for testing"""
    return {
        'spread_bps': 50.0,
        'liquidity_score': 0.1,
        'avg_volume_1min': 100.0,
        'best_bid': Decimal('1.2000'),
        'best_ask': Decimal('1.2600'),  # Very wide spread
        'top_5_liquidity': 500.0,
        'daily_volume': 5000.0,
        'volatility': 0.08,  # High volatility due to illiquidity
        'imbalance': 0.3
    }


class MockExchangeInterface:
    """Mock exchange interface for testing"""

    def __init__(self):
        self.orders = {}
        self.order_counter = 0
        self.execution_delay = 0.01  # 10ms default delay

    async def place_order(self, **kwargs) -> Dict:
        """Mock order placement"""
        self.order_counter += 1
        order_id = f"mock-order-{self.order_counter}"

        # Simulate basic order execution
        filled_qty = kwargs.get('size', 0)
        avg_price = kwargs.get('price', 50000.0)
        commission = float(filled_qty) * float(avg_price) * 0.0004  # 0.04% fee

        result = {
            'order_id': order_id,
            'filled_qty': filled_qty,
            'avg_price': avg_price,
            'commission': commission,
            'status': 'FILLED'
        }

        self.orders[order_id] = result
        return result

    async def cancel_order(self, order_id: str) -> bool:
        """Mock order cancellation"""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELLED'
            return True
        return False

    async def get_order_status(self, order_id: str) -> Dict:
        """Mock order status query"""
        return self.orders.get(order_id, {'status': 'NOT_FOUND'})

    def reset(self):
        """Reset mock state"""
        self.orders.clear()
        self.order_counter = 0


@pytest.fixture
def mock_exchange():
    """Mock exchange interface fixture"""
    return MockExchangeInterface()


@pytest.fixture
def execution_test_data():
    """Comprehensive test data for execution testing"""
    return {
        'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT'],
        'order_sizes': [
            Decimal('0.1'),   # Small
            Decimal('1.0'),   # Medium
            Decimal('10.0'),  # Large
            Decimal('100.0')  # Very large
        ],
        'price_levels': [
            Decimal('1000.0'),
            Decimal('10000.0'),
            Decimal('50000.0')
        ],
        'urgency_levels': [
            OrderUrgency.LOW,
            OrderUrgency.MEDIUM,
            OrderUrgency.HIGH,
            OrderUrgency.IMMEDIATE
        ]
    }