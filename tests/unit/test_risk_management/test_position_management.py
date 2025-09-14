# tests/unit/test_risk_management/test_position_management.py
import pytest
import numpy as np
from datetime import datetime, timedelta
from src.risk_management.position_management import PositionManager
from src.risk_management.risk_management import RiskController


class TestPositionManager:
    """PositionManager 클래스에 대한 TDD 테스트"""

    @pytest.fixture
    def risk_controller(self):
        """테스트용 RiskController 인스턴스"""
        return RiskController(initial_capital_usdt=10000.0)

    @pytest.fixture
    def position_manager(self, risk_controller):
        """테스트용 PositionManager 인스턴스"""
        return PositionManager(risk_controller)

    def test_should_initialize_with_risk_controller(self, risk_controller):
        """PositionManager가 RiskController와 함께 올바르게 초기화되어야 함"""
        # When
        position_manager = PositionManager(risk_controller)

        # Then
        assert position_manager.risk_controller == risk_controller
        assert position_manager.positions == {}
        assert hasattr(position_manager, 'open_position')
        assert hasattr(position_manager, 'close_position')

    def test_should_open_position_correctly(self, position_manager):
        """포지션을 올바르게 오픈해야 함"""
        # Given
        symbol = 'BTCUSDT'
        side = 'LONG'
        size = 0.1  # 0.1 BTC
        price = 50000.0  # 50,000 USDT
        leverage = 5.0

        # When
        position = position_manager.open_position(symbol, side, size, price, leverage)

        # Then
        assert position is not None
        assert position['symbol'] == symbol
        assert position['side'] == side
        assert position['size'] == size
        assert position['entry_price'] == price
        assert position['current_price'] == price
        assert position['leverage'] == leverage
        assert position['margin'] == (size * price) / leverage  # 1000 USDT
        assert position['unrealized_pnl'] == 0
        assert position['realized_pnl'] == 0
        assert 'liquidation_price' in position
        assert 'open_time' in position

        # Position should be stored in manager
        assert symbol in position_manager.positions
        assert position_manager.positions[symbol] == position

    def test_should_calculate_liquidation_price_for_long_position(self, position_manager):
        """롱 포지션의 청산가를 올바르게 계산해야 함"""
        # Given
        symbol = 'BTCUSDT'
        entry_price = 50000.0
        leverage = 10.0  # 10x leverage
        mmr = 0.005  # 0.5% maintenance margin

        # Expected liquidation price for LONG
        # liquidation_price = entry_price * (1 - 1/leverage + mmr)
        expected_liquidation = entry_price * (1 - 1/leverage + mmr)  # 50000 * (1 - 0.1 + 0.005) = 45250

        # When
        position = position_manager.open_position(symbol, 'LONG', 0.1, entry_price, leverage)

        # Then
        assert abs(position['liquidation_price'] - expected_liquidation) < 1.0

    def test_should_calculate_liquidation_price_for_short_position(self, position_manager):
        """숏 포지션의 청산가를 올바르게 계산해야 함"""
        # Given
        symbol = 'BTCUSDT'
        entry_price = 50000.0
        leverage = 10.0  # 10x leverage
        mmr = 0.005  # 0.5% maintenance margin

        # Expected liquidation price for SHORT
        # liquidation_price = entry_price * (1 + 1/leverage - mmr)
        expected_liquidation = entry_price * (1 + 1/leverage - mmr)  # 50000 * (1 + 0.1 - 0.005) = 54750

        # When
        position = position_manager.open_position(symbol, 'SHORT', 0.1, entry_price, leverage)

        # Then
        assert abs(position['liquidation_price'] - expected_liquidation) < 1.0

    def test_should_update_position_pnl_correctly(self, position_manager):
        """포지션 PnL을 올바르게 업데이트해야 함"""
        # Given - LONG position
        symbol = 'BTCUSDT'
        entry_price = 50000.0
        size = 0.1
        current_price = 52000.0  # Price increased by 2000
        expected_pnl = (current_price - entry_price) * size  # (52000 - 50000) * 0.1 = 200 USDT

        # Open position
        position_manager.open_position(symbol, 'LONG', size, entry_price, 5.0)

        # When
        updated_position = position_manager.update_position(symbol, current_price)

        # Then
        assert updated_position is not None
        assert updated_position['current_price'] == current_price
        assert abs(updated_position['unrealized_pnl'] - expected_pnl) < 1e-10

    def test_should_update_short_position_pnl_correctly(self, position_manager):
        """숏 포지션 PnL을 올바르게 업데이트해야 함"""
        # Given - SHORT position
        symbol = 'BTCUSDT'
        entry_price = 50000.0
        size = 0.1
        current_price = 48000.0  # Price decreased by 2000
        expected_pnl = (entry_price - current_price) * size  # (50000 - 48000) * 0.1 = 200 USDT profit

        # Open position
        position_manager.open_position(symbol, 'SHORT', size, entry_price, 5.0)

        # When
        updated_position = position_manager.update_position(symbol, current_price)

        # Then
        assert updated_position is not None
        assert updated_position['current_price'] == current_price
        assert abs(updated_position['unrealized_pnl'] - expected_pnl) < 1e-10

    def test_should_return_none_for_nonexistent_position_update(self, position_manager):
        """존재하지 않는 포지션 업데이트 시 None을 반환해야 함"""
        # When
        result = position_manager.update_position('NONEXISTENT', 50000.0)

        # Then
        assert result is None

    def test_should_close_position_correctly(self, position_manager):
        """포지션을 올바르게 닫아야 함"""
        # Given
        symbol = 'BTCUSDT'
        entry_price = 50000.0
        close_price = 52000.0
        size = 0.1
        expected_pnl = (close_price - entry_price) * size  # 200 USDT profit

        # Open position first
        position_manager.open_position(symbol, 'LONG', size, entry_price, 5.0)

        # When
        closed_position = position_manager.close_position(symbol, close_price, 'MANUAL')

        # Then
        assert closed_position is not None
        assert closed_position['realized_pnl'] == expected_pnl
        assert closed_position['close_price'] == close_price
        assert closed_position['close_reason'] == 'MANUAL'
        assert 'close_time' in closed_position

        # Position should be removed from manager
        assert symbol not in position_manager.positions

    def test_should_return_none_for_nonexistent_position_close(self, position_manager):
        """존재하지 않는 포지션 닫기 시 None을 반환해야 함"""
        # When
        result = position_manager.close_position('NONEXISTENT', 50000.0)

        # Then
        assert result is None

    def test_should_detect_stop_loss_condition(self, position_manager):
        """스톱로스 조건을 감지해야 함"""
        # Given
        symbol = 'BTCUSDT'
        entry_price = 50000.0
        stop_loss = 48000.0
        current_price = 47500.0  # Below stop loss

        # Open position with stop loss
        position = position_manager.open_position(symbol, 'LONG', 0.1, entry_price, 5.0)
        position['stop_loss'] = stop_loss

        # When
        stop_reason = position_manager.check_stop_conditions(symbol, current_price)

        # Then
        assert stop_reason == 'STOP_LOSS'

    def test_should_detect_take_profit_condition(self, position_manager):
        """테이크프로핏 조건을 감지해야 함"""
        # Given
        symbol = 'BTCUSDT'
        entry_price = 50000.0
        take_profit = 55000.0
        current_price = 55500.0  # Above take profit

        # Open position with take profit
        position = position_manager.open_position(symbol, 'LONG', 0.1, entry_price, 5.0)
        position['take_profit'] = take_profit

        # When
        stop_reason = position_manager.check_stop_conditions(symbol, current_price)

        # Then
        assert stop_reason == 'TAKE_PROFIT'

    def test_should_return_none_for_no_stop_conditions(self, position_manager):
        """스톱 조건이 없을 때 None을 반환해야 함"""
        # Given
        symbol = 'BTCUSDT'
        position_manager.open_position(symbol, 'LONG', 0.1, 50000.0, 5.0)

        # When
        stop_reason = position_manager.check_stop_conditions(symbol, 51000.0)

        # Then
        assert stop_reason is None

    def test_should_update_trailing_stop_for_long_position(self, position_manager):
        """롱 포지션의 트레일링 스톱을 업데이트해야 함"""
        # Given
        symbol = 'BTCUSDT'
        entry_price = 50000.0
        trailing_distance = 0.02  # 2%

        # Open position and set trailing stop
        position = position_manager.open_position(symbol, 'LONG', 0.1, entry_price, 5.0)
        position['trailing_stop'] = None
        position['trailing_distance'] = trailing_distance

        # When - Price goes up, trailing stop should move up
        new_price = 52000.0
        position_manager.update_position(symbol, new_price)

        # Then
        expected_trailing_stop = new_price * (1 - trailing_distance)  # 52000 * 0.98 = 50960
        assert abs(position['trailing_stop'] - expected_trailing_stop) < 1.0

    def test_should_not_lower_trailing_stop_for_long_position(self, position_manager):
        """롱 포지션의 트레일링 스톱이 하향 이동하지 않아야 함"""
        # Given
        symbol = 'BTCUSDT'
        entry_price = 50000.0
        trailing_distance = 0.02  # 2%

        # Open position and set initial trailing stop
        position = position_manager.open_position(symbol, 'LONG', 0.1, entry_price, 5.0)
        position['trailing_stop'] = 51000.0  # Already high
        position['trailing_distance'] = trailing_distance

        # When - Price goes down, trailing stop should NOT move down
        lower_price = 50500.0
        position_manager.update_position(symbol, lower_price)

        # Then - Trailing stop should remain at the higher level
        assert position['trailing_stop'] == 51000.0