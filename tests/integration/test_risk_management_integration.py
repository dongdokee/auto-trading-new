# tests/unit/test_risk_management/test_integration.py
"""
Risk Management Module Integration Tests
Tests the interaction between RiskController, PositionSizer, and PositionManager
"""
import pytest
import numpy as np
from src.risk_management.risk_management import RiskController
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.position_management import PositionManager


class TestRiskManagementIntegration:
    """Risk Management 모듈 통합 테스트"""

    @pytest.fixture
    def risk_controller(self):
        """테스트용 RiskController"""
        return RiskController(
            initial_capital_usdt=10000.0,
            var_daily_pct=0.02,
            max_drawdown_pct=0.12,
            max_leverage=5.0,
            allow_short=True
        )

    @pytest.fixture
    def position_sizer(self, risk_controller):
        """테스트용 PositionSizer"""
        return PositionSizer(risk_controller)

    @pytest.fixture
    def position_manager(self, risk_controller):
        """테스트용 PositionManager"""
        return PositionManager(risk_controller)

    @pytest.fixture
    def sample_market_data(self):
        """샘플 마켓 데이터"""
        return {
            'BTCUSDT': {
                'price': 50000.0,
                'atr': 2000.0,
                'daily_volatility': 0.05,
                'regime': 'NEUTRAL',
                'min_notional': 10.0,
                'lot_size': 0.001,
                'symbol_leverage': 10
            },
            'ETHUSDT': {
                'price': 3000.0,
                'atr': 150.0,
                'daily_volatility': 0.06,
                'regime': 'BULL',
                'min_notional': 10.0,
                'lot_size': 0.001,
                'symbol_leverage': 10
            }
        }

    @pytest.fixture
    def sample_portfolio_state(self):
        """샘플 포트폴리오 상태"""
        np.random.seed(42)
        recent_returns = np.random.normal(0.001, 0.02, 35)

        return {
            'equity': 10000.0,
            'recent_returns': recent_returns,
            'positions': [],
            'current_var_usdt': 0.0,
            'symbol_volatilities': {
                'BTCUSDT': 0.05,
                'ETHUSDT': 0.06
            },
            'correlation_matrix': {
                ('BTCUSDT', 'ETHUSDT'): 0.75,
                ('ETHUSDT', 'BTCUSDT'): 0.75
            }
        }

    def test_complete_position_workflow(self, position_sizer, position_manager,
                                       sample_market_data, sample_portfolio_state):
        """전체 포지션 워크플로우 테스트: 사이징 → 오픈 → 업데이트 → 클로즈"""
        # Given - Trading signal for BTC
        signal = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'strength': 0.8,
            'confidence': 0.7
        }

        market_state = sample_market_data['BTCUSDT']

        # Step 1: Calculate optimal position size
        position_size = position_sizer.calculate_position_size(
            signal, market_state, sample_portfolio_state
        )

        assert position_size > 0
        assert position_size < 1.0  # Should be reasonable size (< 1 BTC)

        # Step 2: Open position with calculated size
        position = position_manager.open_position(
            symbol='BTCUSDT',
            side='LONG',
            size=position_size,
            price=market_state['price'],
            leverage=5.0
        )

        assert position is not None
        assert position['size'] == position_size
        assert position['unrealized_pnl'] == 0.0

        # Step 3: Update position with new price (profit scenario)
        new_price = 52000.0  # +4% price increase
        updated_position = position_manager.update_position('BTCUSDT', new_price)

        expected_profit = (new_price - market_state['price']) * position_size
        assert abs(updated_position['unrealized_pnl'] - expected_profit) < 1e-10
        assert updated_position['current_price'] == new_price

        # Step 4: Close position
        close_price = 53000.0  # +6% from entry
        closed_position = position_manager.close_position('BTCUSDT', close_price, 'TAKE_PROFIT')

        expected_final_profit = (close_price - market_state['price']) * position_size
        assert abs(closed_position['realized_pnl'] - expected_final_profit) < 1e-10
        assert closed_position['close_reason'] == 'TAKE_PROFIT'
        assert 'BTCUSDT' not in position_manager.positions

    def test_multi_asset_portfolio_sizing(self, position_sizer, sample_market_data,
                                         sample_portfolio_state):
        """다중 자산 포트폴리오 사이징 테스트"""
        # First position: BTC
        btc_signal = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'strength': 0.8,
            'confidence': 0.7
        }

        btc_size = position_sizer.calculate_position_size(
            btc_signal, sample_market_data['BTCUSDT'], sample_portfolio_state
        )

        # Add BTC position to portfolio state
        sample_portfolio_state['positions'] = [
            {
                'symbol': 'BTCUSDT',
                'size': btc_size,
                'notional': btc_size * sample_market_data['BTCUSDT']['price']
            }
        ]

        # Update VaR usage
        btc_var = btc_size * sample_market_data['BTCUSDT']['price'] * 0.05 * 1.65
        sample_portfolio_state['current_var_usdt'] = btc_var

        # Second position: ETH (correlated with BTC)
        eth_signal = {
            'symbol': 'ETHUSDT',
            'side': 'LONG',
            'strength': 0.9,
            'confidence': 0.8
        }

        eth_size = position_sizer.calculate_position_size(
            eth_signal, sample_market_data['ETHUSDT'], sample_portfolio_state
        )

        # ETH size should be smaller due to correlation with existing BTC position
        correlation_factor = position_sizer._calculate_correlation_adjustment(
            'ETHUSDT', sample_portfolio_state
        )

        assert correlation_factor < 1.0  # Should be reduced due to correlation
        assert eth_size > 0  # But still positive
        assert isinstance(eth_size, float)

    def test_risk_limit_enforcement_in_sizing(self, risk_controller, position_sizer,
                                            sample_market_data, sample_portfolio_state):
        """포지션 사이징에서 리스크 한도 강제 적용 테스트"""
        # Exhaust most of the VaR budget
        sample_portfolio_state['current_var_usdt'] = 180.0  # Out of 200 USDT VaR limit

        signal = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'strength': 1.0,  # Maximum strength
            'confidence': 0.9
        }

        # Calculate position size with limited VaR budget
        position_size = position_sizer.calculate_position_size(
            signal, sample_market_data['BTCUSDT'], sample_portfolio_state
        )

        # Verify VaR constraint is binding
        var_constrained_size = position_sizer._calculate_var_constrained_size(
            'BTCUSDT', sample_market_data['BTCUSDT'], sample_portfolio_state
        )

        # Position size should be limited by VaR constraint
        assert position_size <= var_constrained_size + 1e-10

    def test_leverage_limit_interaction(self, risk_controller, position_sizer,
                                       sample_market_data, sample_portfolio_state):
        """레버리지 한도와 포지션 사이징 상호작용 테스트"""
        # Add existing high leverage position to portfolio
        existing_leverage = 4.0
        existing_notional = sample_portfolio_state['equity'] * existing_leverage

        sample_portfolio_state['positions'] = [
            {
                'symbol': 'ETHUSDT',
                'notional': existing_notional,
                'size': existing_notional / sample_market_data['ETHUSDT']['price']
            }
        ]

        # Check leverage limit
        leverage_violations = risk_controller.check_leverage_limit(sample_portfolio_state)
        assert len(leverage_violations) == 0  # Should be within 5x limit

        # Try to add more leverage
        signal = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'strength': 1.0,
            'confidence': 0.9
        }

        position_size = position_sizer.calculate_position_size(
            signal, sample_market_data['BTCUSDT'], sample_portfolio_state
        )

        # Calculate what total leverage would be with new position
        new_notional = position_size * sample_market_data['BTCUSDT']['price']
        total_notional = existing_notional + new_notional
        total_leverage = total_notional / sample_portfolio_state['equity']

        # Should respect overall leverage limit
        assert total_leverage <= risk_controller.risk_limits['max_leverage'] + 0.1

    def test_drawdown_tracking_with_positions(self, risk_controller, position_manager):
        """포지션과 함께 드로다운 추적 테스트"""
        # Open a position
        position = position_manager.open_position(
            symbol='BTCUSDT',
            side='LONG',
            size=0.1,
            price=50000.0,
            leverage=2.0
        )

        initial_equity = 10000.0

        # Simulate price drop causing loss
        new_price = 45000.0  # -10% price drop
        position_manager.update_position('BTCUSDT', new_price)

        # Calculate new equity (position loss)
        position_loss = (new_price - 50000.0) * 0.1  # -500 USDT
        current_equity = initial_equity + position_loss  # 9500 USDT

        # Update drawdown
        current_drawdown = risk_controller.update_drawdown(current_equity)

        expected_drawdown = (10000.0 - 9500.0) / 10000.0  # 5%
        assert abs(current_drawdown - expected_drawdown) < 1e-10

        # Check drawdown severity
        severity = risk_controller.get_drawdown_severity_level()
        assert severity == 'MODERATE'  # 5% should be MODERATE (5-10%)

    def test_position_sizing_respects_all_constraints(self, position_sizer,
                                                     sample_market_data, sample_portfolio_state):
        """포지션 사이징이 모든 제약 조건을 준수하는지 테스트"""
        signal = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'strength': 1.0,
            'confidence': 0.9
        }

        position_size = position_sizer.calculate_position_size(
            signal, sample_market_data['BTCUSDT'], sample_portfolio_state
        )

        market_state = sample_market_data['BTCUSDT']
        price = market_state['price']
        equity = sample_portfolio_state['equity']

        # Test all constraints are respected
        # 1. Minimum notional
        min_notional = market_state['min_notional']
        assert position_size * price >= min_notional

        # 2. Maximum position limit (20% of equity)
        max_position_value = equity * 0.2
        assert position_size * price <= max_position_value + 1e-6

        # 3. Lot size compliance
        lot_size = market_state['lot_size']
        remainder = position_size % lot_size
        assert abs(remainder) < 1e-10  # Should be properly rounded

        # 4. Position size should be positive
        assert position_size > 0