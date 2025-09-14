# tests/unit/test_risk_management/test_risk_controller.py
import pytest
import numpy as np
from src.risk_management.risk_management import RiskController


class TestRiskController:
    """RiskController 클래스에 대한 TDD 테스트"""

    def test_should_initialize_with_correct_usdt_capital(self):
        """RiskController가 USDT 초기 자본으로 올바르게 초기화되어야 함"""
        # Given
        initial_capital = 10000.0  # 10,000 USDT

        # When
        risk_controller = RiskController(initial_capital)

        # Then
        assert risk_controller.initial_capital == 10000.0
        assert risk_controller.high_water_mark == 10000.0
        assert risk_controller.current_drawdown == 0

    def test_should_set_default_risk_limits_based_on_capital(self):
        """초기 자본에 기반해 기본 리스크 한도가 올바르게 설정되어야 함"""
        # Given
        initial_capital = 10000.0

        # When
        risk_controller = RiskController(initial_capital)

        # Then
        expected_var_usdt = initial_capital * 0.02  # 200 USDT
        expected_cvar_usdt = initial_capital * 0.03  # 300 USDT

        assert risk_controller.risk_limits['var_daily_return'] == 0.02
        assert risk_controller.risk_limits['var_daily_usdt'] == expected_var_usdt
        assert risk_controller.risk_limits['max_drawdown_pct'] == 0.12
        assert risk_controller.risk_limits['max_leverage'] == 10.0
        assert risk_controller.risk_limits['liquidation_prob_24h'] == 0.005

    def test_should_allow_custom_risk_parameters(self):
        """커스텀 리스크 파라미터로 초기화가 가능해야 함"""
        # Given
        initial_capital = 5000.0
        custom_var_pct = 0.015  # 1.5%
        custom_cvar_pct = 0.025  # 2.5%
        custom_max_dd = 0.08  # 8%
        custom_max_leverage = 5.0

        # When
        risk_controller = RiskController(
            initial_capital,
            var_daily_pct=custom_var_pct,
            cvar_daily_pct=custom_cvar_pct,
            max_drawdown_pct=custom_max_dd,
            max_leverage=custom_max_leverage
        )

        # Then
        assert risk_controller.risk_limits['var_daily_return'] == custom_var_pct
        assert risk_controller.risk_limits['var_daily_usdt'] == initial_capital * custom_var_pct
        assert risk_controller.risk_limits['max_drawdown_pct'] == custom_max_dd
        assert risk_controller.risk_limits['max_leverage'] == custom_max_leverage

    def test_should_detect_var_limit_violation(self):
        """VaR 한도 위반을 감지해야 함"""
        # Given
        initial_capital = 10000.0  # 10,000 USDT
        var_limit_pct = 0.02  # 2% = 200 USDT

        risk_controller = RiskController(initial_capital, var_daily_pct=var_limit_pct)

        # Portfolio state with excessive VaR (300 USDT > 200 USDT limit)
        portfolio_state = {
            'current_var_usdt': 300.0,  # 한도 초과
            'equity': 10000.0,
            'total_leverage': 5.0,
            'positions': []
        }

        # When
        violations = risk_controller.check_var_limit(portfolio_state)

        # Then
        assert len(violations) == 1
        assert violations[0][0] == 'VAR_USDT'
        assert violations[0][1] == 300.0  # 위반 값

    def test_should_pass_when_var_within_limit(self):
        """VaR이 한도 내에 있을 때는 위반이 없어야 함"""
        # Given
        initial_capital = 10000.0
        var_limit_pct = 0.02  # 200 USDT 한도

        risk_controller = RiskController(initial_capital, var_daily_pct=var_limit_pct)

        # Portfolio state within limit (150 USDT < 200 USDT)
        portfolio_state = {
            'current_var_usdt': 150.0,  # 한도 내
            'equity': 10000.0,
            'total_leverage': 5.0,
            'positions': []
        }

        # When
        violations = risk_controller.check_var_limit(portfolio_state)

        # Then
        assert len(violations) == 0  # 위반 없음

    def test_should_calculate_kelly_fraction_long_only_default(self):
        """기본 설정(Long-Only)에서 양의 수익률에 대해 Kelly fraction 계산"""
        # Given
        risk_controller = RiskController(10000.0)  # allow_short=False (기본값)

        # Positive expected return: μ ≈ 0.01, σ² ≈ 0.0001
        returns = np.array([0.015, 0.005, 0.020, -0.005, 0.010, 0.008, -0.002, 0.012,
                           0.003, 0.018, -0.001, 0.007, 0.011, 0.002, 0.013, -0.003,
                           0.009, 0.006, 0.014, -0.002, 0.001, 0.016, 0.004, 0.012,
                           0.008, 0.003, 0.017, -0.001, 0.005, 0.009, 0.007, 0.011])

        # When
        kelly_fraction = risk_controller.calculate_optimal_position_fraction(returns)

        # Then
        assert kelly_fraction > 0  # 양의 수익률이므로 양의 fraction
        assert kelly_fraction <= 0.08  # Long-only 기본 캡 (NEUTRAL regime)
        assert isinstance(kelly_fraction, float)

    def test_should_return_zero_for_negative_returns_long_only(self):
        """Long-Only에서 음의 수익률에 대해 0 반환"""
        # Given
        risk_controller = RiskController(10000.0)  # allow_short=False

        # Negative expected returns
        returns = np.array([-0.01, -0.005, -0.015, 0.002, -0.008, -0.012, -0.003, -0.007,
                           -0.009, -0.001, -0.011, 0.001, -0.006, -0.013, -0.002, -0.008,
                           -0.004, -0.010, 0.003, -0.005, -0.014, -0.001, -0.009, -0.007,
                           -0.012, -0.003, -0.006, -0.015, -0.002, -0.008, -0.004, -0.011])

        # When
        kelly_fraction = risk_controller.calculate_optimal_position_fraction(returns)

        # Then
        assert kelly_fraction == 0.0  # Long-only에서 음의 기댓값은 0

    def test_should_allow_short_positions_when_enabled(self):
        """Short 허용 시 음의 수익률에 대해 음의 Kelly fraction 반환"""
        # Given
        risk_controller = RiskController(10000.0, allow_short=True)

        # Negative expected returns (좋은 shorting 기회)
        returns = np.array([-0.01, -0.005, -0.015, 0.002, -0.008, -0.012, -0.003, -0.007,
                           -0.009, -0.001, -0.011, 0.001, -0.006, -0.013, -0.002, -0.008,
                           -0.004, -0.010, 0.003, -0.005, -0.014, -0.001, -0.009, -0.007,
                           -0.012, -0.003, -0.006, -0.015, -0.002, -0.008, -0.004, -0.011])

        # When
        kelly_fraction = risk_controller.calculate_optimal_position_fraction(returns)

        # Then
        assert kelly_fraction < 0  # 음의 수익률이므로 숏 포지션 (음의 fraction)
        assert kelly_fraction >= -0.08  # Short 기본 캡 (NEUTRAL regime)

    def test_should_return_zero_for_insufficient_data(self):
        """데이터가 부족할 때 0 반환"""
        # Given
        risk_controller = RiskController(10000.0)

        # Too few samples (< 30)
        returns = np.array([0.01, 0.02, -0.005])

        # When
        kelly_fraction = risk_controller.calculate_optimal_position_fraction(returns)

        # Then
        assert kelly_fraction == 0.0

    def test_should_detect_leverage_limit_violation(self):
        """레버리지 한도 위반을 감지해야 함"""
        # Given
        initial_capital = 10000.0  # 10,000 USDT
        max_leverage = 5.0  # 최대 레버리지 5x

        risk_controller = RiskController(
            initial_capital,
            max_leverage=max_leverage
        )

        # Portfolio state with excessive leverage (8x > 5x limit)
        portfolio_state = {
            'equity': 10000.0,
            'total_leverage': 8.0,  # 한도 초과
            'positions': [
                {
                    'symbol': 'BTCUSDT',
                    'side': 'LONG',
                    'size': 0.4,  # 0.4 BTC
                    'notional': 40000.0,  # 40,000 USDT notional
                    'leverage': 4.0
                },
                {
                    'symbol': 'ETHUSDT',
                    'side': 'LONG',
                    'size': 20.0,  # 20 ETH
                    'notional': 40000.0,  # 40,000 USDT notional
                    'leverage': 4.0
                }
            ]
        }

        # When
        violations = risk_controller.check_leverage_limit(portfolio_state)

        # Then
        assert len(violations) == 1
        assert violations[0][0] == 'LEVERAGE'
        assert violations[0][1] == 8.0  # 위반된 레버리지 값

    def test_should_pass_when_leverage_within_limit(self):
        """레버리지가 한도 내에 있을 때는 위반이 없어야 함"""
        # Given
        initial_capital = 10000.0
        max_leverage = 10.0  # 기본 최대 레버리지 10x

        risk_controller = RiskController(initial_capital, max_leverage=max_leverage)

        # Portfolio state within leverage limit (3x < 10x)
        portfolio_state = {
            'equity': 10000.0,
            'total_leverage': 3.0,  # 한도 내
            'positions': [
                {
                    'symbol': 'BTCUSDT',
                    'side': 'LONG',
                    'size': 0.3,  # 0.3 BTC
                    'notional': 30000.0,  # 30,000 USDT notional
                    'leverage': 3.0
                }
            ]
        }

        # When
        violations = risk_controller.check_leverage_limit(portfolio_state)

        # Then
        assert len(violations) == 0  # 위반 없음

    def test_should_calculate_total_leverage_correctly(self):
        """포트폴리오의 총 레버리지를 정확히 계산해야 함"""
        # Given
        risk_controller = RiskController(10000.0)

        # Multiple positions with different leverages
        portfolio_state = {
            'equity': 10000.0,  # 10,000 USDT equity
            'positions': [
                {
                    'symbol': 'BTCUSDT',
                    'notional': 20000.0,  # 20,000 USDT notional
                    'leverage': 2.0
                },
                {
                    'symbol': 'ETHUSDT',
                    'notional': 15000.0,  # 15,000 USDT notional
                    'leverage': 1.5
                },
                {
                    'symbol': 'ADAUSDT',
                    'notional': 5000.0,  # 5,000 USDT notional
                    'leverage': 1.0
                }
            ]
        }

        # When
        # 총 notional = 20000 + 15000 + 5000 = 40000
        # 총 레버리지 = 40000 / 10000 = 4.0x
        total_leverage = risk_controller._calculate_total_leverage(portfolio_state)

        # Then
        assert total_leverage == 4.0

    def test_should_handle_empty_portfolio_leverage(self):
        """빈 포트폴리오의 레버리지는 0이어야 함"""
        # Given
        risk_controller = RiskController(10000.0)

        # Empty portfolio
        portfolio_state = {
            'equity': 10000.0,
            'positions': []  # 포지션 없음
        }

        # When
        violations = risk_controller.check_leverage_limit(portfolio_state)
        total_leverage = risk_controller._calculate_total_leverage(portfolio_state)

        # Then
        assert len(violations) == 0  # 위반 없음
        assert total_leverage == 0.0  # 레버리지 0

    def test_should_calculate_safe_leverage_for_liquidation_distance(self):
        """청산 거리 기반 안전 레버리지 계산"""
        # Given
        risk_controller = RiskController(10000.0, max_leverage=10.0)

        # Portfolio with position close to liquidation
        portfolio_state = {
            'equity': 10000.0,
            'positions': [
                {
                    'symbol': 'BTCUSDT',
                    'side': 'LONG',
                    'current_price': 100000.0,
                    'liquidation_price': 90000.0,  # 10% away from liquidation
                    'notional': 50000.0,
                    'daily_volatility': 0.05  # 5% daily volatility
                }
            ]
        }

        # When
        safe_leverage = risk_controller.calculate_safe_leverage_limit(portfolio_state)

        # Then
        # With 10% distance and 5% volatility, safe leverage should be conservative
        assert safe_leverage < 10.0  # Should be less than max leverage
        assert safe_leverage > 0.0   # Should be positive
        assert isinstance(safe_leverage, float)

    def test_should_adjust_leverage_for_high_volatility(self):
        """높은 변동성 시장에서 레버리지 조정"""
        # Given
        risk_controller = RiskController(10000.0, max_leverage=10.0)

        # High volatility market state
        market_state = {
            'daily_volatility': 0.08,  # 8% daily volatility (high)
            'regime': 'VOLATILE'
        }

        # When
        adjusted_leverage = risk_controller.calculate_volatility_adjusted_leverage(
            base_leverage=5.0,
            market_state=market_state
        )

        # Then
        assert adjusted_leverage < 5.0  # Should reduce leverage in high volatility
        assert adjusted_leverage > 0.0