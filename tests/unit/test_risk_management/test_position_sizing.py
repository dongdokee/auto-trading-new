# tests/unit/test_risk_management/test_position_sizing.py
import pytest
import numpy as np
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.risk_management import RiskController


class TestPositionSizer:
    """PositionSizer 클래스에 대한 TDD 테스트"""

    @pytest.fixture
    def risk_controller(self):
        """테스트용 RiskController 인스턴스"""
        return RiskController(initial_capital_usdt=10000.0)

    @pytest.fixture
    def position_sizer(self, risk_controller):
        """테스트용 PositionSizer 인스턴스"""
        return PositionSizer(risk_controller)

    @pytest.fixture
    def sample_market_state(self):
        """샘플 마켓 상태 데이터"""
        return {
            'symbol': 'BTCUSDT',
            'price': 50000.0,  # BTC price in USDT
            'atr': 2000.0,  # 20-day ATR in USDT
            'daily_volatility': 0.05,  # 5% daily volatility
            'regime': 'NEUTRAL',
            'min_notional': 10.0,  # Minimum 10 USDT
            'lot_size': 0.001,  # 0.001 BTC minimum size
            'symbol_leverage': 10  # Max 10x leverage for BTC
        }

    @pytest.fixture
    def sample_portfolio_state(self):
        """샘플 포트폴리오 상태 데이터"""
        # Generate 35 days of realistic returns for Kelly calculation
        np.random.seed(42)  # For reproducible tests
        recent_returns = np.random.normal(0.001, 0.02, 35)  # 35 days, 0.1% mean, 2% std

        return {
            'equity': 10000.0,  # 10,000 USDT
            'recent_returns': recent_returns,  # 35 days of returns
            'positions': [],  # Empty portfolio initially
            'current_var_usdt': 0.0,
            'symbol_volatilities': {'BTCUSDT': 0.05},
            'correlation_matrix': {}
        }

    @pytest.fixture
    def sample_signal(self):
        """샘플 거래 신호"""
        return {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'strength': 0.8,  # 80% signal strength
            'confidence': 0.7
        }

    def test_should_initialize_with_risk_controller(self, risk_controller):
        """PositionSizer가 RiskController와 함께 올바르게 초기화되어야 함"""
        # When
        position_sizer = PositionSizer(risk_controller)

        # Then
        assert position_sizer.risk_controller == risk_controller
        assert hasattr(position_sizer, 'calculate_position_size')

    def test_should_calculate_basic_position_size_for_long_signal(self, position_sizer, sample_signal,
                                                                sample_market_state, sample_portfolio_state):
        """롱 신호에 대해 기본 포지션 크기를 계산해야 함"""
        # When
        position_size = position_sizer.calculate_position_size(
            signal=sample_signal,
            market_state=sample_market_state,
            portfolio_state=sample_portfolio_state
        )

        # Then
        assert isinstance(position_size, float)
        assert position_size > 0  # Should be positive for LONG signal
        assert position_size >= sample_market_state['min_notional'] / sample_market_state['price']  # Above minimum

    def test_should_return_zero_for_zero_signal_strength(self, position_sizer, sample_signal,
                                                       sample_market_state, sample_portfolio_state):
        """신호 강도가 0일 때 포지션 크기 0을 반환해야 함"""
        # Given
        sample_signal['strength'] = 0.0

        # When
        position_size = position_sizer.calculate_position_size(
            signal=sample_signal,
            market_state=sample_market_state,
            portfolio_state=sample_portfolio_state
        )

        # Then
        assert position_size == 0.0

    def test_should_respect_maximum_position_limit(self, position_sizer, sample_signal,
                                                 sample_market_state, sample_portfolio_state):
        """최대 포지션 한도를 준수해야 함"""
        # Given - Very strong signal that might suggest large position
        sample_signal['strength'] = 1.0

        # When
        position_size = position_sizer.calculate_position_size(
            signal=sample_signal,
            market_state=sample_market_state,
            portfolio_state=sample_portfolio_state
        )

        # Then
        max_position_notional = sample_portfolio_state['equity'] * 0.2  # 20% max per position
        max_position_size = max_position_notional / sample_market_state['price']

        assert position_size <= max_position_size

    def test_should_round_to_lot_size(self, position_sizer, sample_signal,
                                    sample_market_state, sample_portfolio_state):
        """거래소 lot size로 반올림해야 함"""
        # When
        position_size = position_sizer.calculate_position_size(
            signal=sample_signal,
            market_state=sample_market_state,
            portfolio_state=sample_portfolio_state
        )

        # Then
        lot_size = sample_market_state['lot_size']
        expected_rounded = np.floor(position_size / lot_size) * lot_size

        # Allow for small floating point differences
        assert abs(position_size - expected_rounded) < 1e-10

    def test_should_handle_insufficient_kelly_data(self, position_sizer, sample_signal,
                                                 sample_market_state, sample_portfolio_state):
        """Kelly 계산을 위한 데이터가 부족할 때 적절히 처리해야 함"""
        # Given - Insufficient return data
        sample_portfolio_state['recent_returns'] = np.array([0.01, -0.005])  # Only 2 days

        # When
        position_size = position_sizer.calculate_position_size(
            signal=sample_signal,
            market_state=sample_market_state,
            portfolio_state=sample_portfolio_state
        )

        # Then
        # Should still return a valid position size based on other methods (ATR, etc.)
        assert isinstance(position_size, float)
        assert position_size >= 0

    # ========== Additional Comprehensive Tests ==========

    def test_should_calculate_atr_based_size_correctly(self, position_sizer, sample_signal,
                                                     sample_market_state, sample_portfolio_state):
        """ATR 기반 포지션 크기를 정확히 계산해야 함"""
        # Given
        equity = 10000.0
        atr = 2000.0  # USDT
        risk_per_trade = equity * 0.01  # 1% = 100 USDT
        stop_distance = 2.0 * atr  # 4000 USDT
        expected_size = risk_per_trade / stop_distance  # 100 / 4000 = 0.025 BTC

        # When
        atr_size = position_sizer._calculate_atr_based_size(
            sample_signal, sample_market_state, sample_portfolio_state
        )

        # Then
        assert abs(atr_size - expected_size) < 1e-10

    def test_should_calculate_var_constrained_size_correctly(self, position_sizer, sample_signal,
                                                           sample_market_state, sample_portfolio_state):
        """VaR 제약 기반 포지션 크기를 정확히 계산해야 함"""
        # Given
        equity = 10000.0
        var_limit = equity * 0.02  # 200 USDT
        price = 50000.0
        sigma = 0.05
        z = 1.65
        expected_size = var_limit / (z * sigma * price)  # 200 / (1.65 * 0.05 * 50000)

        # When
        var_size = position_sizer._calculate_var_constrained_size(
            'BTCUSDT', sample_market_state, sample_portfolio_state
        )

        # Then
        assert abs(var_size - expected_size) < 1e-10

    def test_should_reduce_position_for_high_correlation(self, position_sizer, sample_signal,
                                                       sample_market_state, sample_portfolio_state):
        """높은 상관관계 시 포지션 크기를 감소시켜야 함"""
        # Given - High correlation portfolio
        sample_portfolio_state['positions'] = [
            {'symbol': 'ETHUSDT', 'size': 1.0}
        ]
        sample_portfolio_state['correlation_matrix'] = {
            ('BTCUSDT', 'ETHUSDT'): 0.85  # High correlation
        }

        # When
        correlation_factor = position_sizer._calculate_correlation_adjustment(
            'BTCUSDT', sample_portfolio_state
        )

        # Then
        assert correlation_factor == 0.3  # Should be heavily reduced for correlation > 0.8

    def test_should_calculate_liquidation_safe_leverage(self, position_sizer, sample_signal,
                                                      sample_market_state, sample_portfolio_state):
        """청산 안전 레버리지를 계산해야 함"""
        # When
        liquidation_size = position_sizer._calculate_liquidation_safe_size(
            'BTCUSDT', 'LONG', sample_market_state, sample_portfolio_state
        )

        # Then
        assert isinstance(liquidation_size, float)
        assert liquidation_size > 0
        # Should be reasonable size (not too big or too small)
        equity = sample_portfolio_state['equity']
        price = sample_market_state['price']
        max_reasonable_size = equity / price  # 1x leverage equivalent
        assert liquidation_size <= max_reasonable_size * 20  # Should not exceed 20x equivalent

    def test_should_handle_zero_volatility(self, position_sizer, sample_signal,
                                         sample_market_state, sample_portfolio_state):
        """변동성이 0일 때 적절히 처리해야 함"""
        # Given - Zero volatility
        sample_portfolio_state['symbol_volatilities']['BTCUSDT'] = 0.0

        # When
        var_size = position_sizer._calculate_var_constrained_size(
            'BTCUSDT', sample_market_state, sample_portfolio_state
        )

        # Then
        assert var_size == 0.0  # Should return 0 for zero volatility

    def test_should_handle_zero_atr(self, position_sizer, sample_signal,
                                  sample_market_state, sample_portfolio_state):
        """ATR이 0일 때 적절히 처리해야 함"""
        # Given - Zero ATR
        sample_market_state['atr'] = 0.0

        # When
        atr_size = position_sizer._calculate_atr_based_size(
            sample_signal, sample_market_state, sample_portfolio_state
        )

        # Then
        assert isinstance(atr_size, float)
        assert atr_size > 0  # Should fallback to equity-based sizing

    def test_should_use_minimum_constraint_size(self, position_sizer, sample_signal,
                                              sample_market_state, sample_portfolio_state):
        """여러 제약 조건 중 최소값을 사용해야 함"""
        # Given - Set up different constraints that will return different sizes
        sample_portfolio_state['current_var_usdt'] = 150.0  # Use up most VaR budget

        # When
        position_size = position_sizer.calculate_position_size(
            signal=sample_signal,
            market_state=sample_market_state,
            portfolio_state=sample_portfolio_state
        )

        # Calculate individual components
        kelly_size = position_sizer._calculate_kelly_based_size(sample_signal, sample_market_state, sample_portfolio_state)
        atr_size = position_sizer._calculate_atr_based_size(sample_signal, sample_market_state, sample_portfolio_state)
        liquidation_size = position_sizer._calculate_liquidation_safe_size('BTCUSDT', 'LONG', sample_market_state, sample_portfolio_state)
        var_size = position_sizer._calculate_var_constrained_size('BTCUSDT', sample_market_state, sample_portfolio_state)

        min_size = min(kelly_size, atr_size, liquidation_size, var_size)

        # Then - Final size should be based on the most restrictive constraint
        # (accounting for signal strength and correlation adjustments)
        expected_max = min_size * sample_signal['strength']  # 0.8 signal strength
        assert position_size <= expected_max

    def test_should_apply_signal_strength_correctly(self, position_sizer, sample_signal,
                                                  sample_market_state, sample_portfolio_state):
        """신호 강도를 올바르게 적용해야 함"""
        # Given - Two different signal strengths
        sample_signal_weak = sample_signal.copy()
        sample_signal_weak['strength'] = 0.5

        sample_signal_strong = sample_signal.copy()
        sample_signal_strong['strength'] = 1.0

        # When
        size_weak = position_sizer.calculate_position_size(
            signal=sample_signal_weak,
            market_state=sample_market_state,
            portfolio_state=sample_portfolio_state
        )

        size_strong = position_sizer.calculate_position_size(
            signal=sample_signal_strong,
            market_state=sample_market_state,
            portfolio_state=sample_portfolio_state
        )

        # Then
        assert size_strong > size_weak  # Stronger signal should result in larger position
        assert abs(size_strong / size_weak - 2.0) < 0.1  # Should be roughly 2x (1.0 / 0.5)

    def test_should_handle_maintenance_margin_tiers(self, position_sizer):
        """유지증거금 티어를 올바르게 처리해야 함"""
        # Test different notional amounts
        small_notional = 5000.0
        medium_notional = 30000.0
        large_notional = 100000.0

        # When
        mmr_small = position_sizer._get_maintenance_margin_rate('BTCUSDT', small_notional)
        mmr_medium = position_sizer._get_maintenance_margin_rate('BTCUSDT', medium_notional)
        mmr_large = position_sizer._get_maintenance_margin_rate('BTCUSDT', large_notional)

        # Then
        assert mmr_small == 0.004  # 0.4%
        assert mmr_medium == 0.005  # 0.5%
        assert mmr_large == 0.01   # 1%
        assert mmr_small < mmr_medium < mmr_large  # Should increase with size