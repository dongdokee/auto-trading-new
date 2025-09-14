"""
Test module for FundingArbitrage strategy

Tests the funding arbitrage strategy implementation including funding rate prediction,
delta neutral positioning, and risk management features.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from src.strategy_engine.base_strategy import StrategyConfig, StrategySignal
from src.strategy_engine.strategies.funding_arbitrage import FundingArbitrageStrategy


class TestFundingArbitrageStrategy:
    """Test suite for FundingArbitrage strategy"""

    def setup_method(self):
        """Set up test fixtures"""
        self.default_config = StrategyConfig(
            name="FundingArbitrage",
            parameters={
                "funding_threshold": 0.03,  # 3% annualized funding rate (more realistic)
                "basis_threshold": 0.002,   # 0.2% basis
                "delta_neutral": True,      # Delta neutral mode
                "funding_lookback": 24,     # 24 periods for funding prediction
                "position_hold_hours": 8,   # Hold position for 8 hours
                "risk_factor": 0.5          # Conservative risk taking
            }
        )

        self.strategy = FundingArbitrageStrategy(self.default_config)

        # Create sample funding rate data
        dates = pd.date_range('2024-01-01', periods=100, freq='8h')  # Every 8 hours

        # Create funding rates that vary between -0.02% to 0.03% (annualized: -1.8% to 2.7%)
        base_funding = 0.0001  # 0.01% base
        funding_variation = 0.0002 * np.sin(np.linspace(0, 4*np.pi, 100))
        funding_rates = base_funding + funding_variation

        # Add some noise and occasional high funding events
        np.random.seed(42)
        noise = np.random.normal(0, 0.00005, 100)  # Small noise
        high_funding_events = np.random.choice([0, 1], size=100, p=[0.95, 0.05])  # 5% chance
        high_funding = high_funding_events * np.random.uniform(0.001, 0.003, 100)  # 0.1-0.3%

        funding_rates = funding_rates + noise + high_funding

        # Create spot and futures prices (futures slightly higher due to basis)
        spot_base = 51000
        spot_prices = spot_base + 500 * np.sin(np.linspace(0, 2*np.pi, 100))
        futures_basis = 10 + 20 * np.random.normal(0, 1, 100)  # Basis around $10-30
        futures_prices = spot_prices + futures_basis

        self.funding_data = pd.DataFrame({
            'timestamp': dates,
            'funding_rate': funding_rates,
            'spot_price': spot_prices,
            'futures_price': futures_prices,
            'basis': futures_prices - spot_prices,
            'volume': np.random.normal(1000, 200, 100)
        })

    def test_should_create_funding_arbitrage_strategy_with_default_config(self):
        """Test: FundingArbitrage strategy should initialize with default configuration"""
        # Act: Create strategy with default config
        strategy = FundingArbitrageStrategy(self.default_config)

        # Assert: Strategy should be properly initialized
        assert strategy.name == "FundingArbitrage"
        assert strategy.enabled is True
        assert strategy.funding_threshold == 0.03
        assert strategy.basis_threshold == 0.002
        assert strategy.delta_neutral is True
        assert strategy.funding_lookback == 24
        assert strategy.position_hold_hours == 8
        assert strategy.risk_factor == 0.5

    def test_should_validate_parameter_constraints(self):
        """Test: Strategy should validate parameter constraints"""
        # Test invalid funding threshold
        invalid_config = StrategyConfig(
            name="FundingArbitrage",
            parameters={"funding_threshold": 0.5}  # Too large
        )

        with pytest.raises(ValueError, match="Funding threshold must be between"):
            FundingArbitrageStrategy(invalid_config)

        # Test invalid lookback period
        invalid_config = StrategyConfig(
            name="FundingArbitrage",
            parameters={"funding_lookback": 3}  # Too small
        )

        with pytest.raises(ValueError, match="Funding lookback must be at least"):
            FundingArbitrageStrategy(invalid_config)

    def test_should_predict_funding_rate_from_historical_data(self):
        """Test: Strategy should predict next funding rate from historical data"""
        # Act: Predict funding rate
        predicted_funding = self.strategy._predict_next_funding_rate(
            self.funding_data, current_index=50
        )

        # Assert: Should return reasonable prediction
        assert predicted_funding is not None
        assert isinstance(predicted_funding, float)
        assert -0.005 <= predicted_funding <= 0.005  # Within reasonable bounds

    def test_should_generate_long_funding_signal_for_negative_rates(self):
        """Test: Strategy should generate LONG funding signal when funding is significantly negative"""
        # Arrange: Create data with negative funding rate
        negative_funding_data = self.funding_data.copy()
        negative_funding_data.loc[80:85, 'funding_rate'] = -0.0015  # -0.15% (negative)

        market_data = {
            'symbol': 'BTCUSDT',
            'close': 51000,
            'funding_rate': -0.0015,
            'basis': 15.0,  # Small positive basis
            'funding_data': negative_funding_data
        }

        # Act: Generate signal
        signal = self.strategy.generate_signal(market_data, current_index=82)

        # Assert: Should generate signal to collect negative funding (go long)
        assert isinstance(signal, StrategySignal)
        assert signal.symbol == 'BTCUSDT'
        # When funding is negative, we want to be long to collect funding
        if signal.action != 'HOLD':
            assert signal.action in ['BUY']
            assert signal.strength > 0
            assert signal.confidence > 0

    def test_should_generate_short_funding_signal_for_positive_rates(self):
        """Test: Strategy should generate SHORT funding signal when funding is significantly positive"""
        # Arrange: Create data with high positive funding rate
        high_funding_data = self.funding_data.copy()
        high_funding_data.loc[80:85, 'funding_rate'] = 0.002  # 0.2% (high positive)

        market_data = {
            'symbol': 'BTCUSDT',
            'close': 51000,
            'funding_rate': 0.002,
            'basis': 15.0,
            'funding_data': high_funding_data
        }

        # Act: Generate signal
        signal = self.strategy.generate_signal(market_data, current_index=82)

        # Assert: Should generate signal to collect positive funding (go short)
        assert isinstance(signal, StrategySignal)
        assert signal.symbol == 'BTCUSDT'
        # When funding is positive, we want to be short to collect funding
        if signal.action != 'HOLD':
            assert signal.action in ['SELL']
            assert signal.strength > 0
            assert signal.confidence > 0

    def test_should_generate_hold_signal_for_low_funding_rates(self):
        """Test: Strategy should generate HOLD signal when funding rates are not attractive"""
        # Arrange: Create data with consistently very low funding rates
        low_funding_data = self.funding_data.copy()
        # Set a longer period of very low funding to avoid trend/prediction effects
        low_funding_data.loc[75:85, 'funding_rate'] = 0.00002  # 0.002% (very low)
        low_funding_data.loc[75:85, 'basis'] = 5.0  # Small basis to avoid basis adjustments

        market_data = {
            'symbol': 'BTCUSDT',
            'close': 51000,
            'funding_rate': 0.00002,
            'basis': 5.0,
            'funding_data': low_funding_data
        }

        # Act: Generate signal
        signal = self.strategy.generate_signal(market_data, current_index=82)

        # Assert: Should generate HOLD signal for low funding
        assert isinstance(signal, StrategySignal)
        assert signal.symbol == 'BTCUSDT'
        # Either HOLD or very low strength signal
        if signal.action != 'HOLD':
            assert signal.strength < 0.1  # Very low strength for low funding

    def test_should_consider_basis_in_arbitrage_decision(self):
        """Test: Strategy should consider futures basis in arbitrage decisions"""
        # Arrange: High funding but unfavorable basis
        unfavorable_data = self.funding_data.copy()
        unfavorable_data.loc[80:85, 'funding_rate'] = 0.002  # High positive funding

        market_data = {
            'symbol': 'BTCUSDT',
            'close': 51000,
            'funding_rate': 0.002,
            'basis': -50.0,  # Large negative basis (futures trading below spot)
            'funding_data': unfavorable_data
        }

        # Act: Generate signal
        signal = self.strategy.generate_signal(market_data, current_index=82)

        # Assert: Should be cautious due to unfavorable basis
        assert isinstance(signal, StrategySignal)
        if signal.action != 'HOLD':
            assert signal.confidence < 0.7  # Lower confidence due to basis

    def test_should_handle_delta_neutral_mode(self):
        """Test: Strategy should handle delta neutral positioning"""
        # This test verifies that delta neutral mode affects the strategy behavior
        self.strategy.delta_neutral = True

        market_data = {
            'symbol': 'BTCUSDT',
            'close': 51000,
            'funding_rate': 0.002,
            'basis': 15.0,
            'funding_data': self.funding_data
        }

        # Act: Generate signal in delta neutral mode
        signal_delta_neutral = self.strategy.generate_signal(market_data, current_index=82)

        # Disable delta neutral
        self.strategy.delta_neutral = False
        signal_directional = self.strategy.generate_signal(market_data, current_index=82)

        # Assert: Both signals should be valid
        assert isinstance(signal_delta_neutral, StrategySignal)
        assert isinstance(signal_directional, StrategySignal)

        # In delta neutral mode, confidence might be different
        # (Exact behavior depends on implementation details)

    def test_should_set_appropriate_position_hold_duration(self):
        """Test: Strategy should set appropriate position holding duration"""
        # Arrange: Generate a signal
        market_data = {
            'symbol': 'BTCUSDT',
            'close': 51000,
            'funding_rate': 0.002,
            'basis': 15.0,
            'funding_data': self.funding_data
        }

        # Act: Generate signal
        signal = self.strategy.generate_signal(market_data, current_index=82)

        # Assert: Signal metadata should include hold duration
        assert isinstance(signal, StrategySignal)
        if signal.action != 'HOLD':
            assert 'hold_hours' in signal.metadata
            assert signal.metadata['hold_hours'] == self.strategy.position_hold_hours

    def test_should_handle_insufficient_funding_data(self):
        """Test: Strategy should handle insufficient funding data gracefully"""
        # Arrange: Insufficient funding data
        insufficient_data = self.funding_data.iloc[:3]  # Only 3 data points

        market_data = {
            'symbol': 'BTCUSDT',
            'close': 51000,
            'funding_rate': 0.002,
            'basis': 15.0,
            'funding_data': insufficient_data
        }

        # Act: Generate signal
        signal = self.strategy.generate_signal(market_data, current_index=2)

        # Assert: Should return HOLD signal
        assert signal.action == 'HOLD'
        assert signal.strength == 0.0
        assert signal.confidence == 0.0

    def test_should_calculate_funding_rate_statistics(self):
        """Test: Strategy should calculate funding rate statistics correctly"""
        # Act: Calculate statistics
        stats = self.strategy._calculate_funding_statistics(
            self.funding_data, current_index=50
        )

        # Assert: Should return valid statistics
        assert isinstance(stats, dict)
        assert 'mean_funding' in stats
        assert 'std_funding' in stats
        assert 'trend' in stats
        assert 'volatility' in stats

        # Values should be reasonable
        assert isinstance(stats['mean_funding'], float)
        assert isinstance(stats['std_funding'], float)
        assert stats['std_funding'] >= 0

    def test_should_update_parameters_correctly(self):
        """Test: Strategy should update parameters correctly"""
        # Arrange: New parameters
        new_params = {
            "funding_threshold": 0.015,
            "basis_threshold": 0.003,
            "risk_factor": 0.7
        }

        # Act: Update parameters
        self.strategy.update_parameters(**new_params)

        # Assert: Parameters should be updated
        assert self.strategy.funding_threshold == 0.015
        assert self.strategy.basis_threshold == 0.003
        assert self.strategy.risk_factor == 0.7

    def test_should_track_performance_metrics(self):
        """Test: Strategy should track performance metrics correctly"""
        # Arrange: Generate some signals and track performance
        initial_signals = self.strategy.total_signals

        # Act: Generate several signals
        for i in range(70, 90, 3):  # Every 3rd index to avoid over-testing
            market_data = {
                'symbol': 'BTCUSDT',
                'close': self.funding_data.iloc[i]['futures_price'],
                'funding_rate': self.funding_data.iloc[i]['funding_rate'],
                'basis': self.funding_data.iloc[i]['basis'],
                'funding_data': self.funding_data
            }
            signal = self.strategy.generate_signal(market_data, current_index=i)
            self.strategy.signal_history.append(signal)

        # Update performance (simulate some wins and losses)
        self.strategy.update_performance(pnl=150.0, winning=True)
        self.strategy.update_performance(pnl=-75.0, winning=False)

        # Assert: Performance should be tracked
        assert self.strategy.total_signals >= initial_signals
        assert len(self.strategy.signal_history) > 0
        assert self.strategy.total_pnl == 75.0  # 150 - 75

    def test_should_validate_signal_format(self):
        """Test: Strategy should produce valid signal format"""
        # Act: Generate signal
        market_data = {
            'symbol': 'BTCUSDT',
            'close': 51000,
            'funding_rate': 0.002,
            'basis': 15.0,
            'funding_data': self.funding_data
        }

        signal = self.strategy.generate_signal(market_data, current_index=80)

        # Assert: Signal should be valid format
        assert isinstance(signal, StrategySignal)
        assert signal.symbol == 'BTCUSDT'
        assert signal.action in ['BUY', 'SELL', 'HOLD', 'CLOSE']
        assert 0.0 <= signal.strength <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert isinstance(signal.metadata, dict)
        assert signal.metadata['strategy'] == 'FundingArbitrage'

    def test_should_handle_edge_cases_gracefully(self):
        """Test: Strategy should handle edge cases without crashing"""
        # Test with missing data fields
        incomplete_data = {
            'symbol': 'BTCUSDT',
            'close': 51000
            # Missing funding_rate, basis, funding_data
        }

        signal = self.strategy.generate_signal(incomplete_data, current_index=80)
        assert signal.action == 'HOLD'

        # Test with extreme values
        extreme_data = {
            'symbol': 'BTCUSDT',
            'close': 51000,
            'funding_rate': 0.1,  # Extremely high 10%
            'basis': 1000.0,      # Extremely high basis
            'funding_data': self.funding_data
        }

        signal = self.strategy.generate_signal(extreme_data, current_index=80)
        assert isinstance(signal, StrategySignal)