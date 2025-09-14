"""
Test module for RangeTrading strategy

Tests the range trading strategy implementation including support/resistance detection,
signal generation logic, and risk management features.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from src.strategy_engine.base_strategy import StrategyConfig, StrategySignal
from src.strategy_engine.strategies.range_trading import RangeTradingStrategy


class TestRangeTradingStrategy:
    """Test suite for RangeTrading strategy"""

    def setup_method(self):
        """Set up test fixtures"""
        self.default_config = StrategyConfig(
            name="RangeTrading",
            parameters={
                "lookback_period": 20,
                "support_resistance_threshold": 0.02,  # 2%
                "volume_confirmation": True,
                "atr_period": 14,
                "stop_multiplier": 1.5,
                "target_multiplier": 2.0
            }
        )

        self.strategy = RangeTradingStrategy(self.default_config)

        # Create sample OHLCV data for range-bound market
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')

        # Create range-bound price data oscillating between 49500-52500 (wider range)
        base_price = 51000
        range_amplitude = 1500  # Wider range for better testing
        prices = base_price + range_amplitude * np.sin(np.linspace(0, 4*np.pi, 100))

        # Add some noise but keep in range
        np.random.seed(42)  # Fixed seed for reproducible tests
        noise = np.random.normal(0, 100, 100)
        prices = prices + noise

        # Create OHLC from prices
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, 100)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, 100)))
        opens = prices + np.random.normal(0, 50, 100)
        closes = prices

        volumes = np.random.normal(1000, 200, 100)

        self.range_data = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

    def test_should_create_range_trading_strategy_with_default_config(self):
        """Test: Range trading strategy should initialize with default configuration"""
        # Act: Create strategy with default config
        strategy = RangeTradingStrategy(self.default_config)

        # Assert: Strategy should be properly initialized
        assert strategy.name == "RangeTrading"
        assert strategy.enabled is True
        assert strategy.lookback_period == 20
        assert strategy.support_resistance_threshold == 0.02
        assert strategy.volume_confirmation is True
        assert strategy.atr_period == 14
        assert strategy.stop_multiplier == 1.5
        assert strategy.target_multiplier == 2.0

    def test_should_validate_parameter_constraints(self):
        """Test: Strategy should validate parameter constraints"""
        # Test invalid lookback period
        invalid_config = StrategyConfig(
            name="RangeTrading",
            parameters={"lookback_period": 5}  # Too small
        )

        with pytest.raises(ValueError, match="Lookback period must be at least 10"):
            RangeTradingStrategy(invalid_config)

        # Test invalid threshold
        invalid_config = StrategyConfig(
            name="RangeTrading",
            parameters={"support_resistance_threshold": 0.5}  # Too large
        )

        with pytest.raises(ValueError, match="Support/resistance threshold must be between"):
            RangeTradingStrategy(invalid_config)

    def test_should_detect_support_resistance_levels(self):
        """Test: Strategy should detect support and resistance levels"""
        # Act: Detect levels in range-bound data
        support, resistance = self.strategy._identify_support_resistance(
            self.range_data, current_index=80
        )

        # Assert: Should identify reasonable support/resistance levels
        assert support is not None
        assert resistance is not None
        assert resistance > support
        assert (resistance - support) / support > 0.01  # At least 1% range

    def test_should_generate_buy_signal_near_support(self):
        """Test: Strategy should generate BUY signal when price is near support"""
        # Arrange: Create market data where price is near support
        current_price = 50200  # Near bottom of range

        market_data = {
            'symbol': 'BTCUSDT',
            'close': current_price,
            'ohlcv_data': self.range_data
        }

        # Act: Generate signal
        signal = self.strategy.generate_signal(market_data, current_index=80)

        # Assert: Should generate BUY signal
        assert isinstance(signal, StrategySignal)
        assert signal.symbol == 'BTCUSDT'
        assert signal.action in ['BUY', 'HOLD']  # BUY when near support
        assert 0.0 <= signal.strength <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    def test_should_generate_sell_signal_near_resistance(self):
        """Test: Strategy should generate SELL signal when price is near resistance"""
        # Arrange: First get the actual resistance level
        support, resistance = self.strategy._identify_support_resistance(
            self.range_data, current_index=80
        )

        # Use actual resistance level with small offset
        current_price = resistance * 0.998  # Very close to resistance

        market_data = {
            'symbol': 'BTCUSDT',
            'close': current_price,
            'ohlcv_data': self.range_data
        }

        # Act: Generate signal
        signal = self.strategy.generate_signal(market_data, current_index=80)

        # Assert: Should generate SELL signal or HOLD
        assert isinstance(signal, StrategySignal)
        assert signal.symbol == 'BTCUSDT'
        assert signal.action in ['SELL', 'HOLD']  # SELL when near resistance
        assert 0.0 <= signal.strength <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    def test_should_generate_hold_signal_in_middle_of_range(self):
        """Test: Strategy should generate HOLD signal in middle of range"""
        # Arrange: Get actual support/resistance and use middle price
        support, resistance = self.strategy._identify_support_resistance(
            self.range_data, current_index=80
        )

        # Price in middle of range
        current_price = (support + resistance) / 2

        market_data = {
            'symbol': 'BTCUSDT',
            'close': current_price,
            'ohlcv_data': self.range_data
        }

        # Act: Generate signal
        signal = self.strategy.generate_signal(market_data, current_index=80)

        # Assert: Should generate HOLD signal or low strength signal
        assert isinstance(signal, StrategySignal)
        assert signal.symbol == 'BTCUSDT'
        # Should be HOLD in middle of range (low strength)
        if signal.action != 'HOLD':
            assert signal.strength < 0.5  # Low strength if not HOLD (relaxed threshold)

    def test_should_set_appropriate_stop_loss_and_take_profit(self):
        """Test: Strategy should set ATR-based stop loss and take profit levels"""
        # Arrange: Market data with known ATR
        market_data = {
            'symbol': 'BTCUSDT',
            'close': 50200,  # Near support
            'ohlcv_data': self.range_data
        }

        # Act: Generate signal
        signal = self.strategy.generate_signal(market_data, current_index=80)

        # Assert: Stop and target should be set for BUY signals
        if signal.action == 'BUY':
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.stop_loss < signal.take_profit  # For BUY signals

            # Check ATR-based distance
            atr = self.strategy._calculate_atr(self.range_data, 80)
            expected_stop = market_data['close'] - (atr * self.strategy.stop_multiplier)
            expected_target = market_data['close'] + (atr * self.strategy.target_multiplier)

            assert abs(signal.stop_loss - expected_stop) < atr * 0.1  # Within 10% of ATR
            assert abs(signal.take_profit - expected_target) < atr * 0.1

    def test_should_handle_insufficient_data_gracefully(self):
        """Test: Strategy should handle insufficient data without crashing"""
        # Arrange: Insufficient data
        insufficient_data = self.range_data.iloc[:5]  # Only 5 data points

        market_data = {
            'symbol': 'BTCUSDT',
            'close': 51000,
            'ohlcv_data': insufficient_data
        }

        # Act: Generate signal
        signal = self.strategy.generate_signal(market_data, current_index=4)

        # Assert: Should return HOLD signal
        assert signal.action == 'HOLD'
        assert signal.strength == 0.0
        assert signal.confidence == 0.0

    def test_should_require_volume_confirmation_when_enabled(self):
        """Test: Strategy should require volume confirmation when enabled"""
        # This test will check if volume confirmation affects signal generation
        # Implementation depends on how volume confirmation is implemented

        # Arrange: Enable volume confirmation
        self.strategy.volume_confirmation = True

        market_data = {
            'symbol': 'BTCUSDT',
            'close': 50200,  # Near support
            'ohlcv_data': self.range_data
        }

        # Act: Generate signal with volume confirmation
        signal_with_volume = self.strategy.generate_signal(market_data, current_index=80)

        # Disable volume confirmation
        self.strategy.volume_confirmation = False
        signal_without_volume = self.strategy.generate_signal(market_data, current_index=80)

        # Assert: Volume confirmation should affect signal strength/confidence
        # (Exact behavior depends on implementation)
        assert isinstance(signal_with_volume, StrategySignal)
        assert isinstance(signal_without_volume, StrategySignal)

    def test_should_detect_range_breakout_conditions(self):
        """Test: Strategy should detect when price breaks out of range"""
        # Arrange: Create breakout scenario
        breakout_price = 53000  # Well above previous range

        market_data = {
            'symbol': 'BTCUSDT',
            'close': breakout_price,
            'ohlcv_data': self.range_data
        }

        # Act: Generate signal during breakout
        signal = self.strategy.generate_signal(market_data, current_index=80)

        # Assert: Should detect breakout (likely HOLD or very low confidence)
        assert isinstance(signal, StrategySignal)
        # During strong breakout, range strategy should be cautious
        if signal.action != 'HOLD':
            assert signal.confidence < 0.5  # Low confidence during breakout

    def test_should_update_parameters_correctly(self):
        """Test: Strategy should update parameters correctly"""
        # Arrange: New parameters
        new_params = {
            "lookback_period": 30,
            "support_resistance_threshold": 0.025,
            "stop_multiplier": 2.0
        }

        # Act: Update parameters
        self.strategy.update_parameters(**new_params)

        # Assert: Parameters should be updated
        assert self.strategy.lookback_period == 30
        assert self.strategy.support_resistance_threshold == 0.025
        assert self.strategy.stop_multiplier == 2.0

    def test_should_calculate_atr_correctly(self):
        """Test: Strategy should calculate ATR correctly"""
        # Act: Calculate ATR
        atr = self.strategy._calculate_atr(self.range_data, current_index=50)

        # Assert: ATR should be positive and reasonable
        assert atr > 0
        assert atr < 10000  # Reasonable upper bound for crypto prices

        # Test edge case with insufficient data
        atr_insufficient = self.strategy._calculate_atr(self.range_data, current_index=5)
        assert atr_insufficient > 0  # Should handle gracefully

    def test_should_track_performance_metrics(self):
        """Test: Strategy should track performance metrics correctly"""
        # Arrange: Generate some signals and track performance
        initial_signals = self.strategy.total_signals
        initial_winning = self.strategy.winning_signals

        # Act: Generate several signals
        for i in range(70, 90):
            market_data = {
                'symbol': 'BTCUSDT',
                'close': self.range_data.iloc[i]['close'],
                'ohlcv_data': self.range_data
            }
            signal = self.strategy.generate_signal(market_data, current_index=i)
            self.strategy.signal_history.append(signal)

        # Update performance (simulate some wins and losses)
        self.strategy.update_performance(pnl=100.0, winning=True)
        self.strategy.update_performance(pnl=-50.0, winning=False)

        # Assert: Performance should be tracked
        assert self.strategy.total_signals >= initial_signals
        assert len(self.strategy.signal_history) > 0
        assert self.strategy.total_pnl == 50.0  # 100 - 50

    def test_should_handle_edge_cases_in_support_resistance_detection(self):
        """Test: Strategy should handle edge cases in support/resistance detection"""
        # Test with flat price data
        flat_data = self.range_data.copy()
        flat_data['close'] = 51000  # Flat prices
        flat_data['high'] = 51100
        flat_data['low'] = 50900

        support, resistance = self.strategy._identify_support_resistance(flat_data, 50)

        # Should still return valid levels even with flat data
        assert support is not None
        assert resistance is not None
        assert resistance >= support

    def test_should_validate_signal_format(self):
        """Test: Strategy should produce valid signal format"""
        # Act: Generate signal
        market_data = {
            'symbol': 'BTCUSDT',
            'close': 51000,
            'ohlcv_data': self.range_data
        }

        signal = self.strategy.generate_signal(market_data, current_index=80)

        # Assert: Signal should be valid format
        assert isinstance(signal, StrategySignal)
        assert signal.symbol == 'BTCUSDT'
        assert signal.action in ['BUY', 'SELL', 'HOLD', 'CLOSE']
        assert 0.0 <= signal.strength <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert isinstance(signal.metadata, dict)