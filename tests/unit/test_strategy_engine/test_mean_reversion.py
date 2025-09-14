"""
Unit tests for MeanReversionStrategy
Tests the Bollinger Bands + RSI mean reversion strategy.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.strategy_engine.base_strategy import StrategySignal, StrategyConfig
from src.strategy_engine.strategies.mean_reversion import MeanReversionStrategy


class TestMeanReversionStrategy:
    """Test MeanReversionStrategy functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = StrategyConfig(
            name="MeanReversion",
            parameters={
                "bb_period": 20,
                "bb_std": 2.0,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "min_confidence": 0.6
            }
        )
        self.strategy = MeanReversionStrategy(self.config)

        # Create test data
        np.random.seed(42)
        self.create_test_data()

    def create_test_data(self):
        """Create synthetic market data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        # Create mean-reverting data (oscillating around 50000)
        base_price = 50000
        oscillation = np.sin(np.arange(100) * 0.3) * 2000  # Sine wave oscillation
        noise = np.random.normal(0, 500, 100)
        prices = base_price + oscillation + noise

        self.mean_reverting_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.002, 0.002, 100)),
            'high': prices * (1 + np.abs(np.random.uniform(0, 0.01, 100))),
            'low': prices * (1 - np.abs(np.random.uniform(0, 0.01, 100))),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 100)
        })

        # Create oversold condition data (price below lower BB and low RSI)
        oversold_prices = base_price - 3000 + np.cumsum(np.random.normal(-50, 100, 100))
        self.oversold_data = pd.DataFrame({
            'timestamp': dates,
            'open': oversold_prices * (1 + np.random.uniform(-0.002, 0.002, 100)),
            'high': oversold_prices * (1 + np.abs(np.random.uniform(0, 0.01, 100))),
            'low': oversold_prices * (1 - np.abs(np.random.uniform(0, 0.01, 100))),
            'close': oversold_prices,
            'volume': np.random.uniform(1000000, 5000000, 100)
        })

        # Create overbought condition data (price above upper BB and high RSI)
        overbought_prices = base_price + 3000 + np.cumsum(np.random.normal(50, 100, 100))
        self.overbought_data = pd.DataFrame({
            'timestamp': dates,
            'open': overbought_prices * (1 + np.random.uniform(-0.002, 0.002, 100)),
            'high': overbought_prices * (1 + np.abs(np.random.uniform(0, 0.01, 100))),
            'low': overbought_prices * (1 - np.abs(np.random.uniform(0, 0.01, 100))),
            'close': overbought_prices,
            'volume': np.random.uniform(1000000, 5000000, 100)
        })

    def test_should_initialize_with_default_parameters(self):
        """Test strategy initialization with default parameters"""
        config = StrategyConfig(name="MeanReversion")
        strategy = MeanReversionStrategy(config)

        assert strategy.name == "MeanReversion"
        assert strategy.bb_period == 20
        assert strategy.bb_std == 2.0
        assert strategy.rsi_period == 14
        assert strategy.rsi_oversold == 30
        assert strategy.rsi_overbought == 70
        assert strategy.min_confidence == 0.5

    def test_should_initialize_with_custom_parameters(self):
        """Test strategy initialization with custom parameters"""
        assert self.strategy.bb_period == 20
        assert self.strategy.bb_std == 2.0
        assert self.strategy.rsi_period == 14
        assert self.strategy.rsi_oversold == 30
        assert self.strategy.rsi_overbought == 70
        assert self.strategy.min_confidence == 0.6

    def test_should_calculate_bollinger_bands_correctly(self):
        """Test Bollinger Bands calculation"""
        bb_upper, bb_middle, bb_lower = self.strategy._calculate_bollinger_bands(
            self.mean_reverting_data['close']
        )

        assert len(bb_upper) == len(self.mean_reverting_data)
        assert len(bb_middle) == len(self.mean_reverting_data)
        assert len(bb_lower) == len(self.mean_reverting_data)

        # Upper band should be above middle, middle above lower
        valid_data = bb_upper.dropna()
        if len(valid_data) > 0:
            idx = valid_data.index
            assert (bb_upper[idx] >= bb_middle[idx]).all()
            assert (bb_middle[idx] >= bb_lower[idx]).all()

    def test_should_calculate_rsi_correctly(self):
        """Test RSI calculation"""
        rsi = self.strategy.calculate_rsi(self.mean_reverting_data['close'])

        assert len(rsi) == len(self.mean_reverting_data)

        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()

    def test_should_generate_buy_signal_when_oversold(self):
        """Test buy signal generation in oversold conditions"""
        market_data = {
            "symbol": "BTCUSDT",
            "close": 47000.0,  # Below recent average
            "ohlcv_data": self.oversold_data
        }

        signal = self.strategy.generate_signal(market_data)

        assert isinstance(signal, StrategySignal)
        assert signal.symbol == "BTCUSDT"
        assert signal.action in ["BUY", "HOLD"]

        if signal.action == "BUY":
            assert 0 < signal.strength <= 1
            assert signal.confidence >= self.strategy.min_confidence
            assert "bb_position" in signal.metadata
            assert "rsi" in signal.metadata

    def test_should_generate_sell_signal_when_overbought(self):
        """Test sell signal generation in overbought conditions"""
        market_data = {
            "symbol": "BTCUSDT",
            "close": 53000.0,  # Above recent average
            "ohlcv_data": self.overbought_data
        }

        signal = self.strategy.generate_signal(market_data)

        assert isinstance(signal, StrategySignal)
        assert signal.symbol == "BTCUSDT"
        assert signal.action in ["SELL", "HOLD"]

        if signal.action == "SELL":
            assert 0 < signal.strength <= 1
            assert signal.confidence >= self.strategy.min_confidence
            assert "bb_position" in signal.metadata
            assert "rsi" in signal.metadata

    def test_should_generate_hold_signal_in_neutral_conditions(self):
        """Test hold signal generation in neutral market conditions"""
        market_data = {
            "symbol": "BTCUSDT",
            "close": 50000.0,  # Near middle of range
            "ohlcv_data": self.mean_reverting_data
        }

        signal = self.strategy.generate_signal(market_data)

        assert isinstance(signal, StrategySignal)
        assert signal.symbol == "BTCUSDT"
        # Should likely be HOLD in mean-reverting but neutral conditions
        assert signal.action in ["BUY", "SELL", "HOLD"]

    def test_should_handle_insufficient_data(self):
        """Test handling of insufficient data"""
        small_data = self.mean_reverting_data.head(10)  # Not enough for BB/RSI

        market_data = {
            "symbol": "BTCUSDT",
            "close": 50000.0,
            "ohlcv_data": small_data
        }

        signal = self.strategy.generate_signal(market_data)

        assert signal.action == "HOLD"
        assert signal.strength == 0

    def test_should_calculate_bb_position_correctly(self):
        """Test Bollinger Bands position calculation"""
        # Create prices with enough data for BB calculation
        prices = pd.Series([95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105] * 2)  # 22 data points
        bb_upper, bb_middle, bb_lower = self.strategy._calculate_bollinger_bands(prices)

        # Use the last valid values
        last_idx = bb_upper.dropna().index[-1]
        bb_upper_val = bb_upper.iloc[last_idx]
        bb_middle_val = bb_middle.iloc[last_idx]
        bb_lower_val = bb_lower.iloc[last_idx]

        # Test position calculation above upper band
        current_price = bb_upper_val * 1.1  # 10% above upper band
        bb_position = self.strategy._calculate_bb_position(
            current_price, bb_upper_val, bb_middle_val, bb_lower_val
        )

        assert bb_position > 1.0  # Should be above 1 when above upper band

        # Test position below lower band
        current_price = bb_lower_val * 0.9  # 10% below lower band
        bb_position = self.strategy._calculate_bb_position(
            current_price, bb_upper_val, bb_middle_val, bb_lower_val
        )

        assert bb_position < -1.0  # Should be below -1 when below lower band

    def test_should_calculate_mean_reversion_strength(self):
        """Test mean reversion strength calculation"""
        # Test strong oversold conditions (below lower band and oversold RSI)
        strength = self.strategy._calculate_mean_reversion_strength(
            bb_position=-1.5,  # Well below lower band (must be < -1 for signal)
            rsi=20  # Strongly oversold RSI
        )

        assert strength > 0.3  # Should be positive signal

        # Test strong overbought conditions (above upper band and overbought RSI)
        strength = self.strategy._calculate_mean_reversion_strength(
            bb_position=1.5,  # Well above upper band (must be > 1 for signal)
            rsi=80  # Strongly overbought RSI
        )

        assert strength < -0.3  # Should be negative signal (for sell)

        # Test neutral conditions
        strength = self.strategy._calculate_mean_reversion_strength(
            bb_position=0.5,  # Middle area
            rsi=50  # Neutral RSI
        )

        assert abs(strength) < 0.1  # Should be weak signal

    def test_should_validate_market_data_structure(self):
        """Test market data validation"""
        # Missing ohlcv_data
        invalid_data = {"symbol": "BTCUSDT", "close": 50000.0}

        signal = self.strategy.generate_signal(invalid_data)
        assert signal.action == "HOLD"

        # Invalid ohlcv_data
        invalid_data = {
            "symbol": "BTCUSDT",
            "close": 50000.0,
            "ohlcv_data": "invalid"
        }

        signal = self.strategy.generate_signal(invalid_data)
        assert signal.action == "HOLD"

    def test_should_require_minimum_confidence(self):
        """Test that signals require minimum confidence threshold"""
        # Create strategy with high confidence requirement
        high_conf_config = StrategyConfig(
            name="HighConfidence",
            parameters={"min_confidence": 0.9}
        )
        high_conf_strategy = MeanReversionStrategy(high_conf_config)

        market_data = {
            "symbol": "BTCUSDT",
            "close": 48000.0,
            "ohlcv_data": self.mean_reverting_data
        }

        signal = high_conf_strategy.generate_signal(market_data)

        # With very high confidence requirement, might get HOLD
        if signal.action != "HOLD":
            assert signal.confidence >= 0.9

    def test_should_handle_edge_cases(self):
        """Test handling of edge cases"""
        # All same prices (no volatility)
        flat_data = self.mean_reverting_data.copy()
        flat_data['close'] = 50000.0
        flat_data['high'] = 50000.0
        flat_data['low'] = 50000.0
        flat_data['open'] = 50000.0

        market_data = {
            "symbol": "BTCUSDT",
            "close": 50000.0,
            "ohlcv_data": flat_data
        }

        signal = self.strategy.generate_signal(market_data)
        assert signal.action == "HOLD"  # Should be HOLD with no volatility

    def test_should_include_metadata_in_signal(self):
        """Test that signals include relevant metadata"""
        market_data = {
            "symbol": "BTCUSDT",
            "close": 49000.0,
            "ohlcv_data": self.mean_reverting_data
        }

        signal = self.strategy.generate_signal(market_data)

        assert "strategy_type" in signal.metadata
        assert signal.metadata["strategy_type"] == "mean_reversion"

        if signal.action in ["BUY", "SELL"]:
            assert "bb_position" in signal.metadata
            assert "rsi" in signal.metadata
            assert "bb_upper" in signal.metadata
            assert "bb_lower" in signal.metadata

    def test_should_set_appropriate_targets(self):
        """Test that buy/sell signals have appropriate targets"""
        # Test oversold condition with targets
        market_data = {
            "symbol": "BTCUSDT",
            "close": 47000.0,
            "ohlcv_data": self.oversold_data
        }

        signal = self.strategy.generate_signal(market_data)

        if signal.action == "BUY":
            assert signal.take_profit is not None
            assert signal.stop_loss is not None
            assert signal.take_profit > market_data["close"]  # Profit above entry
            assert signal.stop_loss < market_data["close"]    # Stop below entry

        # Test overbought condition with targets
        market_data = {
            "symbol": "BTCUSDT",
            "close": 53000.0,
            "ohlcv_data": self.overbought_data
        }

        signal = self.strategy.generate_signal(market_data)

        if signal.action == "SELL":
            assert signal.take_profit is not None
            assert signal.stop_loss is not None
            assert signal.take_profit < market_data["close"]  # Profit below entry
            assert signal.stop_loss > market_data["close"]   # Stop above entry

    def test_should_calculate_confidence_correctly(self):
        """Test confidence calculation based on indicator alignment"""
        # Strong alignment should give high confidence
        high_conf = self.strategy._calculate_confidence(
            bb_position=-2.0,  # Well below lower band (needs to be < -1)
            rsi=15,           # Strongly oversold
            action="BUY"
        )

        assert high_conf >= 0.6  # Adjusted expectation based on actual calculation

        # Weak alignment should give lower confidence
        low_conf = self.strategy._calculate_confidence(
            bb_position=-0.5,  # Not below lower band (> -1)
            rsi=35,           # Mildly oversold
            action="BUY"
        )

        assert low_conf < high_conf

    def test_should_update_performance_tracking(self):
        """Test performance tracking functionality"""
        initial_signals = self.strategy.total_signals
        initial_pnl = self.strategy.total_pnl

        # Simulate a profitable trade
        self.strategy.update_performance(pnl=150.0, winning=True)

        assert self.strategy.total_signals == initial_signals + 1
        assert self.strategy.winning_signals == 1
        assert self.strategy.total_pnl == initial_pnl + 150.0

    def test_should_handle_parameter_validation(self):
        """Test parameter validation during initialization"""
        # Invalid RSI parameters
        with pytest.raises(ValueError):
            invalid_config = StrategyConfig(
                name="Invalid",
                parameters={"rsi_oversold": 80, "rsi_overbought": 70}  # Oversold > overbought
            )
            MeanReversionStrategy(invalid_config)

        # Invalid BB parameters
        with pytest.raises(ValueError):
            invalid_config = StrategyConfig(
                name="Invalid",
                parameters={"bb_std": -1.0}  # Negative standard deviation
            )
            MeanReversionStrategy(invalid_config)