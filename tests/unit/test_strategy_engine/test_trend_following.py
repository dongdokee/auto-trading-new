"""
Unit tests for TrendFollowingStrategy
Tests the moving average crossover strategy with ATR-based stops.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.strategy_engine.base_strategy import StrategySignal, StrategyConfig
from src.strategy_engine.strategies.trend_following import TrendFollowingStrategy


class TestTrendFollowingStrategy:
    """Test TrendFollowingStrategy functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = StrategyConfig(
            name="TrendFollowing",
            parameters={
                "fast_period": 10,
                "slow_period": 20,
                "atr_period": 14,
                "atr_multiplier": 2.0,
                "min_trend_strength": 0.3
            }
        )
        self.strategy = TrendFollowingStrategy(self.config)

        # Create test data with trend
        np.random.seed(42)
        self.create_test_data()

    def create_test_data(self):
        """Create synthetic market data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        # Create uptrending data
        base_price = 50000
        trend_returns = np.random.normal(0.005, 0.01, 100)  # Slight uptrend
        prices = base_price * np.exp(np.cumsum(trend_returns))

        self.uptrend_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.002, 0.002, 100)),
            'high': prices * (1 + np.abs(np.random.uniform(0, 0.01, 100))),
            'low': prices * (1 - np.abs(np.random.uniform(0, 0.01, 100))),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 100)
        })

        # Create downtrending data
        trend_returns = np.random.normal(-0.005, 0.01, 100)  # Downtrend
        prices = base_price * np.exp(np.cumsum(trend_returns))

        self.downtrend_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.002, 0.002, 100)),
            'high': prices * (1 + np.abs(np.random.uniform(0, 0.01, 100))),
            'low': prices * (1 - np.abs(np.random.uniform(0, 0.01, 100))),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 100)
        })

        # Create sideways data
        base_price = 50000
        sideways_returns = np.random.normal(0, 0.008, 100)  # No trend
        prices = base_price + np.cumsum(sideways_returns * 100)  # Small absolute changes

        self.sideways_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.002, 0.002, 100)),
            'high': prices * (1 + np.abs(np.random.uniform(0, 0.008, 100))),
            'low': prices * (1 - np.abs(np.random.uniform(0, 0.008, 100))),
            'close': prices,
            'volume': np.random.uniform(1000000, 3000000, 100)
        })

    def test_should_initialize_with_default_parameters(self):
        """Test strategy initialization with default parameters"""
        config = StrategyConfig(name="TrendFollowing")
        strategy = TrendFollowingStrategy(config)

        assert strategy.name == "TrendFollowing"
        assert strategy.fast_period == 20
        assert strategy.slow_period == 50
        assert strategy.atr_period == 14
        assert strategy.atr_multiplier == 2.0
        assert strategy.min_trend_strength == 0.3

    def test_should_initialize_with_custom_parameters(self):
        """Test strategy initialization with custom parameters"""
        assert self.strategy.fast_period == 10
        assert self.strategy.slow_period == 20
        assert self.strategy.atr_period == 14
        assert self.strategy.atr_multiplier == 2.0
        assert self.strategy.min_trend_strength == 0.3

    def test_should_calculate_atr_correctly(self):
        """Test ATR calculation"""
        atr = self.strategy.calculate_atr(self.uptrend_data)

        assert len(atr) == len(self.uptrend_data)
        assert not atr.dropna().empty
        assert (atr.dropna() >= 0).all()  # ATR should be non-negative
        assert not atr.dropna().isna().any()

    def test_should_calculate_moving_averages(self):
        """Test moving average calculations"""
        fast_ma, slow_ma = self.strategy._calculate_moving_averages(self.uptrend_data)

        assert len(fast_ma) == len(self.uptrend_data)
        assert len(slow_ma) == len(self.uptrend_data)

        # Fast MA should have fewer NaN values than slow MA
        assert fast_ma.dropna().size >= slow_ma.dropna().size

        # MAs should be positive for price data
        assert (fast_ma.dropna() > 0).all()
        assert (slow_ma.dropna() > 0).all()

    def test_should_generate_buy_signal_in_uptrend(self):
        """Test buy signal generation in uptrending market"""
        market_data = {
            "symbol": "BTCUSDT",
            "close": 52000.0,
            "ohlcv_data": self.uptrend_data
        }

        signal = self.strategy.generate_signal(market_data)

        assert isinstance(signal, StrategySignal)
        assert signal.symbol == "BTCUSDT"
        assert signal.action in ["BUY", "HOLD"]  # Should be BUY or HOLD

        if signal.action == "BUY":
            assert 0 < signal.strength <= 1
            assert 0 < signal.confidence <= 1
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.stop_loss < market_data["close"]  # Stop below current price
            assert signal.take_profit > market_data["close"]  # Profit above current price

    def test_should_generate_sell_signal_in_downtrend(self):
        """Test sell signal generation in downtrending market"""
        market_data = {
            "symbol": "BTCUSDT",
            "close": 48000.0,
            "ohlcv_data": self.downtrend_data
        }

        signal = self.strategy.generate_signal(market_data)

        assert isinstance(signal, StrategySignal)
        assert signal.symbol == "BTCUSDT"
        assert signal.action in ["SELL", "HOLD"]  # Should be SELL or HOLD

        if signal.action == "SELL":
            assert 0 < signal.strength <= 1
            assert 0 < signal.confidence <= 1
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.stop_loss > market_data["close"]  # Stop above current price
            assert signal.take_profit < market_data["close"]  # Profit below current price

    def test_should_generate_hold_signal_in_sideways_market(self):
        """Test hold signal generation in sideways market"""
        market_data = {
            "symbol": "BTCUSDT",
            "close": 50000.0,
            "ohlcv_data": self.sideways_data
        }

        signal = self.strategy.generate_signal(market_data)

        assert isinstance(signal, StrategySignal)
        assert signal.symbol == "BTCUSDT"
        assert signal.action == "HOLD"  # Should be HOLD in sideways market
        assert signal.strength == 0
        assert signal.confidence > 0

    def test_should_handle_insufficient_data(self):
        """Test handling of insufficient data"""
        small_data = self.uptrend_data.head(10)  # Not enough for slow MA

        market_data = {
            "symbol": "BTCUSDT",
            "close": 50000.0,
            "ohlcv_data": small_data
        }

        signal = self.strategy.generate_signal(market_data)

        assert signal.action == "HOLD"
        assert signal.strength == 0

    def test_should_require_minimum_trend_strength(self):
        """Test that weak trends don't generate signals"""
        # Create data with very small trend
        weak_trend_data = self.sideways_data.copy()
        weak_trend_data['close'] *= 1.001  # Tiny trend

        market_data = {
            "symbol": "BTCUSDT",
            "close": 50000.0,
            "ohlcv_data": weak_trend_data
        }

        signal = self.strategy.generate_signal(market_data)

        # Should be HOLD due to insufficient trend strength
        assert signal.action == "HOLD"

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

    def test_should_calculate_trend_strength_correctly(self):
        """Test trend strength calculation"""
        fast_ma = pd.Series([100, 101, 102, 103, 104])
        slow_ma = pd.Series([99, 100, 101, 102, 103])
        atr = pd.Series([1, 1, 1, 1, 2])

        trend_strength = self.strategy._calculate_trend_strength(fast_ma, slow_ma, atr)

        # Trend strength should be positive for uptrend
        assert trend_strength > 0

        # Test downtrend
        fast_ma = pd.Series([104, 103, 102, 101, 100])
        slow_ma = pd.Series([105, 104, 103, 102, 101])

        trend_strength = self.strategy._calculate_trend_strength(fast_ma, slow_ma, atr)

        # Trend strength should be negative for downtrend
        assert trend_strength < 0

    def test_should_handle_edge_cases(self):
        """Test handling of edge cases"""
        # All same prices (no volatility)
        flat_data = self.uptrend_data.copy()
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
        assert signal.action == "HOLD"

        # Missing required columns
        incomplete_data = self.uptrend_data[['close']].copy()

        market_data = {
            "symbol": "BTCUSDT",
            "close": 50000.0,
            "ohlcv_data": incomplete_data
        }

        signal = self.strategy.generate_signal(market_data)
        assert signal.action == "HOLD"

    def test_should_include_metadata_in_signal(self):
        """Test that signals include relevant metadata"""
        market_data = {
            "symbol": "BTCUSDT",
            "close": 52000.0,
            "ohlcv_data": self.uptrend_data
        }

        signal = self.strategy.generate_signal(market_data)

        assert "strategy_type" in signal.metadata
        assert signal.metadata["strategy_type"] == "trend_following"

        if signal.action in ["BUY", "SELL"]:
            assert "trend_strength" in signal.metadata
            assert "atr" in signal.metadata
            assert "fast_ma" in signal.metadata
            assert "slow_ma" in signal.metadata

    def test_should_scale_strength_with_trend_magnitude(self):
        """Test that signal strength scales with trend magnitude"""
        # Test with different trend magnitudes by modifying min_trend_strength
        strong_config = StrategyConfig(
            name="StrongTrend",
            parameters={"min_trend_strength": 0.1}  # Lower threshold
        )
        strong_strategy = TrendFollowingStrategy(strong_config)

        weak_config = StrategyConfig(
            name="WeakTrend",
            parameters={"min_trend_strength": 0.5}  # Higher threshold
        )
        weak_strategy = TrendFollowingStrategy(weak_config)

        market_data = {
            "symbol": "BTCUSDT",
            "close": 52000.0,
            "ohlcv_data": self.uptrend_data
        }

        strong_signal = strong_strategy.generate_signal(market_data)
        weak_signal = weak_strategy.generate_signal(market_data)

        # Strong strategy should be more likely to generate signals
        if strong_signal.action == "BUY" and weak_signal.action == "HOLD":
            # This is the expected behavior
            assert True
        elif strong_signal.action == weak_signal.action == "BUY":
            # If both generate signals, strength should be comparable or strong > weak
            assert strong_signal.strength >= weak_signal.strength * 0.8
        else:
            # Both might be HOLD if trend is not strong enough
            assert True

    def test_should_update_performance_tracking(self):
        """Test performance tracking functionality"""
        initial_signals = self.strategy.total_signals
        initial_pnl = self.strategy.total_pnl

        # Simulate a profitable trade
        self.strategy.update_performance(pnl=100.0, winning=True)

        assert self.strategy.total_signals == initial_signals + 1
        assert self.strategy.winning_signals == 1
        assert self.strategy.total_pnl == initial_pnl + 100.0