"""
Integration tests for Strategy Engine System

Tests the complete strategy engine system including regime detection,
strategy matrix, strategy manager, and signal aggregation.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.strategy_engine import (
    StrategyManager,
    StrategyConfig,
    NoLookAheadRegimeDetector,
    StrategyMatrix,
    TrendFollowingStrategy,
    MeanReversionStrategy
)


class TestStrategyEngineIntegration:
    """Integration tests for complete strategy engine system"""

    def setup_method(self):
        """Setup test fixtures"""
        # Create strategy manager with default strategies
        self.manager = StrategyManager()

        # Create test market data
        np.random.seed(42)
        self.create_test_data()

    def create_test_data(self):
        """Create synthetic market data for testing"""
        dates = pd.date_range('2024-01-01', periods=200, freq='D')

        # Create trending market data
        base_price = 50000
        trend_returns = np.random.normal(0.003, 0.015, 200)  # Slight uptrend
        prices = base_price * np.exp(np.cumsum(trend_returns))

        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.002, 0.002, 200)),
            'high': prices * (1 + np.abs(np.random.uniform(0, 0.01, 200))),
            'low': prices * (1 - np.abs(np.random.uniform(0, 0.01, 200))),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 200)
        })

        self.market_data = {
            "symbol": "BTCUSDT",
            "close": float(self.test_data['close'].iloc[-1]),
            "ohlcv_data": self.test_data
        }

    def test_should_initialize_strategy_manager_successfully(self):
        """Test strategy manager initialization"""
        assert len(self.manager.strategies) == 2  # Default: TrendFollowing + MeanReversion
        assert "TrendFollowing" in self.manager.strategies
        assert "MeanReversion" in self.manager.strategies

        assert self.manager.regime_detector is not None
        assert self.manager.strategy_matrix is not None

    def test_should_generate_trading_signals_successfully(self):
        """Test complete signal generation workflow"""
        # Generate signals
        result = self.manager.generate_trading_signals(
            market_data=self.market_data,
            current_index=150  # Use sufficient data for regime detection
        )

        # Validate result structure
        assert "primary_signal" in result
        assert "strategy_signals" in result
        assert "regime_info" in result
        assert "allocation" in result
        assert "timestamp" in result

        # Validate primary signal
        primary_signal = result["primary_signal"]
        assert hasattr(primary_signal, 'symbol')
        assert hasattr(primary_signal, 'action')
        assert hasattr(primary_signal, 'strength')
        assert hasattr(primary_signal, 'confidence')

        assert primary_signal.symbol == "BTCUSDT"
        assert primary_signal.action in ["BUY", "SELL", "HOLD"]
        assert 0 <= primary_signal.strength <= 1
        assert 0 <= primary_signal.confidence <= 1

    def test_should_detect_market_regime_correctly(self):
        """Test regime detection integration"""
        regime_detector = NoLookAheadRegimeDetector()

        # Test with sufficient data
        regime_info = regime_detector.detect_regime(self.test_data, current_index=150)

        assert "regime" in regime_info
        assert "confidence" in regime_info
        assert "volatility_forecast" in regime_info
        assert "duration" in regime_info

        assert regime_info["regime"] in ["BULL", "BEAR", "SIDEWAYS", "NEUTRAL"]
        assert 0 <= regime_info["confidence"] <= 1
        assert regime_info["volatility_forecast"] > 0

    def test_should_allocate_strategies_based_on_regime(self):
        """Test strategy matrix allocation"""
        strategy_matrix = StrategyMatrix()

        # Test different regime scenarios
        regime_scenarios = [
            {"regime": "BULL", "volatility_forecast": 0.02, "confidence": 0.8},
            {"regime": "BEAR", "volatility_forecast": 0.04, "confidence": 0.7},
            {"regime": "SIDEWAYS", "volatility_forecast": 0.015, "confidence": 0.9},
            {"regime": "NEUTRAL", "volatility_forecast": 0.025, "confidence": 0.5},
        ]

        for regime_info in regime_scenarios:
            allocation = strategy_matrix.get_strategy_allocation(regime_info)

            # Validate allocation structure
            assert isinstance(allocation, dict)
            assert len(allocation) > 0

            # Check that weights sum approximately to 1
            total_weight = sum(alloc.weight for alloc in allocation.values())
            assert abs(total_weight - 1.0) < 0.01

            # Check all allocations are valid
            for strategy_name, alloc in allocation.items():
                assert 0 <= alloc.weight <= 1
                assert 0.5 <= alloc.confidence_multiplier <= 1.2
                assert isinstance(alloc.enabled, bool)

    def test_should_aggregate_multiple_strategy_signals(self):
        """Test signal aggregation from multiple strategies"""
        # Create custom market data that should trigger signals
        strong_trend_data = self.test_data.copy()
        strong_trend_data['close'] = strong_trend_data['close'] * 1.1  # Strong uptrend

        market_data = {
            "symbol": "ETHUSDT",
            "close": float(strong_trend_data['close'].iloc[-1]),
            "ohlcv_data": strong_trend_data
        }

        result = self.manager.generate_trading_signals(market_data, current_index=180)

        # Should have signals from individual strategies
        strategy_signals = result["strategy_signals"]
        assert len(strategy_signals) >= 1

        # Primary signal should aggregate individual signals
        primary_signal = result["primary_signal"]
        assert primary_signal.action in ["BUY", "SELL", "HOLD"]

        if primary_signal.action in ["BUY", "SELL"]:
            assert primary_signal.strength > 0
            assert "contributing_strategies" in primary_signal.metadata

    def test_should_handle_insufficient_data_gracefully(self):
        """Test handling of insufficient data"""
        # Create minimal data
        small_data = self.test_data.head(10)
        market_data = {
            "symbol": "BTCUSDT",
            "close": 50000.0,
            "ohlcv_data": small_data
        }

        result = self.manager.generate_trading_signals(market_data, current_index=5)

        # Should still generate a result without crashing
        assert "primary_signal" in result
        assert result["primary_signal"].action == "HOLD"  # Should default to HOLD

    def test_should_track_signal_history(self):
        """Test signal history tracking"""
        initial_history_length = len(self.manager.signal_history)

        # Generate multiple signals
        for i in range(5):
            market_data = self.market_data.copy()
            market_data["close"] *= (1 + (i * 0.01))  # Slightly different prices

            self.manager.generate_trading_signals(market_data, current_index=150 + i)

        # Should have recorded all signals
        assert len(self.manager.signal_history) == initial_history_length + 5

        # Each history entry should have required fields
        for entry in self.manager.signal_history[-5:]:
            assert "timestamp" in entry
            assert "symbol" in entry
            assert "primary_signal" in entry
            assert "regime_info" in entry

    def test_should_update_strategy_performance(self):
        """Test strategy performance tracking"""
        strategy_name = "TrendFollowing"
        initial_signals = self.manager.strategies[strategy_name].total_signals

        # Update performance
        self.manager.update_strategy_performance(strategy_name, pnl=100.0, winning=True)

        updated_signals = self.manager.strategies[strategy_name].total_signals
        assert updated_signals == initial_signals + 1
        assert self.manager.strategies[strategy_name].winning_signals == 1
        assert self.manager.strategies[strategy_name].total_pnl == 100.0

    def test_should_provide_comprehensive_system_status(self):
        """Test system status reporting"""
        status = self.manager.get_system_status()

        # Validate status structure
        required_fields = [
            "total_strategies",
            "enabled_strategies",
            "regime_detector_status",
            "strategy_performance",
            "recent_signals"
        ]

        for field in required_fields:
            assert field in status

        # Validate specific fields
        assert status["total_strategies"] >= 2
        assert status["enabled_strategies"] <= status["total_strategies"]
        assert isinstance(status["strategy_performance"], dict)

    def test_should_handle_strategy_errors_gracefully(self):
        """Test error handling when individual strategies fail"""
        # Create invalid market data to trigger errors
        invalid_market_data = {
            "symbol": "BTCUSDT",
            "close": "invalid",  # Invalid price
            "ohlcv_data": "not_a_dataframe"
        }

        # Should not crash, should return error signal
        result = self.manager.generate_trading_signals(invalid_market_data)

        assert "primary_signal" in result
        assert result["primary_signal"].action == "HOLD"

    def test_should_work_with_custom_strategy_configs(self):
        """Test with custom strategy configurations"""
        custom_configs = [
            StrategyConfig(
                name="TrendFollowing",
                parameters={
                    "fast_period": 10,
                    "slow_period": 30,
                    "min_trend_strength": 0.2
                }
            ),
            StrategyConfig(
                name="MeanReversion",
                parameters={
                    "bb_period": 15,
                    "rsi_period": 10,
                    "min_confidence": 0.7
                }
            )
        ]

        custom_manager = StrategyManager(custom_configs)

        # Should initialize with custom parameters
        trend_strategy = custom_manager.strategies["TrendFollowing"]
        assert trend_strategy.fast_period == 10
        assert trend_strategy.slow_period == 30

        mean_strategy = custom_manager.strategies["MeanReversion"]
        assert mean_strategy.bb_period == 15
        assert mean_strategy.rsi_period == 10

        # Should still generate signals
        result = custom_manager.generate_trading_signals(self.market_data, current_index=150)
        assert result["primary_signal"].action in ["BUY", "SELL", "HOLD"]

    def test_should_integrate_with_regime_based_allocation(self):
        """Test integration between regime detection and strategy allocation"""
        # Generate signals multiple times to see allocation changes
        results = []

        for i in range(5):
            # Slightly modify data to potentially change regime
            modified_data = self.market_data.copy()
            modified_data["close"] *= (1 + (i * 0.002))

            result = self.manager.generate_trading_signals(modified_data, current_index=150 + i)
            results.append(result)

        # Should have regime information affecting allocation
        for result in results:
            assert "allocation" in result
            assert len(result["allocation"]) > 0

            # Allocation should change based on regime (at least sometimes)
            regime = result["regime_info"]["regime"]
            allocation = result["allocation"]

            assert regime in ["BULL", "BEAR", "SIDEWAYS", "NEUTRAL"]
            assert sum(allocation.values()) > 0.99  # Should sum to approximately 1

    def test_should_reset_performance_correctly(self):
        """Test performance reset functionality"""
        # Add some performance data
        self.manager.update_strategy_performance("TrendFollowing", 100.0, True)
        self.manager.generate_trading_signals(self.market_data, current_index=150)

        # Verify data exists
        assert self.manager.strategies["TrendFollowing"].total_signals > 0
        assert len(self.manager.signal_history) > 0

        # Reset performance
        self.manager.reset_all_performance()

        # Verify reset
        assert self.manager.strategies["TrendFollowing"].total_signals == 0
        assert self.manager.strategies["TrendFollowing"].total_pnl == 0.0
        assert len(self.manager.signal_history) == 0