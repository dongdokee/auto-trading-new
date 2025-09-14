"""
Unit tests for BaseStrategy interface and StrategySignal
Tests the abstract interface and signal data structures for all trading strategies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import asdict
from typing import Dict, Any
from unittest.mock import MagicMock

from src.strategy_engine.base_strategy import BaseStrategy, StrategySignal, StrategyConfig


class TestStrategySignal:
    """Test StrategySignal dataclass functionality"""

    def test_should_create_strategy_signal_with_required_fields(self):
        """Test basic StrategySignal creation with required fields"""
        signal = StrategySignal(
            symbol="BTCUSDT",
            action="BUY",
            strength=0.8,
            confidence=0.7
        )

        assert signal.symbol == "BTCUSDT"
        assert signal.action == "BUY"
        assert signal.strength == 0.8
        assert signal.confidence == 0.7
        assert signal.stop_loss is None
        assert signal.take_profit is None
        assert signal.metadata == {}

    def test_should_create_strategy_signal_with_optional_fields(self):
        """Test StrategySignal creation with all optional fields"""
        metadata = {"strategy_type": "trend_following", "atr": 1500.0}

        signal = StrategySignal(
            symbol="ETHUSDT",
            action="SELL",
            strength=0.6,
            confidence=0.9,
            stop_loss=1800.0,
            take_profit=1700.0,
            metadata=metadata
        )

        assert signal.symbol == "ETHUSDT"
        assert signal.action == "SELL"
        assert signal.stop_loss == 1800.0
        assert signal.take_profit == 1700.0
        assert signal.metadata == metadata

    def test_should_validate_action_values(self):
        """Test that action field only accepts valid values"""
        # Valid actions should work
        valid_actions = ["BUY", "SELL", "HOLD", "CLOSE"]
        for action in valid_actions:
            signal = StrategySignal("BTCUSDT", action, 0.5, 0.5)
            assert signal.action == action

    def test_should_validate_strength_range(self):
        """Test that strength is between 0 and 1"""
        # Valid strength values
        signal = StrategySignal("BTCUSDT", "BUY", 0.0, 0.5)
        assert signal.strength == 0.0

        signal = StrategySignal("BTCUSDT", "BUY", 1.0, 0.5)
        assert signal.strength == 1.0

        signal = StrategySignal("BTCUSDT", "BUY", 0.5, 0.5)
        assert signal.strength == 0.5

    def test_should_validate_confidence_range(self):
        """Test that confidence is between 0 and 1"""
        # Valid confidence values
        signal = StrategySignal("BTCUSDT", "BUY", 0.5, 0.0)
        assert signal.confidence == 0.0

        signal = StrategySignal("BTCUSDT", "BUY", 0.5, 1.0)
        assert signal.confidence == 1.0

        signal = StrategySignal("BTCUSDT", "BUY", 0.5, 0.8)
        assert signal.confidence == 0.8

    def test_should_convert_to_dict(self):
        """Test conversion to dictionary"""
        signal = StrategySignal(
            symbol="BTCUSDT",
            action="BUY",
            strength=0.8,
            confidence=0.7,
            stop_loss=45000.0,
            metadata={"test": "value"}
        )

        signal_dict = asdict(signal)

        assert signal_dict["symbol"] == "BTCUSDT"
        assert signal_dict["action"] == "BUY"
        assert signal_dict["strength"] == 0.8
        assert signal_dict["stop_loss"] == 45000.0
        assert signal_dict["metadata"]["test"] == "value"


class TestStrategyConfig:
    """Test StrategyConfig dataclass functionality"""

    def test_should_create_strategy_config_with_defaults(self):
        """Test StrategyConfig creation with default values"""
        config = StrategyConfig(name="TestStrategy")

        assert config.name == "TestStrategy"
        assert config.enabled is True
        assert config.weight == 1.0
        assert config.parameters == {}

    def test_should_create_strategy_config_with_custom_values(self):
        """Test StrategyConfig creation with custom values"""
        params = {"fast_ma": 10, "slow_ma": 30}
        config = StrategyConfig(
            name="MA_Strategy",
            enabled=False,
            weight=0.5,
            parameters=params
        )

        assert config.name == "MA_Strategy"
        assert config.enabled is False
        assert config.weight == 0.5
        assert config.parameters == params


class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.generate_signal_calls = []

    def generate_signal(self, market_data: Dict[str, Any]) -> StrategySignal:
        """Mock implementation for testing"""
        self.generate_signal_calls.append(market_data)

        return StrategySignal(
            symbol=market_data.get("symbol", "BTCUSDT"),
            action="BUY",
            strength=0.5,
            confidence=0.6
        )


class TestBaseStrategy:
    """Test BaseStrategy abstract class functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = StrategyConfig(
            name="TestStrategy",
            parameters={"param1": 10, "param2": 0.5}
        )
        self.strategy = ConcreteStrategy(self.config)

    def test_should_initialize_with_config(self):
        """Test BaseStrategy initialization with config"""
        assert self.strategy.name == "TestStrategy"
        assert self.strategy.enabled is True
        assert self.strategy.weight == 1.0
        assert self.strategy.parameters["param1"] == 10
        assert self.strategy.parameters["param2"] == 0.5

    def test_should_track_performance_metrics(self):
        """Test that strategy tracks performance metrics"""
        assert hasattr(self.strategy, 'total_signals')
        assert hasattr(self.strategy, 'winning_signals')
        assert hasattr(self.strategy, 'total_pnl')

        assert self.strategy.total_signals == 0
        assert self.strategy.winning_signals == 0
        assert self.strategy.total_pnl == 0.0

    def test_should_generate_signal_with_market_data(self):
        """Test signal generation with market data"""
        market_data = {
            "symbol": "ETHUSDT",
            "close": 2000.0,
            "volume": 1000000
        }

        signal = self.strategy.generate_signal(market_data)

        assert isinstance(signal, StrategySignal)
        assert signal.symbol == "ETHUSDT"
        assert len(self.strategy.generate_signal_calls) == 1

    def test_should_update_performance_metrics(self):
        """Test updating performance metrics"""
        # Simulate winning trade
        self.strategy.update_performance(pnl=100.0, winning=True)

        assert self.strategy.total_signals == 1
        assert self.strategy.winning_signals == 1
        assert self.strategy.total_pnl == 100.0

        # Simulate losing trade
        self.strategy.update_performance(pnl=-50.0, winning=False)

        assert self.strategy.total_signals == 2
        assert self.strategy.winning_signals == 1  # Still 1
        assert self.strategy.total_pnl == 50.0

    def test_should_calculate_win_rate(self):
        """Test win rate calculation"""
        # No trades yet
        assert self.strategy.get_win_rate() == 0.0

        # Add some trades
        self.strategy.update_performance(100.0, True)
        self.strategy.update_performance(-50.0, False)
        self.strategy.update_performance(75.0, True)

        expected_win_rate = 2 / 3  # 2 winners out of 3 trades
        assert self.strategy.get_win_rate() == pytest.approx(expected_win_rate, rel=1e-6)

    def test_should_calculate_average_pnl(self):
        """Test average PnL calculation"""
        # No trades yet
        assert self.strategy.get_average_pnl() == 0.0

        # Add some trades
        self.strategy.update_performance(100.0, True)
        self.strategy.update_performance(-50.0, False)
        self.strategy.update_performance(25.0, True)

        expected_avg_pnl = 75.0 / 3  # Total PnL / number of trades
        assert self.strategy.get_average_pnl() == pytest.approx(expected_avg_pnl, rel=1e-6)

    def test_should_get_strategy_info(self):
        """Test getting strategy information"""
        info = self.strategy.get_strategy_info()

        assert info["name"] == "TestStrategy"
        assert info["enabled"] is True
        assert info["weight"] == 1.0
        assert info["total_signals"] == 0
        assert info["winning_signals"] == 0
        assert info["total_pnl"] == 0.0
        assert info["win_rate"] == 0.0
        assert info["average_pnl"] == 0.0

    def test_should_disable_and_enable_strategy(self):
        """Test enabling and disabling strategy"""
        assert self.strategy.enabled is True

        self.strategy.disable()
        assert self.strategy.enabled is False

        self.strategy.enable()
        assert self.strategy.enabled is True

    def test_should_update_weight(self):
        """Test updating strategy weight"""
        assert self.strategy.weight == 1.0

        self.strategy.update_weight(0.75)
        assert self.strategy.weight == 0.75

        # Test bounds
        self.strategy.update_weight(-0.5)  # Should be clamped to 0
        assert self.strategy.weight == 0.0

        self.strategy.update_weight(1.5)   # Should be clamped to 1
        assert self.strategy.weight == 1.0


class TestBaseStrategyAbstract:
    """Test that BaseStrategy is properly abstract"""

    def test_should_not_instantiate_base_strategy_directly(self):
        """Test that BaseStrategy cannot be instantiated directly"""
        config = StrategyConfig(name="Test")

        with pytest.raises(TypeError):
            BaseStrategy(config)  # Should fail because generate_signal is abstract