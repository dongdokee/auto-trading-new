"""
Strategy Manager - Central Coordination System

Coordinates all trading strategies, regime detection, and signal aggregation.
Manages the complete strategy execution pipeline from market data to final trading signals.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, StrategySignal, StrategyConfig
from .regime_detector import NoLookAheadRegimeDetector
from .strategy_matrix import StrategyMatrix, StrategyAllocation
from .strategies import TrendFollowingStrategy, MeanReversionStrategy
from src.core.patterns import BaseManager, LoggerFactory


class StrategyManager(BaseManager):
    """
    Central coordinator for all trading strategies

    Responsibilities:
    - Manage multiple strategy instances
    - Coordinate with regime detection system
    - Aggregate signals from multiple strategies
    - Apply dynamic strategy allocation weights
    - Handle strategy lifecycle and performance tracking
    """

    def __init__(self, strategy_configs: Optional[List[StrategyConfig]] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy manager

        Args:
            strategy_configs: List of strategy configurations. If None, uses defaults.
            config: Manager configuration
        """
        super().__init__("StrategyManager", config)

        # Core components
        self.regime_detector = NoLookAheadRegimeDetector()
        self.strategy_matrix = StrategyMatrix()

        # Strategy instances
        self.strategies: Dict[str, BaseStrategy] = {}

        # Initialize default strategies if no configs provided
        if strategy_configs is None:
            strategy_configs = self._get_default_configs()

        # Initialize strategies
        for config in strategy_configs:
            self._add_strategy(config)

        # Signal aggregation settings
        self.min_strategy_agreement = self.config.get('min_strategy_agreement', 0.6)  # Minimum agreement threshold for signals
        self.max_position_size = self.config.get('max_position_size', 1.0)  # Maximum position size multiplier

        # Performance tracking
        self.signal_history: List[Dict[str, Any]] = []

        # Setup logger
        self.logger = LoggerFactory.get_strategy_logger("StrategyManager")

    async def _do_initialize(self) -> None:
        """Initialize strategy manager"""
        self.logger.info("Initializing strategy manager with {} strategies".format(len(self.strategies)))

    async def _do_start(self) -> None:
        """Start strategy manager"""
        self.logger.info("Starting strategy manager")

    async def _do_stop(self) -> None:
        """Stop strategy manager"""
        self.logger.info("Stopping strategy manager")

    def _get_default_configs(self) -> List[StrategyConfig]:
        """Get default strategy configurations"""
        return [
            StrategyConfig(
                name="TrendFollowing",
                parameters={
                    "fast_period": 20,
                    "slow_period": 50,
                    "min_trend_strength": 0.3
                }
            ),
            StrategyConfig(
                name="MeanReversion",
                parameters={
                    "bb_period": 20,
                    "rsi_period": 14,
                    "min_confidence": 0.6
                }
            )
        ]

    def _add_strategy(self, config: StrategyConfig) -> None:
        """
        Add a strategy instance

        Args:
            config: Strategy configuration
        """
        # Map strategy names to classes
        strategy_classes = {
            "TrendFollowing": TrendFollowingStrategy,
            "MeanReversion": MeanReversionStrategy,
        }

        strategy_class = strategy_classes.get(config.name)
        if strategy_class is None:
            raise ValueError(f"Unknown strategy type: {config.name}")

        strategy_instance = strategy_class(config)
        self.strategies[config.name] = strategy_instance

    def generate_trading_signals(
        self,
        market_data: Dict[str, Any],
        current_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate aggregated trading signals from all strategies

        Args:
            market_data: Market data containing OHLCV and current price
            current_index: Current time index for regime detection

        Returns:
            dict: Aggregated signal information including:
                - primary_signal: Main trading signal
                - strategy_signals: Individual strategy signals
                - regime_info: Current market regime information
                - allocation: Strategy allocation weights
        """
        try:
            # Step 1: Detect current market regime
            ohlcv_data = market_data.get("ohlcv_data")
            if current_index is not None and isinstance(ohlcv_data, pd.DataFrame):
                regime_info = self.regime_detector.detect_regime(ohlcv_data, current_index)
            else:
                regime_info = {
                    'regime': 'NEUTRAL',
                    'confidence': 0.5,
                    'volatility_forecast': 0.02,
                    'duration': 0
                }

            # Step 2: Get strategy allocation based on regime
            allocation = self.strategy_matrix.get_strategy_allocation(regime_info)

            # Step 3: Generate signals from individual strategies
            strategy_signals = {}
            for strategy_name, strategy in self.strategies.items():
                if strategy.enabled:
                    try:
                        signal = strategy.generate_signal(market_data)
                        strategy_signals[strategy_name] = signal
                    except Exception as e:
                        # Handle strategy-specific errors gracefully
                        print(f"Error in strategy {strategy_name}: {e}")
                        strategy_signals[strategy_name] = self._error_signal(market_data.get("symbol", "UNKNOWN"))

            # Step 4: Aggregate signals using allocation weights
            primary_signal = self._aggregate_signals(strategy_signals, allocation, regime_info)

            # Step 5: Log and track performance
            signal_data = {
                "timestamp": pd.Timestamp.now(),
                "symbol": market_data.get("symbol", "UNKNOWN"),
                "primary_signal": primary_signal,
                "strategy_signals": strategy_signals,
                "regime_info": regime_info,
                "allocation": {name: alloc.weight for name, alloc in allocation.items()}
            }

            self.signal_history.append(signal_data)

            # Keep only recent history
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]

            return signal_data

        except Exception as e:
            # Handle any system-level errors
            print(f"Error in strategy manager: {e}")
            return {
                "timestamp": pd.Timestamp.now(),
                "symbol": market_data.get("symbol", "UNKNOWN"),
                "primary_signal": self._error_signal(market_data.get("symbol", "UNKNOWN")),
                "strategy_signals": {},
                "regime_info": {"regime": "NEUTRAL", "confidence": 0.0},
                "allocation": {}
            }

    def _aggregate_signals(
        self,
        strategy_signals: Dict[str, StrategySignal],
        allocation: Dict[str, StrategyAllocation],
        regime_info: Dict[str, Any]
    ) -> StrategySignal:
        """
        Aggregate individual strategy signals into a primary signal

        Args:
            strategy_signals: Individual strategy signals
            allocation: Strategy allocation weights
            regime_info: Current regime information

        Returns:
            StrategySignal: Aggregated primary trading signal
        """
        if not strategy_signals:
            return self._hold_signal("UNKNOWN")

        symbol = list(strategy_signals.values())[0].symbol

        # Separate signals by action
        buy_signals = []
        sell_signals = []
        hold_signals = []

        for strategy_name, signal in strategy_signals.items():
            allocation_info = allocation.get(strategy_name)
            if allocation_info and allocation_info.enabled and allocation_info.weight > 0:
                weight = allocation_info.weight
                confidence_mult = allocation_info.confidence_multiplier

                # Adjust signal strength by allocation weight and confidence
                adjusted_strength = signal.strength * weight * confidence_mult

                if signal.action == "BUY":
                    buy_signals.append((signal, adjusted_strength, strategy_name))
                elif signal.action == "SELL":
                    sell_signals.append((signal, adjusted_strength, strategy_name))
                else:  # HOLD
                    hold_signals.append((signal, weight, strategy_name))

        # Aggregate buy signals
        total_buy_strength = sum(strength for _, strength, _ in buy_signals)
        total_sell_strength = sum(strength for _, strength, _ in sell_signals)
        total_hold_weight = sum(weight for _, weight, _ in hold_signals)

        # Determine primary action based on aggregated strengths
        if total_buy_strength > total_sell_strength and total_buy_strength > 0.3:
            # Generate BUY signal
            return self._create_aggregated_buy_signal(buy_signals, symbol, regime_info)

        elif total_sell_strength > total_buy_strength and total_sell_strength > 0.3:
            # Generate SELL signal
            return self._create_aggregated_sell_signal(sell_signals, symbol, regime_info)

        else:
            # Generate HOLD signal
            return self._hold_signal(symbol)

    def _create_aggregated_buy_signal(
        self,
        buy_signals: List[Tuple[StrategySignal, float, str]],
        symbol: str,
        regime_info: Dict[str, Any]
    ) -> StrategySignal:
        """Create aggregated BUY signal"""
        if not buy_signals:
            return self._hold_signal(symbol)

        # Weight-averaged metrics
        total_weight = sum(strength for _, strength, _ in buy_signals)
        if total_weight == 0:
            return self._hold_signal(symbol)

        avg_confidence = sum(signal.confidence * strength for signal, strength, _ in buy_signals) / total_weight
        total_strength = min(1.0, total_weight)

        # Get stop loss and take profit levels (use most conservative)
        stop_losses = [signal.stop_loss for signal, _, _ in buy_signals if signal.stop_loss is not None]
        take_profits = [signal.take_profit for signal, _, _ in buy_signals if signal.take_profit is not None]

        stop_loss = max(stop_losses) if stop_losses else None  # Most conservative stop
        take_profit = min(take_profits) if take_profits else None  # Most conservative target

        return StrategySignal(
            symbol=symbol,
            action="BUY",
            strength=total_strength,
            confidence=avg_confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "strategy_type": "aggregated",
                "contributing_strategies": [name for _, _, name in buy_signals],
                "regime": regime_info.get("regime", "NEUTRAL"),
                "regime_confidence": regime_info.get("confidence", 0.5)
            }
        )

    def _create_aggregated_sell_signal(
        self,
        sell_signals: List[Tuple[StrategySignal, float, str]],
        symbol: str,
        regime_info: Dict[str, Any]
    ) -> StrategySignal:
        """Create aggregated SELL signal"""
        if not sell_signals:
            return self._hold_signal(symbol)

        # Weight-averaged metrics
        total_weight = sum(strength for _, strength, _ in sell_signals)
        if total_weight == 0:
            return self._hold_signal(symbol)

        avg_confidence = sum(signal.confidence * strength for signal, strength, _ in sell_signals) / total_weight
        total_strength = min(1.0, total_weight)

        # Get stop loss and take profit levels (use most conservative)
        stop_losses = [signal.stop_loss for signal, _, _ in sell_signals if signal.stop_loss is not None]
        take_profits = [signal.take_profit for signal, _, _ in sell_signals if signal.take_profit is not None]

        stop_loss = min(stop_losses) if stop_losses else None  # Most conservative stop
        take_profit = max(take_profits) if take_profits else None  # Most conservative target

        return StrategySignal(
            symbol=symbol,
            action="SELL",
            strength=total_strength,
            confidence=avg_confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "strategy_type": "aggregated",
                "contributing_strategies": [name for _, _, name in sell_signals],
                "regime": regime_info.get("regime", "NEUTRAL"),
                "regime_confidence": regime_info.get("confidence", 0.5)
            }
        )

    def _hold_signal(self, symbol: str) -> StrategySignal:
        """Create HOLD signal"""
        return StrategySignal(
            symbol=symbol,
            action="HOLD",
            strength=0.0,
            confidence=0.5,
            metadata={"strategy_type": "aggregated"}
        )

    def _error_signal(self, symbol: str) -> StrategySignal:
        """Create error signal"""
        return StrategySignal(
            symbol=symbol,
            action="HOLD",
            strength=0.0,
            confidence=0.0,
            metadata={"strategy_type": "error", "error": True}
        )

    def update_strategy_performance(self, strategy_name: str, pnl: float, winning: bool) -> None:
        """
        Update strategy performance metrics

        Args:
            strategy_name: Name of the strategy
            pnl: Profit/loss from the trade
            winning: Whether the trade was profitable
        """
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update_performance(pnl, winning)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "total_strategies": len(self.strategies),
            "enabled_strategies": sum(1 for s in self.strategies.values() if s.enabled),
            "regime_detector_status": {
                "models_trained": self.regime_detector.hmm_model is not None,
                "last_train_index": self.regime_detector.last_train_index,
                "current_regime": self.regime_detector.current_regime
            },
            "strategy_performance": {},
            "recent_signals": len(self.signal_history)
        }

        # Add individual strategy status
        for name, strategy in self.strategies.items():
            status["strategy_performance"][name] = strategy.get_strategy_info()

        return status

    def reset_all_performance(self) -> None:
        """Reset performance tracking for all strategies"""
        for strategy in self.strategies.values():
            strategy.reset_performance()

        self.strategy_matrix.reset_performance_tracking()
        self.signal_history.clear()