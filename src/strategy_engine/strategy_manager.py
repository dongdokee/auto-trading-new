"""
Strategy Manager - Central Coordination System

Coordinates all trading strategies, regime detection, and signal aggregation.
Manages the complete strategy execution pipeline from market data to final trading signals.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import uuid
from datetime import datetime

from .base_strategy import BaseStrategy, StrategySignal, StrategyConfig
from .regime_detector import NoLookAheadRegimeDetector
from .strategy_matrix import StrategyMatrix, StrategyAllocation
from .strategies import TrendFollowingStrategy, MeanReversionStrategy
from src.core.patterns import BaseManager, LoggerFactory

# Import enhanced logging if available
try:
    from src.utils.trading_logger import TradingMode, LogCategory
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False


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

        # Enhanced logging setup
        self._setup_enhanced_logging()

        # Core components
        self.regime_detector = NoLookAheadRegimeDetector()
        self.strategy_matrix = StrategyMatrix()

        # Strategy instances
        self.strategies: Dict[str, BaseStrategy] = {}

        # Trading session tracking
        self.current_session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()

        # Signal aggregation tracking
        self.signal_count = 0
        self.last_regime_info = None
        self.last_allocation = None

        # Initialize default strategies if no configs provided
        if strategy_configs is None:
            strategy_configs = self._get_default_configs()

        self.logger.info(
            "StrategyManager initialized",
            session_id=self.current_session_id,
            total_strategies=len(strategy_configs),
            strategy_names=[config.name for config in strategy_configs]
        )

        # Initialize strategies
        for config in strategy_configs:
            self._add_strategy(config)

        # Signal aggregation settings
        self.min_strategy_agreement = self.config.get('min_strategy_agreement', 0.6)  # Minimum agreement threshold for signals
        self.max_position_size = self.config.get('max_position_size', 1.0)  # Maximum position size multiplier

        # Performance tracking
        self.signal_history: List[Dict[str, Any]] = []

        # Enhanced logging methods - set up after logger initialization
        self._setup_enhanced_logging_methods()

    def _setup_enhanced_logging(self):
        """Setup enhanced logging for strategy manager"""
        if ENHANCED_LOGGING_AVAILABLE:
            # Use enhanced logger factory for strategy manager
            self.logger = LoggerFactory.get_component_trading_logger(
                component="strategy_engine",
                strategy="manager"
            )
        else:
            # Fallback to standard logging
            self.logger = LoggerFactory.get_strategy_logger("StrategyManager")

    def _setup_enhanced_logging_methods(self):
        """Setup enhanced logging methods for strategy manager"""
        if hasattr(self.logger, 'log_signal'):
            # Enhanced logger available
            self.log_signal_workflow = self._enhanced_log_signal_workflow
            self.log_regime_detection = self._enhanced_log_regime_detection
            self.log_strategy_allocation = self._enhanced_log_strategy_allocation
            self.log_signal_aggregation = self._enhanced_log_signal_aggregation
        else:
            # Standard logger - use basic methods
            self.log_signal_workflow = self._basic_log_signal_workflow
            self.log_regime_detection = self._basic_log_regime_detection
            self.log_strategy_allocation = self._basic_log_strategy_allocation
            self.log_signal_aggregation = self._basic_log_signal_aggregation

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
            # Generate correlation ID for this signal workflow
            import uuid
            correlation_id = str(uuid.uuid4())

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

            # Log regime detection with change detection
            regime_change = (self.last_regime_info is None or
                           self.last_regime_info.get('regime') != regime_info.get('regime'))

            self.log_regime_detection(
                regime_info,
                correlation_id=correlation_id,
                symbol=market_data.get('symbol', 'UNKNOWN'),
                previous_regime=self.last_regime_info.get('regime') if self.last_regime_info else None,
                regime_change=regime_change,
                hmm_models_available=hasattr(self.regime_detector, 'hmm_model') and self.regime_detector.hmm_model is not None
            )

            self.last_regime_info = regime_info

            # Step 2: Get strategy allocation based on regime
            allocation = self.strategy_matrix.get_strategy_allocation(regime_info)

            # Log strategy allocation with change detection
            allocation_change = (self.last_allocation is None or
                               self.last_allocation != {name: getattr(alloc, 'weight', alloc) for name, alloc in allocation.items()})

            self.log_strategy_allocation(
                allocation,
                regime_info,
                correlation_id=correlation_id,
                allocation_change=allocation_change,
                previous_allocation=self.last_allocation
            )

            self.last_allocation = {name: getattr(alloc, 'weight', alloc) for name, alloc in allocation.items()}

            # Step 3: Generate signals from individual strategies
            strategy_signals = {}
            for strategy_name, strategy in self.strategies.items():
                if strategy.enabled:
                    try:
                        # Set trading session context for individual strategy
                        strategy.set_trading_session(
                            session_id=self.current_session_id,
                            correlation_id=correlation_id
                        )
                        signal = strategy.generate_signal(market_data)
                        strategy_signals[strategy_name] = signal
                    except Exception as e:
                        # Handle strategy-specific errors gracefully
                        self.logger.error(
                            f"Strategy {strategy_name} failed to generate signal: {e}",
                            strategy_name=strategy_name,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            correlation_id=correlation_id,
                            session_id=self.current_session_id,
                            symbol=market_data.get("symbol", "UNKNOWN")
                        )
                        strategy_signals[strategy_name] = self._error_signal(market_data.get("symbol", "UNKNOWN"))

            # Step 4: Aggregate signals using allocation weights
            primary_signal = self._aggregate_signals(strategy_signals, allocation, regime_info)

            # Log signal aggregation
            self.log_signal_aggregation(
                strategy_signals,
                primary_signal,
                correlation_id=correlation_id,
                regime=regime_info.get('regime', 'UNKNOWN'),
                enabled_strategies=[name for name, alloc in allocation.items() if getattr(alloc, 'enabled', True)],
                disabled_strategies=[name for name, alloc in allocation.items() if not getattr(alloc, 'enabled', True)]
            )

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

            # Log complete workflow with performance metrics
            self.signal_count += 1
            workflow_end_time = pd.Timestamp.now()
            workflow_duration = (workflow_end_time - signal_data["timestamp"]).total_seconds() * 1000  # milliseconds

            self.log_signal_workflow(
                market_data,
                signal_data,
                correlation_id=correlation_id,
                workflow_duration_ms=workflow_duration,
                signal_sequence_number=self.signal_count,
                enabled_strategy_count=len([name for name, alloc in allocation.items() if getattr(alloc, 'enabled', True)]),
                total_strategy_count=len(self.strategies)
            )

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

    # Enhanced Logging Methods

    def _enhanced_log_signal_workflow(self, market_data: Dict[str, Any], result: Dict[str, Any], **context):
        """Log complete signal generation workflow using enhanced logger"""
        try:
            self.logger.log_signal(
                message=f"Signal workflow completed for {market_data.get('symbol', 'UNKNOWN')}",
                symbol=market_data.get('symbol', 'UNKNOWN'),
                signal_type=result['primary_signal'].action,
                strength=result['primary_signal'].strength,
                confidence=result['primary_signal'].confidence,
                strategy="manager",
                correlation_id=context.get('correlation_id'),
                session_id=self.current_session_id,
                regime=result['regime_info'].get('regime', 'UNKNOWN'),
                regime_confidence=result['regime_info'].get('confidence', 0.0),
                participating_strategies=list(result['strategy_signals'].keys()),
                allocation_weights={name: alloc for name, alloc in result['allocation'].items()},
                market_price=market_data.get('close'),
                workflow_duration_ms=context.get('workflow_duration_ms', 0),
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced workflow logging failed: {e}")
            self._basic_log_signal_workflow(market_data, result, **context)

    def _basic_log_signal_workflow(self, market_data: Dict[str, Any], result: Dict[str, Any], **context):
        """Log signal workflow using basic logger"""
        primary_signal = result['primary_signal']
        regime_info = result['regime_info']

        self.logger.info(
            f"[StrategyManager] Signal workflow: {primary_signal.action} {market_data.get('symbol', 'UNKNOWN')} "
            f"(strength: {primary_signal.strength:.3f}, confidence: {primary_signal.confidence:.3f}, "
            f"regime: {regime_info.get('regime', 'UNKNOWN')})",
            extra={
                'symbol': market_data.get('symbol', 'UNKNOWN'),
                'signal_action': primary_signal.action,
                'signal_strength': primary_signal.strength,
                'signal_confidence': primary_signal.confidence,
                'regime': regime_info.get('regime', 'UNKNOWN'),
                'regime_confidence': regime_info.get('confidence', 0.0),
                'strategy_count': len(result['strategy_signals']),
                'session_id': self.current_session_id,
                **context
            }
        )

    def _enhanced_log_regime_detection(self, regime_info: Dict[str, Any], **context):
        """Log regime detection using enhanced logger"""
        try:
            self.logger.log_analysis(
                message=f"Regime detected: {regime_info.get('regime', 'UNKNOWN')}",
                analysis_type="regime_detection",
                symbol=context.get('symbol', 'UNKNOWN'),
                correlation_id=context.get('correlation_id'),
                session_id=self.current_session_id,
                regime=regime_info.get('regime', 'UNKNOWN'),
                regime_confidence=regime_info.get('confidence', 0.0),
                volatility_forecast=regime_info.get('volatility_forecast', 0.0),
                regime_duration=regime_info.get('duration', 0),
                previous_regime=context.get('previous_regime'),
                regime_change=context.get('regime_change', False),
                hmm_models_available=context.get('hmm_models_available', False),
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced regime logging failed: {e}")
            self._basic_log_regime_detection(regime_info, **context)

    def _basic_log_regime_detection(self, regime_info: Dict[str, Any], **context):
        """Log regime detection using basic logger"""
        regime_change_msg = " (CHANGE)" if context.get('regime_change', False) else ""

        self.logger.info(
            f"[StrategyManager] Regime: {regime_info.get('regime', 'UNKNOWN')}{regime_change_msg} "
            f"(confidence: {regime_info.get('confidence', 0.0):.3f}, "
            f"volatility: {regime_info.get('volatility_forecast', 0.0):.4f})",
            extra={
                'regime': regime_info.get('regime', 'UNKNOWN'),
                'regime_confidence': regime_info.get('confidence', 0.0),
                'volatility_forecast': regime_info.get('volatility_forecast', 0.0),
                'regime_duration': regime_info.get('duration', 0),
                'regime_change': context.get('regime_change', False),
                'session_id': self.current_session_id,
                **context
            }
        )

    def _enhanced_log_strategy_allocation(self, allocation: Dict[str, Any], regime_info: Dict[str, Any], **context):
        """Log strategy allocation using enhanced logger"""
        try:
            # Prepare allocation data
            allocation_data = {}
            total_weight = 0.0

            for strategy_name, alloc_info in allocation.items():
                if hasattr(alloc_info, 'weight'):
                    weight = alloc_info.weight
                    enabled = alloc_info.enabled
                else:
                    weight = alloc_info
                    enabled = True

                allocation_data[f"{strategy_name}_weight"] = weight
                allocation_data[f"{strategy_name}_enabled"] = enabled
                total_weight += weight if enabled else 0.0

            self.logger.log_analysis(
                message=f"Strategy allocation updated for {regime_info.get('regime', 'UNKNOWN')} regime",
                analysis_type="strategy_allocation",
                correlation_id=context.get('correlation_id'),
                session_id=self.current_session_id,
                regime=regime_info.get('regime', 'UNKNOWN'),
                regime_confidence=regime_info.get('confidence', 0.0),
                total_allocated_weight=total_weight,
                allocation_count=len(allocation),
                **allocation_data,
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced allocation logging failed: {e}")
            self._basic_log_strategy_allocation(allocation, regime_info, **context)

    def _basic_log_strategy_allocation(self, allocation: Dict[str, Any], regime_info: Dict[str, Any], **context):
        """Log strategy allocation using basic logger"""
        allocation_str = ", ".join([
            f"{name}: {getattr(alloc, 'weight', alloc):.2f}"
            for name, alloc in allocation.items()
        ])

        self.logger.info(
            f"[StrategyManager] Allocation for {regime_info.get('regime', 'UNKNOWN')}: {allocation_str}",
            extra={
                'regime': regime_info.get('regime', 'UNKNOWN'),
                'allocation_data': {name: getattr(alloc, 'weight', alloc) for name, alloc in allocation.items()},
                'allocation_count': len(allocation),
                'session_id': self.current_session_id,
                **context
            }
        )

    def _enhanced_log_signal_aggregation(self, individual_signals: Dict[str, Any], primary_signal: Any, **context):
        """Log signal aggregation using enhanced logger"""
        try:
            # Analyze signal distribution
            buy_count = sum(1 for signal in individual_signals.values() if hasattr(signal, 'action') and signal.action == 'BUY')
            sell_count = sum(1 for signal in individual_signals.values() if hasattr(signal, 'action') and signal.action == 'SELL')
            hold_count = len(individual_signals) - buy_count - sell_count

            avg_confidence = sum(
                getattr(signal, 'confidence', 0.0) for signal in individual_signals.values()
            ) / len(individual_signals) if individual_signals else 0.0

            self.logger.log_analysis(
                message=f"Signal aggregation: {len(individual_signals)} strategies → {primary_signal.action}",
                analysis_type="signal_aggregation",
                correlation_id=context.get('correlation_id'),
                session_id=self.current_session_id,
                primary_action=primary_signal.action,
                primary_strength=primary_signal.strength,
                primary_confidence=primary_signal.confidence,
                individual_signals_count=len(individual_signals),
                buy_signals_count=buy_count,
                sell_signals_count=sell_count,
                hold_signals_count=hold_count,
                average_individual_confidence=avg_confidence,
                aggregation_method="weighted_average",
                stop_loss=primary_signal.stop_loss,
                take_profit=primary_signal.take_profit,
                **context
            )
        except Exception as e:
            self.logger.error(f"Enhanced aggregation logging failed: {e}")
            self._basic_log_signal_aggregation(individual_signals, primary_signal, **context)

    def _basic_log_signal_aggregation(self, individual_signals: Dict[str, Any], primary_signal: Any, **context):
        """Log signal aggregation using basic logger"""
        signal_summary = []
        for name, signal in individual_signals.items():
            if hasattr(signal, 'action'):
                signal_summary.append(f"{name}:{signal.action[:1]}")  # First letter of action

        self.logger.info(
            f"[StrategyManager] Aggregation: [{', '.join(signal_summary)}] → {primary_signal.action} "
            f"(strength: {primary_signal.strength:.3f})",
            extra={
                'primary_action': primary_signal.action,
                'primary_strength': primary_signal.strength,
                'primary_confidence': primary_signal.confidence,
                'individual_count': len(individual_signals),
                'session_id': self.current_session_id,
                **context
            }
        )