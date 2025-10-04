"""
Base Strategy Interface

Defines the abstract interface that all trading strategies must implement.
Provides common functionality for performance tracking, configuration management,
and signal generation standardization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging

# Import enhanced logging if available
try:
    from src.core.patterns.logging import LoggerFactory
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False
    LoggerFactory = None


@dataclass
class StrategySignal:
    """
    Standardized signal output from trading strategies

    Represents a trading signal with all necessary information for position management.
    All strategies must output signals in this format for consistency.
    """

    symbol: str  # Trading pair (e.g., "BTCUSDT")
    action: str  # Action: "BUY", "SELL", "HOLD", "CLOSE"
    strength: float  # Signal strength [0, 1]
    confidence: float  # Confidence in signal [0, 1]
    stop_loss: Optional[float] = None  # Stop loss price (optional)
    take_profit: Optional[float] = None  # Take profit price (optional)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional strategy data

    def __post_init__(self):
        """Validate signal parameters after initialization"""
        if self.action not in ["BUY", "SELL", "HOLD", "CLOSE"]:
            raise ValueError(f"Invalid action: {self.action}")

        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"Strength must be between 0 and 1: {self.strength}")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0 and 1: {self.confidence}")


@dataclass
class StrategyConfig:
    """
    Configuration for trading strategies

    Contains all parameters needed to initialize and configure a trading strategy.
    """

    name: str  # Strategy name
    enabled: bool = True  # Whether strategy is active
    weight: float = 1.0  # Strategy weight in portfolio [0, 1]
    parameters: Dict[str, Any] = field(default_factory=dict)  # Strategy-specific parameters


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies

    Provides common functionality and defines the interface that all strategies must implement.
    Each concrete strategy must implement the generate_signal method.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize base strategy with configuration

        Args:
            config: Strategy configuration including name, parameters, etc.
        """
        self.name = config.name
        self.enabled = config.enabled
        self.weight = config.weight
        self.parameters = config.parameters

        # Enhanced logging setup
        self._setup_logging()

        # Performance tracking
        self.total_signals = 0
        self.winning_signals = 0
        self.total_pnl = 0.0

        # Signal history (for analysis)
        self.signal_history: List[StrategySignal] = []

        # Trading session context
        self.current_session_id = None
        self.current_correlation_id = None

        # Log strategy initialization
        self.logger.info(
            f"Strategy {self.name} initialized",
            strategy_name=self.name,
            enabled=self.enabled,
            weight=self.weight,
            parameters=self.parameters
        )

    def _setup_logging(self):
        """Setup enhanced logging for strategy"""
        if ENHANCED_LOGGING_AVAILABLE:
            # Use enhanced logger factory for strategy-specific logging
            self.logger = LoggerFactory.get_component_trading_logger(
                component="strategy",
                strategy=self.name.lower().replace(" ", "_")
            )
        else:
            # Fallback to standard logging
            self.logger = logging.getLogger(f"strategy.{self.name}")

        # Strategy-specific log methods
        self._setup_strategy_logging_methods()

    def _setup_strategy_logging_methods(self):
        """Setup strategy-specific logging methods"""
        # These methods provide strategy-specific logging functionality
        if hasattr(self.logger, 'log_signal'):
            # Enhanced logger available - use its methods
            self.log_signal_generation = self._enhanced_log_signal
            self.log_performance_update = self._enhanced_log_performance
            self.log_parameter_update = self._enhanced_log_parameter_update
        else:
            # Standard logger - use basic methods
            self.log_signal_generation = self._basic_log_signal
            self.log_performance_update = self._basic_log_performance
            self.log_parameter_update = self._basic_log_parameter_update

    def _enhanced_log_signal(self, signal: StrategySignal, market_data: Dict[str, Any], **context):
        """Log signal generation using enhanced logger"""
        try:
            self.logger.log_signal(
                message=f"Signal generated: {signal.action} {signal.symbol}",
                symbol=signal.symbol,
                signal_type=signal.action,
                strength=signal.strength,
                strategy=self.name,
                confidence=signal.confidence,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                metadata=signal.metadata,
                market_price=market_data.get('close'),
                market_volume=market_data.get('volume'),
                strategy_weight=self.weight,
                **context
            )
        except Exception as e:
            # Fallback to basic logging
            self.logger.error(f"Enhanced signal logging failed: {e}")
            self._basic_log_signal(signal, market_data, **context)

    def _basic_log_signal(self, signal: StrategySignal, market_data: Dict[str, Any], **context):
        """Log signal generation using basic logger"""
        self.logger.info(
            f"[{self.name}] Signal: {signal.action} {signal.symbol} "
            f"(strength: {signal.strength:.3f}, confidence: {signal.confidence:.3f})",
            extra={
                'strategy_name': self.name,
                'signal_action': signal.action,
                'signal_symbol': signal.symbol,
                'signal_strength': signal.strength,
                'signal_confidence': signal.confidence,
                'market_price': market_data.get('close'),
                **context
            }
        )

    def _enhanced_log_performance(self, pnl: float, winning: bool, **context):
        """Log performance update using enhanced logger"""
        try:
            self.logger.log_performance(
                message=f"Performance updated: {'+' if winning else '-'}{abs(pnl):.4f}",
                metric_name="trade_pnl",
                metric_value=pnl,
                metric_unit="USDT",
                strategy=self.name,
                winning_trade=winning,
                total_trades=self.total_signals,
                win_rate=self.get_win_rate(),
                total_pnl=self.total_pnl,
                avg_pnl=self.get_average_pnl(),
                **context
            )
        except Exception as e:
            # Fallback to basic logging
            self.logger.error(f"Enhanced performance logging failed: {e}")
            self._basic_log_performance(pnl, winning, **context)

    def _basic_log_performance(self, pnl: float, winning: bool, **context):
        """Log performance update using basic logger"""
        self.logger.info(
            f"[{self.name}] Performance: {'+' if winning else '-'}{abs(pnl):.4f} "
            f"(Total: {self.total_pnl:.4f}, Win Rate: {self.get_win_rate():.1%})",
            extra={
                'strategy_name': self.name,
                'trade_pnl': pnl,
                'winning_trade': winning,
                'total_pnl': self.total_pnl,
                'win_rate': self.get_win_rate(),
                'total_trades': self.total_signals,
                **context
            }
        )

    def _enhanced_log_parameter_update(self, **parameters):
        """Log parameter update using enhanced logger"""
        try:
            self.logger.log_validation(
                message=f"Strategy parameters updated",
                validation_type="parameter_update",
                result=True,
                details={
                    'strategy': self.name,
                    'old_parameters': self.parameters.copy(),
                    'new_parameters': parameters,
                    'updated_fields': list(parameters.keys())
                }
            )
        except Exception as e:
            # Fallback to basic logging
            self.logger.error(f"Enhanced parameter logging failed: {e}")
            self._basic_log_parameter_update(**parameters)

    def _basic_log_parameter_update(self, **parameters):
        """Log parameter update using basic logger"""
        self.logger.info(
            f"[{self.name}] Parameters updated: {parameters}",
            extra={
                'strategy_name': self.name,
                'parameter_updates': parameters,
                'old_parameters': self.parameters.copy()
            }
        )

    def set_trading_session(self, session_id: str, correlation_id: str = None):
        """
        Set trading session context for logging

        Args:
            session_id: Trading session identifier
            correlation_id: Optional correlation identifier
        """
        self.current_session_id = session_id
        self.current_correlation_id = correlation_id

        # Update logger context if enhanced logging is available
        if hasattr(self.logger, 'base_logger') and hasattr(self.logger.base_logger, 'set_context'):
            self.logger.base_logger.set_context(
                session_id=session_id,
                correlation_id=correlation_id,
                strategy=self.name
            )

    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> StrategySignal:
        """
        Generate trading signal based on market data

        Args:
            market_data: Dictionary containing market information including:
                - symbol: Trading pair
                - price/close: Current price
                - volume: Trading volume
                - ohlcv_data: Historical OHLCV DataFrame (if needed)
                - Additional market state information

        Returns:
            StrategySignal: Trading signal with action, strength, confidence

        Note:
            This method must be implemented by each concrete strategy.
            Should return HOLD signal when no clear trading opportunity exists.
        """
        pass

    def _generate_signal_with_logging(self, market_data: Dict[str, Any], **context) -> StrategySignal:
        """
        Helper method to generate signal with comprehensive logging

        Args:
            market_data: Market data for signal generation
            **context: Additional context for logging

        Returns:
            StrategySignal: Generated signal with logging
        """
        # Validate market data
        if not self._validate_market_data(market_data):
            self.logger.error(
                f"Invalid market data for strategy {self.name}",
                strategy_name=self.name,
                market_data_keys=list(market_data.keys()) if market_data else None
            )
            # Return default HOLD signal
            return StrategySignal(
                symbol=market_data.get('symbol', 'UNKNOWN'),
                action='HOLD',
                strength=0.0,
                confidence=0.0,
                metadata={'error': 'invalid_market_data'}
            )

        try:
            # Generate signal using abstract method
            signal = self.generate_signal(market_data)

            # Log signal generation
            self.log_signal_generation(
                signal=signal,
                market_data=market_data,
                correlation_id=self.current_correlation_id,
                session_id=self.current_session_id,
                **context
            )

            # Store signal in history
            self._log_signal(signal)

            return signal

        except Exception as e:
            self.logger.error(
                f"Signal generation failed for strategy {self.name}: {e}",
                strategy_name=self.name,
                error_type=type(e).__name__,
                error_message=str(e),
                market_data_symbol=market_data.get('symbol'),
                market_data_price=market_data.get('close')
            )

            # Return safe HOLD signal on error
            return StrategySignal(
                symbol=market_data.get('symbol', 'UNKNOWN'),
                action='HOLD',
                strength=0.0,
                confidence=0.0,
                metadata={'error': 'signal_generation_failed', 'error_details': str(e)}
            )

    @abstractmethod
    def update_parameters(self, **kwargs) -> None:
        """
        Update strategy parameters

        Args:
            **kwargs: Parameter updates

        Note:
            This method must be implemented by each concrete strategy.
            Should validate parameters and log updates.
        """
        pass

    def _update_parameters_with_logging(self, **kwargs) -> None:
        """
        Helper method to update parameters with logging

        Args:
            **kwargs: Parameter updates
        """
        # Store old parameters for comparison
        old_parameters = self.parameters.copy()

        try:
            # Update parameters using abstract method
            self.update_parameters(**kwargs)

            # Log parameter update
            self.log_parameter_update(**kwargs)

            self.logger.info(
                f"Strategy {self.name} parameters updated successfully",
                strategy_name=self.name,
                old_parameters=old_parameters,
                new_parameters=self.parameters,
                updated_fields=list(kwargs.keys())
            )

        except Exception as e:
            self.logger.error(
                f"Parameter update failed for strategy {self.name}: {e}",
                strategy_name=self.name,
                error_type=type(e).__name__,
                error_message=str(e),
                attempted_updates=kwargs
            )
            raise

    def update_performance(self, pnl: float, winning: bool) -> None:
        """
        Update strategy performance metrics

        Args:
            pnl: Profit/loss from the trade
            winning: Whether the trade was profitable
        """
        # Store previous state for logging
        previous_total = self.total_signals
        previous_pnl = self.total_pnl
        previous_win_rate = self.get_win_rate()

        # Update metrics
        self.total_signals += 1
        self.total_pnl += pnl

        if winning:
            self.winning_signals += 1

        # Log performance update
        self.log_performance_update(
            pnl=pnl,
            winning=winning,
            previous_total_trades=previous_total,
            previous_total_pnl=previous_pnl,
            previous_win_rate=previous_win_rate,
            correlation_id=self.current_correlation_id,
            session_id=self.current_session_id
        )

    def get_win_rate(self) -> float:
        """
        Calculate current win rate

        Returns:
            float: Win rate as percentage [0, 1]
        """
        if self.total_signals == 0:
            return 0.0

        return self.winning_signals / self.total_signals

    def get_average_pnl(self) -> float:
        """
        Calculate average PnL per trade

        Returns:
            float: Average profit/loss per trade
        """
        if self.total_signals == 0:
            return 0.0

        return self.total_pnl / self.total_signals

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information

        Returns:
            dict: Strategy statistics and configuration
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "weight": self.weight,
            "total_signals": self.total_signals,
            "winning_signals": self.winning_signals,
            "total_pnl": self.total_pnl,
            "win_rate": self.get_win_rate(),
            "average_pnl": self.get_average_pnl(),
            "parameters": self.parameters
        }

    def enable(self) -> None:
        """Enable strategy for signal generation"""
        previous_state = self.enabled
        self.enabled = True

        if not previous_state:  # Only log if state changed
            self.logger.info(
                f"Strategy {self.name} enabled",
                strategy_name=self.name,
                state_change="disabled_to_enabled"
            )

    def disable(self) -> None:
        """Disable strategy (will not generate signals)"""
        previous_state = self.enabled
        self.enabled = False

        if previous_state:  # Only log if state changed
            self.logger.warning(
                f"Strategy {self.name} disabled",
                strategy_name=self.name,
                state_change="enabled_to_disabled"
            )

    def update_weight(self, new_weight: float) -> None:
        """
        Update strategy weight with bounds checking

        Args:
            new_weight: New weight value [0, 1]
        """
        old_weight = self.weight
        self.weight = max(0.0, min(1.0, new_weight))

        # Log weight change
        if abs(old_weight - self.weight) > 0.001:  # Only log significant changes
            self.logger.info(
                f"Strategy {self.name} weight updated: {old_weight:.3f} -> {self.weight:.3f}",
                strategy_name=self.name,
                old_weight=old_weight,
                new_weight=self.weight,
                weight_change=self.weight - old_weight
            )

    def reset_performance(self) -> None:
        """Reset all performance metrics to zero"""
        self.total_signals = 0
        self.winning_signals = 0
        self.total_pnl = 0.0
        self.signal_history.clear()

    def get_recent_signals(self, count: int = 10) -> List[StrategySignal]:
        """
        Get recent signals for analysis

        Args:
            count: Number of recent signals to return

        Returns:
            list: Recent StrategySignal objects
        """
        return self.signal_history[-count:] if self.signal_history else []

    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """
        Validate that market data contains required fields

        Args:
            market_data: Market data dictionary to validate

        Returns:
            bool: True if data is valid, False otherwise
        """
        required_fields = ["symbol"]

        for field in required_fields:
            if field not in market_data:
                return False

        # Additional validation for specific data types
        if "close" in market_data:
            try:
                float(market_data["close"])
            except (ValueError, TypeError):
                return False

        return True

    def _log_signal(self, signal: StrategySignal) -> None:
        """
        Log signal to history for analysis

        Args:
            signal: Generated signal to log
        """
        self.signal_history.append(signal)

        # Keep only last 1000 signals to prevent memory issues
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]