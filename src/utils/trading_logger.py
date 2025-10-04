"""
Unified Trading Logger for Paper and Live Trading

Enhanced logging system that supports both paper trading validation and live trading operations.
Provides comprehensive trade tracking, performance analytics, and regulatory compliance logging.

Key Features:
- Dual-mode support (Paper/Live trading)
- Comprehensive trade journal
- Performance metrics collection
- Correlation ID tracking
- Compliance and audit logging
- Real-time analytics integration
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List, Union, Callable
from contextlib import asynccontextmanager
from enum import Enum
from dataclasses import dataclass, asdict
import structlog

from .logger import TradingLogger, TradeContext, SensitiveDataFilter
from .time_utils import get_utc_now, get_epoch_timestamp


class TradingMode(str, Enum):
    """Trading mode enumeration"""
    PAPER = "paper"
    LIVE = "live"
    DEMO = "demo"
    BACKTEST = "backtest"


class LogCategory(str, Enum):
    """Extended log categories for trading system"""
    SIGNAL = "signal"
    VALIDATION = "validation"
    EXECUTION = "execution"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    MARKET_DATA = "market_data"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"
    SECURITY = "security"
    DEBUG = "debug"


@dataclass
class TradeMetrics:
    """Trading performance metrics"""
    correlation_id: str
    trade_id: Optional[str] = None
    symbol: str = ""
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    side: str = ""
    quantity: Optional[Decimal] = None
    entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    pnl: Optional[Decimal] = None
    fees: Optional[Decimal] = None
    slippage_bps: Optional[Decimal] = None
    execution_latency_ms: Optional[float] = None
    signal_strength: Optional[float] = None
    risk_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result


@dataclass
class PerformanceSnapshot:
    """Performance snapshot for analytics"""
    timestamp: datetime
    mode: TradingMode
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: Decimal
    total_fees: Decimal
    win_rate: float
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, TradingMode):
                result[key] = value.value
            else:
                result[key] = value
        return result


class UnifiedTradingLogger:
    """
    Unified logging system for paper and live trading

    Provides comprehensive logging with trade tracking, performance analytics,
    and regulatory compliance features. Supports both paper trading validation
    and live trading operations.
    """

    def __init__(
        self,
        name: str,
        mode: TradingMode = TradingMode.PAPER,
        log_level: str = "INFO",
        enable_trade_journal: bool = True,
        enable_performance_tracking: bool = True,
        enable_compliance_logging: bool = None,
        log_to_file: bool = True,
        log_dir: str = "logs",
        correlation_id: Optional[str] = None
    ):
        """
        Initialize unified trading logger

        Args:
            name: Logger name
            mode: Trading mode (paper/live/demo/backtest)
            log_level: Logging level
            enable_trade_journal: Enable trade journal recording
            enable_performance_tracking: Enable performance metrics
            enable_compliance_logging: Enable compliance logging (auto-detect if None)
            log_to_file: Enable file logging
            log_dir: Log directory
            correlation_id: Optional correlation ID for tracking
        """
        self.name = name
        self.mode = mode
        self.enable_trade_journal = enable_trade_journal
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_compliance_logging = (
            enable_compliance_logging if enable_compliance_logging is not None
            else mode == TradingMode.LIVE
        )

        # Initialize base logger
        self.base_logger = TradingLogger(
            name=f"trading_{mode.value}_{name}",
            log_level=log_level,
            log_to_file=log_to_file,
            log_dir=log_dir
        )

        # Session management
        self.session_id = str(uuid.uuid4())
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.start_time = get_utc_now()

        # Trade tracking
        self.active_trades: Dict[str, TradeMetrics] = {}
        self.completed_trades: List[TradeMetrics] = []

        # Performance tracking
        self.trade_count = 0
        self.total_pnl = Decimal('0')
        self.total_fees = Decimal('0')
        self.winning_trades = 0
        self.losing_trades = 0

        # Analytics callbacks
        self.analytics_callbacks: List[Callable] = []

        # Context data
        self.persistent_context = {
            'mode': mode.value,
            'session_id': self.session_id,
            'correlation_id': self.correlation_id,
            'is_paper_trading': mode in [TradingMode.PAPER, TradingMode.DEMO],
            'testnet': mode in [TradingMode.PAPER, TradingMode.DEMO],
            'compliance_enabled': self.enable_compliance_logging,
            'journal_enabled': self.enable_trade_journal
        }

        # Set persistent context
        self.base_logger.set_context(**self.persistent_context)

        # Initialize components
        if self.enable_trade_journal:
            self._initialize_trade_journal()

        self.base_logger.info(
            "UnifiedTradingLogger initialized",
            mode=mode.value,
            session_id=self.session_id,
            trade_journal=self.enable_trade_journal,
            performance_tracking=self.enable_performance_tracking,
            compliance_logging=self.enable_compliance_logging
        )

    def _initialize_trade_journal(self):
        """Initialize trade journal system"""
        # This will be implemented when we create the TradeJournal class
        pass

    # Core logging methods with enhanced context

    def debug(self, message: str, category: LogCategory = LogCategory.DEBUG, **kwargs):
        """Debug level logging with category"""
        kwargs.update({
            'log_category': category.value,
            'timestamp': get_utc_now().isoformat()
        })
        self.base_logger.debug(message, **kwargs)

    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Info level logging with category"""
        kwargs.update({
            'log_category': category.value,
            'timestamp': get_utc_now().isoformat()
        })
        self.base_logger.info(message, **kwargs)

    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Warning level logging with category"""
        kwargs.update({
            'log_category': category.value,
            'timestamp': get_utc_now().isoformat()
        })
        self.base_logger.warning(message, **kwargs)

    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Error level logging with category"""
        kwargs.update({
            'log_category': category.value,
            'timestamp': get_utc_now().isoformat()
        })
        self.base_logger.error(message, **kwargs)

    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Critical level logging with category"""
        kwargs.update({
            'log_category': category.value,
            'timestamp': get_utc_now().isoformat()
        })
        self.base_logger.critical(message, **kwargs)

    # Trading-specific logging methods

    def log_signal(
        self,
        message: str,
        symbol: str,
        signal_type: str,
        strength: float,
        strategy: str,
        **kwargs
    ):
        """Log trading signal generation"""
        signal_data = {
            'signal_type': signal_type,
            'symbol': symbol,
            'strength': strength,
            'strategy': strategy,
            'signal_time': get_utc_now().isoformat(),
            **kwargs
        }

        self.base_logger.log_trade(
            message,
            log_category=LogCategory.SIGNAL.value,
            **signal_data
        )

    def log_validation(
        self,
        message: str,
        validation_type: str,
        result: bool,
        details: Dict[str, Any],
        **kwargs
    ):
        """Log validation events (risk checks, compliance, etc.)"""
        validation_data = {
            'validation_type': validation_type,
            'validation_result': result,
            'validation_details': details,
            'validation_time': get_utc_now().isoformat(),
            **kwargs
        }

        level = "info" if result else "warning"
        log_method = getattr(self.base_logger, level)
        log_method(
            message,
            log_category=LogCategory.VALIDATION.value,
            **validation_data
        )

    def log_execution(
        self,
        message: str,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        execution_time_ms: Optional[float] = None,
        **kwargs
    ):
        """Log order execution events"""
        execution_data = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': float(quantity),
            'price': float(price) if price else None,
            'execution_time_ms': execution_time_ms,
            'execution_timestamp': get_utc_now().isoformat(),
            **kwargs
        }

        self.base_logger.log_execution(
            message,
            log_category=LogCategory.EXECUTION.value,
            **execution_data
        )

    def log_performance(
        self,
        message: str,
        metric_name: str,
        metric_value: Union[float, Decimal],
        metric_unit: str = "",
        **kwargs
    ):
        """Log performance metrics"""
        performance_data = {
            'metric_name': metric_name,
            'metric_value': float(metric_value) if isinstance(metric_value, Decimal) else metric_value,
            'metric_unit': metric_unit,
            'measurement_time': get_utc_now().isoformat(),
            **kwargs
        }

        self.base_logger.info(
            message,
            log_category=LogCategory.PERFORMANCE.value,
            **performance_data
        )

    def log_compliance(
        self,
        message: str,
        compliance_type: str,
        status: str,
        details: Dict[str, Any],
        **kwargs
    ):
        """Log compliance events (only in live mode unless forced)"""
        if not self.enable_compliance_logging and 'force' not in kwargs:
            return

        compliance_data = {
            'compliance_type': compliance_type,
            'compliance_status': status,
            'compliance_details': details,
            'compliance_timestamp': get_utc_now().isoformat(),
            **kwargs
        }

        # Remove 'force' from kwargs if present
        compliance_data.pop('force', None)

        self.base_logger.info(
            message,
            log_category=LogCategory.COMPLIANCE.value,
            **compliance_data
        )

    # Trade tracking methods

    def start_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        signal_strength: Optional[float] = None,
        strategy: Optional[str] = None,
        **kwargs
    ) -> str:
        """Start tracking a new trade"""
        correlation_id = kwargs.get('correlation_id', str(uuid.uuid4()))

        trade_metrics = TradeMetrics(
            correlation_id=correlation_id,
            trade_id=trade_id,
            symbol=symbol,
            entry_time=get_utc_now(),
            side=side,
            quantity=quantity,
            signal_strength=signal_strength
        )

        self.active_trades[trade_id] = trade_metrics

        self.base_logger.log_trade(
            f"Trade started: {side} {quantity} {symbol}",
            trade_id=trade_id,
            correlation_id=correlation_id,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            signal_strength=signal_strength,
            strategy=strategy,
            log_category=LogCategory.EXECUTION.value,
            **kwargs
        )

        return correlation_id

    def update_trade(
        self,
        trade_id: str,
        **updates
    ):
        """Update trade metrics"""
        if trade_id not in self.active_trades:
            self.warning(f"Trade {trade_id} not found for update", trade_id=trade_id)
            return

        trade = self.active_trades[trade_id]

        # Update trade metrics
        for key, value in updates.items():
            if hasattr(trade, key):
                setattr(trade, key, value)

        self.debug(
            f"Trade updated: {trade_id}",
            trade_id=trade_id,
            correlation_id=trade.correlation_id,
            updates=updates,
            log_category=LogCategory.EXECUTION.value
        )

    def complete_trade(
        self,
        trade_id: str,
        exit_price: Decimal,
        pnl: Decimal,
        fees: Decimal = Decimal('0'),
        **kwargs
    ):
        """Complete and finalize a trade"""
        if trade_id not in self.active_trades:
            self.error(f"Trade {trade_id} not found for completion", trade_id=trade_id)
            return

        trade = self.active_trades[trade_id]
        trade.exit_time = get_utc_now()
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.fees = fees

        # Calculate execution latency if entry_time is available
        if trade.entry_time:
            execution_duration = (trade.exit_time - trade.entry_time).total_seconds() * 1000
            trade.execution_latency_ms = execution_duration

        # Update performance tracking
        if self.enable_performance_tracking:
            self._update_performance_metrics(trade)

        # Move to completed trades
        self.completed_trades.append(trade)
        del self.active_trades[trade_id]

        # Log completion
        self.base_logger.log_trade(
            f"Trade completed: {trade.side} {trade.quantity} {trade.symbol}",
            **trade.to_dict(),
            log_category=LogCategory.EXECUTION.value,
            **kwargs
        )

        # Trigger analytics callbacks
        for callback in self.analytics_callbacks:
            try:
                callback(trade)
            except Exception as e:
                self.error(f"Analytics callback failed: {e}", callback_error=str(e))

    def _update_performance_metrics(self, trade: TradeMetrics):
        """Update internal performance metrics"""
        self.trade_count += 1
        self.total_pnl += trade.pnl or Decimal('0')
        self.total_fees += trade.fees or Decimal('0')

        if trade.pnl and trade.pnl > 0:
            self.winning_trades += 1
        elif trade.pnl and trade.pnl < 0:
            self.losing_trades += 1

    # Performance analytics methods

    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot"""
        if self.trade_count == 0:
            win_rate = 0.0
            avg_win = Decimal('0')
            avg_loss = Decimal('0')
            profit_factor = 0.0
        else:
            win_rate = self.winning_trades / self.trade_count

            winning_pnl = sum(t.pnl for t in self.completed_trades if t.pnl and t.pnl > 0)
            losing_pnl = sum(abs(t.pnl) for t in self.completed_trades if t.pnl and t.pnl < 0)

            avg_win = winning_pnl / self.winning_trades if self.winning_trades > 0 else Decimal('0')
            avg_loss = losing_pnl / self.losing_trades if self.losing_trades > 0 else Decimal('0')
            profit_factor = float(winning_pnl / losing_pnl) if losing_pnl > 0 else float('inf')

        return PerformanceSnapshot(
            timestamp=get_utc_now(),
            mode=self.mode,
            total_trades=self.trade_count,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            total_pnl=self.total_pnl,
            total_fees=self.total_fees,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor
        )

    def log_performance_snapshot(self):
        """Log current performance snapshot"""
        snapshot = self.get_performance_snapshot()

        self.log_performance(
            "Performance snapshot",
            metric_name="performance_summary",
            metric_value=0,  # Placeholder value
            **snapshot.to_dict()
        )

    # Context management

    @asynccontextmanager
    async def trade_context(self, **context_data):
        """Async context manager for trade operations"""
        original_context = dict(self.base_logger._context_data)

        # Add new context
        enhanced_context = {
            **context_data,
            'context_timestamp': get_utc_now().isoformat(),
            'context_correlation_id': self.correlation_id
        }
        self.base_logger.set_context(**enhanced_context)

        try:
            yield self
        finally:
            # Restore original context
            self.base_logger._context_data = original_context

    # Analytics integration

    def add_analytics_callback(self, callback: Callable[[TradeMetrics], None]):
        """Add callback for trade completion analytics"""
        self.analytics_callbacks.append(callback)

    def remove_analytics_callback(self, callback: Callable):
        """Remove analytics callback"""
        if callback in self.analytics_callbacks:
            self.analytics_callbacks.remove(callback)

    # Utility methods

    def get_active_trades(self) -> Dict[str, TradeMetrics]:
        """Get currently active trades"""
        return self.active_trades.copy()

    def get_completed_trades(self) -> List[TradeMetrics]:
        """Get completed trades"""
        return self.completed_trades.copy()

    def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        return {
            'session_id': self.session_id,
            'correlation_id': self.correlation_id,
            'mode': self.mode.value,
            'start_time': self.start_time.isoformat(),
            'uptime_seconds': (get_utc_now() - self.start_time).total_seconds(),
            'trade_count': self.trade_count,
            'active_trades': len(self.active_trades),
            'total_pnl': float(self.total_pnl),
            'total_fees': float(self.total_fees)
        }


# Convenience functions

def get_paper_trading_logger(
    name: str,
    log_level: str = "DEBUG",
    **kwargs
) -> UnifiedTradingLogger:
    """Get paper trading logger with optimal settings"""
    return UnifiedTradingLogger(
        name=name,
        mode=TradingMode.PAPER,
        log_level=log_level,
        enable_trade_journal=True,
        enable_performance_tracking=True,
        enable_compliance_logging=False,
        **kwargs
    )


def get_live_trading_logger(
    name: str,
    log_level: str = "INFO",
    **kwargs
) -> UnifiedTradingLogger:
    """Get live trading logger with production settings"""
    return UnifiedTradingLogger(
        name=name,
        mode=TradingMode.LIVE,
        log_level=log_level,
        enable_trade_journal=True,
        enable_performance_tracking=True,
        enable_compliance_logging=True,
        **kwargs
    )