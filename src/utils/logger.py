"""
Trading System Logging Framework
구조화된 로깅 시스템 with 금융 특화 기능

Key Features:
- Structured logging with structlog
- Custom financial log levels (TRADE, RISK, PORTFOLIO, EXECUTION)
- Context management for trade tracking
- Sensitive data filtering for security
- High-performance logging for trading systems
"""

import logging
import structlog
import json
import time
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
import re


class CustomLogLevels:
    """Custom log levels for financial trading system"""

    EXECUTION = 22    # Order execution events (below INFO)
    TRADE = 25        # Trade-specific events (above INFO)
    RISK = 35         # Risk management events (above WARNING)
    PORTFOLIO = 45    # Portfolio events (above ERROR)


# Register custom log levels
logging.addLevelName(CustomLogLevels.EXECUTION, "EXECUTION")
logging.addLevelName(CustomLogLevels.TRADE, "TRADE")
logging.addLevelName(CustomLogLevels.RISK, "RISK")
logging.addLevelName(CustomLogLevels.PORTFOLIO, "PORTFOLIO")


class TradingLogger:
    """
    주요 로깅 클래스 for cryptocurrency trading system
    Provides structured logging with context management
    """

    def __init__(
        self,
        name: str,
        log_level: str = "INFO",
        log_to_file: bool = True,
        log_dir: str = "logs",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5
    ):
        """
        Initialize trading logger

        Args:
            name: Logger name (usually system/module name)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_to_file: Whether to log to files
            log_dir: Directory for log files
            max_file_size: Maximum size per log file (bytes)
            backup_count: Number of backup log files
        """
        self.name = name
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self.max_file_size = max_file_size
        self.backup_count = backup_count

        # Initialize standard logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level))

        # Configure structured logger
        self._configure_structured_logger()

        # Initialize context storage
        self._context_data = {}

        # Initialize sensitive data filter
        self.sensitive_filter = SensitiveDataFilter()

    def _configure_structured_logger(self):
        """Configure structlog for structured logging"""

        # Configure structlog processors
        structlog.configure(
            processors=[
                # Add timestamp
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                # Custom processor for sensitive data filtering
                self._sensitive_data_processor,
                # JSON formatter for structured output
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )

        self.structured_logger = structlog.get_logger(self.name)

    def _sensitive_data_processor(self, logger, method_name, event_dict):
        """Process logs to filter sensitive data"""
        return self.sensitive_filter.filter_sensitive_data(event_dict)

    # Core logging methods
    def debug(self, message: str, **kwargs):
        """Debug level logging"""
        self.structured_logger.debug(message, **self._add_context(kwargs))

    def info(self, message: str, **kwargs):
        """Info level logging"""
        self.structured_logger.info(message, **self._add_context(kwargs))

    def warning(self, message: str, **kwargs):
        """Warning level logging"""
        self.structured_logger.warning(message, **self._add_context(kwargs))

    def error(self, message: str, **kwargs):
        """Error level logging"""
        self.structured_logger.error(message, **self._add_context(kwargs))

    def critical(self, message: str, **kwargs):
        """Critical level logging"""
        self.structured_logger.critical(message, **self._add_context(kwargs))

    # Financial-specific logging methods
    def log_trade(self, message: str, **kwargs):
        """Log trade-specific events"""
        self.structured_logger.info(
            message,
            log_type="TRADE",
            **self._add_context(kwargs)
        )

    def log_risk(self, message: str, level: str = "WARNING", **kwargs):
        """Log risk management events"""
        log_method = getattr(self.structured_logger, level.lower())
        log_method(
            message,
            log_type="RISK",
            **self._add_context(kwargs)
        )

    def log_portfolio(self, message: str, **kwargs):
        """Log portfolio-related events"""
        self.structured_logger.error(  # Using error level for high priority
            message,
            log_type="PORTFOLIO",
            **self._add_context(kwargs)
        )

    def log_execution(self, message: str, **kwargs):
        """Log order execution events"""
        self.structured_logger.info(
            message,
            log_type="EXECUTION",
            **self._add_context(kwargs)
        )

    def _add_context(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Add current context data to log kwargs"""
        combined = dict(self._context_data)
        combined.update(kwargs)
        return combined

    def set_context(self, **context_data):
        """Set persistent context data for all logs"""
        self._context_data.update(context_data)

    def clear_context(self):
        """Clear all context data"""
        self._context_data.clear()


class TradeContext:
    """
    Context manager for trade-specific logging
    Automatically adds trade metadata to all logs within context
    """

    def __init__(self, logger: TradingLogger, **context_data):
        """
        Initialize trade context

        Args:
            logger: TradingLogger instance
            **context_data: Context data to add to all logs
        """
        self.logger = logger
        self.context_data = context_data
        self.original_context = {}

    def __enter__(self):
        # Store original context
        self.original_context = dict(self.logger._context_data)

        # Add new context
        self.logger.set_context(**self.context_data)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original context
        self.logger._context_data = self.original_context


class SensitiveDataFilter:
    """
    Filter for masking sensitive data in logs
    Prevents API keys, secrets, and personal data from being logged
    """

    SENSITIVE_PATTERNS = [
        # API keys and tokens
        (re.compile(r'api_key', re.IGNORECASE), 'api_key'),
        (re.compile(r'secret_key', re.IGNORECASE), 'secret_key'),
        (re.compile(r'access_token', re.IGNORECASE), 'access_token'),
        (re.compile(r'webhook_secret', re.IGNORECASE), 'webhook_secret'),
        (re.compile(r'private_key', re.IGNORECASE), 'private_key'),
        (re.compile(r'secret', re.IGNORECASE), 'secret'),
        (re.compile(r'password', re.IGNORECASE), 'password'),
    ]

    def filter_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter sensitive data from log dictionary

        Args:
            data: Log data dictionary

        Returns:
            Filtered dictionary with sensitive data masked
        """
        if not isinstance(data, dict):
            return data

        filtered_data = {}

        for key, value in data.items():
            # Check if key matches sensitive patterns
            if self._is_sensitive_key(key):
                filtered_data[key] = self._mask_value(str(value))
            elif isinstance(value, dict):
                # Recursively filter nested dictionaries
                filtered_data[key] = self.filter_sensitive_data(value)
            else:
                # Keep non-sensitive data as-is
                filtered_data[key] = value

        return filtered_data

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if key is sensitive"""
        for pattern, _ in self.SENSITIVE_PATTERNS:
            if pattern.search(key):
                return True
        return False

    def _mask_value(self, value: str) -> str:
        """Mask sensitive value"""
        if len(value) <= 6:
            return "***masked***"

        # Show first word or 6 characters, whichever comes first
        if '_' in value:
            first_part = value.split('_')[0] + '_'
        else:
            first_part = value[:6]

        return f"{first_part}***masked***"


# Convenience function to create logger
def get_trading_logger(
    name: str,
    log_level: str = "INFO",
    **kwargs
) -> TradingLogger:
    """
    Factory function to create TradingLogger instance

    Args:
        name: Logger name
        log_level: Logging level
        **kwargs: Additional configuration

    Returns:
        Configured TradingLogger instance
    """
    return TradingLogger(name, log_level, **kwargs)


# Global logger instance for easy access
default_logger = get_trading_logger("trading_system")