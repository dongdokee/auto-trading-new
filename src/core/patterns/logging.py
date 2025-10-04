"""
Centralized logger factory for consistent logging across the application.

Provides standardized logger creation with:
- Component-specific loggers
- Structured logging support
- Performance optimized with caching
- Trading-specific logger types
"""

import logging
import sys
from typing import Dict, Any, Optional
from functools import lru_cache
from datetime import datetime

# Try to import structlog for structured logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None


class LoggerFactory:
    """
    Centralized factory for creating and managing loggers.

    Provides consistent logger configuration across the application
    with support for structured logging when available.
    """

    _initialized = False
    _log_level = logging.INFO
    _log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    _logger_cache: Dict[str, Any] = {}

    @classmethod
    def initialize(
        cls,
        log_level: int = logging.INFO,
        log_format: Optional[str] = None,
        enable_structured: bool = True
    ) -> None:
        """
        Initialize the logger factory with global settings.

        Args:
            log_level: Default logging level
            log_format: Log message format string
            enable_structured: Whether to use structured logging if available
        """
        if cls._initialized:
            return

        cls._log_level = log_level
        if log_format:
            cls._log_format = log_format

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=cls._log_format,
            stream=sys.stdout
        )

        # Configure structured logging if available and enabled
        if STRUCTLOG_AVAILABLE and enable_structured:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="ISO"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )

        cls._initialized = True

    @staticmethod
    def get_logger(name: str, context: Optional[Dict[str, Any]] = None):
        """
        Get a logger instance with optional context.

        Args:
            name: Logger name (typically module name)
            context: Additional context to bind to logger

        Returns:
            Logger instance (structlog.BoundLogger if available, else logging.Logger)
        """
        if not LoggerFactory._initialized:
            LoggerFactory.initialize()

        # Create cache key
        cache_key = name
        if context:
            cache_key = f"{name}_{hash(frozenset(context.items()) if context else frozenset())}"

        # Check cache
        if cache_key in LoggerFactory._logger_cache:
            return LoggerFactory._logger_cache[cache_key]

        # Create new logger
        if STRUCTLOG_AVAILABLE:
            logger = structlog.get_logger(name)
            if context:
                logger = logger.bind(**context)
        else:
            # Fallback to standard logging
            logger = logging.getLogger(name)
            # Store context in logger for potential future use
            if context and not hasattr(logger, '_context'):
                logger._context = context

        # Cache and return
        LoggerFactory._logger_cache[cache_key] = logger
        return logger

    @staticmethod
    def get_trading_logger(
        component: str,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None
    ):
        """
        Get a trading-specific logger with trading context.

        Args:
            component: Trading component name (e.g., 'execution', 'strategy', 'risk')
            symbol: Trading symbol (e.g., 'BTCUSDT')
            strategy: Strategy name (e.g., 'TrendFollowing')

        Returns:
            Logger instance with trading context
        """
        context = {
            "component": component,
            "timestamp": datetime.now().isoformat()
        }

        if symbol:
            context["symbol"] = symbol
        if strategy:
            context["strategy"] = strategy

        logger_name = f"trading.{component}"
        return LoggerFactory.get_logger(logger_name, context)

    @staticmethod
    def get_performance_logger():
        """
        Get a performance measurement logger.

        Returns:
            Logger instance optimized for performance metrics
        """
        context = {
            "component": "performance",
            "timestamp": datetime.now().isoformat()
        }
        return LoggerFactory.get_logger("performance", context)

    @staticmethod
    def get_risk_logger(risk_type: str = "general"):
        """
        Get a risk management logger.

        Args:
            risk_type: Type of risk being logged (e.g., 'position', 'portfolio', 'market')

        Returns:
            Logger instance for risk management
        """
        context = {
            "component": "risk_management",
            "risk_type": risk_type,
            "timestamp": datetime.now().isoformat()
        }
        return LoggerFactory.get_logger("risk", context)

    @staticmethod
    def get_api_logger(exchange: str = "unknown"):
        """
        Get an API interaction logger.

        Args:
            exchange: Exchange name (e.g., 'binance', 'coinbase')

        Returns:
            Logger instance for API interactions
        """
        context = {
            "component": "api",
            "exchange": exchange,
            "timestamp": datetime.now().isoformat()
        }
        return LoggerFactory.get_logger("api", context)

    @staticmethod
    def get_strategy_logger(strategy_name: str):
        """
        Get a strategy-specific logger.

        Args:
            strategy_name: Name of the trading strategy

        Returns:
            Logger instance for strategy operations
        """
        context = {
            "component": "strategy",
            "strategy_name": strategy_name,
            "timestamp": datetime.now().isoformat()
        }
        return LoggerFactory.get_logger(f"strategy.{strategy_name}", context)

    @staticmethod
    def get_execution_logger():
        """
        Get an order execution logger.

        Returns:
            Logger instance for order execution
        """
        context = {
            "component": "execution",
            "timestamp": datetime.now().isoformat()
        }
        return LoggerFactory.get_logger("execution", context)

    @staticmethod
    def get_market_data_logger():
        """
        Get a market data logger.

        Returns:
            Logger instance for market data processing
        """
        context = {
            "component": "market_data",
            "timestamp": datetime.now().isoformat()
        }
        return LoggerFactory.get_logger("market_data", context)

    @staticmethod
    def get_portfolio_logger():
        """
        Get a portfolio management logger.

        Returns:
            Logger instance for portfolio operations
        """
        context = {
            "component": "portfolio",
            "timestamp": datetime.now().isoformat()
        }
        return LoggerFactory.get_logger("portfolio", context)

    @classmethod
    def set_log_level(cls, level: int) -> None:
        """
        Update the global log level.

        Args:
            level: New logging level
        """
        cls._log_level = level
        logging.getLogger().setLevel(level)

    @classmethod
    def is_structured_logging_available(cls) -> bool:
        """
        Check if structured logging is available.

        Returns:
            True if structlog is available, False otherwise
        """
        return STRUCTLOG_AVAILABLE

    @classmethod
    def get_logger_info(cls) -> Dict[str, Any]:
        """
        Get information about the logger factory configuration.

        Returns:
            Dictionary with logger factory status
        """
        return {
            'initialized': cls._initialized,
            'log_level': cls._log_level,
            'log_format': cls._log_format,
            'structured_logging_available': STRUCTLOG_AVAILABLE,
            'cached_loggers': len(cls._logger_cache)
        }