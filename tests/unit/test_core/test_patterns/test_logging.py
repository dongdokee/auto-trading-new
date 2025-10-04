"""
Tests for LoggerFactory pattern.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock

from src.core.patterns.logging import LoggerFactory


class TestLoggerFactory:
    """Test cases for LoggerFactory"""

    def setup_method(self):
        """Reset factory state before each test"""
        LoggerFactory._initialized = False
        LoggerFactory._logger_cache.clear()

    def test_initialization(self):
        """Test logger factory initialization"""
        assert not LoggerFactory._initialized

        LoggerFactory.initialize(log_level=logging.DEBUG)

        assert LoggerFactory._initialized
        assert LoggerFactory._log_level == logging.DEBUG

    def test_initialization_idempotent(self):
        """Test that initialization is idempotent"""
        LoggerFactory.initialize(log_level=logging.INFO)
        original_level = LoggerFactory._log_level

        LoggerFactory.initialize(log_level=logging.DEBUG)

        # Should not change after first initialization
        assert LoggerFactory._log_level == original_level

    def test_get_logger_basic(self):
        """Test basic logger creation"""
        logger = LoggerFactory.get_logger("test_module")

        assert logger is not None
        # Should work with standard logging
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')

    def test_get_logger_with_context(self):
        """Test logger creation with context"""
        context = {"component": "test", "module": "example"}
        logger = LoggerFactory.get_logger("test_module", context)

        assert logger is not None

    def test_get_logger_caching(self):
        """Test that loggers are cached"""
        logger1 = LoggerFactory.get_logger("test_module")
        logger2 = LoggerFactory.get_logger("test_module")

        # Should return the same logger instance due to caching
        assert logger1 is logger2

    def test_trading_logger(self):
        """Test trading-specific logger creation"""
        logger = LoggerFactory.get_trading_logger("execution", "BTCUSDT", "TrendFollowing")

        assert logger is not None

    def test_trading_logger_minimal(self):
        """Test trading logger with minimal parameters"""
        logger = LoggerFactory.get_trading_logger("risk")

        assert logger is not None

    def test_performance_logger(self):
        """Test performance logger creation"""
        logger = LoggerFactory.get_performance_logger()

        assert logger is not None

    def test_risk_logger(self):
        """Test risk logger creation"""
        logger = LoggerFactory.get_risk_logger("position")

        assert logger is not None

    def test_risk_logger_default(self):
        """Test risk logger with default type"""
        logger = LoggerFactory.get_risk_logger()

        assert logger is not None

    def test_api_logger(self):
        """Test API logger creation"""
        logger = LoggerFactory.get_api_logger("binance")

        assert logger is not None

    def test_api_logger_default(self):
        """Test API logger with default exchange"""
        logger = LoggerFactory.get_api_logger()

        assert logger is not None

    def test_strategy_logger(self):
        """Test strategy logger creation"""
        logger = LoggerFactory.get_strategy_logger("TrendFollowing")

        assert logger is not None

    def test_execution_logger(self):
        """Test execution logger creation"""
        logger = LoggerFactory.get_execution_logger()

        assert logger is not None

    def test_market_data_logger(self):
        """Test market data logger creation"""
        logger = LoggerFactory.get_market_data_logger()

        assert logger is not None

    def test_portfolio_logger(self):
        """Test portfolio logger creation"""
        logger = LoggerFactory.get_portfolio_logger()

        assert logger is not None

    def test_set_log_level(self):
        """Test log level setting"""
        LoggerFactory.initialize()
        original_level = LoggerFactory._log_level

        LoggerFactory.set_log_level(logging.ERROR)

        assert LoggerFactory._log_level == logging.ERROR

    def test_is_structured_logging_available(self):
        """Test structured logging availability check"""
        result = LoggerFactory.is_structured_logging_available()

        # Should return a boolean
        assert isinstance(result, bool)

    def test_get_logger_info(self):
        """Test logger factory info retrieval"""
        LoggerFactory.initialize()

        info = LoggerFactory.get_logger_info()

        assert isinstance(info, dict)
        assert 'initialized' in info
        assert 'log_level' in info
        assert 'log_format' in info
        assert 'structured_logging_available' in info
        assert info['initialized'] is True

    def test_logger_info_before_initialization(self):
        """Test logger info before initialization"""
        info = LoggerFactory.get_logger_info()

        assert info['initialized'] is False

    @patch('src.core.patterns.logging.STRUCTLOG_AVAILABLE', False)
    def test_fallback_to_standard_logging(self):
        """Test fallback to standard logging when structlog unavailable"""
        logger = LoggerFactory.get_logger("test_module")

        # Should still work with standard logging
        assert logger is not None
        assert hasattr(logger, 'info')

    def test_context_handling(self):
        """Test context handling in loggers"""
        context = {"key": "value", "number": 42}
        logger = LoggerFactory.get_logger("test_module", context)

        # Should not raise an exception
        assert logger is not None

    def test_auto_initialization(self):
        """Test that get_logger auto-initializes if not initialized"""
        assert not LoggerFactory._initialized

        logger = LoggerFactory.get_logger("test_module")

        assert LoggerFactory._initialized
        assert logger is not None

    def test_custom_log_format(self):
        """Test custom log format initialization"""
        custom_format = "%(name)s - %(levelname)s - %(message)s"

        LoggerFactory.initialize(log_format=custom_format)

        assert LoggerFactory._log_format == custom_format

    def test_logger_names_unique(self):
        """Test that different logger names create different entries"""
        logger1 = LoggerFactory.get_logger("module1")
        logger2 = LoggerFactory.get_logger("module2")

        # Should be different logger instances
        assert logger1 is not logger2

    def test_cache_info_availability(self):
        """Test that cache info is available"""
        LoggerFactory.get_logger("test1")
        LoggerFactory.get_logger("test2")

        info = LoggerFactory.get_logger_info()

        # Cache info should be available
        assert 'cached_loggers' in info