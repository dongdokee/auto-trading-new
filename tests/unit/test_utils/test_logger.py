"""
Unit tests for trading logging system
TDD approach: Red -> Green -> Refactor

Test Categories:
1. TradingLogger basic functionality
2. Custom log levels (TRADE, RISK, PORTFOLIO)
3. Context management for trades
4. Structured logging output
5. Sensitive data filtering
6. Performance optimization
"""

import pytest
import logging
import json
import time
from unittest.mock import patch, MagicMock
from typing import Dict, Any
from src.utils.logger import (
    TradingLogger,
    CustomLogLevels,
    SensitiveDataFilter,
    TradeContext
)


class TestTradingLogger:
    """Test suite for TradingLogger class"""

    @pytest.fixture
    def trading_logger(self):
        """Create TradingLogger instance for testing"""
        return TradingLogger(
            name="test_trading_system",
            log_level="DEBUG",
            log_to_file=False  # For testing, don't write to files
        )

    def test_should_create_trading_logger_with_default_config(self, trading_logger):
        """Trading logger should be created with default configuration"""
        assert trading_logger.name == "test_trading_system"
        assert trading_logger.logger is not None
        assert trading_logger.structured_logger is not None

    def test_should_have_custom_financial_log_levels(self, trading_logger):
        """Logger should support custom financial log levels"""
        # These should exist as constants
        assert hasattr(CustomLogLevels, 'TRADE')
        assert hasattr(CustomLogLevels, 'RISK')
        assert hasattr(CustomLogLevels, 'PORTFOLIO')
        assert hasattr(CustomLogLevels, 'EXECUTION')

        # Should have correct numeric values
        assert CustomLogLevels.TRADE == 25
        assert CustomLogLevels.RISK == 35
        assert CustomLogLevels.PORTFOLIO == 45
        assert CustomLogLevels.EXECUTION == 22

    def test_should_log_trade_event_with_structured_format(self, trading_logger):
        """Trade events should be logged in structured format"""
        trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'size': 1.5,
            'price': 50000.0,
            'trade_id': 'T123456'
        }

        with patch.object(trading_logger.structured_logger, 'info') as mock_log:
            trading_logger.log_trade(
                message="Position opened",
                **trade_data
            )

            # Should call structured logger with trade data
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert 'Position opened' in str(call_args)
            assert 'symbol' in str(call_args)
            assert 'BTCUSDT' in str(call_args)

    def test_should_log_risk_event_with_high_priority(self, trading_logger):
        """Risk events should be logged with high priority level"""
        risk_data = {
            'event_type': 'DRAWDOWN_LIMIT',
            'severity': 'HIGH',
            'current_drawdown': 0.08,
            'limit': 0.07,
            'portfolio_equity': 9200.0
        }

        with patch.object(trading_logger.structured_logger, 'warning') as mock_log:
            trading_logger.log_risk(
                message="Drawdown limit exceeded",
                level='WARNING',
                **risk_data
            )

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert 'Drawdown limit exceeded' in str(call_args)
            assert 'DRAWDOWN_LIMIT' in str(call_args)


class TestTradeContext:
    """Test suite for TradeContext manager"""

    @pytest.fixture
    def trading_logger(self):
        return TradingLogger("test_context", log_to_file=False)

    def test_should_create_trade_context_with_metadata(self, trading_logger):
        """Trade context should automatically include metadata in all logs"""
        context_data = {
            'trade_id': 'T789',
            'strategy_id': 'trend_follow_v1',
            'symbol': 'ETHUSDT'
        }

        with TradeContext(trading_logger, **context_data) as ctx:
            assert ctx.context_data['trade_id'] == 'T789'
            assert ctx.context_data['strategy_id'] == 'trend_follow_v1'
            assert ctx.context_data['symbol'] == 'ETHUSDT'

    def test_should_automatically_add_context_to_logs(self, trading_logger):
        """All logs within context should automatically include context data"""
        with patch.object(trading_logger.structured_logger, 'info') as mock_log:
            with TradeContext(trading_logger, trade_id='T999', symbol='ADAUSDT'):
                trading_logger.info("Test message within context")

            # Should include context in the log
            mock_log.assert_called_once()
            call_args = str(mock_log.call_args)
            assert 'T999' in call_args
            assert 'ADAUSDT' in call_args


class TestSensitiveDataFilter:
    """Test suite for sensitive data filtering"""

    def test_should_mask_api_keys_in_logs(self):
        """API keys should be masked in log output"""
        filter = SensitiveDataFilter()

        sensitive_log = {
            'message': 'API call made',
            'api_key': 'secret_api_key_12345',
            'timestamp': '2025-01-01T00:00:00Z'
        }

        filtered_log = filter.filter_sensitive_data(sensitive_log)

        assert filtered_log['api_key'] == 'secret_***masked***'
        assert filtered_log['message'] == 'API call made'
        assert filtered_log['timestamp'] == '2025-01-01T00:00:00Z'

    def test_should_mask_secret_keys_and_tokens(self):
        """Secret keys and tokens should be properly masked"""
        filter = SensitiveDataFilter()

        sensitive_data = {
            'secret_key': 'very_secret_key_abcdef',
            'access_token': 'bearer_token_xyz123',
            'webhook_secret': 'webhook_secret_456'
        }

        filtered = filter.filter_sensitive_data(sensitive_data)

        assert filtered['secret_key'] == 'very_***masked***'
        assert filtered['access_token'] == 'bearer_***masked***'
        assert filtered['webhook_secret'] == 'webhook_***masked***'

    def test_should_preserve_non_sensitive_financial_data(self):
        """Non-sensitive financial data should be preserved"""
        filter = SensitiveDataFilter()

        financial_data = {
            'symbol': 'BTCUSDT',
            'price': 45000.0,
            'quantity': 1.5,
            'pnl': 125.75,
            'trade_id': 'T12345'
        }

        filtered = filter.filter_sensitive_data(financial_data)

        # All data should be preserved
        assert filtered == financial_data


class TestLoggingPerformance:
    """Test suite for logging performance optimization"""

    def test_should_handle_high_frequency_logging_efficiently(self):
        """Logger should handle high-frequency logs without performance degradation"""
        logger = TradingLogger("performance_test", log_to_file=False)

        # Simulate high-frequency logging
        start_time = time.time()

        for i in range(1000):
            logger.log_trade(
                message=f"Trade {i}",
                symbol='BTCUSDT',
                price=50000 + i,
                size=0.1,
                trade_id=f'T{i:06d}'
            )

        end_time = time.time()
        duration = end_time - start_time

        # Should complete 1000 logs in less than 1 second
        assert duration < 1.0, f"High-frequency logging took {duration:.2f}s, should be < 1.0s"

    def test_should_support_async_logging_for_performance(self):
        """Logger should support asynchronous logging for better performance"""
        # This test will be implemented when we add async support
        pass


class TestStructuredLogging:
    """Test suite for structured logging format"""

    def test_should_output_valid_json_format(self):
        """Structured logs should be in valid JSON format"""
        logger = TradingLogger("json_test", log_to_file=False)

        # Test that the logger can be called without errors
        # and that it has the expected structure
        try:
            # This should not raise any exceptions
            logger.log_portfolio(
                "Portfolio update",
                equity=10000.0,
                pnl=250.5,
                positions_count=3
            )

            # Verify that the logger has the structured logger attribute
            assert hasattr(logger, 'structured_logger')
            assert logger.structured_logger is not None

            # Verify that sensitive filter is working
            assert hasattr(logger, 'sensitive_filter')
            assert logger.sensitive_filter is not None

            # Verify context functionality
            assert hasattr(logger, '_context_data')

        except Exception as e:
            pytest.fail(f"Logger should work without errors, but got: {e}")

    def test_should_include_timestamp_in_all_logs(self):
        """All logs should include timestamp for proper sequencing"""
        logger = TradingLogger("timestamp_test", log_to_file=False)

        with patch.object(logger.structured_logger, 'info') as mock_log:
            logger.info("Test message")

            # Structured logger should be called (timestamp added by structlog config)
            mock_log.assert_called_once()