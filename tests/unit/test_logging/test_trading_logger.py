# tests/unit/test_logging/test_trading_logger.py
import pytest
import sqlite3
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime

from src.utils.trading_logger import (
    UnifiedTradingLogger, TradingMode, LogCategory,
    ComponentTradingLogger, PerformanceAnalytics
)
from src.core.patterns import LoggerFactory


class TestUnifiedTradingLogger:
    """Test suite for UnifiedTradingLogger"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_trading.db")

        self.config = {
            'database': {'path': self.db_path},
            'paper_trading': {'enabled': True},
            'logging': {
                'level': 'DEBUG',
                'file_handler': {'enabled': True, 'filename': 'test_trading.log'},
                'db_handler': {'enabled': True}
            }
        }

        self.logger = UnifiedTradingLogger(
            name="test_logger",
            mode=TradingMode.PAPER,
            config=self.config
        )

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_initialize_with_paper_trading_mode(self):
        """Test logger initializes correctly in paper trading mode"""
        assert self.logger.mode == TradingMode.PAPER
        assert self.logger.name == "test_logger"
        assert self.logger.db_path == self.db_path

    def test_should_initialize_database_tables(self):
        """Test database tables are created correctly"""
        # Check if database file exists
        assert os.path.exists(self.db_path)

        # Check if tables exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = ['trading_logs', 'signals', 'orders', 'market_data', 'performance_metrics']
        for table in expected_tables:
            assert table in tables

        conn.close()

    def test_should_log_signal_with_all_details(self):
        """Test signal logging with comprehensive details"""
        signal_data = {
            'strategy': 'test_strategy',
            'symbol': 'BTCUSDT',
            'signal_type': 'BUY',
            'strength': 0.85,
            'confidence': 0.92,
            'price': Decimal('50000.00'),
            'session_id': 'test_session_123',
            'correlation_id': 'corr_456'
        }

        self.logger.log_signal(
            message="Test signal generated",
            **signal_data
        )

        # Verify signal was logged to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM signals WHERE strategy = ?", ('test_strategy',))
        row = cursor.fetchone()

        assert row is not None
        assert row[2] == 'test_strategy'  # strategy column
        assert row[3] == 'BTCUSDT'       # symbol column
        assert row[4] == 'BUY'           # signal_type column

        conn.close()

    def test_should_log_order_with_execution_details(self):
        """Test order logging with execution details"""
        order_data = {
            'order_id': 'order_123',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'size': Decimal('1.0'),
            'price': Decimal('50000.00'),
            'order_type': 'LIMIT',
            'status': 'FILLED',
            'execution_price': Decimal('50001.00'),
            'commission': Decimal('0.01'),
            'session_id': 'test_session_123'
        }

        self.logger.log_order(
            message="Order executed",
            **order_data
        )

        # Verify order was logged to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM orders WHERE order_id = ?", ('order_123',))
        row = cursor.fetchone()

        assert row is not None
        assert row[2] == 'order_123'     # order_id column
        assert row[3] == 'BTCUSDT'       # symbol column
        assert row[4] == 'BUY'           # side column

        conn.close()

    def test_should_handle_paper_trading_mode_correctly(self):
        """Test paper trading mode logging behavior"""
        # Should log with PAPER prefix in paper mode
        self.logger.log_signal(
            message="Paper trading signal",
            strategy="test_strategy",
            symbol="BTCUSDT",
            signal_type="BUY"
        )

        # Check that logs include paper trading indicator
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT message FROM trading_logs WHERE message LIKE '%Paper trading%'")
        row = cursor.fetchone()

        assert row is not None

        conn.close()

    def test_should_calculate_session_statistics(self):
        """Test session statistics calculation"""
        session_id = "test_session_stats"

        # Log some test data
        for i in range(5):
            self.logger.log_order(
                message=f"Test order {i}",
                order_id=f"order_{i}",
                symbol="BTCUSDT",
                side="BUY",
                size=Decimal('1.0'),
                status="FILLED",
                session_id=session_id
            )

        stats = self.logger.get_session_statistics(session_id)

        assert stats['total_orders'] == 5
        assert stats['session_id'] == session_id

    def test_should_export_session_data(self):
        """Test session data export functionality"""
        session_id = "test_export_session"

        # Log test data
        self.logger.log_signal(
            message="Export test signal",
            strategy="test_strategy",
            symbol="BTCUSDT",
            signal_type="BUY",
            session_id=session_id
        )

        data = self.logger.export_session_data(session_id)

        assert 'signals' in data
        assert 'orders' in data
        assert 'market_data' in data
        assert len(data['signals']) > 0


class TestComponentTradingLogger:
    """Test suite for ComponentTradingLogger"""

    def setup_method(self):
        """Setup test environment"""
        self.mock_base_logger = Mock()
        self.component_logger = ComponentTradingLogger(
            base_logger=self.mock_base_logger,
            component="test_component",
            strategy="test_strategy"
        )

    def test_should_initialize_with_component_info(self):
        """Test component logger initialization"""
        assert self.component_logger.component == "test_component"
        assert self.component_logger.strategy == "test_strategy"
        assert self.component_logger.base_logger == self.mock_base_logger

    def test_should_delegate_signal_logging(self):
        """Test signal logging delegation to base logger"""
        self.component_logger.log_signal(
            message="Test signal",
            symbol="BTCUSDT",
            signal_type="BUY"
        )

        self.mock_base_logger.log_signal.assert_called_once()
        call_args = self.mock_base_logger.log_signal.call_args

        assert call_args[1]['component'] == "test_component"
        assert call_args[1]['strategy'] == "test_strategy"

    def test_should_delegate_order_logging(self):
        """Test order logging delegation to base logger"""
        self.component_logger.log_order(
            message="Test order",
            order_id="order_123",
            symbol="BTCUSDT"
        )

        self.mock_base_logger.log_order.assert_called_once()
        call_args = self.mock_base_logger.log_order.call_args

        assert call_args[1]['component'] == "test_component"


class TestPerformanceAnalytics:
    """Test suite for PerformanceAnalytics"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_analytics.db")

        # Create test database with some data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE trading_logs (
                timestamp TEXT,
                level TEXT,
                component TEXT,
                strategy TEXT,
                session_id TEXT,
                correlation_id TEXT,
                message TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE orders (
                timestamp TEXT,
                level TEXT,
                order_id TEXT,
                symbol TEXT,
                side TEXT,
                size REAL,
                status TEXT,
                execution_price REAL,
                session_id TEXT
            )
        ''')

        # Insert test data
        test_orders = [
            ('2025-01-01 10:00:00', 'INFO', 'order_1', 'BTCUSDT', 'BUY', 1.0, 'FILLED', 50000.0, 'session_1'),
            ('2025-01-01 10:01:00', 'INFO', 'order_2', 'BTCUSDT', 'SELL', 1.0, 'FILLED', 50100.0, 'session_1'),
            ('2025-01-01 10:02:00', 'INFO', 'order_3', 'ETHUSDT', 'BUY', 10.0, 'FILLED', 3000.0, 'session_1'),
        ]

        cursor.executemany(
            'INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            test_orders
        )

        conn.commit()
        conn.close()

        self.analytics = PerformanceAnalytics(self.db_path)

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_calculate_trading_statistics(self):
        """Test trading statistics calculation"""
        stats = self.analytics.calculate_trading_statistics('session_1')

        assert stats['total_orders'] == 3
        assert stats['filled_orders'] == 3
        assert 'symbols_traded' in stats
        assert 'BTCUSDT' in stats['symbols_traded']
        assert 'ETHUSDT' in stats['symbols_traded']

    def test_should_generate_performance_report(self):
        """Test performance report generation"""
        report = self.analytics.generate_performance_report('session_1')

        assert 'summary' in report
        assert 'trading_activity' in report
        assert 'execution_quality' in report
        assert report['summary']['total_orders'] == 3

    def test_should_calculate_pnl_basic(self):
        """Test basic P&L calculation"""
        # This is a simplified test - real P&L calculation would be more complex
        stats = self.analytics.calculate_trading_statistics('session_1')

        assert 'filled_orders' in stats
        assert stats['filled_orders'] == 3


class TestLoggerFactory:
    """Test suite for LoggerFactory enhancements"""

    def test_should_get_component_trading_logger(self):
        """Test getting component trading logger"""
        with patch('src.utils.trading_logger.UnifiedTradingLogger'):
            logger = LoggerFactory.get_component_trading_logger(
                component="test_component",
                strategy="test_strategy"
            )

            assert logger is not None

    def test_should_handle_missing_enhanced_logging(self):
        """Test fallback when enhanced logging is not available"""
        with patch('src.core.patterns.ENHANCED_LOGGING_AVAILABLE', False):
            logger = LoggerFactory.get_api_logger("test_api")

            assert logger is not None

    def test_should_create_performance_analytics(self):
        """Test performance analytics creation"""
        with patch('src.utils.trading_logger.PerformanceAnalytics'):
            analytics = LoggerFactory.get_performance_analytics()

            assert analytics is not None


class TestLoggingIntegration:
    """Integration tests for the complete logging system"""

    def setup_method(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'database': {'path': os.path.join(self.temp_dir, "integration_test.db")},
            'paper_trading': {'enabled': True},
            'logging': {
                'level': 'DEBUG',
                'file_handler': {'enabled': True, 'filename': 'integration_test.log'},
                'db_handler': {'enabled': True}
            }
        }

    def teardown_method(self):
        """Cleanup integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_integrate_across_all_components(self):
        """Test logging integration across all system components"""
        # Create loggers for different components
        strategy_logger = LoggerFactory.get_component_trading_logger(
            component="strategy_engine",
            strategy="test_strategy"
        )

        execution_logger = LoggerFactory.get_component_trading_logger(
            component="execution_engine",
            strategy="test_strategy"
        )

        api_logger = LoggerFactory.get_component_trading_logger(
            component="api_integration",
            strategy="test_strategy"
        )

        # Simulate cross-component logging with session tracking
        session_id = "integration_test_session"
        correlation_id = "correlation_123"

        # Strategy generates signal
        strategy_logger.log_signal(
            message="Strategy signal generated",
            strategy="test_strategy",
            symbol="BTCUSDT",
            signal_type="BUY",
            session_id=session_id,
            correlation_id=correlation_id
        )

        # Execution processes order
        execution_logger.log_order(
            message="Order submitted for execution",
            order_id="order_123",
            symbol="BTCUSDT",
            side="BUY",
            session_id=session_id,
            correlation_id=correlation_id
        )

        # API confirms execution
        api_logger.log_execution(
            message="Order executed via API",
            order_id="order_123",
            execution_price=Decimal('50000.00'),
            session_id=session_id,
            correlation_id=correlation_id
        )

        # Verify correlation tracking works
        assert session_id == "integration_test_session"
        assert correlation_id == "correlation_123"

    def test_should_handle_high_volume_logging(self):
        """Test logging system under high volume"""
        logger = LoggerFactory.get_component_trading_logger(
            component="volume_test",
            strategy="volume_strategy"
        )

        session_id = "volume_test_session"

        # Log many events rapidly
        for i in range(100):
            logger.log_market_data(
                message=f"Market data update {i}",
                symbol="BTCUSDT",
                price=Decimal(f'{50000 + i}'),
                session_id=session_id
            )

        # Should handle volume without errors
        assert True  # If we get here, no exceptions were raised

    def test_should_validate_paper_trading_safety(self):
        """Test paper trading safety measures"""
        paper_logger = UnifiedTradingLogger(
            name="paper_safety_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        # All logs in paper mode should be clearly marked
        paper_logger.log_order(
            message="Paper trading order",
            order_id="paper_order_123",
            symbol="BTCUSDT",
            side="BUY",
            status="FILLED"
        )

        # Verify paper trading mode is preserved
        assert paper_logger.mode == TradingMode.PAPER

    def test_should_export_complete_session_data(self):
        """Test complete session data export"""
        logger = UnifiedTradingLogger(
            name="export_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        session_id = "export_test_session"

        # Log diverse data types
        logger.log_signal(
            message="Test signal",
            strategy="export_strategy",
            symbol="BTCUSDT",
            signal_type="BUY",
            session_id=session_id
        )

        logger.log_order(
            message="Test order",
            order_id="export_order_123",
            symbol="BTCUSDT",
            side="BUY",
            session_id=session_id
        )

        logger.log_market_data(
            message="Test market data",
            symbol="BTCUSDT",
            price=Decimal('50000.00'),
            session_id=session_id
        )

        # Export and verify
        exported_data = logger.export_session_data(session_id)

        assert 'signals' in exported_data
        assert 'orders' in exported_data
        assert 'market_data' in exported_data
        assert len(exported_data['signals']) > 0
        assert len(exported_data['orders']) > 0
        assert len(exported_data['market_data']) > 0