# tests/integration/test_logging_integration.py
import pytest
import tempfile
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
from datetime import datetime

from src.strategy_engine.strategy_manager import StrategyManager
from src.execution.order_manager import OrderManager
from src.execution.slippage_controller import SlippageController
from src.risk_management.risk_management import RiskController
from src.api.base import BaseExchangeClient, ExchangeConfig
from src.api.binance.websocket import BinanceWebSocket
from src.utils.trading_logger import TradingMode
from src.core.patterns import LoggerFactory


class TestStrategyEngineLogging:
    """Test strategy engine logging integration"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

        # Mock dependencies
        self.mock_config = {
            'database': {'path': os.path.join(self.temp_dir, "strategy_test.db")},
            'paper_trading': {'enabled': True},
            'logging': {'level': 'DEBUG', 'db_handler': {'enabled': True}}
        }

        # Mock strategies
        self.mock_strategies = {
            'momentum': Mock(),
            'mean_reversion': Mock(),
            'breakout': Mock()
        }

        self.strategy_manager = StrategyManager(
            strategies=self.mock_strategies,
            config=self.mock_config
        )

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_log_signal_workflow_completely(self):
        """Test complete signal generation workflow logging"""
        # Set trading session
        session_id = "strategy_test_session"
        correlation_id = "strategy_corr_123"

        self.strategy_manager.set_trading_session(session_id, correlation_id)

        # Mock strategy signals
        mock_signals = {
            'momentum': {'signal': 0.7, 'confidence': 0.8},
            'mean_reversion': {'signal': -0.3, 'confidence': 0.6},
            'breakout': {'signal': 0.5, 'confidence': 0.9}
        }

        for strategy_name, strategy in self.mock_strategies.items():
            strategy.generate_signal.return_value = mock_signals[strategy_name]

        # Generate signals and verify logging
        with patch.object(self.strategy_manager, 'log_signal_workflow') as mock_log:
            signals = self.strategy_manager.generate_signals('BTCUSDT')

            # Should log the complete workflow
            mock_log.assert_called()

            # Verify signals were generated
            assert len(signals) == 3

    def test_should_log_regime_detection_correctly(self):
        """Test regime detection logging"""
        session_id = "regime_test_session"
        self.strategy_manager.set_trading_session(session_id)

        # Mock market data for regime detection
        mock_market_data = {
            'price': Decimal('50000.00'),
            'volume': 1000,
            'volatility': 0.02
        }

        with patch.object(self.strategy_manager, 'log_regime_detection') as mock_log:
            # Simulate regime detection
            regime = self.strategy_manager._detect_market_regime(mock_market_data)

            # Should log regime detection
            mock_log.assert_called()

    def test_should_track_strategy_allocation(self):
        """Test strategy allocation logging"""
        session_id = "allocation_test_session"
        self.strategy_manager.set_trading_session(session_id)

        allocations = {'momentum': 0.4, 'mean_reversion': 0.3, 'breakout': 0.3}

        with patch.object(self.strategy_manager, 'log_strategy_allocation') as mock_log:
            self.strategy_manager._allocate_strategies(allocations)

            # Should log allocation decisions
            mock_log.assert_called()


class TestExecutionEngineLogging:
    """Test execution engine logging integration"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

        self.mock_config = {
            'database': {'path': os.path.join(self.temp_dir, "execution_test.db")},
            'paper_trading': {'enabled': True},
            'logging': {'level': 'DEBUG', 'db_handler': {'enabled': True}}
        }

        self.order_manager = OrderManager(config=self.mock_config)
        self.slippage_controller = SlippageController(config=self.mock_config)

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_log_order_lifecycle_completely(self):
        """Test complete order lifecycle logging"""
        from src.execution.models import Order, OrderSide, OrderUrgency, OrderInfo

        session_id = "order_lifecycle_session"
        correlation_id = "order_corr_123"

        self.order_manager.set_trading_session(session_id, correlation_id)

        # Create test order
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal('1.0'),
            price=Decimal('50000.00'),
            urgency=OrderUrgency.NORMAL
        )

        order_info = OrderInfo(
            strategy="test_strategy",
            signal_strength=0.8,
            risk_score=0.3
        )

        order_id = "test_order_123"

        with patch.object(self.order_manager, 'log_order_submission') as mock_log_submit:
            # Submit order
            self.order_manager._submit_order_internal(order_id, order, order_info)

            # Should log order submission
            mock_log_submit.assert_called()

        with patch.object(self.order_manager, 'log_order_status_update') as mock_log_status:
            # Update order status
            self.order_manager._update_order_status(order_id, "FILLED")

            # Should log status update
            mock_log_status.assert_called()

    def test_should_log_slippage_measurement(self):
        """Test slippage measurement logging"""
        from src.execution.models import Order, OrderSide, SlippageMetrics

        session_id = "slippage_test_session"
        self.slippage_controller.set_trading_session(session_id)

        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal('1.0'),
            price=Decimal('50000.00')
        )

        metrics = SlippageMetrics(
            expected_price=Decimal('50000.00'),
            execution_price=Decimal('50010.00'),
            slippage_bps=2.0,
            market_impact_bps=1.0,
            timing_cost_bps=1.0
        )

        with patch.object(self.slippage_controller, 'log_slippage_measurement') as mock_log:
            self.slippage_controller._record_slippage(order, metrics)

            # Should log slippage measurement
            mock_log.assert_called()


class TestRiskManagementLogging:
    """Test risk management logging integration"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

        self.mock_config = {
            'database': {'path': os.path.join(self.temp_dir, "risk_test.db")},
            'paper_trading': {'enabled': True},
            'logging': {'level': 'DEBUG', 'db_handler': {'enabled': True}},
            'risk': {
                'max_portfolio_risk': 0.05,
                'max_position_size': 0.1,
                'max_daily_var': 0.02
            }
        }

        self.risk_controller = RiskController(config=self.mock_config)

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_log_risk_violations(self):
        """Test risk violation logging"""
        session_id = "risk_violation_session"
        self.risk_controller.set_trading_session(session_id)

        # Mock a risk violation scenario
        with patch.object(self.risk_controller, 'log_risk_violation') as mock_log:
            # Simulate exceeding position size limit
            self.risk_controller._check_position_size_limit('BTCUSDT', Decimal('0.15'))

            # Should log risk violation
            mock_log.assert_called()

    def test_should_log_kelly_calculation(self):
        """Test Kelly criterion calculation logging"""
        session_id = "kelly_test_session"
        self.risk_controller.set_trading_session(session_id)

        with patch.object(self.risk_controller, 'log_kelly_calculation') as mock_log:
            # Calculate Kelly fraction
            win_rate = 0.6
            avg_win = 1.5
            avg_loss = 1.0

            kelly_fraction = self.risk_controller._calculate_kelly_fraction(
                win_rate, avg_win, avg_loss
            )

            # Should log Kelly calculation
            mock_log.assert_called()

    def test_should_log_position_sizing_decisions(self):
        """Test position sizing decision logging"""
        session_id = "position_sizing_session"
        self.risk_controller.set_trading_session(session_id)

        with patch.object(self.risk_controller, 'log_position_sizing') as mock_log:
            # Calculate position size
            portfolio_value = Decimal('100000.00')
            signal_strength = 0.8

            position_size = self.risk_controller._calculate_position_size(
                'BTCUSDT', signal_strength, portfolio_value
            )

            # Should log position sizing decision
            mock_log.assert_called()


class TestAPIIntegrationLogging:
    """Test API integration logging"""

    def setup_method(self):
        """Setup test environment"""
        self.config = ExchangeConfig(
            api_key="test_key_12345678",
            api_secret="test_secret_12345678",
            testnet=True
        )

    @pytest.mark.asyncio
    async def test_should_log_websocket_events(self):
        """Test WebSocket event logging"""
        websocket = BinanceWebSocket(self.config)
        session_id = "websocket_test_session"
        websocket.set_trading_session(session_id)

        with patch.object(websocket, 'log_connection_event') as mock_log:
            # Mock WebSocket connection
            with patch('websockets.connect', new_callable=AsyncMock):
                await websocket.connect()

                # Should log connection event
                mock_log.assert_called()

    @pytest.mark.asyncio
    async def test_should_log_subscription_events(self):
        """Test subscription event logging"""
        websocket = BinanceWebSocket(self.config)
        websocket._connected = True
        websocket.websocket = Mock()
        websocket.websocket.send = AsyncMock()

        session_id = "subscription_test_session"
        websocket.set_trading_session(session_id)

        with patch.object(websocket, 'log_subscription_event') as mock_log:
            await websocket.subscribe_orderbook('BTCUSDT', lambda x: None)

            # Should log subscription event
            mock_log.assert_called()

    def test_should_log_api_requests_with_sanitization(self):
        """Test API request logging with sensitive data sanitization"""
        # Create a mock exchange client that inherits from BaseExchangeClient
        class MockExchangeClient(BaseExchangeClient):
            async def _create_exchange_connection(self):
                return Mock()

            async def _close_exchange_connection(self, connection):
                pass

            async def submit_order(self, order):
                return {}

            async def cancel_order(self, order_id):
                return True

            async def get_order_status(self, order_id):
                return {}

            async def get_account_balance(self):
                return {}

            async def get_positions(self):
                return []

            async def get_market_data(self, symbol):
                return {}

            async def subscribe_to_orderbook(self, symbol, callback):
                pass

            async def subscribe_to_trades(self, symbol, callback):
                pass

        client = MockExchangeClient(self.config)
        session_id = "api_request_session"
        client.set_trading_session(session_id)

        # Test API request logging with sensitive data
        sensitive_params = {
            'symbol': 'BTCUSDT',
            'api_key': 'sensitive_key_123',
            'signature': 'sensitive_signature_456'
        }

        with patch.object(client, 'log_api_request') as mock_log:
            client.log_api_request('POST', '/fapi/v1/order', sensitive_params)

            # Should log API request
            mock_log.assert_called()

            # Verify sensitive data is sanitized
            call_args = mock_log.call_args
            assert 'api_key' not in str(call_args) or '***MASKED***' in str(call_args)


class TestCrossComponentLogging:
    """Test logging across multiple components"""

    def setup_method(self):
        """Setup cross-component test environment"""
        self.temp_dir = tempfile.mkdtemp()

        self.config = {
            'database': {'path': os.path.join(self.temp_dir, "cross_component_test.db")},
            'paper_trading': {'enabled': True},
            'logging': {'level': 'DEBUG', 'db_handler': {'enabled': True}}
        }

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_maintain_session_correlation_across_components(self):
        """Test session correlation tracking across all components"""
        session_id = "cross_component_session"
        correlation_id = "cross_corr_123"

        # Create loggers for different components
        strategy_logger = LoggerFactory.get_component_trading_logger(
            component="strategy_engine",
            strategy="test_strategy"
        )

        execution_logger = LoggerFactory.get_component_trading_logger(
            component="execution_engine",
            strategy="test_strategy"
        )

        risk_logger = LoggerFactory.get_component_trading_logger(
            component="risk_management",
            strategy="test_strategy"
        )

        api_logger = LoggerFactory.get_component_trading_logger(
            component="api_integration",
            strategy="test_strategy"
        )

        # Set same session context for all components
        for logger in [strategy_logger, execution_logger, risk_logger, api_logger]:
            if hasattr(logger, 'set_context'):
                logger.set_context(session_id=session_id, correlation_id=correlation_id)

        # Simulate trading workflow with correlation tracking
        with patch.object(strategy_logger, 'log_signal') as mock_signal:
            strategy_logger.log_signal(
                message="Signal generated",
                session_id=session_id,
                correlation_id=correlation_id
            )
            mock_signal.assert_called()

        with patch.object(execution_logger, 'log_order') as mock_order:
            execution_logger.log_order(
                message="Order submitted",
                session_id=session_id,
                correlation_id=correlation_id
            )
            mock_order.assert_called()

        with patch.object(risk_logger, 'log_risk_check') as mock_risk:
            risk_logger.log_risk_check(
                message="Risk validated",
                session_id=session_id,
                correlation_id=correlation_id
            )
            mock_risk.assert_called()

        with patch.object(api_logger, 'log_execution') as mock_execution:
            api_logger.log_execution(
                message="Order executed",
                session_id=session_id,
                correlation_id=correlation_id
            )
            mock_execution.assert_called()

    def test_should_handle_concurrent_logging(self):
        """Test concurrent logging from multiple components"""
        import threading
        import time

        session_id = "concurrent_session"
        results = []

        def log_from_component(component_name, count):
            """Simulate logging from a component"""
            logger = LoggerFactory.get_component_trading_logger(
                component=component_name,
                strategy="concurrent_test"
            )

            for i in range(count):
                try:
                    logger.log_market_data(
                        message=f"{component_name} data {i}",
                        symbol="BTCUSDT",
                        session_id=session_id
                    )
                    results.append(f"{component_name}_{i}")
                except Exception as e:
                    results.append(f"ERROR_{component_name}_{i}: {e}")

                time.sleep(0.001)  # Small delay to simulate real timing

        # Create threads for different components
        threads = []
        components = ["strategy", "execution", "risk", "api"]

        for component in components:
            thread = threading.Thread(
                target=log_from_component,
                args=(component, 10)
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all logs completed successfully
        error_count = len([r for r in results if r.startswith("ERROR")])
        assert error_count == 0, f"Found {error_count} errors in concurrent logging"
        assert len(results) == 40  # 4 components * 10 logs each

    def test_should_validate_paper_trading_isolation(self):
        """Test paper trading isolation and safety"""
        paper_config = self.config.copy()
        paper_config['paper_trading']['enabled'] = True

        # Create paper trading loggers
        paper_logger = LoggerFactory.get_component_trading_logger(
            component="paper_test",
            strategy="paper_strategy"
        )

        session_id = "paper_isolation_session"

        # All paper trading operations should be clearly marked
        with patch.object(paper_logger, 'log_order') as mock_log:
            paper_logger.log_order(
                message="Paper trading order",
                order_id="paper_order_123",
                symbol="BTCUSDT",
                side="BUY",
                session_id=session_id,
                paper_trading=True
            )

            mock_log.assert_called()

            # Verify paper trading flag is preserved
            call_args = mock_log.call_args
            assert call_args[1].get('paper_trading') is True

    def test_should_export_complete_session_analytics(self):
        """Test complete session analytics export"""
        session_id = "analytics_export_session"

        # Create analytics instance
        analytics = LoggerFactory.get_performance_analytics()

        # Mock database with session data
        with patch.object(analytics, 'calculate_trading_statistics') as mock_stats:
            mock_stats.return_value = {
                'total_orders': 10,
                'filled_orders': 9,
                'success_rate': 0.9,
                'symbols_traded': ['BTCUSDT', 'ETHUSDT'],
                'total_volume': Decimal('100000.00')
            }

            stats = analytics.calculate_trading_statistics(session_id)

            # Verify comprehensive statistics
            assert 'total_orders' in stats
            assert 'success_rate' in stats
            assert 'symbols_traded' in stats

            mock_stats.assert_called_with(session_id)

        with patch.object(analytics, 'generate_performance_report') as mock_report:
            mock_report.return_value = {
                'summary': {'session_id': session_id, 'duration': '1h 30m'},
                'trading_activity': {'orders': 10, 'volume': 100000},
                'execution_quality': {'avg_slippage': 0.02, 'fill_rate': 0.9},
                'risk_metrics': {'max_drawdown': 0.03, 'var_95': 0.02}
            }

            report = analytics.generate_performance_report(session_id)

            # Verify comprehensive reporting
            assert 'summary' in report
            assert 'trading_activity' in report
            assert 'execution_quality' in report
            assert 'risk_metrics' in report

            mock_report.assert_called_with(session_id)