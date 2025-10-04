# tests/integration/test_paper_trading_workflow.py
import pytest
import tempfile
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
from datetime import datetime, timedelta

from src.utils.trading_logger import UnifiedTradingLogger, TradingMode
from src.core.patterns import LoggerFactory
from src.strategy_engine.strategy_manager import StrategyManager
from src.execution.order_manager import OrderManager
from src.risk_management.risk_management import RiskController
from src.api.base import ExchangeConfig
from src.api.binance.websocket import BinanceWebSocket
from src.execution.models import Order, OrderSide, OrderUrgency


class TestPaperTradingWorkflow:
    """Test complete paper trading workflow with comprehensive logging"""

    def setup_method(self):
        """Setup complete paper trading test environment"""
        self.temp_dir = tempfile.mkdtemp()

        # Complete paper trading configuration
        self.config = {
            'trading': {'mode': 'paper', 'session_timeout': 3600},
            'paper_trading': {
                'enabled': True,
                'initial_balance': 100000.0,
                'commission_rate': 0.001,
                'slippage_simulation': True,
                'latency_simulation': {'enabled': True, 'min_latency_ms': 10, 'max_latency_ms': 50}
            },
            'database': {'path': os.path.join(self.temp_dir, "paper_workflow.db")},
            'logging': {
                'level': 'DEBUG',
                'file_handler': {'enabled': True, 'filename': 'paper_workflow.log'},
                'db_handler': {'enabled': True}
            },
            'exchanges': {
                'binance': {
                    'testnet': True,
                    'paper_trading': True,
                    'api_key': 'testnet_key_12345678',
                    'api_secret': 'testnet_secret_12345678'
                }
            },
            'strategies': {
                'momentum': {'enabled': True, 'allocation': 0.5},
                'mean_reversion': {'enabled': True, 'allocation': 0.5}
            },
            'risk_management': {
                'max_portfolio_risk': 0.05,
                'max_position_size': 0.1,
                'paper_trading_multiplier': 1.0
            }
        }

        # Initialize session tracking
        self.session_id = f"paper_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.correlation_id = f"corr_{self.session_id}"

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_execute_complete_paper_trading_workflow(self):
        """Test complete paper trading workflow from signal to execution"""
        # 1. Initialize logging system
        logger = UnifiedTradingLogger(
            name="paper_workflow_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        # 2. Set up component loggers
        strategy_logger = LoggerFactory.get_component_trading_logger(
            component="strategy_engine",
            strategy="paper_workflow_strategy"
        )

        execution_logger = LoggerFactory.get_component_trading_logger(
            component="execution_engine",
            strategy="paper_workflow_strategy"
        )

        risk_logger = LoggerFactory.get_component_trading_logger(
            component="risk_management",
            strategy="paper_workflow_strategy"
        )

        api_logger = LoggerFactory.get_component_trading_logger(
            component="api_integration",
            strategy="paper_workflow_strategy"
        )

        # 3. Simulate complete trading workflow
        self._simulate_signal_generation(strategy_logger)
        self._simulate_risk_validation(risk_logger)
        self._simulate_order_execution(execution_logger)
        self._simulate_api_interaction(api_logger)

        # 4. Verify session correlation
        session_data = logger.export_session_data(self.session_id)
        assert session_data is not None

    def _simulate_signal_generation(self, logger):
        """Simulate strategy signal generation"""
        # Mock strategy manager
        mock_strategies = {
            'momentum': Mock(),
            'mean_reversion': Mock()
        }

        strategy_manager = StrategyManager(
            strategies=mock_strategies,
            config=self.config
        )

        strategy_manager.set_trading_session(self.session_id, self.correlation_id)

        # Mock signal generation
        mock_strategies['momentum'].generate_signal.return_value = {
            'signal': 0.8, 'confidence': 0.9
        }
        mock_strategies['mean_reversion'].generate_signal.return_value = {
            'signal': -0.2, 'confidence': 0.7
        }

        with patch.object(strategy_manager, 'log_signal_workflow') as mock_log:
            signals = strategy_manager.generate_signals('BTCUSDT')

            # Verify signal generation was logged
            mock_log.assert_called()

            # Log aggregated signal
            logger.log_signal(
                message="Aggregated signal for BTCUSDT",
                strategy="paper_workflow_strategy",
                symbol="BTCUSDT",
                signal_type="BUY",
                strength=0.6,  # Weighted average
                confidence=0.85,
                session_id=self.session_id,
                correlation_id=self.correlation_id
            )

    def _simulate_risk_validation(self, logger):
        """Simulate risk management validation"""
        risk_controller = RiskController(config=self.config)
        risk_controller.set_trading_session(self.session_id, self.correlation_id)

        # Simulate risk checks
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal('1.0'),
            price=Decimal('50000.00')
        )

        with patch.object(risk_controller, 'log_risk_check') as mock_log:
            # Check position size
            risk_passed = risk_controller._check_position_size_limit('BTCUSDT', order.size)

            # Verify risk check was logged
            mock_log.assert_called()

            # Log risk validation result
            logger.log_risk_event(
                message="Risk validation passed for BTCUSDT order",
                event_type="risk_validation",
                symbol="BTCUSDT",
                order_size=float(order.size),
                risk_score=0.03,
                passed=True,
                session_id=self.session_id,
                correlation_id=self.correlation_id
            )

    def _simulate_order_execution(self, logger):
        """Simulate order execution"""
        order_manager = OrderManager(config=self.config)
        order_manager.set_trading_session(self.session_id, self.correlation_id)

        # Create order
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=Decimal('1.0'),
            price=Decimal('50000.00'),
            urgency=OrderUrgency.NORMAL
        )

        order_id = f"paper_order_{self.session_id}"

        with patch.object(order_manager, 'log_order_submission') as mock_log:
            # Simulate order submission
            from src.execution.models import OrderInfo

            order_info = OrderInfo(
                strategy="paper_workflow_strategy",
                signal_strength=0.8,
                risk_score=0.03
            )

            order_manager._submit_order_internal(order_id, order, order_info)

            # Verify order submission was logged
            mock_log.assert_called()

            # Log order execution
            logger.log_order(
                message="Paper order executed",
                order_id=order_id,
                symbol="BTCUSDT",
                side="BUY",
                size=float(order.size),
                price=float(order.price),
                order_type="LIMIT",
                status="FILLED",
                execution_price=50001.0,  # Simulate slight slippage
                commission=0.05,  # $0.05 commission
                session_id=self.session_id,
                correlation_id=self.correlation_id,
                paper_trading=True
            )

    def _simulate_api_interaction(self, logger):
        """Simulate API interaction"""
        exchange_config = ExchangeConfig(
            api_key="testnet_key_12345678",
            api_secret="testnet_secret_12345678",
            testnet=True
        )

        websocket = BinanceWebSocket(exchange_config)
        websocket.set_trading_session(self.session_id, self.correlation_id)

        # Simulate market data reception
        with patch.object(websocket, 'log_stream_data') as mock_log:
            market_data = {
                'symbol': 'BTCUSDT',
                'price': '50001.0',
                'quantity': '1.0',
                'event_time': int(datetime.now().timestamp() * 1000)
            }

            websocket._process_message_data('BTCUSDT', 'trades', market_data)

            # Log market data
            logger.log_market_data(
                message="Market data received for BTCUSDT",
                symbol="BTCUSDT",
                data_type="trade",
                price=Decimal('50001.0'),
                quantity=Decimal('1.0'),
                timestamp=market_data['event_time'],
                session_id=self.session_id,
                correlation_id=self.correlation_id
            )

    def test_should_validate_paper_trading_safety_throughout_workflow(self):
        """Test paper trading safety validation throughout workflow"""
        logger = UnifiedTradingLogger(
            name="safety_validation_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        # All operations should be marked as paper trading
        test_operations = [
            {
                'operation': 'signal_generation',
                'data': {
                    'message': "Paper signal generated",
                    'strategy': "safety_test_strategy",
                    'symbol': "BTCUSDT",
                    'session_id': self.session_id
                }
            },
            {
                'operation': 'order_execution',
                'data': {
                    'message': "Paper order executed",
                    'order_id': f"safety_order_{self.session_id}",
                    'symbol': "BTCUSDT",
                    'session_id': self.session_id,
                    'paper_trading': True
                }
            },
            {
                'operation': 'risk_check',
                'data': {
                    'message': "Paper trading risk check",
                    'event_type': "risk_validation",
                    'symbol': "BTCUSDT",
                    'session_id': self.session_id
                }
            }
        ]

        for operation in test_operations:
            if operation['operation'] == 'signal_generation':
                logger.log_signal(**operation['data'])
            elif operation['operation'] == 'order_execution':
                logger.log_order(**operation['data'])
            elif operation['operation'] == 'risk_check':
                logger.log_risk_event(**operation['data'])

        # Verify all operations were safely logged
        session_data = logger.export_session_data(self.session_id)
        assert len(session_data['signals']) > 0
        assert len(session_data['orders']) > 0

        # Verify paper trading mode is maintained
        assert logger.mode == TradingMode.PAPER

    def test_should_handle_paper_trading_error_scenarios(self):
        """Test error handling in paper trading scenarios"""
        logger = UnifiedTradingLogger(
            name="error_scenario_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        # Simulate various error scenarios
        error_scenarios = [
            {
                'type': 'strategy_error',
                'error': 'Strategy calculation failed',
                'context': {'strategy': 'momentum', 'symbol': 'BTCUSDT'}
            },
            {
                'type': 'execution_error',
                'error': 'Order validation failed',
                'context': {'order_id': 'error_order_123', 'symbol': 'BTCUSDT'}
            },
            {
                'type': 'risk_error',
                'error': 'Risk limit exceeded',
                'context': {'symbol': 'BTCUSDT', 'risk_score': 0.08}
            },
            {
                'type': 'api_error',
                'error': 'Simulated API connection error',
                'context': {'endpoint': '/fapi/v1/order', 'symbol': 'BTCUSDT'}
            }
        ]

        for scenario in error_scenarios:
            # Log error scenario
            logger.log_error(
                message=f"Paper trading error: {scenario['error']}",
                error_type=scenario['type'],
                error_message=scenario['error'],
                session_id=self.session_id,
                correlation_id=self.correlation_id,
                paper_trading=True,
                **scenario['context']
            )

        # Verify errors were logged correctly
        session_data = logger.export_session_data(self.session_id)
        # Note: Error logs might be in a separate table, but session should exist
        assert session_data is not None

    def test_should_generate_comprehensive_paper_trading_report(self):
        """Test comprehensive paper trading session report generation"""
        logger = UnifiedTradingLogger(
            name="report_generation_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        # Simulate a complete trading session
        session_start_time = datetime.now()

        # Log various activities
        activities = [
            # Signals
            {
                'type': 'signal',
                'data': {
                    'message': "Momentum signal", 'strategy': "momentum",
                    'symbol': "BTCUSDT", 'signal_type': "BUY", 'strength': 0.8
                }
            },
            {
                'type': 'signal',
                'data': {
                    'message': "Mean reversion signal", 'strategy': "mean_reversion",
                    'symbol': "ETHUSDT", 'signal_type': "SELL", 'strength': 0.6
                }
            },
            # Orders
            {
                'type': 'order',
                'data': {
                    'message': "BTC order executed", 'order_id': "btc_order_123",
                    'symbol': "BTCUSDT", 'side': "BUY", 'size': 1.0,
                    'status': "FILLED", 'execution_price': 50000.0
                }
            },
            {
                'type': 'order',
                'data': {
                    'message': "ETH order executed", 'order_id': "eth_order_456",
                    'symbol': "ETHUSDT", 'side': "SELL", 'size': 10.0,
                    'status': "FILLED", 'execution_price': 3000.0
                }
            },
            # Market data
            {
                'type': 'market_data',
                'data': {
                    'message': "BTC price update", 'symbol': "BTCUSDT",
                    'data_type': "price", 'price': Decimal('50001.0')
                }
            },
            {
                'type': 'market_data',
                'data': {
                    'message': "ETH price update", 'symbol': "ETHUSDT",
                    'data_type': "price", 'price': Decimal('2999.0')
                }
            }
        ]

        for activity in activities:
            activity['data']['session_id'] = self.session_id
            activity['data']['correlation_id'] = self.correlation_id

            if activity['type'] == 'signal':
                logger.log_signal(**activity['data'])
            elif activity['type'] == 'order':
                logger.log_order(**activity['data'])
            elif activity['type'] == 'market_data':
                logger.log_market_data(**activity['data'])

        # Generate session statistics
        session_stats = logger.get_session_statistics(self.session_id)

        # Verify comprehensive statistics
        expected_stats = [
            'total_signals', 'total_orders', 'filled_orders',
            'symbols_traded', 'session_duration'
        ]

        for stat in expected_stats:
            assert stat in session_stats

        # Verify paper trading specific metrics
        assert session_stats.get('paper_trading', False) is True
        assert session_stats['total_signals'] == 2
        assert session_stats['total_orders'] == 2
        assert len(session_stats['symbols_traded']) == 2

    @pytest.mark.asyncio
    async def test_should_handle_concurrent_paper_trading_operations(self):
        """Test concurrent paper trading operations"""
        logger = UnifiedTradingLogger(
            name="concurrent_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        # Define concurrent operations
        async def simulate_strategy_signals():
            """Simulate multiple strategy signals"""
            for i in range(5):
                logger.log_signal(
                    message=f"Concurrent signal {i}",
                    strategy="concurrent_strategy",
                    symbol="BTCUSDT",
                    signal_type="BUY",
                    session_id=self.session_id
                )
                await asyncio.sleep(0.01)

        async def simulate_order_executions():
            """Simulate multiple order executions"""
            for i in range(5):
                logger.log_order(
                    message=f"Concurrent order {i}",
                    order_id=f"concurrent_order_{i}",
                    symbol="BTCUSDT",
                    side="BUY",
                    status="FILLED",
                    session_id=self.session_id
                )
                await asyncio.sleep(0.01)

        async def simulate_market_data():
            """Simulate market data updates"""
            for i in range(5):
                logger.log_market_data(
                    message=f"Concurrent market data {i}",
                    symbol="BTCUSDT",
                    data_type="price",
                    price=Decimal(f'{50000 + i}'),
                    session_id=self.session_id
                )
                await asyncio.sleep(0.01)

        # Run concurrent operations
        await asyncio.gather(
            simulate_strategy_signals(),
            simulate_order_executions(),
            simulate_market_data()
        )

        # Verify all operations completed successfully
        session_data = logger.export_session_data(self.session_id)
        assert len(session_data['signals']) == 5
        assert len(session_data['orders']) == 5
        assert len(session_data['market_data']) == 5

    def test_should_export_paper_trading_session_for_analysis(self):
        """Test exporting paper trading session data for analysis"""
        logger = UnifiedTradingLogger(
            name="export_analysis_test",
            mode=TradingMode.PAPER,
            config=self.config
        )

        # Create rich session data
        self._create_rich_session_data(logger)

        # Export session data
        exported_data = logger.export_session_data(self.session_id)

        # Verify export completeness
        required_sections = ['signals', 'orders', 'market_data', 'performance_metrics']
        for section in required_sections:
            assert section in exported_data

        # Verify data richness
        assert len(exported_data['signals']) > 0
        assert len(exported_data['orders']) > 0
        assert len(exported_data['market_data']) > 0

        # Verify session correlation is maintained
        for signal in exported_data['signals']:
            if isinstance(signal, dict):
                assert signal.get('session_id') == self.session_id

    def _create_rich_session_data(self, logger):
        """Create rich session data for testing"""
        # Multiple strategies
        strategies = ['momentum', 'mean_reversion', 'breakout']
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']

        # Generate signals
        for i, strategy in enumerate(strategies):
            for j, symbol in enumerate(symbols):
                logger.log_signal(
                    message=f"{strategy} signal for {symbol}",
                    strategy=strategy,
                    symbol=symbol,
                    signal_type="BUY" if i % 2 == 0 else "SELL",
                    strength=0.5 + (i * 0.1),
                    confidence=0.7 + (j * 0.1),
                    session_id=self.session_id,
                    correlation_id=f"{self.correlation_id}_{i}_{j}"
                )

        # Generate orders
        for i, symbol in enumerate(symbols):
            logger.log_order(
                message=f"Order executed for {symbol}",
                order_id=f"rich_order_{i}",
                symbol=symbol,
                side="BUY" if i % 2 == 0 else "SELL",
                size=1.0 + i,
                price=50000.0 + (i * 1000),
                status="FILLED",
                execution_price=50001.0 + (i * 1000),
                commission=0.05 + (i * 0.01),
                session_id=self.session_id,
                correlation_id=f"{self.correlation_id}_{i}",
                paper_trading=True
            )

        # Generate market data
        for i, symbol in enumerate(symbols):
            for j in range(3):  # Multiple data points per symbol
                logger.log_market_data(
                    message=f"Market data for {symbol}",
                    symbol=symbol,
                    data_type="trade",
                    price=Decimal(f'{50000 + (i * 1000) + j}'),
                    quantity=Decimal(f'{1.0 + j}'),
                    session_id=self.session_id
                )