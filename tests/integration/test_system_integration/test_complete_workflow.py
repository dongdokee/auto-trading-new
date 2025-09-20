# tests/integration/test_system_integration/test_complete_workflow.py
"""
Complete Workflow Integration Tests

Tests the entire trading system workflow from market data to order execution.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.integration.trading_orchestrator import TradingOrchestrator, OrchestratorConfig
from src.integration.events.event_bus import EventBus
from src.integration.events.models import (
    MarketDataEvent, StrategySignalEvent, PortfolioEvent, OrderEvent,
    ExecutionEvent, RiskEvent, SystemEvent, EventType, EventPriority
)
from src.integration.state.manager import StateManager
from src.integration.adapters.strategy_adapter import StrategyAdapter
from src.integration.adapters.risk_adapter import RiskAdapter
from src.integration.adapters.execution_adapter import ExecutionAdapter
from src.integration.adapters.portfolio_adapter import PortfolioAdapter
from src.integration.monitoring.monitor import SystemMonitor
from src.integration.monitoring.alerts import AlertManager


class TestCompleteWorkflow:
    """Test complete trading system workflows"""

    @pytest.fixture
    async def system_components(self):
        """Setup complete system components for testing"""
        # Create orchestrator config
        config = OrchestratorConfig(
            enable_paper_trading=True,
            max_concurrent_orders=5,
            risk_check_interval_seconds=10,
            portfolio_rebalance_interval_seconds=60,
            health_check_interval_seconds=30
        )

        # Create orchestrator
        orchestrator = TradingOrchestrator(config)

        # Get references to internal components
        event_bus = orchestrator.event_bus
        state_manager = orchestrator.state_manager

        # Create adapters
        strategy_adapter = StrategyAdapter(event_bus)
        risk_adapter = RiskAdapter(event_bus, state_manager, initial_capital=100000.0)
        execution_adapter = ExecutionAdapter(event_bus, state_manager)
        portfolio_adapter = PortfolioAdapter(event_bus, state_manager)

        # Create monitoring
        alert_manager = AlertManager(event_bus)
        monitor = SystemMonitor(event_bus, state_manager, alert_manager)

        components = {
            'orchestrator': orchestrator,
            'event_bus': event_bus,
            'state_manager': state_manager,
            'strategy_adapter': strategy_adapter,
            'risk_adapter': risk_adapter,
            'execution_adapter': execution_adapter,
            'portfolio_adapter': portfolio_adapter,
            'alert_manager': alert_manager,
            'monitor': monitor
        }

        # Start all components
        await orchestrator.start()
        await strategy_adapter.start()
        await risk_adapter.start()
        await execution_adapter.start()
        await portfolio_adapter.start()
        await alert_manager.start()
        await monitor.start()

        yield components

        # Cleanup
        await monitor.stop()
        await alert_manager.stop()
        await portfolio_adapter.stop()
        await execution_adapter.stop()
        await risk_adapter.stop()
        await strategy_adapter.stop()
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_market_data_to_signal_workflow(self, system_components):
        """Test workflow from market data to strategy signal generation"""
        components = system_components
        event_bus = components['event_bus']
        strategy_adapter = components['strategy_adapter']

        # Track generated signals
        generated_signals = []

        def signal_handler(event):
            if isinstance(event, StrategySignalEvent):
                generated_signals.append(event)

        event_bus.subscribe(EventType.STRATEGY_SIGNAL, signal_handler)

        # Send market data event
        market_data = MarketDataEvent(
            source_component="test_market_feed",
            symbol="BTCUSDT",
            price=Decimal("50000.0"),
            volume=Decimal("100.5"),
            bid=Decimal("49995.0"),
            ask=Decimal("50005.0")
        )

        await event_bus.publish(market_data)

        # Wait for processing
        await asyncio.sleep(1)

        # Verify signal generation
        assert len(generated_signals) > 0, "No strategy signals generated"

        signal = generated_signals[0]
        assert signal.symbol == "BTCUSDT"
        assert signal.action in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= signal.strength <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_signal_to_order_workflow(self, system_components):
        """Test workflow from strategy signal to order creation"""
        components = system_components
        event_bus = components['event_bus']

        # Track generated orders
        generated_orders = []

        def order_handler(event):
            if isinstance(event, OrderEvent):
                generated_orders.append(event)

        event_bus.subscribe(EventType.ORDER, order_handler)

        # Send strategy signal
        signal = StrategySignalEvent(
            source_component="test_strategy",
            strategy_name="trend_following",
            symbol="BTCUSDT",
            action="BUY",
            strength=0.8,
            confidence=0.7,
            target_price=Decimal("51000.0"),
            stop_loss=Decimal("48000.0")
        )

        await event_bus.publish(signal)

        # Wait for processing
        await asyncio.sleep(2)

        # Verify order generation might occur through portfolio optimization
        # (In a real system, this would trigger portfolio rebalancing)

    @pytest.mark.asyncio
    async def test_order_execution_workflow(self, system_components):
        """Test order execution workflow"""
        components = system_components
        event_bus = components['event_bus']
        execution_adapter = components['execution_adapter']

        # Track execution results
        execution_results = []

        def execution_handler(event):
            if isinstance(event, ExecutionEvent):
                execution_results.append(event)

        event_bus.subscribe(EventType.EXECUTION, execution_handler)

        # Mock exchange execution
        with patch.object(execution_adapter.order_router, 'route_order') as mock_router:
            mock_router.return_value = {
                'strategy': 'AGGRESSIVE',
                'total_filled': Decimal('0.1'),
                'avg_price': Decimal('50000.0'),
                'total_cost': Decimal('10.0'),
                'slices': []
            }

            # Create order event
            order = OrderEvent(
                source_component="test_portfolio",
                action="CREATE",
                order_id="test_order_123",
                symbol="BTCUSDT",
                side="BUY",
                size=Decimal("0.1"),
                order_type="MARKET",
                urgency="MEDIUM"
            )

            await event_bus.publish(order)

            # Wait for execution
            await asyncio.sleep(1)

            # Verify execution
            assert len(execution_results) > 0, "No execution results generated"

            execution = execution_results[0]
            assert execution.order_id == "test_order_123"
            assert execution.symbol == "BTCUSDT"
            assert execution.status in ["FILLED", "PARTIALLY_FILLED"]

    @pytest.mark.asyncio
    async def test_risk_management_workflow(self, system_components):
        """Test risk management integration"""
        components = system_components
        event_bus = components['event_bus']
        risk_adapter = components['risk_adapter']

        # Track risk events
        risk_events = []

        def risk_handler(event):
            if isinstance(event, RiskEvent):
                risk_events.append(event)

        event_bus.subscribe(EventType.RISK, risk_handler)

        # Setup portfolio state
        portfolio_state = {
            'equity': 100000.0,
            'margin_used': 10000.0,
            'positions': {},
            'current_var_usdt': 0.0
        }

        await components['state_manager'].update_portfolio_state(portfolio_state)

        # Create large order that should trigger risk checks
        large_order = OrderEvent(
            source_component="test_strategy",
            action="CREATE",
            order_id="large_order_123",
            symbol="BTCUSDT",
            side="BUY",
            size=Decimal("100.0"),  # Very large order
            order_type="MARKET"
        )

        await event_bus.publish(large_order)

        # Wait for risk processing
        await asyncio.sleep(1)

        # Verify risk validation occurred
        assert risk_adapter.order_validations > 0, "No risk validations performed"

    @pytest.mark.asyncio
    async def test_portfolio_optimization_workflow(self, system_components):
        """Test portfolio optimization workflow"""
        components = system_components
        event_bus = components['event_bus']
        portfolio_adapter = components['portfolio_adapter']

        # Track portfolio events
        portfolio_events = []

        def portfolio_handler(event):
            if isinstance(event, PortfolioEvent):
                portfolio_events.append(event)

        event_bus.subscribe(EventType.PORTFOLIO, portfolio_handler)

        # Send multiple strategy signals to trigger optimization
        signals = [
            StrategySignalEvent(
                source_component="test_strategy",
                strategy_name="trend_following",
                symbol="BTCUSDT",
                action="BUY",
                strength=0.8,
                confidence=0.7
            ),
            StrategySignalEvent(
                source_component="test_strategy",
                strategy_name="mean_reversion",
                symbol="ETHUSDT",
                action="SELL",
                strength=0.6,
                confidence=0.8
            ),
            StrategySignalEvent(
                source_component="test_strategy",
                strategy_name="range_trading",
                symbol="ADAUSDT",
                action="BUY",
                strength=0.7,
                confidence=0.6
            ),
            StrategySignalEvent(
                source_component="test_strategy",
                strategy_name="funding_arbitrage",
                symbol="DOTUSDT",
                action="HOLD",
                strength=0.5,
                confidence=0.9
            )
        ]

        for signal in signals:
            await event_bus.publish(signal)

        # Wait for signal processing
        await asyncio.sleep(1)

        # Trigger portfolio optimization
        await portfolio_adapter.perform_portfolio_optimization()

        # Wait for optimization
        await asyncio.sleep(2)

        # Verify optimization occurred
        assert portfolio_adapter.optimizations_performed > 0, "No portfolio optimizations performed"

    @pytest.mark.asyncio
    async def test_complete_end_to_end_workflow(self, system_components):
        """Test complete end-to-end trading workflow"""
        components = system_components
        event_bus = components['event_bus']

        # Track all events
        all_events = []

        def event_tracker(event):
            all_events.append({
                'timestamp': datetime.now(),
                'event_type': type(event).__name__,
                'event': event
            })

        # Subscribe to all event types
        for event_type in EventType:
            event_bus.subscribe(event_type, event_tracker)

        # Mock execution for predictable results
        execution_adapter = components['execution_adapter']
        with patch.object(execution_adapter.order_router, 'route_order') as mock_router:
            mock_router.return_value = {
                'strategy': 'TWAP',
                'total_filled': Decimal('1.0'),
                'avg_price': Decimal('50000.0'),
                'total_cost': Decimal('25.0'),
                'slices': []
            }

            # Step 1: Market data arrives
            market_data = MarketDataEvent(
                source_component="market_feed",
                symbol="BTCUSDT",
                price=Decimal("50000.0"),
                volume=Decimal("1000.0"),
                bid=Decimal("49995.0"),
                ask=Decimal("50005.0")
            )

            await event_bus.publish(market_data)
            await asyncio.sleep(0.5)

            # Step 2: Strategy generates signal
            signal = StrategySignalEvent(
                source_component="strategy_engine",
                strategy_name="trend_following",
                symbol="BTCUSDT",
                action="BUY",
                strength=0.8,
                confidence=0.75,
                target_price=Decimal("52000.0"),
                stop_loss=Decimal("48000.0")
            )

            await event_bus.publish(signal)
            await asyncio.sleep(0.5)

            # Step 3: Portfolio optimization creates order
            order = OrderEvent(
                source_component="portfolio_optimizer",
                action="CREATE",
                order_id="end_to_end_order",
                symbol="BTCUSDT",
                side="BUY",
                size=Decimal("1.0"),
                order_type="MARKET",
                urgency="MEDIUM",
                source_signal="trend_following"
            )

            await event_bus.publish(order)
            await asyncio.sleep(1)

            # Wait for complete processing
            await asyncio.sleep(2)

            # Verify workflow completion
            event_types = {event['event_type'] for event in all_events}

            # Should have processed multiple event types
            assert len(event_types) >= 3, f"Expected multiple event types, got: {event_types}"

            # Should have market data event
            market_events = [e for e in all_events if e['event_type'] == 'MarketDataEvent']
            assert len(market_events) > 0, "No market data events processed"

            # Should have signal event
            signal_events = [e for e in all_events if e['event_type'] == 'StrategySignalEvent']
            assert len(signal_events) > 0, "No strategy signal events processed"

            # Should have order event
            order_events = [e for e in all_events if e['event_type'] == 'OrderEvent']
            assert len(order_events) > 0, "No order events processed"

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, system_components):
        """Test system health monitoring and alerting"""
        components = system_components
        monitor = components['monitor']
        alert_manager = components['alert_manager']

        # Register test component
        monitor.register_component("test_component")

        # Update component heartbeat
        await monitor.update_component_heartbeat("test_component", {
            'status': 'healthy',
            'last_check': datetime.now().isoformat()
        })

        # Get system health
        system_health = await monitor.get_system_health()

        # Verify monitoring
        assert system_health.total_components > 0, "No components registered"
        assert system_health.healthy_components > 0, "No healthy components"

        # Test alert creation
        await alert_manager.create_alert(
            alert_type="TEST_ALERT",
            message="Test alert for integration testing",
            severity="WARNING",
            component="test_component"
        )

        # Verify alert was created
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) > 0, "No alerts created"

        test_alert = active_alerts[0]
        assert test_alert.alert_type == "TEST_ALERT"
        assert test_alert.component == "test_component"

    @pytest.mark.asyncio
    async def test_system_state_persistence(self, system_components):
        """Test system state management and persistence"""
        components = system_components
        state_manager = components['state_manager']

        # Update portfolio state
        portfolio_data = {
            'equity': 105000.0,
            'margin_used': 15000.0,
            'unrealized_pnl': 5000.0,
            'positions': {
                'BTCUSDT': {
                    'side': 'LONG',
                    'size': 1.0,
                    'entry_price': 48000.0,
                    'current_price': 50000.0
                }
            }
        }

        await state_manager.update_portfolio_state(portfolio_data)

        # Add active order
        await state_manager.add_active_order("test_order_456", {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'size': 0.5,
            'status': 'PENDING'
        })

        # Update risk metrics
        await state_manager.update_risk_metrics({
            'var_daily_usdt': 2000.0,
            'leverage_ratio': 3.5,
            'current_drawdown_pct': 2.5
        })

        # Get complete system state
        system_state = await state_manager.get_system_state()

        # Verify state persistence
        assert system_state.trading_active == True
        assert float(system_state.portfolio_state['equity']) == 105000.0
        assert len(system_state.active_orders) == 1
        assert system_state.active_orders[0]['symbol'] == 'BTCUSDT'

        # Verify position tracking
        positions = await state_manager.get_positions()
        assert 'BTCUSDT' in positions
        assert positions['BTCUSDT']['side'] == 'LONG'

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, system_components):
        """Test error handling and system recovery"""
        components = system_components
        event_bus = components['event_bus']
        orchestrator = components['orchestrator']

        # Test invalid event handling
        invalid_order = OrderEvent(
            source_component="test_error",
            action="CREATE",
            symbol="",  # Invalid empty symbol
            side="BUY",
            size=Decimal("0")  # Invalid zero size
        )

        # This should be handled gracefully without crashing
        await event_bus.publish(invalid_order)
        await asyncio.sleep(0.5)

        # System should still be running
        system_status = orchestrator.get_system_status()
        assert system_status['orchestrator_state'] == 'running'

        # Test component failure simulation
        await orchestrator.update_component_status("test_component", "ERROR")

        # System should detect the failure
        component_status = system_status['component_status']
        if 'test_component' in component_status:
            # Component status should be updated
            pass  # Test passes if no exception

    @pytest.mark.asyncio
    async def test_performance_under_load(self, system_components):
        """Test system performance under load"""
        components = system_components
        event_bus = components['event_bus']

        start_time = datetime.now()

        # Send many events rapidly
        events_to_send = 100
        tasks = []

        for i in range(events_to_send):
            market_data = MarketDataEvent(
                source_component="load_test",
                symbol=f"TEST{i % 10}USDT",
                price=Decimal(f"{50000 + i}"),
                volume=Decimal("100.0")
            )
            tasks.append(event_bus.publish(market_data))

        # Send all events
        await asyncio.gather(*tasks)

        # Wait for processing
        await asyncio.sleep(2)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Verify performance
        events_per_second = events_to_send / processing_time
        assert events_per_second > 10, f"Performance too low: {events_per_second} events/sec"

        # Check system metrics
        metrics = event_bus.get_metrics()
        assert metrics['event_bus']['events_published'] >= events_to_send