# src/integration/trading_orchestrator.py
"""
Trading Orchestrator

Central coordination system that manages the complete trading workflow.
Orchestrates all components through event-driven architecture.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from .events.event_bus import EventBus
from .events.models import (
    BaseEvent, MarketDataEvent, StrategySignalEvent, PortfolioEvent,
    OrderEvent, ExecutionEvent, RiskEvent, SystemEvent, EventType, EventPriority
)
from .events.handlers import BaseEventHandler, HandlerResult
from .state.manager import StateManager


class TradingState(Enum):
    """Trading system state"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class OrchestratorConfig:
    """Configuration for trading orchestrator"""
    enable_paper_trading: bool = True
    max_concurrent_orders: int = 10
    risk_check_interval_seconds: int = 30
    portfolio_rebalance_interval_seconds: int = 300  # 5 minutes
    health_check_interval_seconds: int = 60
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    emergency_stop_on_risk_breach: bool = True


class TradingOrchestrator:
    """
    Central orchestrator for the automated trading system

    Responsibilities:
    - Coordinate all trading components
    - Manage system state and lifecycle
    - Handle error recovery and system health
    - Orchestrate complete trading workflows
    - Monitor and enforce risk limits
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.state = TradingState.INITIALIZING

        # Core components
        self.event_bus = EventBus()
        self.state_manager = StateManager()

        # Component tracking
        self.registered_components: Set[str] = set()
        self.component_status: Dict[str, str] = {}
        self.last_health_check = {}

        # Workflow tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_metrics: Dict[str, Any] = {}

        # Recovery tracking
        self.recovery_attempts: Dict[str, int] = {}
        self.last_recovery_time: Dict[str, datetime] = {}

        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()

        # Logger
        self.logger = logging.getLogger("trading_orchestrator")

        # Setup event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup event handlers for orchestration"""

        # Market data handler
        market_data_handler = MarketDataOrchestrationHandler(self)
        self.event_bus.register_handler(EventType.MARKET_DATA, market_data_handler)

        # Strategy signal handler
        signal_handler = StrategySignalOrchestrationHandler(self)
        self.event_bus.register_handler(EventType.STRATEGY_SIGNAL, signal_handler)

        # Portfolio handler
        portfolio_handler = PortfolioOrchestrationHandler(self)
        self.event_bus.register_handler(EventType.PORTFOLIO, portfolio_handler)

        # Order handler
        order_handler = OrderOrchestrationHandler(self)
        self.event_bus.register_handler(EventType.ORDER, order_handler)

        # Execution handler
        execution_handler = ExecutionOrchestrationHandler(self)
        self.event_bus.register_handler(EventType.EXECUTION, execution_handler)

        # Risk handler
        risk_handler = RiskOrchestrationHandler(self)
        self.event_bus.register_handler(EventType.RISK, risk_handler)

        # System handler
        system_handler = SystemOrchestrationHandler(self)
        self.event_bus.register_handler(EventType.SYSTEM, system_handler)

    async def start(self) -> bool:
        """
        Start the trading orchestrator

        Returns:
            True if startup successful
        """
        try:
            self.logger.info("Starting trading orchestrator...")
            self.state = TradingState.INITIALIZING

            # Start event bus
            await self.event_bus.start()

            # Start state manager
            await self.state_manager.start()

            # Start background tasks
            await self._start_background_tasks()

            # Initialize components
            await self._initialize_components()

            # Transition to running state
            self.state = TradingState.RUNNING

            # Send system start event
            start_event = SystemEvent(
                source_component="orchestrator",
                system_action="START",
                status="RUNNING",
                message="Trading orchestrator started successfully"
            )
            await self.event_bus.publish(start_event)

            self.logger.info("Trading orchestrator started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start trading orchestrator: {e}")
            self.state = TradingState.ERROR
            return False

    async def stop(self) -> bool:
        """
        Stop the trading orchestrator

        Returns:
            True if shutdown successful
        """
        try:
            self.logger.info("Stopping trading orchestrator...")
            self.state = TradingState.STOPPING

            # Send system stop event
            stop_event = SystemEvent(
                source_component="orchestrator",
                system_action="STOP",
                status="STOPPING",
                message="Trading orchestrator stopping"
            )
            await self.event_bus.publish(stop_event)

            # Stop background tasks
            await self._stop_background_tasks()

            # Stop components
            await self._shutdown_components()

            # Stop state manager
            await self.state_manager.stop()

            # Stop event bus
            await self.event_bus.stop()

            self.state = TradingState.STOPPED
            self.logger.info("Trading orchestrator stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during orchestrator shutdown: {e}")
            self.state = TradingState.ERROR
            return False

    async def pause(self) -> bool:
        """Pause trading operations"""
        if self.state != TradingState.RUNNING:
            self.logger.warning(f"Cannot pause orchestrator in state {self.state}")
            return False

        self.state = TradingState.PAUSED

        pause_event = SystemEvent(
            source_component="orchestrator",
            system_action="PAUSE",
            status="PAUSED",
            message="Trading operations paused"
        )
        await self.event_bus.publish(pause_event)

        self.logger.info("Trading operations paused")
        return True

    async def resume(self) -> bool:
        """Resume trading operations"""
        if self.state != TradingState.PAUSED:
            self.logger.warning(f"Cannot resume orchestrator in state {self.state}")
            return False

        self.state = TradingState.RUNNING

        resume_event = SystemEvent(
            source_component="orchestrator",
            system_action="RESUME",
            status="RUNNING",
            message="Trading operations resumed"
        )
        await self.event_bus.publish(resume_event)

        self.logger.info("Trading operations resumed")
        return True

    async def emergency_stop(self, reason: str):
        """Emergency stop of all trading operations"""
        self.logger.critical(f"EMERGENCY STOP triggered: {reason}")

        # Cancel all active orders
        await self._cancel_all_orders()

        # Pause trading
        await self.pause()

        # Send critical alert
        emergency_event = SystemEvent(
            source_component="orchestrator",
            system_action="ERROR",
            status="ERROR",
            message=f"Emergency stop: {reason}",
            priority=EventPriority.CRITICAL
        )
        await self.event_bus.publish(emergency_event)

    def register_component(self, component_name: str, status: str = "INITIALIZING"):
        """Register a component with the orchestrator"""
        self.registered_components.add(component_name)
        self.component_status[component_name] = status
        self.last_health_check[component_name] = datetime.now()
        self.logger.info(f"Registered component: {component_name}")

    def update_component_status(self, component_name: str, status: str):
        """Update component status"""
        if component_name in self.registered_components:
            self.component_status[component_name] = status
            self.last_health_check[component_name] = datetime.now()
            self.logger.debug(f"Component {component_name} status: {status}")

    async def _start_background_tasks(self):
        """Start background monitoring tasks"""

        # Risk monitoring task
        risk_task = asyncio.create_task(self._risk_monitoring_loop())
        self.background_tasks.add(risk_task)

        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self.background_tasks.add(health_task)

        # Portfolio rebalancing task
        portfolio_task = asyncio.create_task(self._portfolio_monitoring_loop())
        self.background_tasks.add(portfolio_task)

        # Cleanup completed tasks
        cleanup_task = asyncio.create_task(self._cleanup_tasks())
        self.background_tasks.add(cleanup_task)

    async def _stop_background_tasks(self):
        """Stop all background tasks"""
        for task in self.background_tasks:
            task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()

    async def _risk_monitoring_loop(self):
        """Background risk monitoring"""
        while self.state in [TradingState.RUNNING, TradingState.PAUSED]:
            try:
                # Check portfolio risk metrics
                portfolio_state = await self.state_manager.get_portfolio_state()

                if portfolio_state:
                    # Create risk check event
                    risk_event = RiskEvent(
                        source_component="orchestrator",
                        risk_type="PERIODIC_CHECK",
                        severity="INFO",
                        risk_metrics=portfolio_state
                    )
                    await self.event_bus.publish(risk_event)

                await asyncio.sleep(self.config.risk_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retry

    async def _health_check_loop(self):
        """Background health monitoring"""
        while self.state in [TradingState.RUNNING, TradingState.PAUSED]:
            try:
                # Check component health
                now = datetime.now()

                for component, last_check in self.last_health_check.items():
                    time_since_check = now - last_check

                    if time_since_check > timedelta(minutes=5):
                        self.logger.warning(f"Component {component} not responding")

                        # Attempt recovery if enabled
                        if self.config.enable_auto_recovery:
                            await self._attempt_component_recovery(component)

                await asyncio.sleep(self.config.health_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)

    async def _portfolio_monitoring_loop(self):
        """Background portfolio monitoring"""
        while self.state in [TradingState.RUNNING, TradingState.PAUSED]:
            try:
                if self.state == TradingState.RUNNING:
                    # Trigger portfolio optimization check
                    portfolio_event = PortfolioEvent(
                        source_component="orchestrator",
                        action="OPTIMIZE"
                    )
                    await self.event_bus.publish(portfolio_event)

                await asyncio.sleep(self.config.portfolio_rebalance_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in portfolio monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _cleanup_tasks(self):
        """Cleanup completed background tasks"""
        while self.state != TradingState.STOPPED:
            try:
                # Remove completed tasks
                completed_tasks = {task for task in self.background_tasks if task.done()}
                self.background_tasks -= completed_tasks

                await asyncio.sleep(60)  # Cleanup every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task cleanup: {e}")

    async def _initialize_components(self):
        """Initialize all trading components"""
        # This would initialize all the trading components
        # For now, we'll create placeholder events

        init_event = SystemEvent(
            source_component="orchestrator",
            system_action="HEALTH_CHECK",
            status="INITIALIZING",
            message="Initializing trading components"
        )
        await self.event_bus.publish(init_event)

    async def _shutdown_components(self):
        """Shutdown all components gracefully"""
        shutdown_event = SystemEvent(
            source_component="orchestrator",
            system_action="STOP",
            status="STOPPING",
            message="Shutting down trading components"
        )
        await self.event_bus.publish(shutdown_event)

    async def _cancel_all_orders(self):
        """Cancel all active orders in emergency"""
        self.logger.warning("Cancelling all active orders")

        # This would interact with the order management system
        # For now, create a cancel event
        cancel_event = OrderEvent(
            source_component="orchestrator",
            action="CANCEL",
            symbol="ALL",
            side="ALL",
            priority=EventPriority.CRITICAL
        )
        await self.event_bus.publish(cancel_event)

    async def _attempt_component_recovery(self, component_name: str):
        """Attempt to recover a failed component"""
        attempts = self.recovery_attempts.get(component_name, 0)

        if attempts >= self.config.max_recovery_attempts:
            self.logger.error(f"Maximum recovery attempts reached for {component_name}")
            return

        self.recovery_attempts[component_name] = attempts + 1
        self.last_recovery_time[component_name] = datetime.now()

        recovery_event = SystemEvent(
            source_component="orchestrator",
            system_action="HEALTH_CHECK",
            component=component_name,
            status="RECOVERING",
            message=f"Attempting recovery of {component_name}, attempt {attempts + 1}"
        )
        await self.event_bus.publish(recovery_event)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'orchestrator_state': self.state.value,
            'component_count': len(self.registered_components),
            'component_status': self.component_status.copy(),
            'active_workflows': len(self.active_workflows),
            'background_tasks': len(self.background_tasks),
            'event_bus_metrics': self.event_bus.get_metrics(),
            'recovery_attempts': self.recovery_attempts.copy(),
            'config': {
                'paper_trading': self.config.enable_paper_trading,
                'max_concurrent_orders': self.config.max_concurrent_orders,
                'auto_recovery': self.config.enable_auto_recovery
            }
        }


# Event handler classes for orchestration

class MarketDataOrchestrationHandler(BaseEventHandler):
    """Orchestration handler for market data events"""

    def __init__(self, orchestrator: TradingOrchestrator):
        super().__init__("market_data_orchestration")
        self.orchestrator = orchestrator

    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """Handle market data event"""
        # Update component health
        self.orchestrator.update_component_status("market_data", "RUNNING")

        # Forward to strategy engine (in production, this would be done by adapters)
        return HandlerResult(success=True)


class StrategySignalOrchestrationHandler(BaseEventHandler):
    """Orchestration handler for strategy signal events"""

    def __init__(self, orchestrator: TradingOrchestrator):
        super().__init__("strategy_signal_orchestration")
        self.orchestrator = orchestrator

    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """Handle strategy signal event"""
        signal_event = event

        # Update component health
        self.orchestrator.update_component_status("strategy_engine", "RUNNING")

        # Create portfolio optimization event
        portfolio_event = PortfolioEvent(
            source_component="orchestrator",
            action="UPDATE_ALLOCATION",
            correlation_id=signal_event.event_id
        )

        return HandlerResult(
            success=True,
            additional_events=[portfolio_event]
        )


class PortfolioOrchestrationHandler(BaseEventHandler):
    """Orchestration handler for portfolio events"""

    def __init__(self, orchestrator: TradingOrchestrator):
        super().__init__("portfolio_orchestration")
        self.orchestrator = orchestrator

    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """Handle portfolio event"""
        # Update component health
        self.orchestrator.update_component_status("portfolio_optimizer", "RUNNING")

        # In production, this would create order events based on rebalancing
        return HandlerResult(success=True)


class OrderOrchestrationHandler(BaseEventHandler):
    """Orchestration handler for order events"""

    def __init__(self, orchestrator: TradingOrchestrator):
        super().__init__("order_orchestration")
        self.orchestrator = orchestrator

    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """Handle order event"""
        # Update component health
        self.orchestrator.update_component_status("order_manager", "RUNNING")

        # Track active orders (simplified)
        return HandlerResult(success=True)


class ExecutionOrchestrationHandler(BaseEventHandler):
    """Orchestration handler for execution events"""

    def __init__(self, orchestrator: TradingOrchestrator):
        super().__init__("execution_orchestration")
        self.orchestrator = orchestrator

    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """Handle execution event"""
        # Update component health
        self.orchestrator.update_component_status("execution_engine", "RUNNING")

        # Update position state in state manager
        return HandlerResult(success=True)


class RiskOrchestrationHandler(BaseEventHandler):
    """Orchestration handler for risk events"""

    def __init__(self, orchestrator: TradingOrchestrator):
        super().__init__("risk_orchestration")
        self.orchestrator = orchestrator

    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """Handle risk event"""
        risk_event = event

        # Update component health
        self.orchestrator.update_component_status("risk_manager", "RUNNING")

        # Handle critical risk events
        if risk_event.severity == "CRITICAL" and self.orchestrator.config.emergency_stop_on_risk_breach:
            await self.orchestrator.emergency_stop(f"Critical risk event: {risk_event.risk_type}")

        return HandlerResult(success=True)


class SystemOrchestrationHandler(BaseEventHandler):
    """Orchestration handler for system events"""

    def __init__(self, orchestrator: TradingOrchestrator):
        super().__init__("system_orchestration")
        self.orchestrator = orchestrator

    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """Handle system event"""
        system_event = event

        # Update component status if specified
        if system_event.component:
            self.orchestrator.update_component_status(
                system_event.component,
                system_event.status
            )

        return HandlerResult(success=True)