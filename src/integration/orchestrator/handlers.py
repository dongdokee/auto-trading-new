# src/integration/orchestrator/handlers.py
"""
Trading Orchestrator Event Handlers

Handles all event types for the orchestration workflow.
"""

from ..events.handlers import BaseEventHandler, HandlerResult
from ..events.models import BaseEvent, EventType, PortfolioEvent


class HandlerManager:
    """Manages event handlers for the trading orchestrator"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def setup_event_handlers(self):
        """Setup event handlers for orchestration"""

        # Market data handler
        market_data_handler = MarketDataOrchestrationHandler(self.orchestrator)
        self.orchestrator.event_bus.register_handler(EventType.MARKET_DATA, market_data_handler)

        # Strategy signal handler
        signal_handler = StrategySignalOrchestrationHandler(self.orchestrator)
        self.orchestrator.event_bus.register_handler(EventType.STRATEGY_SIGNAL, signal_handler)

        # Portfolio handler
        portfolio_handler = PortfolioOrchestrationHandler(self.orchestrator)
        self.orchestrator.event_bus.register_handler(EventType.PORTFOLIO, portfolio_handler)

        # Order handler
        order_handler = OrderOrchestrationHandler(self.orchestrator)
        self.orchestrator.event_bus.register_handler(EventType.ORDER, order_handler)

        # Execution handler
        execution_handler = ExecutionOrchestrationHandler(self.orchestrator)
        self.orchestrator.event_bus.register_handler(EventType.EXECUTION, execution_handler)

        # Risk handler
        risk_handler = RiskOrchestrationHandler(self.orchestrator)
        self.orchestrator.event_bus.register_handler(EventType.RISK, risk_handler)

        # System handler
        system_handler = SystemOrchestrationHandler(self.orchestrator)
        self.orchestrator.event_bus.register_handler(EventType.SYSTEM, system_handler)


class MarketDataOrchestrationHandler(BaseEventHandler):
    """Orchestration handler for market data events"""

    def __init__(self, orchestrator):
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

    def __init__(self, orchestrator):
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

    def __init__(self, orchestrator):
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

    def __init__(self, orchestrator):
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

    def __init__(self, orchestrator):
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

    def __init__(self, orchestrator):
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

    def __init__(self, orchestrator):
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