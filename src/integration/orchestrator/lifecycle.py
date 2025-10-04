# src/integration/orchestrator/lifecycle.py
"""
Trading Orchestrator Lifecycle Management

Handles system lifecycle operations including start, stop, pause, resume, and emergency stop.
"""

import logging
from ..events.models import SystemEvent, OrderEvent, EventPriority
from .models import TradingState


class LifecycleManager:
    """Manages the lifecycle of the trading orchestrator"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("lifecycle_manager")

    async def start(self) -> bool:
        """
        Start the trading orchestrator

        Returns:
            True if startup successful
        """
        try:
            self.logger.info("Starting trading orchestrator...")
            self.orchestrator.state = TradingState.INITIALIZING

            # Start event bus
            await self.orchestrator.event_bus.start()

            # Start state manager
            await self.orchestrator.state_manager.start()

            # Start background tasks
            await self.orchestrator.monitoring_manager.start_background_tasks()

            # Initialize components
            await self._initialize_components()

            # Transition to running state
            self.orchestrator.state = TradingState.RUNNING

            # Send system start event
            start_event = SystemEvent(
                source_component="orchestrator",
                system_action="START",
                status="RUNNING",
                message="Trading orchestrator started successfully"
            )
            await self.orchestrator.event_bus.publish(start_event)

            self.logger.info("Trading orchestrator started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start trading orchestrator: {e}")
            self.orchestrator.state = TradingState.ERROR
            return False

    async def stop(self) -> bool:
        """
        Stop the trading orchestrator

        Returns:
            True if shutdown successful
        """
        try:
            self.logger.info("Stopping trading orchestrator...")
            self.orchestrator.state = TradingState.STOPPING

            # Send system stop event
            stop_event = SystemEvent(
                source_component="orchestrator",
                system_action="STOP",
                status="STOPPING",
                message="Trading orchestrator stopping"
            )
            await self.orchestrator.event_bus.publish(stop_event)

            # Stop background tasks
            await self.orchestrator.monitoring_manager.stop_background_tasks()

            # Stop components
            await self._shutdown_components()

            # Stop state manager
            await self.orchestrator.state_manager.stop()

            # Stop event bus
            await self.orchestrator.event_bus.stop()

            self.orchestrator.state = TradingState.STOPPED
            self.logger.info("Trading orchestrator stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during orchestrator shutdown: {e}")
            self.orchestrator.state = TradingState.ERROR
            return False

    async def pause(self) -> bool:
        """Pause trading operations"""
        if self.orchestrator.state != TradingState.RUNNING:
            self.logger.warning(f"Cannot pause orchestrator in state {self.orchestrator.state}")
            return False

        self.orchestrator.state = TradingState.PAUSED

        pause_event = SystemEvent(
            source_component="orchestrator",
            system_action="PAUSE",
            status="PAUSED",
            message="Trading operations paused"
        )
        await self.orchestrator.event_bus.publish(pause_event)

        self.logger.info("Trading operations paused")
        return True

    async def resume(self) -> bool:
        """Resume trading operations"""
        if self.orchestrator.state != TradingState.PAUSED:
            self.logger.warning(f"Cannot resume orchestrator in state {self.orchestrator.state}")
            return False

        self.orchestrator.state = TradingState.RUNNING

        resume_event = SystemEvent(
            source_component="orchestrator",
            system_action="RESUME",
            status="RUNNING",
            message="Trading operations resumed"
        )
        await self.orchestrator.event_bus.publish(resume_event)

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
        await self.orchestrator.event_bus.publish(emergency_event)

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
        await self.orchestrator.event_bus.publish(init_event)

    async def _shutdown_components(self):
        """Shutdown all components gracefully"""
        shutdown_event = SystemEvent(
            source_component="orchestrator",
            system_action="STOP",
            status="STOPPING",
            message="Shutting down trading components"
        )
        await self.orchestrator.event_bus.publish(shutdown_event)

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
        await self.orchestrator.event_bus.publish(cancel_event)