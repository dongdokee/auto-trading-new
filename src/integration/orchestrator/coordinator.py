# src/integration/orchestrator/coordinator.py
"""
Trading Orchestrator Coordinator

Core coordination system that manages the complete trading workflow.
Orchestrates all components through event-driven architecture.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

from src.integration.events.event_bus import EventBus
from src.integration.state.manager import StateManager
from .models import TradingState, OrchestratorConfig
from .lifecycle import LifecycleManager
from .monitoring import MonitoringManager
from .handlers import HandlerManager


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

        # Managers
        self.lifecycle_manager = LifecycleManager(self)
        self.monitoring_manager = MonitoringManager(self)
        self.handler_manager = HandlerManager(self)

        # Setup event handlers
        self.handler_manager.setup_event_handlers()

    async def start(self) -> bool:
        """
        Start the trading orchestrator

        Returns:
            True if startup successful
        """
        return await self.lifecycle_manager.start()

    async def stop(self) -> bool:
        """
        Stop the trading orchestrator

        Returns:
            True if shutdown successful
        """
        return await self.lifecycle_manager.stop()

    async def pause(self) -> bool:
        """Pause trading operations"""
        return await self.lifecycle_manager.pause()

    async def resume(self) -> bool:
        """Resume trading operations"""
        return await self.lifecycle_manager.resume()

    async def emergency_stop(self, reason: str):
        """Emergency stop of all trading operations"""
        await self.lifecycle_manager.emergency_stop(reason)

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