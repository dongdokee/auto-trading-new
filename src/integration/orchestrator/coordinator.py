# src/integration/orchestrator/coordinator.py
"""
Trading Orchestrator Coordinator

Core coordination system that manages the complete trading workflow.
Orchestrates all components through event-driven architecture.

Phase 8 Optimizations:
- Parallel component initialization and management
- Batch processing for component operations
- Memory-efficient component tracking
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Callable, Tuple

from src.integration.events.event_bus import EventBus
from src.integration.state.manager import StateManager
from src.core.patterns.async_utils import (
    BatchProcessor, ConcurrentExecutor, process_concurrently, BatchResult
)
from src.core.patterns.memory_utils import CircularBuffer, MemoryMonitor
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

        # Component tracking with memory optimization
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

        # Phase 8 Optimizations: Async and Memory utilities
        self.batch_processor = BatchProcessor(
            batch_size=getattr(config, 'component_batch_size', 10),
            max_concurrent_batches=getattr(config, 'max_concurrent_batches', 3),
            timeout_seconds=getattr(config, 'component_timeout_seconds', 30.0)
        )

        self.concurrent_executor = ConcurrentExecutor(
            max_concurrent=getattr(config, 'max_concurrent_operations', 20),
            timeout_seconds=getattr(config, 'operation_timeout_seconds', 30.0)
        )

        # Memory-efficient circular buffers for metrics
        self.component_metrics_buffer = CircularBuffer(maxsize=1000)
        self.workflow_metrics_buffer = CircularBuffer(maxsize=500)
        self.memory_monitor = MemoryMonitor(
            alert_threshold_mb=getattr(config, 'memory_alert_threshold_mb', 1000.0)
        )

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

    async def initialize_components_parallel(self, components: List[Tuple[str, Callable]]) -> BatchResult:
        """
        Initialize multiple components in parallel with batch processing

        Args:
            components: List of (component_name, init_function) tuples

        Returns:
            BatchResult with initialization results
        """
        self.logger.info(f"Initializing {len(components)} components in parallel")

        async def init_component(component_data: Tuple[str, Callable]) -> str:
            name, init_func = component_data
            try:
                self.register_component(name, "INITIALIZING")

                # Execute initialization
                if asyncio.iscoroutinefunction(init_func):
                    await init_func()
                else:
                    init_func()

                self.update_component_status(name, "RUNNING")
                self.logger.debug(f"Component {name} initialized successfully")
                return name

            except Exception as e:
                self.update_component_status(name, "FAILED")
                self.logger.error(f"Failed to initialize component {name}: {e}")
                raise

        # Use batch processor for parallel initialization
        result = await self.batch_processor.process_batch(
            items=components,
            processor=init_component,
            return_exceptions=True
        )

        self.logger.info(
            f"Component initialization complete: {result.success_count}/{result.total_items} "
            f"successful ({result.success_rate:.2%})"
        )

        return result

    async def health_check_all_components_parallel(self) -> Dict[str, Any]:
        """
        Perform health checks on all components in parallel

        Returns:
            Dictionary with health check results
        """
        if not self.registered_components:
            return {}

        async def check_component_health(component_name: str) -> Tuple[str, Dict[str, Any]]:
            try:
                # Simulate health check - in practice, this would call component-specific health check
                start_time = datetime.now()

                # Component health check logic would go here
                # For now, we'll check if component is in good state
                status = self.component_status.get(component_name, "UNKNOWN")
                last_check = self.last_health_check.get(component_name)

                health_data = {
                    'status': status,
                    'last_check': last_check,
                    'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'healthy': status in ["RUNNING", "PAUSED"]
                }

                self.last_health_check[component_name] = datetime.now()
                return component_name, health_data

            except Exception as e:
                return component_name, {
                    'status': 'ERROR',
                    'error': str(e),
                    'healthy': False,
                    'last_check': datetime.now()
                }

        # Execute health checks concurrently
        components_list = list(self.registered_components)
        results = await process_concurrently(
            items=components_list,
            processor=check_component_health,
            max_concurrent=10,
            timeout=5.0
        )

        # Process results
        health_results = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed: {result}")
                continue

            component_name, health_data = result
            health_results[component_name] = health_data

        return health_results

    async def batch_update_component_metrics(self, metrics_updates: List[Tuple[str, Dict[str, Any]]]):
        """
        Update component metrics in batch for efficiency

        Args:
            metrics_updates: List of (component_name, metrics) tuples
        """
        timestamp = datetime.now()

        for component_name, metrics in metrics_updates:
            # Add to circular buffer for memory efficiency
            metric_entry = {
                'component': component_name,
                'metrics': metrics,
                'timestamp': timestamp
            }
            self.component_metrics_buffer.append(metric_entry)

        self.logger.debug(f"Updated metrics for {len(metrics_updates)} components")

    def get_recent_component_metrics(self, component_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent component metrics from circular buffer

        Args:
            component_name: Optional component name filter
            limit: Maximum number of metrics to return

        Returns:
            List of recent metrics
        """
        recent_metrics = self.component_metrics_buffer.get_recent(limit)

        if component_name:
            # Filter by component name
            recent_metrics = [
                metric for metric in recent_metrics
                if metric.get('component') == component_name
            ]

        return recent_metrics

    async def graceful_shutdown_parallel(self, components: Optional[List[str]] = None) -> BatchResult:
        """
        Shutdown components in parallel with graceful handling

        Args:
            components: Optional list of specific components to shutdown

        Returns:
            BatchResult with shutdown results
        """
        if components is None:
            components = list(self.registered_components)

        self.logger.info(f"Shutting down {len(components)} components in parallel")

        async def shutdown_component(component_name: str) -> str:
            try:
                self.update_component_status(component_name, "SHUTTING_DOWN")

                # In practice, this would call component-specific shutdown logic
                await asyncio.sleep(0.1)  # Simulate shutdown time

                self.update_component_status(component_name, "STOPPED")
                self.logger.debug(f"Component {component_name} shutdown successfully")
                return component_name

            except Exception as e:
                self.update_component_status(component_name, "ERROR")
                self.logger.error(f"Failed to shutdown component {component_name}: {e}")
                raise

        # Use batch processor for parallel shutdown
        result = await self.batch_processor.process_batch(
            items=components,
            processor=shutdown_component,
            return_exceptions=True
        )

        # Cleanup executor
        await self.concurrent_executor.shutdown()

        self.logger.info(
            f"Component shutdown complete: {result.success_count}/{result.total_items} "
            f"successful ({result.success_rate:.2%})"
        )

        return result

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with optimization metrics"""
        # Get memory statistics
        memory_stats = self.memory_monitor.get_current_stats()

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
            },
            # Phase 8 optimization metrics
            'optimization_metrics': {
                'batch_processor': self.batch_processor.get_metrics(),
                'concurrent_executor': self.concurrent_executor.get_metrics(),
                'component_metrics_buffer': self.component_metrics_buffer.get_stats(),
                'workflow_metrics_buffer': self.workflow_metrics_buffer.get_stats(),
                'memory_stats': {
                    'current_rss_mb': memory_stats.rss_mb,
                    'current_percent': memory_stats.percent,
                    'available_mb': memory_stats.available_mb
                },
                'memory_summary': self.memory_monitor.get_summary()
            }
        }

    async def cleanup_resources(self):
        """Cleanup optimization resources"""
        try:
            # Shutdown concurrent executor
            await self.concurrent_executor.shutdown(timeout=5.0)

            # Clear buffers
            self.component_metrics_buffer.clear()
            self.workflow_metrics_buffer.clear()

            self.logger.info("Optimization resources cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")