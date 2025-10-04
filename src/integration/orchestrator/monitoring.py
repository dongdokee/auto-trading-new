# src/integration/orchestrator/monitoring.py
"""
Trading Orchestrator Monitoring Manager

Handles background monitoring tasks including risk checks, health monitoring,
and portfolio rebalancing.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from ..events.models import RiskEvent, PortfolioEvent, SystemEvent
from .models import TradingState


class MonitoringManager:
    """Manages background monitoring tasks for the trading orchestrator"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("monitoring_manager")

    async def start_background_tasks(self):
        """Start background monitoring tasks"""

        # Risk monitoring task
        risk_task = asyncio.create_task(self._risk_monitoring_loop())
        self.orchestrator.background_tasks.add(risk_task)

        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self.orchestrator.background_tasks.add(health_task)

        # Portfolio rebalancing task
        portfolio_task = asyncio.create_task(self._portfolio_monitoring_loop())
        self.orchestrator.background_tasks.add(portfolio_task)

        # Cleanup completed tasks
        cleanup_task = asyncio.create_task(self._cleanup_tasks())
        self.orchestrator.background_tasks.add(cleanup_task)

    async def stop_background_tasks(self):
        """Stop all background tasks"""
        for task in self.orchestrator.background_tasks:
            task.cancel()

        if self.orchestrator.background_tasks:
            await asyncio.gather(*self.orchestrator.background_tasks, return_exceptions=True)

        self.orchestrator.background_tasks.clear()

    async def _risk_monitoring_loop(self):
        """Background risk monitoring"""
        while self.orchestrator.state in [TradingState.RUNNING, TradingState.PAUSED]:
            try:
                # Check portfolio risk metrics
                portfolio_state = await self.orchestrator.state_manager.get_portfolio_state()

                if portfolio_state:
                    # Create risk check event
                    risk_event = RiskEvent(
                        source_component="orchestrator",
                        risk_type="PERIODIC_CHECK",
                        severity="INFO",
                        risk_metrics=portfolio_state
                    )
                    await self.orchestrator.event_bus.publish(risk_event)

                await asyncio.sleep(self.orchestrator.config.risk_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retry

    async def _health_check_loop(self):
        """Background health monitoring"""
        while self.orchestrator.state in [TradingState.RUNNING, TradingState.PAUSED]:
            try:
                # Check component health
                now = datetime.now()

                for component, last_check in self.orchestrator.last_health_check.items():
                    time_since_check = now - last_check

                    if time_since_check > timedelta(minutes=5):
                        self.logger.warning(f"Component {component} not responding")

                        # Attempt recovery if enabled
                        if self.orchestrator.config.enable_auto_recovery:
                            await self._attempt_component_recovery(component)

                await asyncio.sleep(self.orchestrator.config.health_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)

    async def _portfolio_monitoring_loop(self):
        """Background portfolio monitoring"""
        while self.orchestrator.state in [TradingState.RUNNING, TradingState.PAUSED]:
            try:
                if self.orchestrator.state == TradingState.RUNNING:
                    # Trigger portfolio optimization check
                    portfolio_event = PortfolioEvent(
                        source_component="orchestrator",
                        action="OPTIMIZE"
                    )
                    await self.orchestrator.event_bus.publish(portfolio_event)

                await asyncio.sleep(self.orchestrator.config.portfolio_rebalance_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in portfolio monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _cleanup_tasks(self):
        """Cleanup completed background tasks"""
        while self.orchestrator.state != TradingState.STOPPED:
            try:
                # Remove completed tasks
                completed_tasks = {task for task in self.orchestrator.background_tasks if task.done()}
                self.orchestrator.background_tasks -= completed_tasks

                await asyncio.sleep(60)  # Cleanup every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task cleanup: {e}")

    async def _attempt_component_recovery(self, component_name: str):
        """Attempt to recover a failed component"""
        attempts = self.orchestrator.recovery_attempts.get(component_name, 0)

        if attempts >= self.orchestrator.config.max_recovery_attempts:
            self.logger.error(f"Maximum recovery attempts reached for {component_name}")
            return

        self.orchestrator.recovery_attempts[component_name] = attempts + 1
        self.orchestrator.last_recovery_time[component_name] = datetime.now()

        recovery_event = SystemEvent(
            source_component="orchestrator",
            system_action="HEALTH_CHECK",
            component=component_name,
            status="RECOVERING",
            message=f"Attempting recovery of {component_name}, attempt {attempts + 1}"
        )
        await self.orchestrator.event_bus.publish(recovery_event)