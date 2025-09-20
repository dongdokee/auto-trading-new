# src/integration/monitoring/monitor.py
"""
System Monitor

Comprehensive monitoring system for trading system health, performance, and alerts.
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from ..events.event_bus import EventBus
from ..events.models import SystemEvent, RiskEvent, EventPriority
from ..state.manager import StateManager
from .alerts import AlertManager
from .metrics import MetricsCollector


class HealthStatus(Enum):
    """Component health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a component"""
    component_name: str
    status: HealthStatus
    last_heartbeat: datetime
    response_time_ms: float
    error_count: int
    error_rate_pct: float
    message: str = ""
    metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class SystemHealth:
    """Overall system health summary"""
    overall_status: HealthStatus
    healthy_components: int
    warning_components: int
    critical_components: int
    total_components: int
    uptime_seconds: float
    last_update: datetime
    critical_issues: List[str]


class SystemMonitor:
    """
    Comprehensive system monitoring and health checking

    Features:
    - Component health monitoring
    - Performance metrics collection
    - Resource utilization tracking
    - Automated alerting
    - Health status aggregation
    """

    def __init__(self,
                 event_bus: EventBus,
                 state_manager: StateManager,
                 alert_manager: AlertManager,
                 check_interval_seconds: int = 30):

        self.event_bus = event_bus
        self.state_manager = state_manager
        self.alert_manager = alert_manager
        self.check_interval = check_interval_seconds

        # Component registry
        self.registered_components: Set[str] = set()
        self.component_health: Dict[str, ComponentHealth] = {}
        self.component_checkers: Dict[str, callable] = {}

        # System metrics
        self.metrics_collector = MetricsCollector()
        self.start_time = datetime.now()

        # Monitoring state
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Health thresholds
        self.response_time_warning_ms = 1000  # 1 second
        self.response_time_critical_ms = 5000  # 5 seconds
        self.error_rate_warning_pct = 5.0      # 5%
        self.error_rate_critical_pct = 15.0    # 15%
        self.heartbeat_timeout_seconds = 120   # 2 minutes

        # Alert tracking
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []

        # Logger
        self.logger = logging.getLogger("system_monitor")

        # Register default component checkers
        self._register_default_checkers()

    def _register_default_checkers(self):
        """Register default health checkers for core components"""

        # Event bus health checker
        async def check_event_bus() -> ComponentHealth:
            try:
                start_time = time.time()
                metrics = self.event_bus.get_metrics()
                response_time = (time.time() - start_time) * 1000

                # Determine health based on queue size and processing rate
                queue_size = metrics['queue_status']['current_size']
                max_queue_size = metrics['queue_status']['max_queue_size']
                queue_utilization = queue_size / max_queue_size

                if queue_utilization > 0.9:
                    status = HealthStatus.CRITICAL
                    message = f"Event queue {queue_utilization*100:.1f}% full"
                elif queue_utilization > 0.7:
                    status = HealthStatus.WARNING
                    message = f"Event queue {queue_utilization*100:.1f}% full"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Event bus operating normally"

                return ComponentHealth(
                    component_name="event_bus",
                    status=status,
                    last_heartbeat=datetime.now(),
                    response_time_ms=response_time,
                    error_count=metrics['event_bus']['events_failed'],
                    error_rate_pct=0.0,  # Calculate from metrics
                    message=message,
                    metrics=metrics
                )

            except Exception as e:
                return ComponentHealth(
                    component_name="event_bus",
                    status=HealthStatus.CRITICAL,
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.0,
                    error_count=0,
                    error_rate_pct=100.0,
                    message=f"Health check failed: {str(e)}"
                )

        # State manager health checker
        async def check_state_manager() -> ComponentHealth:
            try:
                start_time = time.time()
                metrics = self.state_manager.get_state_metrics()
                response_time = (time.time() - start_time) * 1000

                status = HealthStatus.HEALTHY if metrics['is_running'] else HealthStatus.CRITICAL
                message = "State manager running" if metrics['is_running'] else "State manager stopped"

                return ComponentHealth(
                    component_name="state_manager",
                    status=status,
                    last_heartbeat=datetime.now(),
                    response_time_ms=response_time,
                    error_count=0,
                    error_rate_pct=0.0,
                    message=message,
                    metrics=metrics
                )

            except Exception as e:
                return ComponentHealth(
                    component_name="state_manager",
                    status=HealthStatus.CRITICAL,
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.0,
                    error_count=0,
                    error_rate_pct=100.0,
                    message=f"Health check failed: {str(e)}"
                )

        # System resources health checker
        async def check_system_resources() -> ComponentHealth:
            try:
                start_time = time.time()

                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                response_time = (time.time() - start_time) * 1000

                # Determine health based on resource usage
                if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                    status = HealthStatus.CRITICAL
                    message = f"High resource usage: CPU {cpu_percent:.1f}%, RAM {memory.percent:.1f}%, Disk {disk.percent:.1f}%"
                elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 80:
                    status = HealthStatus.WARNING
                    message = f"Elevated resource usage: CPU {cpu_percent:.1f}%, RAM {memory.percent:.1f}%, Disk {disk.percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Resource usage normal: CPU {cpu_percent:.1f}%, RAM {memory.percent:.1f}%, Disk {disk.percent:.1f}%"

                return ComponentHealth(
                    component_name="system_resources",
                    status=status,
                    last_heartbeat=datetime.now(),
                    response_time_ms=response_time,
                    error_count=0,
                    error_rate_pct=0.0,
                    message=message,
                    metrics={
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_available_gb': memory.available / (1024**3),
                        'disk_percent': disk.percent,
                        'disk_free_gb': disk.free / (1024**3)
                    }
                )

            except Exception as e:
                return ComponentHealth(
                    component_name="system_resources",
                    status=HealthStatus.CRITICAL,
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.0,
                    error_count=0,
                    error_rate_pct=100.0,
                    message=f"Resource check failed: {str(e)}"
                )

        # Register checkers
        self.register_component_checker("event_bus", check_event_bus)
        self.register_component_checker("state_manager", check_state_manager)
        self.register_component_checker("system_resources", check_system_resources)

    async def start(self):
        """Start the system monitor"""
        if self.is_running:
            self.logger.warning("System monitor is already running")
            return

        self.is_running = True
        self.start_time = datetime.now()

        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        # Start metrics collection
        await self.metrics_collector.start()

        self.logger.info("System monitor started")

    async def stop(self):
        """Stop the system monitor"""
        if not self.is_running:
            self.logger.warning("System monitor is not running")
            return

        self.is_running = False

        # Stop monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop metrics collection
        await self.metrics_collector.stop()

        self.logger.info("System monitor stopped")

    def register_component(self, component_name: str):
        """Register a component for monitoring"""
        self.registered_components.add(component_name)
        self.logger.info(f"Registered component for monitoring: {component_name}")

    def register_component_checker(self, component_name: str, checker_func: callable):
        """Register a custom health checker for a component"""
        self.component_checkers[component_name] = checker_func
        self.register_component(component_name)
        self.logger.info(f"Registered health checker for: {component_name}")

    async def check_component_health(self, component_name: str) -> ComponentHealth:
        """Check health of a specific component"""
        if component_name in self.component_checkers:
            try:
                health = await self.component_checkers[component_name]()
                self.component_health[component_name] = health
                return health

            except Exception as e:
                self.logger.error(f"Error checking health of {component_name}: {e}")
                health = ComponentHealth(
                    component_name=component_name,
                    status=HealthStatus.CRITICAL,
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.0,
                    error_count=0,
                    error_rate_pct=100.0,
                    message=f"Health check error: {str(e)}"
                )
                self.component_health[component_name] = health
                return health

        else:
            # Default health check (just heartbeat)
            last_heartbeat = self.component_health.get(component_name, ComponentHealth(
                component_name=component_name,
                status=HealthStatus.UNKNOWN,
                last_heartbeat=datetime.min,
                response_time_ms=0.0,
                error_count=0,
                error_rate_pct=0.0
            )).last_heartbeat

            time_since_heartbeat = datetime.now() - last_heartbeat

            if time_since_heartbeat.total_seconds() > self.heartbeat_timeout_seconds:
                status = HealthStatus.CRITICAL
                message = f"No heartbeat for {time_since_heartbeat.total_seconds():.0f} seconds"
            else:
                status = HealthStatus.HEALTHY
                message = "Component responsive"

            health = ComponentHealth(
                component_name=component_name,
                status=status,
                last_heartbeat=last_heartbeat,
                response_time_ms=0.0,
                error_count=0,
                error_rate_pct=0.0,
                message=message
            )

            self.component_health[component_name] = health
            return health

    async def check_all_components(self) -> Dict[str, ComponentHealth]:
        """Check health of all registered components"""
        health_results = {}

        for component_name in self.registered_components:
            health = await self.check_component_health(component_name)
            health_results[component_name] = health

        return health_results

    async def get_system_health(self) -> SystemHealth:
        """Get overall system health summary"""
        health_results = await self.check_all_components()

        healthy_count = sum(1 for h in health_results.values() if h.status == HealthStatus.HEALTHY)
        warning_count = sum(1 for h in health_results.values() if h.status == HealthStatus.WARNING)
        critical_count = sum(1 for h in health_results.values() if h.status == HealthStatus.CRITICAL)
        total_count = len(health_results)

        # Determine overall status
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY

        # Collect critical issues
        critical_issues = [
            f"{h.component_name}: {h.message}"
            for h in health_results.values()
            if h.status == HealthStatus.CRITICAL
        ]

        uptime = (datetime.now() - self.start_time).total_seconds()

        return SystemHealth(
            overall_status=overall_status,
            healthy_components=healthy_count,
            warning_components=warning_count,
            critical_components=critical_count,
            total_components=total_count,
            uptime_seconds=uptime,
            last_update=datetime.now(),
            critical_issues=critical_issues
        )

    async def update_component_heartbeat(self, component_name: str, metrics: Optional[Dict[str, Any]] = None):
        """Update component heartbeat"""
        if component_name not in self.component_health:
            self.component_health[component_name] = ComponentHealth(
                component_name=component_name,
                status=HealthStatus.HEALTHY,
                last_heartbeat=datetime.now(),
                response_time_ms=0.0,
                error_count=0,
                error_rate_pct=0.0,
                message="Heartbeat received",
                metrics=metrics or {}
            )
        else:
            health = self.component_health[component_name]
            health.last_heartbeat = datetime.now()
            health.status = HealthStatus.HEALTHY
            health.message = "Heartbeat received"
            if metrics:
                health.metrics.update(metrics)

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Check system health
                system_health = await self.get_system_health()

                # Collect system metrics
                await self.metrics_collector.collect_metrics()

                # Process alerts
                await self._process_health_alerts(system_health)

                # Update state manager with health status
                await self._update_health_state(system_health)

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    async def _process_health_alerts(self, system_health: SystemHealth):
        """Process health-related alerts"""
        try:
            # Generate alerts for critical components
            for component_name, health in self.component_health.items():
                alert_key = f"component_health_{component_name}"

                if health.status == HealthStatus.CRITICAL:
                    if alert_key not in self.active_alerts:
                        # New critical alert
                        alert_data = {
                            'alert_type': 'COMPONENT_CRITICAL',
                            'component': component_name,
                            'message': health.message,
                            'timestamp': datetime.now(),
                            'severity': 'CRITICAL'
                        }

                        await self.alert_manager.create_alert(
                            alert_type="COMPONENT_CRITICAL",
                            message=f"Component {component_name} is in critical state: {health.message}",
                            severity="CRITICAL",
                            metadata=alert_data
                        )

                        self.active_alerts[alert_key] = alert_data

                elif health.status == HealthStatus.WARNING:
                    if alert_key not in self.active_alerts:
                        # New warning alert
                        alert_data = {
                            'alert_type': 'COMPONENT_WARNING',
                            'component': component_name,
                            'message': health.message,
                            'timestamp': datetime.now(),
                            'severity': 'WARNING'
                        }

                        await self.alert_manager.create_alert(
                            alert_type="COMPONENT_WARNING",
                            message=f"Component {component_name} has warnings: {health.message}",
                            severity="WARNING",
                            metadata=alert_data
                        )

                        self.active_alerts[alert_key] = alert_data

                else:
                    # Component is healthy, resolve any active alerts
                    if alert_key in self.active_alerts:
                        await self.alert_manager.resolve_alert(alert_key)
                        del self.active_alerts[alert_key]

            # Generate system-level alerts
            if system_health.overall_status == HealthStatus.CRITICAL:
                system_alert_key = "system_health_critical"
                if system_alert_key not in self.active_alerts:
                    await self.alert_manager.create_alert(
                        alert_type="SYSTEM_CRITICAL",
                        message=f"System health is critical: {len(system_health.critical_issues)} critical issues",
                        severity="CRITICAL",
                        metadata={
                            'critical_components': system_health.critical_components,
                            'critical_issues': system_health.critical_issues
                        }
                    )
                    self.active_alerts[system_alert_key] = {
                        'alert_type': 'SYSTEM_CRITICAL',
                        'timestamp': datetime.now()
                    }

        except Exception as e:
            self.logger.error(f"Error processing health alerts: {e}")

    async def _update_health_state(self, system_health: SystemHealth):
        """Update health status in state manager"""
        try:
            health_data = {
                'overall_status': system_health.overall_status.value,
                'healthy_components': system_health.healthy_components,
                'warning_components': system_health.warning_components,
                'critical_components': system_health.critical_components,
                'total_components': system_health.total_components,
                'uptime_seconds': system_health.uptime_seconds,
                'last_update': system_health.last_update.isoformat(),
                'critical_issues': system_health.critical_issues
            }

            await self.state_manager.update_component_state("system_health", health_data)

        except Exception as e:
            self.logger.error(f"Error updating health state: {e}")

    def get_monitor_metrics(self) -> Dict[str, Any]:
        """Get monitoring system metrics"""
        return {
            'is_running': self.is_running,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'registered_components': len(self.registered_components),
            'component_checkers': len(self.component_checkers),
            'active_alerts': len(self.active_alerts),
            'alert_history_size': len(self.alert_history),
            'check_interval_seconds': self.check_interval
        }