# src/integration/monitoring/alerts.py
"""
Alert Manager

Comprehensive alerting system for trading system events and conditions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json

from src.integration.events.event_bus import EventBus
from src.integration.events.models import SystemEvent, RiskEvent, EventPriority


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    created_at: datetime
    updated_at: datetime
    component: Optional[str] = None
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AlertRule:
    """Alert rule definition"""

    def __init__(self,
                 rule_id: str,
                 name: str,
                 condition_func: Callable,
                 alert_type: str,
                 severity: AlertSeverity,
                 message_template: str,
                 cooldown_minutes: int = 15):

        self.rule_id = rule_id
        self.name = name
        self.condition_func = condition_func
        self.alert_type = alert_type
        self.severity = severity
        self.message_template = message_template
        self.cooldown_minutes = cooldown_minutes

        # State tracking
        self.last_triggered = None
        self.trigger_count = 0
        self.is_enabled = True


class AlertManager:
    """
    Comprehensive alert management system

    Features:
    - Alert creation and management
    - Rule-based alerting
    - Alert escalation
    - Notification delivery
    - Alert history and analytics
    """

    def __init__(self,
                 event_bus: Optional[EventBus] = None,
                 max_active_alerts: int = 1000,
                 alert_retention_days: int = 30):

        self.event_bus = event_bus
        self.max_active_alerts = max_active_alerts
        self.alert_retention_days = alert_retention_days

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}

        # Notification handlers
        self.notification_handlers: List[Callable] = []

        # Alert statistics
        self.alerts_created = 0
        self.alerts_resolved = 0
        self.alerts_acknowledged = 0

        # Background tasks
        self.is_running = False
        self.cleanup_task: Optional[asyncio.Task] = None

        # Logger
        self.logger = logging.getLogger("alert_manager")

        # Setup default alert rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default alert rules"""

        # High error rate rule
        def high_error_rate_condition(metrics: Dict[str, Any]) -> bool:
            component_metrics = metrics.get('component_metrics', {})
            for component, metric in component_metrics.items():
                error_rate = metric.get('error_rate_pct', 0)
                if error_rate > 10:  # 10% error rate
                    return True
            return False

        self.add_alert_rule(AlertRule(
            rule_id="high_error_rate",
            name="High Error Rate",
            condition_func=high_error_rate_condition,
            alert_type="HIGH_ERROR_RATE",
            severity=AlertSeverity.WARNING,
            message_template="High error rate detected: {error_rate}%",
            cooldown_minutes=10
        ))

        # Low system performance rule
        def low_performance_condition(metrics: Dict[str, Any]) -> bool:
            system_metrics = metrics.get('system_resources', {})
            cpu_percent = system_metrics.get('cpu_percent', 0)
            memory_percent = system_metrics.get('memory_percent', 0)
            return cpu_percent > 90 or memory_percent > 90

        self.add_alert_rule(AlertRule(
            rule_id="low_system_performance",
            name="Low System Performance",
            condition_func=low_performance_condition,
            alert_type="SYSTEM_PERFORMANCE",
            severity=AlertSeverity.CRITICAL,
            message_template="System performance degraded: CPU {cpu_percent}%, Memory {memory_percent}%",
            cooldown_minutes=5
        ))

        # Trading halted rule
        def trading_halted_condition(metrics: Dict[str, Any]) -> bool:
            trading_metrics = metrics.get('trading_status', {})
            return not trading_metrics.get('is_active', True)

        self.add_alert_rule(AlertRule(
            rule_id="trading_halted",
            name="Trading Halted",
            condition_func=trading_halted_condition,
            alert_type="TRADING_HALTED",
            severity=AlertSeverity.CRITICAL,
            message_template="Trading has been halted",
            cooldown_minutes=1
        ))

    async def start(self):
        """Start the alert manager"""
        if self.is_running:
            self.logger.warning("Alert manager is already running")
            return

        self.is_running = True

        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.info("Alert manager started")

    async def stop(self):
        """Stop the alert manager"""
        if not self.is_running:
            self.logger.warning("Alert manager is not running")
            return

        self.is_running = False

        # Stop cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Alert manager stopped")

    async def create_alert(self,
                          alert_type: str,
                          message: str,
                          severity: str = "WARNING",
                          component: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert"""

        try:
            # Convert severity string to enum
            severity_enum = AlertSeverity(severity.lower())

            # Generate alert ID
            alert_id = f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_alerts)}"

            # Create alert
            alert = Alert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity_enum,
                status=AlertStatus.ACTIVE,
                message=message,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                component=component,
                metadata=metadata or {}
            )

            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.alerts_created += 1

            # Limit active alerts
            if len(self.active_alerts) > self.max_active_alerts:
                oldest_alert_id = min(self.active_alerts.keys(),
                                    key=lambda x: self.active_alerts[x].created_at)
                await self.resolve_alert(oldest_alert_id, "Auto-resolved due to alert limit")

            # Send notifications
            await self._send_notifications(alert)

            # Publish system event if event bus is available
            if self.event_bus:
                await self._publish_alert_event(alert)

            self.logger.info(f"Created alert: {alert_type} - {message}")
            return alert

        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
            raise

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.active_alerts:
            self.logger.warning(f"Cannot acknowledge alert {alert_id}: Alert not found")
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        alert.updated_at = datetime.now()

        self.alerts_acknowledged += 1

        self.logger.info(f"Acknowledged alert {alert_id} by {acknowledged_by}")
        return True

    async def resolve_alert(self, alert_id: str, resolution_note: str = "") -> bool:
        """Resolve an alert"""
        if alert_id not in self.active_alerts:
            self.logger.warning(f"Cannot resolve alert {alert_id}: Alert not found")
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.updated_at = datetime.now()

        if resolution_note:
            alert.metadata['resolution_note'] = resolution_note

        # Remove from active alerts
        del self.active_alerts[alert_id]
        self.alerts_resolved += 1

        self.logger.info(f"Resolved alert {alert_id}: {resolution_note}")
        return True

    async def suppress_alert(self, alert_id: str, suppress_until: datetime) -> bool:
        """Suppress an alert until a specific time"""
        if alert_id not in self.active_alerts:
            self.logger.warning(f"Cannot suppress alert {alert_id}: Alert not found")
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED
        alert.updated_at = datetime.now()
        alert.metadata['suppressed_until'] = suppress_until.isoformat()

        self.logger.info(f"Suppressed alert {alert_id} until {suppress_until}")
        return True

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")

    def enable_alert_rule(self, rule_id: str):
        """Enable an alert rule"""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].is_enabled = True
            self.logger.info(f"Enabled alert rule: {rule_id}")

    def disable_alert_rule(self, rule_id: str):
        """Disable an alert rule"""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].is_enabled = False
            self.logger.info(f"Disabled alert rule: {rule_id}")

    async def evaluate_rules(self, metrics: Dict[str, Any]):
        """Evaluate all alert rules against current metrics"""
        for rule in self.alert_rules.values():
            if not rule.is_enabled:
                continue

            try:
                # Check cooldown
                if rule.last_triggered:
                    time_since_trigger = datetime.now() - rule.last_triggered
                    if time_since_trigger.total_seconds() < rule.cooldown_minutes * 60:
                        continue

                # Evaluate condition
                if rule.condition_func(metrics):
                    # Condition met, create alert
                    message = rule.message_template.format(**metrics.get('format_data', {}))

                    await self.create_alert(
                        alert_type=rule.alert_type,
                        message=message,
                        severity=rule.severity.value,
                        metadata={'rule_id': rule.rule_id, 'metrics': metrics}
                    )

                    rule.last_triggered = datetime.now()
                    rule.trigger_count += 1

            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.rule_id}: {e}")

    def add_notification_handler(self, handler: Callable):
        """Add a notification handler"""
        self.notification_handlers.append(handler)
        self.logger.info("Added notification handler")

    def remove_notification_handler(self, handler: Callable):
        """Remove a notification handler"""
        if handler in self.notification_handlers:
            self.notification_handlers.remove(handler)
            self.logger.info("Removed notification handler")

    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Error in notification handler: {e}")

    async def _publish_alert_event(self, alert: Alert):
        """Publish alert as system event"""
        try:
            priority = EventPriority.CRITICAL if alert.severity == AlertSeverity.CRITICAL else EventPriority.HIGH

            event = SystemEvent(
                source_component="alert_manager",
                system_action="ERROR" if alert.severity == AlertSeverity.CRITICAL else "HEALTH_CHECK",
                component=alert.component,
                status="ERROR" if alert.severity == AlertSeverity.CRITICAL else "WARNING",
                message=alert.message,
                priority=priority,
                error_details=alert.metadata
            )

            await self.event_bus.publish(event)

        except Exception as e:
            self.logger.error(f"Error publishing alert event: {e}")

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.is_running:
            try:
                await self._cleanup_old_alerts()
                await asyncio.sleep(3600)  # Run every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _cleanup_old_alerts(self):
        """Clean up old alerts from history"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.alert_retention_days)

            # Remove old alerts from history
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.created_at > cutoff_date
            ]

            # Un-suppress alerts that are past their suppression time
            current_time = datetime.now()
            for alert in list(self.active_alerts.values()):
                if alert.status == AlertStatus.SUPPRESSED:
                    suppress_until_str = alert.metadata.get('suppressed_until')
                    if suppress_until_str:
                        suppress_until = datetime.fromisoformat(suppress_until_str)
                        if current_time > suppress_until:
                            alert.status = AlertStatus.ACTIVE
                            alert.updated_at = current_time
                            self.logger.info(f"Un-suppressed alert {alert.alert_id}")

        except Exception as e:
            self.logger.error(f"Error cleaning up alerts: {e}")

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return sorted(alerts, key=lambda x: x.created_at, reverse=True)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        active_by_severity = {}
        for severity in AlertSeverity:
            count = sum(1 for alert in self.active_alerts.values() if alert.severity == severity)
            active_by_severity[severity.value] = count

        return {
            'total_created': self.alerts_created,
            'total_resolved': self.alerts_resolved,
            'total_acknowledged': self.alerts_acknowledged,
            'active_count': len(self.active_alerts),
            'active_by_severity': active_by_severity,
            'history_size': len(self.alert_history),
            'rules_count': len(self.alert_rules),
            'enabled_rules': sum(1 for rule in self.alert_rules.values() if rule.is_enabled)
        }

    def get_alert_history(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alert history with optional filters"""
        alerts = self.alert_history

        if start_time:
            alerts = [alert for alert in alerts if alert.created_at >= start_time]

        if end_time:
            alerts = [alert for alert in alerts if alert.created_at <= end_time]

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return sorted(alerts, key=lambda x: x.created_at, reverse=True)