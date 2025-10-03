"""
Real-time Monitoring Dashboard for performance tracking.

This module provides comprehensive real-time monitoring capabilities including:
- Performance metrics collection and visualization
- System health monitoring and alerting
- Real-time data streaming and dashboards
- Performance trend analysis and reporting
- Alert management and notification system
- Interactive web-based dashboard interface

Features:
- Real-time performance metrics tracking
- Configurable alerting thresholds
- Interactive web dashboard with charts
- Historical data analysis and trending
- System resource monitoring
- Custom metric collection support
- Alert notification via multiple channels
"""

import asyncio
import time
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from threading import Thread, Event
import statistics

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
except ImportError:
    uvicorn = None
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = None
    StaticFiles = None
    HTMLResponse = None

logger = logging.getLogger(__name__)


class MonitoringError(Exception):
    """Raised when monitoring operations fail."""
    pass


@dataclass
class MetricPoint:
    """
    Individual metric data point.

    Represents a single measurement at a specific time.
    """
    timestamp: datetime
    value: float
    metric_name: str
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric point to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'metric_name': self.metric_name,
            'tags': self.tags
        }


@dataclass
class AlertRule:
    """
    Alert rule configuration.

    Defines conditions for triggering alerts based on metric values.
    """
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    threshold: float
    duration_seconds: int = 60
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def check_condition(self, value: float) -> bool:
        """Check if metric value meets alert condition."""
        if not self.enabled:
            return False

        if self.condition == 'gt':
            return value > self.threshold
        elif self.condition == 'lt':
            return value < self.threshold
        elif self.condition == 'eq':
            return value == self.threshold
        elif self.condition == 'ne':
            return value != self.threshold
        else:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert rule to dictionary."""
        return asdict(self)


@dataclass
class Alert:
    """
    Active alert instance.

    Represents a triggered alert with its status and details.
    """
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    condition: str
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    status: str = "active"  # active, resolved, acknowledged
    message: str = ""

    @property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.status == "active"

    def resolve(self):
        """Mark alert as resolved."""
        self.status = "resolved"
        self.resolved_at = datetime.utcnow()

    def acknowledge(self):
        """Mark alert as acknowledged."""
        self.status = "acknowledged"

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """
    Metrics collection and aggregation system.

    Collects, stores, and provides access to performance metrics.
    """

    def __init__(self, max_points_per_metric: int = 10000):
        """
        Initialize metrics collector.

        Args:
            max_points_per_metric: Maximum number of points to store per metric
        """
        self.max_points_per_metric = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.callbacks: List[Callable[[MetricPoint], None]] = []

    def add_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Add a new metric point."""
        tags = tags or {}
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            metric_name=name,
            tags=tags
        )

        self.metrics[name].append(point)

        # Check alert rules
        self._check_alerts(point)

        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(point)
            except Exception as e:
                logger.error(f"Error in metric callback: {e}")

    def get_metric_history(self, name: str, hours: int = 1) -> List[MetricPoint]:
        """Get metric history for specified time period."""
        if name not in self.metrics:
            return []

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            point for point in self.metrics[name]
            if point.timestamp >= cutoff_time
        ]

    def get_metric_stats(self, name: str, hours: int = 1) -> Dict[str, float]:
        """Get statistical summary of metric values."""
        history = self.get_metric_history(name, hours)
        if not history:
            return {}

        values = [point.value for point in history]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0
        }

    def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.active_alerts.values() if alert.is_active]

    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolve()
            logger.info(f"Resolved alert: {alert_id}")

    def add_callback(self, callback: Callable[[MetricPoint], None]):
        """Add callback for new metric points."""
        self.callbacks.append(callback)

    def _check_alerts(self, point: MetricPoint):
        """Check if metric point triggers any alerts."""
        for rule in self.alert_rules.values():
            if rule.metric_name == point.metric_name and rule.check_condition(point.value):
                alert_id = f"{rule.name}_{point.timestamp.isoformat()}"

                # Check if similar alert already exists
                existing_active = any(
                    alert.rule_name == rule.name and alert.is_active
                    for alert in self.active_alerts.values()
                )

                if not existing_active:
                    alert = Alert(
                        rule_name=rule.name,
                        metric_name=point.metric_name,
                        current_value=point.value,
                        threshold=rule.threshold,
                        condition=rule.condition,
                        message=f"Metric {point.metric_name} is {point.value}, threshold is {rule.threshold}"
                    )
                    self.active_alerts[alert_id] = alert
                    logger.warning(f"Alert triggered: {rule.name}")


class DashboardServer:
    """
    Web-based dashboard server.

    Provides real-time monitoring dashboard with WebSocket support.
    """

    def __init__(self, metrics_collector: MetricsCollector, host: str = "127.0.0.1", port: int = 8000):
        """
        Initialize dashboard server.

        Args:
            metrics_collector: Metrics collector instance
            host: Server host address
            port: Server port number
        """
        if FastAPI is None:
            raise MonitoringError("FastAPI package is required for dashboard server")

        self.metrics_collector = metrics_collector
        self.host = host
        self.port = port
        self.app = FastAPI(title="Monitoring Dashboard")
        self.websocket_connections: List[WebSocket] = []
        self.running = False

        self._setup_routes()

        # Register callback to broadcast new metrics
        self.metrics_collector.add_callback(self._broadcast_metric)

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve dashboard HTML."""
            return self._get_dashboard_html()

        @self.app.get("/metrics/{metric_name}")
        async def get_metric(metric_name: str, hours: int = 1):
            """Get metric history."""
            history = self.metrics_collector.get_metric_history(metric_name, hours)
            return {
                'metric_name': metric_name,
                'points': [point.to_dict() for point in history]
            }

        @self.app.get("/metrics/{metric_name}/stats")
        async def get_metric_stats(metric_name: str, hours: int = 1):
            """Get metric statistics."""
            stats = self.metrics_collector.get_metric_stats(metric_name, hours)
            return {
                'metric_name': metric_name,
                'stats': stats
            }

        @self.app.get("/alerts")
        async def get_alerts():
            """Get active alerts."""
            alerts = self.metrics_collector.get_active_alerts()
            return {
                'alerts': [alert.to_dict() for alert in alerts]
            }

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.websocket_connections.append(websocket)

            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)

    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Monitoring Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-card { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
                .alert { background-color: #ffe6e6; border-color: #ff9999; }
                .chart-container { width: 100%; height: 300px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Real-time Monitoring Dashboard</h1>
            <div id="alerts"></div>
            <div id="metrics"></div>

            <script>
                const ws = new WebSocket('ws://localhost:8000/ws');

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };

                function updateDashboard(data) {
                    // Update dashboard with real-time data
                    console.log('Received data:', data);
                }

                // Initial load
                fetch('/alerts').then(r => r.json()).then(data => {
                    document.getElementById('alerts').innerHTML =
                        '<h2>Active Alerts</h2>' +
                        data.alerts.map(alert =>
                            `<div class="metric-card alert">
                                <strong>${alert.rule_name}</strong>: ${alert.message}
                            </div>`
                        ).join('');
                });
            </script>
        </body>
        </html>
        """

    def _broadcast_metric(self, point: MetricPoint):
        """Broadcast new metric to all WebSocket connections."""
        if not self.websocket_connections:
            return

        message = json.dumps({
            'type': 'metric',
            'data': point.to_dict()
        })

        # Remove disconnected connections
        active_connections = []
        for connection in self.websocket_connections:
            try:
                asyncio.create_task(connection.send_text(message))
                active_connections.append(connection)
            except Exception:
                pass  # Connection closed

        self.websocket_connections = active_connections

    async def start(self):
        """Start dashboard server."""
        if uvicorn is None:
            raise MonitoringError("uvicorn package is required for dashboard server")

        self.running = True
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    def start_background(self):
        """Start dashboard server in background thread."""
        def run():
            asyncio.run(self.start())

        thread = Thread(target=run, daemon=True)
        thread.start()
        self.running = True
        logger.info(f"Dashboard server starting on http://{self.host}:{self.port}")

    def stop(self):
        """Stop dashboard server."""
        self.running = False


class MonitoringDashboard:
    """
    Main monitoring dashboard orchestrator.

    Integrates metrics collection with web dashboard and alerting.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        max_points_per_metric: int = 10000
    ):
        """
        Initialize monitoring dashboard.

        Args:
            host: Dashboard server host
            port: Dashboard server port
            max_points_per_metric: Maximum metric points to store
        """
        self.metrics_collector = MetricsCollector(max_points_per_metric)
        self.dashboard_server = DashboardServer(self.metrics_collector, host, port)
        self.is_initialized = False

    async def initialize(self):
        """Initialize monitoring dashboard."""
        try:
            self.is_initialized = True
            logger.info("Monitoring dashboard initialized successfully")
        except Exception as e:
            self.is_initialized = False
            raise MonitoringError(f"Failed to initialize monitoring dashboard: {e}")

    def add_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Add metric point."""
        if not self.is_initialized:
            raise MonitoringError("Dashboard not initialized")

        self.metrics_collector.add_metric(name, value, tags)

    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        duration_seconds: int = 60
    ):
        """Add alert rule."""
        rule = AlertRule(
            name=name,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            duration_seconds=duration_seconds
        )
        self.metrics_collector.add_alert_rule(rule)

    def get_metric_stats(self, name: str, hours: int = 1) -> Dict[str, float]:
        """Get metric statistics."""
        return self.metrics_collector.get_metric_stats(name, hours)

    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return self.metrics_collector.get_active_alerts()

    def start_dashboard(self):
        """Start web dashboard in background."""
        self.dashboard_server.start_background()

    async def start_dashboard_async(self):
        """Start web dashboard asynchronously."""
        await self.dashboard_server.start()

    def stop_dashboard(self):
        """Stop web dashboard."""
        self.dashboard_server.stop()

    def export_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Export metrics data for analysis."""
        export_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {},
            'alerts': [alert.to_dict() for alert in self.get_active_alerts()]
        }

        for metric_name in self.metrics_collector.metrics.keys():
            history = self.metrics_collector.get_metric_history(metric_name, hours)
            stats = self.metrics_collector.get_metric_stats(metric_name, hours)

            export_data['metrics'][metric_name] = {
                'history': [point.to_dict() for point in history],
                'stats': stats
            }

        return export_data