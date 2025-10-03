"""
Comprehensive tests for the Real-time Monitoring Dashboard.

Tests cover:
- Metrics collection and storage
- Alert rule configuration and triggering
- Web dashboard functionality
- Real-time data streaming
- Performance monitoring
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from src.optimization.monitoring_dashboard import (
    MetricPoint,
    AlertRule,
    Alert,
    MetricsCollector,
    DashboardServer,
    MonitoringDashboard,
    MonitoringError
)


class TestMetricPoint:
    """Test MetricPoint data structure."""

    def test_should_initialize_with_required_fields(self):
        """Test that MetricPoint initializes with required fields."""
        timestamp = datetime.utcnow()
        point = MetricPoint(
            timestamp=timestamp,
            value=42.5,
            metric_name="cpu_usage"
        )

        assert point.timestamp == timestamp
        assert point.value == 42.5
        assert point.metric_name == "cpu_usage"
        assert point.tags == {}

    def test_should_initialize_with_tags(self):
        """Test that MetricPoint can initialize with tags."""
        timestamp = datetime.utcnow()
        tags = {"host": "server1", "region": "us-east"}
        point = MetricPoint(
            timestamp=timestamp,
            value=42.5,
            metric_name="cpu_usage",
            tags=tags
        )

        assert point.tags == tags

    def test_should_convert_to_dictionary(self):
        """Test that MetricPoint can convert to dictionary."""
        timestamp = datetime.utcnow()
        tags = {"host": "server1"}
        point = MetricPoint(
            timestamp=timestamp,
            value=42.5,
            metric_name="cpu_usage",
            tags=tags
        )

        result = point.to_dict()

        assert isinstance(result, dict)
        assert result['timestamp'] == timestamp.isoformat()
        assert result['value'] == 42.5
        assert result['metric_name'] == "cpu_usage"
        assert result['tags'] == tags


class TestAlertRule:
    """Test AlertRule configuration and logic."""

    def test_should_initialize_with_required_fields(self):
        """Test that AlertRule initializes with required fields."""
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0
        )

        assert rule.name == "high_cpu"
        assert rule.metric_name == "cpu_usage"
        assert rule.condition == "gt"
        assert rule.threshold == 80.0
        assert rule.duration_seconds == 60
        assert rule.enabled is True
        assert rule.notification_channels == []

    def test_should_check_greater_than_condition(self):
        """Test that AlertRule checks greater than condition correctly."""
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0
        )

        assert rule.check_condition(85.0) is True
        assert rule.check_condition(75.0) is False
        assert rule.check_condition(80.0) is False

    def test_should_check_less_than_condition(self):
        """Test that AlertRule checks less than condition correctly."""
        rule = AlertRule(
            name="low_memory",
            metric_name="memory_free",
            condition="lt",
            threshold=20.0
        )

        assert rule.check_condition(15.0) is True
        assert rule.check_condition(25.0) is False
        assert rule.check_condition(20.0) is False

    def test_should_check_equal_condition(self):
        """Test that AlertRule checks equal condition correctly."""
        rule = AlertRule(
            name="exact_value",
            metric_name="queue_size",
            condition="eq",
            threshold=0.0
        )

        assert rule.check_condition(0.0) is True
        assert rule.check_condition(1.0) is False

    def test_should_ignore_disabled_rules(self):
        """Test that disabled rules don't trigger."""
        rule = AlertRule(
            name="disabled_rule",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0,
            enabled=False
        )

        assert rule.check_condition(85.0) is False

    def test_should_convert_to_dictionary(self):
        """Test that AlertRule can convert to dictionary."""
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0
        )

        result = rule.to_dict()

        assert isinstance(result, dict)
        assert result['name'] == "high_cpu"
        assert result['metric_name'] == "cpu_usage"
        assert result['condition'] == "gt"
        assert result['threshold'] == 80.0


class TestAlert:
    """Test Alert instances and management."""

    def test_should_initialize_with_required_fields(self):
        """Test that Alert initializes with required fields."""
        alert = Alert(
            rule_name="high_cpu",
            metric_name="cpu_usage",
            current_value=85.0,
            threshold=80.0,
            condition="gt"
        )

        assert alert.rule_name == "high_cpu"
        assert alert.metric_name == "cpu_usage"
        assert alert.current_value == 85.0
        assert alert.threshold == 80.0
        assert alert.condition == "gt"
        assert alert.status == "active"
        assert alert.resolved_at is None

    def test_should_check_active_status(self):
        """Test that Alert correctly reports active status."""
        alert = Alert(
            rule_name="high_cpu",
            metric_name="cpu_usage",
            current_value=85.0,
            threshold=80.0,
            condition="gt"
        )

        assert alert.is_active is True

        alert.resolve()
        assert alert.is_active is False

    def test_should_resolve_alert(self):
        """Test that Alert can be resolved."""
        alert = Alert(
            rule_name="high_cpu",
            metric_name="cpu_usage",
            current_value=85.0,
            threshold=80.0,
            condition="gt"
        )

        assert alert.status == "active"
        assert alert.resolved_at is None

        alert.resolve()

        assert alert.status == "resolved"
        assert alert.resolved_at is not None

    def test_should_acknowledge_alert(self):
        """Test that Alert can be acknowledged."""
        alert = Alert(
            rule_name="high_cpu",
            metric_name="cpu_usage",
            current_value=85.0,
            threshold=80.0,
            condition="gt"
        )

        alert.acknowledge()

        assert alert.status == "acknowledged"

    def test_should_convert_to_dictionary(self):
        """Test that Alert can convert to dictionary."""
        alert = Alert(
            rule_name="high_cpu",
            metric_name="cpu_usage",
            current_value=85.0,
            threshold=80.0,
            condition="gt"
        )

        result = alert.to_dict()

        assert isinstance(result, dict)
        assert result['rule_name'] == "high_cpu"
        assert result['current_value'] == 85.0
        assert result['status'] == "active"


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for testing."""
        return MetricsCollector(max_points_per_metric=100)

    def test_should_initialize_with_configuration(self, metrics_collector):
        """Test that MetricsCollector initializes correctly."""
        assert metrics_collector.max_points_per_metric == 100
        assert len(metrics_collector.metrics) == 0
        assert len(metrics_collector.alert_rules) == 0
        assert len(metrics_collector.active_alerts) == 0

    def test_should_add_metric_points(self, metrics_collector):
        """Test that MetricsCollector can add metric points."""
        metrics_collector.add_metric("cpu_usage", 75.0, {"host": "server1"})

        assert "cpu_usage" in metrics_collector.metrics
        assert len(metrics_collector.metrics["cpu_usage"]) == 1

        point = metrics_collector.metrics["cpu_usage"][0]
        assert point.value == 75.0
        assert point.metric_name == "cpu_usage"
        assert point.tags == {"host": "server1"}

    def test_should_limit_metric_history(self, metrics_collector):
        """Test that MetricsCollector limits metric history."""
        # Add more points than the limit
        for i in range(150):
            metrics_collector.add_metric("test_metric", float(i))

        assert len(metrics_collector.metrics["test_metric"]) == 100

    def test_should_get_metric_history(self, metrics_collector):
        """Test that MetricsCollector can get metric history."""
        # Add some points
        for i in range(10):
            metrics_collector.add_metric("cpu_usage", float(i))

        history = metrics_collector.get_metric_history("cpu_usage", hours=1)

        assert len(history) == 10
        assert all(isinstance(point, MetricPoint) for point in history)

    def test_should_get_metric_statistics(self, metrics_collector):
        """Test that MetricsCollector can calculate statistics."""
        # Add some test data
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            metrics_collector.add_metric("test_metric", value)

        stats = metrics_collector.get_metric_stats("test_metric", hours=1)

        assert stats['count'] == 5
        assert stats['min'] == 10.0
        assert stats['max'] == 50.0
        assert stats['mean'] == 30.0
        assert stats['median'] == 30.0
        assert stats['stdev'] > 0

    def test_should_handle_empty_metrics(self, metrics_collector):
        """Test that MetricsCollector handles empty metrics gracefully."""
        history = metrics_collector.get_metric_history("nonexistent", hours=1)
        assert history == []

        stats = metrics_collector.get_metric_stats("nonexistent", hours=1)
        assert stats == {}

    def test_should_add_alert_rules(self, metrics_collector):
        """Test that MetricsCollector can add alert rules."""
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0
        )

        metrics_collector.add_alert_rule(rule)

        assert "high_cpu" in metrics_collector.alert_rules
        assert metrics_collector.alert_rules["high_cpu"] == rule

    def test_should_remove_alert_rules(self, metrics_collector):
        """Test that MetricsCollector can remove alert rules."""
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0
        )

        metrics_collector.add_alert_rule(rule)
        metrics_collector.remove_alert_rule("high_cpu")

        assert "high_cpu" not in metrics_collector.alert_rules

    def test_should_trigger_alerts(self, metrics_collector):
        """Test that MetricsCollector triggers alerts correctly."""
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0
        )

        metrics_collector.add_alert_rule(rule)
        metrics_collector.add_metric("cpu_usage", 85.0)

        active_alerts = metrics_collector.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].rule_name == "high_cpu"

    def test_should_not_duplicate_alerts(self, metrics_collector):
        """Test that MetricsCollector doesn't create duplicate alerts."""
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0
        )

        metrics_collector.add_alert_rule(rule)

        # Add multiple metrics that trigger the same alert
        metrics_collector.add_metric("cpu_usage", 85.0)
        metrics_collector.add_metric("cpu_usage", 90.0)

        active_alerts = metrics_collector.get_active_alerts()
        assert len(active_alerts) == 1

    def test_should_resolve_alerts(self, metrics_collector):
        """Test that MetricsCollector can resolve alerts."""
        rule = AlertRule(
            name="high_cpu",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0
        )

        metrics_collector.add_alert_rule(rule)
        metrics_collector.add_metric("cpu_usage", 85.0)

        active_alerts = metrics_collector.get_active_alerts()
        assert len(active_alerts) == 1

        # Resolve the alert
        alert_id = list(metrics_collector.active_alerts.keys())[0]
        metrics_collector.resolve_alert(alert_id)

        active_alerts = metrics_collector.get_active_alerts()
        assert len(active_alerts) == 0

    def test_should_support_callbacks(self, metrics_collector):
        """Test that MetricsCollector supports metric callbacks."""
        callback_called = False
        received_point = None

        def test_callback(point):
            nonlocal callback_called, received_point
            callback_called = True
            received_point = point

        metrics_collector.add_callback(test_callback)
        metrics_collector.add_metric("test_metric", 42.0)

        assert callback_called is True
        assert received_point.value == 42.0


class TestDashboardServer:
    """Test DashboardServer web interface."""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for testing."""
        return MetricsCollector()

    @pytest.fixture
    def dashboard_server(self, metrics_collector):
        """Create dashboard server for testing."""
        with patch('src.optimization.monitoring_dashboard.FastAPI'):
            return DashboardServer(metrics_collector)

    def test_should_initialize_with_configuration(self, dashboard_server):
        """Test that DashboardServer initializes correctly."""
        assert dashboard_server.host == "127.0.0.1"
        assert dashboard_server.port == 8000
        assert dashboard_server.running is False
        assert len(dashboard_server.websocket_connections) == 0

    def test_should_generate_dashboard_html(self, dashboard_server):
        """Test that DashboardServer generates HTML."""
        html = dashboard_server._get_dashboard_html()

        assert isinstance(html, str)
        assert "Monitoring Dashboard" in html
        assert "chart.js" in html
        assert "WebSocket" in html

    def test_should_handle_websocket_connections(self, dashboard_server):
        """Test that DashboardServer handles WebSocket connections."""
        mock_websocket = Mock()
        dashboard_server.websocket_connections.append(mock_websocket)

        assert len(dashboard_server.websocket_connections) == 1

    def test_should_broadcast_metrics(self, dashboard_server):
        """Test that DashboardServer broadcasts metrics."""
        mock_websocket = Mock()
        dashboard_server.websocket_connections.append(mock_websocket)

        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=42.0,
            metric_name="test_metric"
        )

        # This should not raise an error
        dashboard_server._broadcast_metric(point)

    def test_should_start_in_background(self, dashboard_server):
        """Test that DashboardServer can start in background."""
        with patch('src.optimization.monitoring_dashboard.uvicorn') as mock_uvicorn, \
             patch('src.optimization.monitoring_dashboard.Thread') as mock_thread:

            dashboard_server.start_background()

            assert dashboard_server.running is True
            mock_thread.assert_called_once()


class TestMonitoringDashboard:
    """Test MonitoringDashboard integration."""

    @pytest.fixture
    def monitoring_dashboard(self):
        """Create monitoring dashboard for testing."""
        with patch('src.optimization.monitoring_dashboard.FastAPI'):
            return MonitoringDashboard()

    @pytest.mark.asyncio
    async def test_should_initialize_successfully(self, monitoring_dashboard):
        """Test that MonitoringDashboard initializes successfully."""
        await monitoring_dashboard.initialize()

        assert monitoring_dashboard.is_initialized is True

    def test_should_require_initialization(self, monitoring_dashboard):
        """Test that MonitoringDashboard requires initialization."""
        with pytest.raises(MonitoringError, match="Dashboard not initialized"):
            monitoring_dashboard.add_metric("test_metric", 42.0)

    @pytest.mark.asyncio
    async def test_should_add_metrics_after_initialization(self, monitoring_dashboard):
        """Test that MonitoringDashboard can add metrics after initialization."""
        await monitoring_dashboard.initialize()

        monitoring_dashboard.add_metric("cpu_usage", 75.0, {"host": "server1"})

        stats = monitoring_dashboard.get_metric_stats("cpu_usage")
        assert stats['count'] == 1

    @pytest.mark.asyncio
    async def test_should_add_alert_rules(self, monitoring_dashboard):
        """Test that MonitoringDashboard can add alert rules."""
        await monitoring_dashboard.initialize()

        monitoring_dashboard.add_alert_rule(
            name="high_cpu",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0
        )

        # Add metric that triggers alert
        monitoring_dashboard.add_metric("cpu_usage", 85.0)

        alerts = monitoring_dashboard.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].rule_name == "high_cpu"

    @pytest.mark.asyncio
    async def test_should_export_metrics_data(self, monitoring_dashboard):
        """Test that MonitoringDashboard can export metrics data."""
        await monitoring_dashboard.initialize()

        # Add some test data
        monitoring_dashboard.add_metric("cpu_usage", 75.0)
        monitoring_dashboard.add_metric("memory_usage", 60.0)

        export_data = monitoring_dashboard.export_metrics(hours=1)

        assert isinstance(export_data, dict)
        assert 'timestamp' in export_data
        assert 'metrics' in export_data
        assert 'alerts' in export_data
        assert 'cpu_usage' in export_data['metrics']
        assert 'memory_usage' in export_data['metrics']

    def test_should_start_dashboard_in_background(self, monitoring_dashboard):
        """Test that MonitoringDashboard can start dashboard in background."""
        with patch.object(monitoring_dashboard.dashboard_server, 'start_background') as mock_start:
            monitoring_dashboard.start_dashboard()
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_start_dashboard_async(self, monitoring_dashboard):
        """Test that MonitoringDashboard can start dashboard asynchronously."""
        with patch.object(monitoring_dashboard.dashboard_server, 'start', new_callable=AsyncMock) as mock_start:
            await monitoring_dashboard.start_dashboard_async()
            mock_start.assert_called_once()

    def test_should_stop_dashboard(self, monitoring_dashboard):
        """Test that MonitoringDashboard can stop dashboard."""
        with patch.object(monitoring_dashboard.dashboard_server, 'stop') as mock_stop:
            monitoring_dashboard.stop_dashboard()
            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_handle_initialization_errors(self):
        """Test that MonitoringDashboard handles initialization errors."""
        with patch('src.optimization.monitoring_dashboard.FastAPI'):
            dashboard = MonitoringDashboard()

            # Patch the logger to simulate an error during initialization
            with patch('src.optimization.monitoring_dashboard.logger.info', side_effect=Exception("Test error")):
                with pytest.raises(MonitoringError, match="Failed to initialize monitoring dashboard"):
                    await dashboard.initialize()

    @pytest.mark.asyncio
    async def test_should_integrate_all_components(self, monitoring_dashboard):
        """Test that MonitoringDashboard integrates all components correctly."""
        await monitoring_dashboard.initialize()

        # Add alert rule
        monitoring_dashboard.add_alert_rule(
            name="high_cpu",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0
        )

        # Add normal metric
        monitoring_dashboard.add_metric("cpu_usage", 70.0)
        assert len(monitoring_dashboard.get_active_alerts()) == 0

        # Add metric that triggers alert
        monitoring_dashboard.add_metric("cpu_usage", 85.0)
        alerts = monitoring_dashboard.get_active_alerts()
        assert len(alerts) == 1

        # Check statistics
        stats = monitoring_dashboard.get_metric_stats("cpu_usage")
        assert stats['count'] == 2
        assert stats['max'] == 85.0

        # Export data
        export_data = monitoring_dashboard.export_metrics()
        assert len(export_data['alerts']) == 1
        assert 'cpu_usage' in export_data['metrics']