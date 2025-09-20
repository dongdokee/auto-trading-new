# src/integration/monitoring/metrics.py
"""
Metrics Collector

Comprehensive metrics collection and aggregation for system monitoring.
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    labels: Dict[str, str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class MetricSummary:
    """Summary statistics for a metric"""
    metric_name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    median_value: float
    p95_value: float
    p99_value: float
    last_value: float
    last_updated: datetime


class MetricsCollector:
    """
    Comprehensive metrics collection and aggregation system

    Features:
    - System resource metrics
    - Application performance metrics
    - Custom metric registration
    - Time-series data storage
    - Statistical aggregation
    """

    def __init__(self,
                 retention_minutes: int = 1440,  # 24 hours
                 collection_interval_seconds: int = 30):

        self.retention_minutes = retention_minutes
        self.collection_interval = collection_interval_seconds

        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.retention_minutes * 2))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}

        # Collection state
        self.is_running = False
        self.collection_task: Optional[asyncio.Task] = None

        # Performance tracking
        self.collection_count = 0
        self.last_collection = None
        self.collection_errors = 0

        # Logger
        self.logger = logging.getLogger("metrics_collector")

        # Register default metrics
        self._register_default_metrics()

    def _register_default_metrics(self):
        """Register default system metrics"""
        self.register_metric("system.cpu.percent", "CPU utilization percentage")
        self.register_metric("system.memory.percent", "Memory utilization percentage")
        self.register_metric("system.memory.available_gb", "Available memory in GB")
        self.register_metric("system.disk.percent", "Disk utilization percentage")
        self.register_metric("system.disk.free_gb", "Free disk space in GB")
        self.register_metric("system.network.bytes_sent", "Network bytes sent")
        self.register_metric("system.network.bytes_recv", "Network bytes received")

        # Application metrics
        self.register_metric("app.uptime_seconds", "Application uptime in seconds")
        self.register_metric("app.memory_usage_mb", "Application memory usage in MB")
        self.register_metric("app.thread_count", "Number of threads")

        # Trading-specific metrics
        self.register_metric("trading.orders.total", "Total orders processed")
        self.register_metric("trading.orders.success_rate", "Order success rate percentage")
        self.register_metric("trading.execution.latency_ms", "Order execution latency in milliseconds")
        self.register_metric("trading.portfolio.equity", "Portfolio equity value")
        self.register_metric("trading.portfolio.pnl", "Portfolio P&L")
        self.register_metric("trading.risk.var_utilization", "VaR utilization percentage")

    async def start(self):
        """Start metrics collection"""
        if self.is_running:
            self.logger.warning("Metrics collector is already running")
            return

        self.is_running = True

        # Start collection task
        self.collection_task = asyncio.create_task(self._collection_loop())

        self.logger.info("Metrics collector started")

    async def stop(self):
        """Stop metrics collection"""
        if not self.is_running:
            self.logger.warning("Metrics collector is not running")
            return

        self.is_running = False

        # Stop collection task
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Metrics collector stopped")

    def register_metric(self, metric_name: str, description: str, unit: str = ""):
        """Register a new metric"""
        self.metric_metadata[metric_name] = {
            'description': description,
            'unit': unit,
            'registered_at': datetime.now()
        }

        # Initialize storage if not exists
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.retention_minutes * 2)

        self.logger.debug(f"Registered metric: {metric_name}")

    def record_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        try:
            # Ensure metric is registered
            if metric_name not in self.metric_metadata:
                self.register_metric(metric_name, f"Auto-registered metric: {metric_name}")

            # Create metric point
            point = MetricPoint(
                timestamp=datetime.now(),
                metric_name=metric_name,
                value=value,
                labels=labels or {}
            )

            # Store metric
            self.metrics[metric_name].append(point)

        except Exception as e:
            self.logger.error(f"Error recording metric {metric_name}: {e}")

    async def collect_metrics(self):
        """Collect all registered metrics"""
        try:
            start_time = time.time()

            # Collect system metrics
            await self._collect_system_metrics()

            # Collect application metrics
            await self._collect_application_metrics()

            # Update collection statistics
            self.collection_count += 1
            self.last_collection = datetime.now()

            collection_time = (time.time() - start_time) * 1000
            self.record_metric("metrics.collection.duration_ms", collection_time)

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            self.collection_errors += 1

    async def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.record_metric("system.cpu.percent", cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system.memory.percent", memory.percent)
            self.record_metric("system.memory.available_gb", memory.available / (1024**3))

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric("system.disk.percent", disk.percent)
            self.record_metric("system.disk.free_gb", disk.free / (1024**3))

            # Network metrics
            network = psutil.net_io_counters()
            if network:
                self.record_metric("system.network.bytes_sent", network.bytes_sent)
                self.record_metric("system.network.bytes_recv", network.bytes_recv)

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    async def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # Process metrics
            process = psutil.Process()

            # Memory usage
            memory_info = process.memory_info()
            self.record_metric("app.memory_usage_mb", memory_info.rss / (1024**2))

            # Thread count
            self.record_metric("app.thread_count", process.num_threads())

            # Uptime (simplified - would need to track from app start)
            self.record_metric("app.uptime_seconds", time.time())

        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")

    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.is_running:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(10)

    def get_metric_summary(self, metric_name: str, window_minutes: Optional[int] = None) -> Optional[MetricSummary]:
        """Get summary statistics for a metric"""
        if metric_name not in self.metrics:
            return None

        points = list(self.metrics[metric_name])

        if not points:
            return None

        # Filter by time window if specified
        if window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            points = [p for p in points if p.timestamp >= cutoff_time]

        if not points:
            return None

        values = [p.value for p in points]

        try:
            return MetricSummary(
                metric_name=metric_name,
                count=len(values),
                min_value=min(values),
                max_value=max(values),
                avg_value=statistics.mean(values),
                median_value=statistics.median(values),
                p95_value=self._percentile(values, 95),
                p99_value=self._percentile(values, 99),
                last_value=values[-1],
                last_updated=points[-1].timestamp
            )
        except Exception as e:
            self.logger.error(f"Error calculating metric summary for {metric_name}: {e}")
            return None

    def get_metric_values(self,
                         metric_name: str,
                         window_minutes: Optional[int] = None) -> List[MetricPoint]:
        """Get raw metric values"""
        if metric_name not in self.metrics:
            return []

        points = list(self.metrics[metric_name])

        if window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            points = [p for p in points if p.timestamp >= cutoff_time]

        return points

    def get_all_metrics_summary(self, window_minutes: Optional[int] = None) -> Dict[str, MetricSummary]:
        """Get summary for all metrics"""
        summaries = {}

        for metric_name in self.metrics.keys():
            summary = self.get_metric_summary(metric_name, window_minutes)
            if summary:
                summaries[metric_name] = summary

        return summaries

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def get_metrics_health(self) -> Dict[str, Any]:
        """Get metrics collection health status"""
        last_collection_age = None
        if self.last_collection:
            last_collection_age = (datetime.now() - self.last_collection).total_seconds()

        error_rate = (self.collection_errors / self.collection_count * 100) if self.collection_count > 0 else 0

        return {
            'is_running': self.is_running,
            'collection_count': self.collection_count,
            'collection_errors': self.collection_errors,
            'error_rate_pct': error_rate,
            'last_collection_age_seconds': last_collection_age,
            'registered_metrics': len(self.metric_metadata),
            'metrics_with_data': len([m for m in self.metrics.values() if len(m) > 0]),
            'collection_interval_seconds': self.collection_interval,
            'retention_minutes': self.retention_minutes
        }

    def get_registered_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get list of registered metrics with metadata"""
        return self.metric_metadata.copy()

    def clear_metric_data(self, metric_name: str):
        """Clear data for a specific metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name].clear()
            self.logger.info(f"Cleared data for metric: {metric_name}")

    def clear_all_metrics(self):
        """Clear all metric data"""
        for metric_queue in self.metrics.values():
            metric_queue.clear()
        self.logger.info("Cleared all metric data")

    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format"""
        if format_type.lower() == "json":
            return self._export_json()
        elif format_type.lower() == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _export_json(self) -> str:
        """Export metrics as JSON"""
        import json

        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metadata': self.metric_metadata,
            'summaries': {}
        }

        for metric_name in self.metrics.keys():
            summary = self.get_metric_summary(metric_name)
            if summary:
                export_data['summaries'][metric_name] = asdict(summary)

        return json.dumps(export_data, default=str, indent=2)

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        for metric_name in self.metrics.keys():
            summary = self.get_metric_summary(metric_name)
            if summary:
                # Convert metric name to Prometheus format
                prom_name = metric_name.replace('.', '_')

                # Add help and type
                metadata = self.metric_metadata.get(metric_name, {})
                description = metadata.get('description', '')
                lines.append(f"# HELP {prom_name} {description}")
                lines.append(f"# TYPE {prom_name} gauge")

                # Add metric value
                lines.append(f"{prom_name} {summary.last_value}")

        return '\n'.join(lines)