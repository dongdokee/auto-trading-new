"""
Resource manager for monitoring and controlling system resources.

This module provides system resource monitoring and management capabilities
including CPU, memory, and concurrent operation limits with real-time monitoring.
"""

import asyncio
import logging
import threading
from typing import Optional, Callable

try:
    import psutil
except ImportError:
    psutil = None

from .models import PerformanceMetrics, OptimizationError

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Resource manager for monitoring and controlling system resources.

    Manages CPU, memory, and concurrent operation limits with real-time monitoring.
    """

    def __init__(
        self,
        max_cpu_percent: float = 80.0,
        max_memory_mb: float = 1000.0,
        max_concurrent_operations: int = 50
    ):
        """
        Initialize resource manager.

        Args:
            max_cpu_percent: Maximum CPU usage percentage
            max_memory_mb: Maximum memory usage in MB
            max_concurrent_operations: Maximum concurrent operations
        """
        if psutil is None:
            raise OptimizationError("psutil package is required for resource management")

        self.max_cpu_percent = max_cpu_percent
        self.max_memory_mb = max_memory_mb
        self.max_concurrent_operations = max_concurrent_operations
        self.current_operations = 0
        self._operation_lock = threading.Lock()
        self._process = psutil.Process()

    def can_allocate_resources(self) -> bool:
        """Check if resources can be allocated for new operations."""
        try:
            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage > self.max_cpu_percent:
                return False

            # Check memory usage
            memory_info = psutil.virtual_memory()
            memory_usage_mb = memory_info.used / (1024 * 1024)
            if memory_usage_mb > self.max_memory_mb:
                return False

            # Check concurrent operations
            if self.current_operations >= self.max_concurrent_operations:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking resource availability: {e}")
            return False

    async def acquire_operation_slot(self) -> bool:
        """
        Acquire a slot for operation execution.

        Returns:
            True if slot acquired, False otherwise
        """
        with self._operation_lock:
            if self.current_operations < self.max_concurrent_operations:
                if self.can_allocate_resources():
                    self.current_operations += 1
                    return True
        return False

    async def release_operation_slot(self) -> None:
        """Release an operation slot."""
        with self._operation_lock:
            if self.current_operations > 0:
                self.current_operations -= 1

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)

            # Memory usage
            memory_info = psutil.virtual_memory()
            system_memory_mb = memory_info.used / (1024 * 1024)

            # Process memory usage
            process_memory_mb = self._process.memory_info().rss / (1024 * 1024)

            return PerformanceMetrics(
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=system_memory_mb,
                concurrent_operations=self.current_operations
            )

        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return PerformanceMetrics()

    async def start_monitoring(
        self,
        interval_seconds: float = 1.0,
        callback: Optional[Callable[[PerformanceMetrics], None]] = None
    ) -> None:
        """
        Start monitoring system resources.

        Args:
            interval_seconds: Monitoring interval
            callback: Optional callback for metrics
        """
        while True:
            try:
                metrics = self.get_current_metrics()

                if callback:
                    callback(metrics)

                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(interval_seconds)