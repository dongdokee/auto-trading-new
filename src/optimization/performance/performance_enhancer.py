"""
Comprehensive performance enhancement system.

This module provides automatic optimization for function execution, batch processing,
resource management, and performance monitoring with auto-tuning capabilities.
"""

import asyncio
import time
import gc
import logging
import statistics
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from abc import ABC, abstractmethod

from .models import PerformanceMetrics, OptimizationError
from .resource_manager import ResourceManager
from .parallel_processor import ParallelProcessor

logger = logging.getLogger(__name__)


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""

    @abstractmethod
    async def optimize(self, func: Callable, *args, **kwargs) -> Any:
        """Apply optimization strategy to function execution."""
        pass


class PerformanceEnhancer:
    """
    Comprehensive performance enhancement system.

    Provides automatic optimization for function execution, batch processing,
    resource management, and performance monitoring.
    """

    def __init__(
        self,
        enable_parallel_processing: bool = True,
        enable_memory_optimization: bool = True,
        enable_cpu_optimization: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize performance enhancer.

        Args:
            enable_parallel_processing: Enable parallel processing
            enable_memory_optimization: Enable memory optimization
            enable_cpu_optimization: Enable CPU optimization
            max_workers: Maximum worker threads/processes
        """
        self.enable_parallel_processing = enable_parallel_processing
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_cpu_optimization = enable_cpu_optimization

        # Initialize components
        self.resource_manager = ResourceManager()
        self.parallel_processor = ParallelProcessor(max_workers=max_workers)
        self.optimization_strategies: Dict[str, OptimizationStrategy] = {}

        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.is_running = False
        self._monitoring_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start performance enhancement systems."""
        if self.is_running:
            return

        self.is_running = True

        # Start parallel processor
        if self.enable_parallel_processing:
            await self.parallel_processor.start()

        # Start resource monitoring
        self._monitoring_task = asyncio.create_task(
            self._monitor_performance()
        )

        logger.info("Performance enhancement systems started")

    async def stop(self) -> None:
        """Stop performance enhancement systems."""
        if not self.is_running:
            return

        self.is_running = False

        # Stop monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop parallel processor
        if self.enable_parallel_processing:
            await self.parallel_processor.stop()

        logger.info("Performance enhancement systems stopped")

    async def optimize_execution(
        self,
        func: Callable,
        *args,
        strategy: str = 'default',
        **kwargs
    ) -> Any:
        """
        Optimize function execution with automatic performance enhancement.

        Args:
            func: Function to execute
            *args: Function arguments
            strategy: Optimization strategy to use
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        if not self.is_running:
            raise OptimizationError("Performance enhancer not started")

        # Check resource availability
        if not await self.resource_manager.acquire_operation_slot():
            logger.warning("Resource limit reached, executing without optimization")
            return func(*args, **kwargs)

        try:
            start_time = time.time()

            # Apply custom strategy if available
            if strategy in self.optimization_strategies:
                result = await self.optimization_strategies[strategy].optimize(
                    func, *args, **kwargs
                )
            else:
                # Default optimization
                result = await self._apply_default_optimization(func, *args, **kwargs)

            execution_time = time.time() - start_time

            # Memory cleanup if enabled
            if self.enable_memory_optimization:
                gc.collect()

            # Record metrics
            self._record_execution_metrics(execution_time)

            return result

        finally:
            await self.resource_manager.release_operation_slot()

    async def optimize_batch_processing(
        self,
        func: Callable,
        items: List[Any],
        batch_size: int = None
    ) -> List[Any]:
        """
        Optimize batch processing with parallel execution.

        Args:
            func: Function to apply to each item
            items: List of items to process
            batch_size: Optional batch size for processing

        Returns:
            List of results
        """
        if not self.is_running:
            raise OptimizationError("Performance enhancer not started")

        if not self.enable_parallel_processing:
            # Sequential processing
            return [func(item) for item in items]

        # Determine optimal batch size
        if batch_size is None:
            batch_size = min(len(items), self.parallel_processor.max_workers * 2)

        return await self.parallel_processor.batch_process(func, items, batch_size)

    async def _apply_default_optimization(self, func: Callable, *args, **kwargs) -> Any:
        """Apply default optimization strategy."""
        # For CPU-intensive tasks, use parallel processing if available
        if self.enable_parallel_processing and self._is_cpu_intensive(func):
            try:
                # Execute in thread pool for CPU-bound tasks
                if self.parallel_processor.executor:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        self.parallel_processor.executor, func, *args
                    )
            except Exception as e:
                logger.warning(f"Parallel execution failed, falling back to sequential: {e}")

        # Default sequential execution
        return func(*args, **kwargs)

    def _is_cpu_intensive(self, func: Callable) -> bool:
        """Heuristic to determine if function is CPU-intensive."""
        # Simple heuristic - could be enhanced with profiling
        func_name = getattr(func, '__name__', '')
        cpu_keywords = ['compute', 'calculate', 'process', 'heavy', 'intensive']
        return any(keyword in func_name.lower() for keyword in cpu_keywords)

    async def _monitor_performance(self) -> None:
        """Monitor system performance and collect metrics."""
        while self.is_running:
            try:
                metrics = self.resource_manager.get_current_metrics()
                self.metrics_history.append(metrics)

                # Keep only recent metrics (last 1000 entries)
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]

                await asyncio.sleep(5.0)  # Monitor every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(5.0)

    def _record_execution_metrics(self, execution_time: float) -> None:
        """Record execution metrics."""
        if self.metrics_history:
            # Update latest metrics with execution info
            latest_metrics = self.metrics_history[-1]
            latest_metrics.latency_ms = execution_time * 1000

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on performance history."""
        if not self.metrics_history:
            return {
                'cpu_optimization': 'No data available',
                'memory_optimization': 'No data available',
                'parallel_processing': 'No data available',
                'resource_allocation': 'No data available'
            }

        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history

        avg_cpu = statistics.mean(m.cpu_usage_percent for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_usage_mb for m in recent_metrics)
        avg_latency = statistics.mean(m.latency_ms for m in recent_metrics if m.latency_ms > 0)

        recommendations = {}

        # CPU optimization
        if avg_cpu > 80:
            recommendations['cpu_optimization'] = 'Consider reducing CPU load or scaling horizontally'
        elif avg_cpu < 30:
            recommendations['cpu_optimization'] = 'CPU underutilized, consider increasing concurrency'
        else:
            recommendations['cpu_optimization'] = 'CPU usage optimal'

        # Memory optimization
        if avg_memory > 800:
            recommendations['memory_optimization'] = 'High memory usage, consider memory cleanup'
        else:
            recommendations['memory_optimization'] = 'Memory usage acceptable'

        # Parallel processing
        if avg_cpu < 50 and avg_latency > 50:
            recommendations['parallel_processing'] = 'Consider increasing parallel processing'
        else:
            recommendations['parallel_processing'] = 'Parallel processing configuration optimal'

        # Resource allocation
        max_concurrent = max(m.concurrent_operations for m in recent_metrics)
        if max_concurrent >= self.resource_manager.max_concurrent_operations * 0.9:
            recommendations['resource_allocation'] = 'Consider increasing concurrent operation limits'
        else:
            recommendations['resource_allocation'] = 'Resource allocation optimal'

        return recommendations

    async def auto_tune_performance(self) -> Dict[str, Any]:
        """Auto-tune performance parameters based on collected metrics."""
        if len(self.metrics_history) < 10:
            return {
                'tuning_applied': False,
                'reason': 'Insufficient data for auto-tuning'
            }

        recommendations = self.get_optimization_recommendations()
        tuning_result = {
            'tuning_applied': False,
            'performance_improvement': 0.0,
            'new_parameters': {}
        }

        # Auto-tune based on recommendations
        if 'increasing concurrent operation limits' in recommendations.get('resource_allocation', ''):
            old_limit = self.resource_manager.max_concurrent_operations
            new_limit = min(old_limit + 10, 100)  # Increase by 10, max 100
            self.resource_manager.max_concurrent_operations = new_limit

            tuning_result['tuning_applied'] = True
            tuning_result['new_parameters']['max_concurrent_operations'] = new_limit
            tuning_result['performance_improvement'] = 0.1  # Estimated 10% improvement

        if 'increasing parallel processing' in recommendations.get('parallel_processing', ''):
            # Increase worker count if not at maximum
            current_workers = self.parallel_processor.max_workers
            if current_workers < 8:
                await self.parallel_processor.stop()
                self.parallel_processor.max_workers = current_workers + 2
                await self.parallel_processor.start()

                tuning_result['tuning_applied'] = True
                tuning_result['new_parameters']['max_workers'] = current_workers + 2
                tuning_result['performance_improvement'] += 0.15  # Additional 15% improvement

        return tuning_result

    def register_optimization_strategy(self, name: str, strategy: OptimizationStrategy) -> None:
        """Register a custom optimization strategy."""
        self.optimization_strategies[name] = strategy
        logger.info(f"Registered optimization strategy: {name}")

    def export_performance_report(self) -> Dict[str, Any]:
        """Export comprehensive performance report."""
        if not self.metrics_history:
            return {'error': 'No performance data available'}

        recent_metrics = self.metrics_history[-100:] if len(self.metrics_history) >= 100 else self.metrics_history

        # Calculate statistics
        cpu_values = [m.cpu_usage_percent for m in recent_metrics]
        memory_values = [m.memory_usage_mb for m in recent_metrics]
        latency_values = [m.latency_ms for m in recent_metrics if m.latency_ms > 0]

        report = {
            'summary': {
                'total_metrics_collected': len(self.metrics_history),
                'monitoring_duration_hours': (
                    (datetime.utcnow() - self.metrics_history[0].timestamp).total_seconds() / 3600
                    if self.metrics_history else 0
                ),
                'average_efficiency_score': statistics.mean(
                    m.calculate_efficiency_score() for m in recent_metrics
                )
            },
            'optimization_impact': {
                'parallel_processing_enabled': self.enable_parallel_processing,
                'memory_optimization_enabled': self.enable_memory_optimization,
                'cpu_optimization_enabled': self.enable_cpu_optimization,
                'custom_strategies_registered': len(self.optimization_strategies)
            },
            'resource_utilization': {
                'avg_cpu_usage': statistics.mean(cpu_values) if cpu_values else 0,
                'max_cpu_usage': max(cpu_values) if cpu_values else 0,
                'avg_memory_usage': statistics.mean(memory_values) if memory_values else 0,
                'max_memory_usage': max(memory_values) if memory_values else 0,
                'avg_latency': statistics.mean(latency_values) if latency_values else 0
            },
            'recommendations': self.get_optimization_recommendations(),
            'metrics_history': [m.to_dict() for m in recent_metrics[-20:]]  # Last 20 entries
        }

        return report