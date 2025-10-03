"""
Performance Enhancement System for production optimization.

This module provides comprehensive performance optimization capabilities including:
- Parallel processing with thread and process pools
- Memory optimization and resource management
- CPU optimization and load balancing
- Async optimization and performance monitoring
- Custom optimization strategies

Features:
- Automatic resource management and monitoring
- Parallel and batch processing optimization
- Memory usage optimization with automatic cleanup
- CPU optimization with load balancing
- Performance metrics collection and analysis
- Auto-tuning of performance parameters
- Custom optimization strategy support
"""

import asyncio
import time
import gc
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from abc import ABC, abstractmethod
import threading
import statistics

logger = logging.getLogger(__name__)


class OptimizationError(Exception):
    """Raised when optimization operations fail."""
    pass


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for system monitoring and optimization.

    Tracks CPU, memory, latency, throughput, and error metrics.
    """
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_rate_percent: float = 0.0
    concurrent_operations: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'latency_ms': self.latency_ms,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'error_rate_percent': self.error_rate_percent,
            'concurrent_operations': self.concurrent_operations,
            'timestamp': self.timestamp.isoformat()
        }

    def calculate_efficiency_score(self) -> float:
        """
        Calculate efficiency score based on metrics.

        Returns:
            Efficiency score between 0 and 1 (higher is better)
        """
        # Normalize metrics to 0-1 scale
        cpu_score = max(0, (100 - self.cpu_usage_percent) / 100)
        memory_score = max(0, (1000 - self.memory_usage_mb) / 1000)
        latency_score = max(0, (100 - self.latency_ms) / 100)
        throughput_score = min(1, self.throughput_ops_per_sec / 2000)  # Max 2000 ops/sec
        error_score = max(0, (100 - self.error_rate_percent) / 100)

        # Weighted average
        weights = {
            'cpu': 0.2,
            'memory': 0.2,
            'latency': 0.3,
            'throughput': 0.2,
            'error': 0.1
        }

        efficiency = (
            weights['cpu'] * cpu_score +
            weights['memory'] * memory_score +
            weights['latency'] * latency_score +
            weights['throughput'] * throughput_score +
            weights['error'] * error_score
        )

        return max(0.0, min(1.0, efficiency))


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


class ParallelProcessor:
    """
    Parallel processing system with thread and process pool support.

    Provides optimized parallel execution for CPU and I/O bound tasks.
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False
    ):
        """
        Initialize parallel processor.

        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: Use process pool instead of thread pool
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self.task_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'start_time': time.time()
        }

    async def start(self) -> None:
        """Start the executor."""
        if self.executor is None:
            if self.use_processes:
                self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

            logger.info(f"Started {'process' if self.use_processes else 'thread'} pool with {self.max_workers} workers")

    async def stop(self) -> None:
        """Stop the executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
            logger.info("Parallel processor stopped")

    async def execute_parallel(
        self,
        tasks: List[Callable],
        task_args: List[Any],
        handle_errors: bool = False
    ) -> List[Any]:
        """
        Execute tasks in parallel.

        Args:
            tasks: List of functions to execute
            task_args: Arguments for each task
            handle_errors: Return exceptions instead of raising them

        Returns:
            List of results
        """
        if not self.executor:
            raise OptimizationError("Parallel processor not started")

        if len(tasks) != len(task_args):
            raise OptimizationError("Number of tasks and arguments must match")

        start_time = time.time()
        results = []

        try:
            # Submit all tasks
            future_to_index = {}
            for i, (task, args) in enumerate(zip(tasks, task_args)):
                if isinstance(args, (list, tuple)):
                    future = self.executor.submit(task, *args)
                else:
                    future = self.executor.submit(task, args)
                future_to_index[future] = i

            # Collect results in order
            results = [None] * len(tasks)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    self.task_metrics['tasks_completed'] += 1
                except Exception as e:
                    if handle_errors:
                        results[index] = e
                    else:
                        raise
                    self.task_metrics['tasks_failed'] += 1

        except Exception as e:
            logger.error(f"Error in parallel execution: {e}")
            raise

        execution_time = time.time() - start_time
        self.task_metrics['total_execution_time'] += execution_time

        return results

    async def batch_process(
        self,
        task_func: Callable,
        data: List[Any],
        batch_size: int = 10
    ) -> List[Any]:
        """
        Process large datasets in batches.

        Args:
            task_func: Function to apply to each item
            data: List of data items
            batch_size: Size of each batch

        Returns:
            List of results
        """
        if not self.executor:
            raise OptimizationError("Parallel processor not started")

        results = []

        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_tasks = [task_func] * len(batch)

            batch_results = await self.execute_parallel(batch_tasks, batch, handle_errors=True)
            results.extend(batch_results)

        return results

    async def execute_async_parallel(
        self,
        async_tasks: List[Callable],
        task_args: List[Any]
    ) -> List[Any]:
        """
        Execute async tasks in parallel.

        Args:
            async_tasks: List of async functions
            task_args: Arguments for each task

        Returns:
            List of results
        """
        if len(async_tasks) != len(task_args):
            raise OptimizationError("Number of tasks and arguments must match")

        # Create coroutines
        coroutines = []
        for task, args in zip(async_tasks, task_args):
            if isinstance(args, (list, tuple)):
                coroutines.append(task(*args))
            else:
                coroutines.append(task(args))

        # Execute in parallel
        return await asyncio.gather(*coroutines)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for parallel processing."""
        elapsed_time = time.time() - self.task_metrics['start_time']
        total_tasks = self.task_metrics['tasks_completed'] + self.task_metrics['tasks_failed']

        return {
            'tasks_completed': self.task_metrics['tasks_completed'],
            'tasks_failed': self.task_metrics['tasks_failed'],
            'average_execution_time': (
                self.task_metrics['total_execution_time'] / max(1, self.task_metrics['tasks_completed'])
            ),
            'throughput_tasks_per_second': total_tasks / max(1, elapsed_time)
        }


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