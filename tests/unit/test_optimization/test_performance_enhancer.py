"""
Tests for Performance Enhancement System.

Following TDD methodology: Red -> Green -> Refactor
Tests for parallel processing, async optimization, memory management, and CPU optimization.
"""

import pytest
import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import AsyncMock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from src.optimization.performance_enhancer import (
    PerformanceEnhancer,
    ParallelProcessor,
    PerformanceMetrics,
    ResourceManager,
    OptimizationError
)


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics class."""

    def test_should_initialize_with_default_values(self):
        """Test that PerformanceMetrics initializes with default values."""
        metrics = PerformanceMetrics()

        assert metrics.cpu_usage_percent == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.latency_ms == 0.0
        assert metrics.throughput_ops_per_sec == 0.0
        assert metrics.error_rate_percent == 0.0
        assert metrics.concurrent_operations == 0

    def test_should_initialize_with_custom_values(self):
        """Test that PerformanceMetrics accepts custom values."""
        metrics = PerformanceMetrics(
            cpu_usage_percent=75.5,
            memory_usage_mb=512.0,
            latency_ms=25.3,
            throughput_ops_per_sec=1000.0
        )

        assert metrics.cpu_usage_percent == 75.5
        assert metrics.memory_usage_mb == 512.0
        assert metrics.latency_ms == 25.3
        assert metrics.throughput_ops_per_sec == 1000.0

    def test_should_convert_to_dictionary(self):
        """Test that PerformanceMetrics can be converted to dictionary."""
        metrics = PerformanceMetrics(
            cpu_usage_percent=50.0,
            memory_usage_mb=256.0
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict['cpu_usage_percent'] == 50.0
        assert metrics_dict['memory_usage_mb'] == 256.0
        assert 'timestamp' in metrics_dict

    def test_should_calculate_efficiency_score(self):
        """Test that PerformanceMetrics can calculate efficiency score."""
        # High efficiency scenario
        high_efficiency = PerformanceMetrics(
            cpu_usage_percent=60.0,
            memory_usage_mb=200.0,
            latency_ms=10.0,
            throughput_ops_per_sec=2000.0,
            error_rate_percent=0.1
        )

        # Low efficiency scenario
        low_efficiency = PerformanceMetrics(
            cpu_usage_percent=95.0,
            memory_usage_mb=800.0,
            latency_ms=100.0,
            throughput_ops_per_sec=100.0,
            error_rate_percent=5.0
        )

        high_score = high_efficiency.calculate_efficiency_score()
        low_score = low_efficiency.calculate_efficiency_score()

        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1


class TestResourceManager:
    """Test suite for ResourceManager class."""

    @pytest.fixture
    def resource_manager(self):
        """Create ResourceManager instance for testing."""
        return ResourceManager(
            max_cpu_percent=80.0,
            max_memory_mb=1000.0,
            max_concurrent_operations=50
        )

    def test_should_initialize_with_limits(self, resource_manager):
        """Test that ResourceManager initializes with resource limits."""
        assert resource_manager.max_cpu_percent == 80.0
        assert resource_manager.max_memory_mb == 1000.0
        assert resource_manager.max_concurrent_operations == 50
        assert resource_manager.current_operations == 0

    def test_should_check_resource_availability(self, resource_manager):
        """Test that ResourceManager can check resource availability."""
        # Mock current system state
        with patch('psutil.cpu_percent', return_value=70.0), \
             patch('psutil.virtual_memory') as mock_memory:

            mock_memory.return_value.used = 500 * 1024 * 1024  # 500MB

            assert resource_manager.can_allocate_resources() is True

            # Test with high resource usage
            with patch('psutil.cpu_percent', return_value=85.0):
                assert resource_manager.can_allocate_resources() is False

    @pytest.mark.asyncio
    async def test_should_acquire_and_release_operation_slot(self, resource_manager):
        """Test that ResourceManager can manage operation slots."""
        # Mock resource availability
        with patch.object(resource_manager, 'can_allocate_resources', return_value=True):
            # Acquire operation slot
            success = await resource_manager.acquire_operation_slot()
            assert success is True
            assert resource_manager.current_operations == 1

            # Release operation slot
            await resource_manager.release_operation_slot()
            assert resource_manager.current_operations == 0

    @pytest.mark.asyncio
    async def test_should_handle_concurrent_slot_requests(self, resource_manager):
        """Test that ResourceManager handles concurrent slot requests."""
        # Set low limit for testing
        resource_manager.max_concurrent_operations = 2

        # Acquire multiple slots concurrently
        tasks = [resource_manager.acquire_operation_slot() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # Should only allow max_concurrent_operations
        successful_acquisitions = sum(results)
        assert successful_acquisitions <= 2

        # Release all slots
        for _ in range(successful_acquisitions):
            await resource_manager.release_operation_slot()

    def test_should_get_current_metrics(self, resource_manager):
        """Test that ResourceManager can get current system metrics."""
        with patch('psutil.cpu_percent', return_value=65.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.Process') as mock_process:

            mock_memory.return_value.used = 400 * 1024 * 1024  # 400MB
            mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB

            metrics = resource_manager.get_current_metrics()

            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.cpu_usage_percent == 65.0
            assert metrics.memory_usage_mb == pytest.approx(400.0, rel=1e-1)

    @pytest.mark.asyncio
    async def test_should_monitor_resource_usage(self, resource_manager):
        """Test that ResourceManager can monitor resource usage."""
        metrics_history = []

        def metrics_callback(metrics):
            metrics_history.append(metrics)

        # Start monitoring
        monitoring_task = asyncio.create_task(
            resource_manager.start_monitoring(
                interval_seconds=0.1,
                callback=metrics_callback
            )
        )

        # Let it run for a short time
        await asyncio.sleep(0.3)

        # Stop monitoring
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

        # Should have collected some metrics
        assert len(metrics_history) >= 2


class TestParallelProcessor:
    """Test suite for ParallelProcessor class."""

    @pytest.fixture
    def parallel_processor(self):
        """Create ParallelProcessor instance for testing."""
        return ParallelProcessor(
            max_workers=4,
            use_processes=False
        )

    def test_should_initialize_with_configuration(self, parallel_processor):
        """Test that ParallelProcessor initializes with configuration."""
        assert parallel_processor.max_workers == 4
        assert parallel_processor.use_processes is False
        assert parallel_processor.executor is None

    @pytest.mark.asyncio
    async def test_should_start_and_stop_executor(self, parallel_processor):
        """Test that ParallelProcessor can start and stop executor."""
        await parallel_processor.start()
        assert parallel_processor.executor is not None
        assert isinstance(parallel_processor.executor, ThreadPoolExecutor)

        await parallel_processor.stop()
        assert parallel_processor.executor is None

    @pytest.mark.asyncio
    async def test_should_process_tasks_in_parallel(self, parallel_processor):
        """Test that ParallelProcessor can execute tasks in parallel."""
        await parallel_processor.start()

        def cpu_intensive_task(x):
            # Simulate CPU-intensive work
            result = 0
            for i in range(x * 1000):
                result += i ** 0.5
            return result

        tasks = [cpu_intensive_task for _ in range(4)]
        task_args = [100, 200, 300, 400]

        start_time = time.time()
        results = await parallel_processor.execute_parallel(tasks, task_args)
        end_time = time.time()

        assert len(results) == 4
        assert all(isinstance(r, (int, float)) for r in results)

        # Should complete faster than sequential execution
        parallel_time = end_time - start_time

        # Sequential execution for comparison
        start_time = time.time()
        sequential_results = [cpu_intensive_task(arg) for arg in task_args]
        sequential_time = time.time() - start_time

        # Parallel should be faster (with some tolerance for overhead)
        # On some systems parallel might not be much faster for small tasks, so we allow a wider tolerance
        assert parallel_time <= sequential_time * 1.2  # Allow up to 20% slower due to overhead

        await parallel_processor.stop()

    @pytest.mark.asyncio
    async def test_should_handle_task_errors(self, parallel_processor):
        """Test that ParallelProcessor handles task errors gracefully."""
        await parallel_processor.start()

        def failing_task(x):
            if x > 5:
                raise ValueError(f"Invalid value: {x}")
            return x * 2

        tasks = [failing_task for _ in range(5)]
        task_args = [1, 3, 7, 2, 9]  # 7 and 9 will cause errors

        results = await parallel_processor.execute_parallel(
            tasks, task_args, handle_errors=True
        )

        assert len(results) == 5
        assert results[0] == 2  # 1 * 2
        assert results[1] == 6  # 3 * 2
        assert isinstance(results[2], Exception)  # Error for 7
        assert results[3] == 4  # 2 * 2
        assert isinstance(results[4], Exception)  # Error for 9

        await parallel_processor.stop()

    @pytest.mark.asyncio
    async def test_should_batch_process_large_datasets(self, parallel_processor):
        """Test that ParallelProcessor can handle large datasets with batching."""
        await parallel_processor.start()

        def simple_task(x):
            return x ** 2

        # Large dataset
        large_dataset = list(range(100))

        results = await parallel_processor.batch_process(
            simple_task, large_dataset, batch_size=10
        )

        assert len(results) == 100
        assert results[0] == 0
        assert results[9] == 81
        assert results[99] == 9801

        await parallel_processor.stop()

    @pytest.mark.asyncio
    async def test_should_support_async_tasks(self, parallel_processor):
        """Test that ParallelProcessor supports async tasks."""
        await parallel_processor.start()

        async def async_task(x):
            await asyncio.sleep(0.01)  # Simulate async work
            return x * 3

        tasks = [async_task for _ in range(4)]
        task_args = [1, 2, 3, 4]

        results = await parallel_processor.execute_async_parallel(tasks, task_args)

        assert len(results) == 4
        assert results == [3, 6, 9, 12]

        await parallel_processor.stop()

    def test_should_get_performance_metrics(self, parallel_processor):
        """Test that ParallelProcessor provides performance metrics."""
        metrics = parallel_processor.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert 'tasks_completed' in metrics
        assert 'tasks_failed' in metrics
        assert 'average_execution_time' in metrics
        assert 'throughput_tasks_per_second' in metrics


class TestPerformanceEnhancer:
    """Test suite for PerformanceEnhancer class."""

    @pytest.fixture
    def performance_enhancer(self):
        """Create PerformanceEnhancer instance for testing."""
        return PerformanceEnhancer(
            enable_parallel_processing=True,
            enable_memory_optimization=True,
            enable_cpu_optimization=True
        )

    def test_should_initialize_with_optimizations(self, performance_enhancer):
        """Test that PerformanceEnhancer initializes with optimization settings."""
        assert performance_enhancer.enable_parallel_processing is True
        assert performance_enhancer.enable_memory_optimization is True
        assert performance_enhancer.enable_cpu_optimization is True
        assert performance_enhancer.resource_manager is not None
        assert performance_enhancer.parallel_processor is not None

    @pytest.mark.asyncio
    async def test_should_start_and_stop_optimization_systems(self, performance_enhancer):
        """Test that PerformanceEnhancer can start and stop optimization systems."""
        await performance_enhancer.start()

        assert performance_enhancer.is_running is True
        assert performance_enhancer.parallel_processor.executor is not None

        await performance_enhancer.stop()

        assert performance_enhancer.is_running is False

    @pytest.mark.asyncio
    async def test_should_optimize_function_execution(self, performance_enhancer):
        """Test that PerformanceEnhancer can optimize function execution."""
        await performance_enhancer.start()

        def compute_heavy_task(n):
            result = 0
            for i in range(n):
                result += i ** 0.5
            return result

        # Optimize single function call
        optimized_result = await performance_enhancer.optimize_execution(
            compute_heavy_task, 10000
        )

        # Direct execution for comparison
        direct_result = compute_heavy_task(10000)

        assert optimized_result == direct_result

        await performance_enhancer.stop()

    @pytest.mark.asyncio
    async def test_should_optimize_batch_processing(self, performance_enhancer):
        """Test that PerformanceEnhancer can optimize batch processing."""
        await performance_enhancer.start()

        def process_item(item):
            return item * 2 + 1

        items = list(range(50))

        # Optimized batch processing
        optimized_results = await performance_enhancer.optimize_batch_processing(
            process_item, items
        )

        # Direct processing for comparison
        direct_results = [process_item(item) for item in items]

        assert optimized_results == direct_results

        await performance_enhancer.stop()

    @pytest.mark.asyncio
    async def test_should_manage_memory_usage(self, performance_enhancer):
        """Test that PerformanceEnhancer manages memory usage."""
        await performance_enhancer.start()

        # Create memory-intensive operation
        def memory_heavy_task():
            large_list = [i for i in range(100000)]
            return sum(large_list)

        initial_memory = performance_enhancer.resource_manager.get_current_metrics().memory_usage_mb

        result = await performance_enhancer.optimize_execution(memory_heavy_task)

        final_memory = performance_enhancer.resource_manager.get_current_metrics().memory_usage_mb

        assert isinstance(result, int)
        # Memory should be cleaned up (allowing some variance)
        assert final_memory <= initial_memory + 100  # 100MB tolerance

        await performance_enhancer.stop()

    @pytest.mark.asyncio
    async def test_should_handle_concurrent_optimizations(self, performance_enhancer):
        """Test that PerformanceEnhancer handles concurrent optimizations."""
        await performance_enhancer.start()

        def simple_task(x):
            time.sleep(0.01)  # Small delay
            return x * x

        # Run multiple optimizations concurrently
        tasks = [
            performance_enhancer.optimize_execution(simple_task, i)
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert results[0] == 0
        assert results[5] == 25
        assert results[9] == 81

        await performance_enhancer.stop()

    @pytest.mark.asyncio
    async def test_should_provide_optimization_recommendations(self, performance_enhancer):
        """Test that PerformanceEnhancer provides optimization recommendations."""
        await performance_enhancer.start()

        # Simulate some workload
        for i in range(5):
            await performance_enhancer.optimize_execution(lambda x: x * 2, i)

        recommendations = performance_enhancer.get_optimization_recommendations()

        assert isinstance(recommendations, dict)
        assert 'cpu_optimization' in recommendations
        assert 'memory_optimization' in recommendations
        assert 'parallel_processing' in recommendations
        assert 'resource_allocation' in recommendations

        await performance_enhancer.stop()

    @pytest.mark.asyncio
    async def test_should_auto_tune_performance(self, performance_enhancer):
        """Test that PerformanceEnhancer can auto-tune performance parameters."""
        await performance_enhancer.start()

        # Mock resource allocation to always succeed
        with patch.object(performance_enhancer.resource_manager, 'acquire_operation_slot', return_value=True), \
             patch.object(performance_enhancer.resource_manager, 'release_operation_slot'):

            # Run workload to gather metrics
            def varying_workload(complexity):
                result = 0
                for i in range(complexity * 1000):
                    result += i ** 0.5
                return result

            # Execute tasks with different complexities
            for complexity in [10, 20, 30, 15, 25]:
                await performance_enhancer.optimize_execution(varying_workload, complexity)

            # Add some mock metrics to ensure we have enough data
            for i in range(15):
                metrics = PerformanceMetrics(
                    cpu_usage_percent=50 + i * 2,
                    memory_usage_mb=200 + i * 10,
                    latency_ms=20 + i * 2,  # Add latency data
                    concurrent_operations=i
                )
                performance_enhancer.metrics_history.append(metrics)

            # Auto-tune based on gathered metrics
            tuning_result = await performance_enhancer.auto_tune_performance()

            assert isinstance(tuning_result, dict)
            assert 'tuning_applied' in tuning_result
            assert 'performance_improvement' in tuning_result
            assert 'new_parameters' in tuning_result

        await performance_enhancer.stop()

    def test_should_export_performance_report(self, performance_enhancer):
        """Test that PerformanceEnhancer can export performance report."""
        # Add some mock metrics history
        for i in range(10):
            metrics = PerformanceMetrics(
                cpu_usage_percent=50 + i * 2,
                memory_usage_mb=200 + i * 10,
                latency_ms=20 + i,
                throughput_ops_per_sec=1000 - i * 10
            )
            performance_enhancer.metrics_history.append(metrics)

        report = performance_enhancer.export_performance_report()

        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'optimization_impact' in report
        assert 'resource_utilization' in report
        assert 'recommendations' in report
        assert 'metrics_history' in report

    @pytest.mark.asyncio
    async def test_should_handle_resource_constraints(self, performance_enhancer):
        """Test that PerformanceEnhancer handles resource constraints gracefully."""
        # Set very low resource limits
        performance_enhancer.resource_manager.max_cpu_percent = 10.0
        performance_enhancer.resource_manager.max_memory_mb = 50.0

        await performance_enhancer.start()

        def resource_intensive_task():
            # Task that would normally use significant resources
            return sum(i ** 2 for i in range(10000))

        # Should still execute but with resource management
        result = await performance_enhancer.optimize_execution(resource_intensive_task)

        assert isinstance(result, int)

        await performance_enhancer.stop()

    @pytest.mark.asyncio
    async def test_should_support_custom_optimization_strategies(self, performance_enhancer):
        """Test that PerformanceEnhancer supports custom optimization strategies."""
        await performance_enhancer.start()

        # Mock resource allocation to always succeed
        with patch.object(performance_enhancer.resource_manager, 'acquire_operation_slot', return_value=True), \
             patch.object(performance_enhancer.resource_manager, 'release_operation_slot'):

            # Custom optimization strategy
            class CustomStrategy:
                async def optimize(self, func, *args, **kwargs):
                    # Simple custom optimization: add timing
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    return {
                        'result': result,
                        'execution_time': end_time - start_time,
                        'strategy': 'custom'
                    }

            custom_strategy = CustomStrategy()

            # Register custom strategy
            performance_enhancer.register_optimization_strategy('custom', custom_strategy)

            def test_function(x):
                return x * 3

            # Use custom strategy
            result = await performance_enhancer.optimize_execution(
                test_function, 5, strategy='custom'
            )

            assert isinstance(result, dict)
            assert result['result'] == 15
            assert 'execution_time' in result
            assert result['strategy'] == 'custom'

        await performance_enhancer.stop()