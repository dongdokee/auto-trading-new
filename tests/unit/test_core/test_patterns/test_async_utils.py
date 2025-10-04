# tests/unit/test_core/test_patterns/test_async_utils.py
"""
Test suite for async utility patterns (Phase 8 optimizations)
"""

import pytest
import asyncio
import time
from typing import List
from unittest.mock import AsyncMock

from src.core.patterns.async_utils import (
    BatchProcessor, ConcurrentExecutor, ThrottledExecutor,
    process_concurrently, process_in_batches,
    CircuitBreakerOpen, SemaphoreTimeoutError
)


class TestBatchProcessor:
    """Test BatchProcessor functionality"""

    @pytest.fixture
    def batch_processor(self):
        return BatchProcessor(batch_size=5, max_concurrent_batches=2, timeout_seconds=5.0)

    @pytest.mark.asyncio
    async def test_should_process_items_in_batches(self, batch_processor):
        """Test basic batch processing"""
        items = list(range(12))  # 12 items, should create 3 batches of 5, 5, 2

        async def processor(item: int) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            return f"processed_{item}"

        result = await batch_processor.process_batch(items, processor)

        assert result.total_items == 12
        assert result.success_count == 12
        assert result.failure_count == 0
        assert result.success_rate == 1.0
        assert len(result.successful) == 12
        assert all(res.startswith("processed_") for res in result.successful)

    @pytest.mark.asyncio
    async def test_should_handle_failures_gracefully(self, batch_processor):
        """Test batch processing with some failures"""
        items = list(range(10))

        async def processor(item: int) -> str:
            await asyncio.sleep(0.01)
            if item % 3 == 0:  # Fail every 3rd item
                raise ValueError(f"Failed on {item}")
            return f"processed_{item}"

        result = await batch_processor.process_batch(items, processor)

        assert result.total_items == 10
        assert result.failure_count == 4  # Items 0, 3, 6, 9
        assert result.success_count == 6
        assert result.success_rate == 0.6

    @pytest.mark.asyncio
    async def test_should_handle_empty_input(self, batch_processor):
        """Test batch processing with empty input"""
        async def processor(item: int) -> str:
            return f"processed_{item}"

        result = await batch_processor.process_batch([], processor)

        assert result.total_items == 0
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.success_rate == 1.0

    def test_should_track_metrics(self, batch_processor):
        """Test metrics collection"""
        metrics = batch_processor.get_metrics()

        assert 'total_processed' in metrics
        assert 'total_failures' in metrics
        assert 'total_batches' in metrics
        assert 'avg_processing_time' in metrics
        assert 'overall_success_rate' in metrics


class TestConcurrentExecutor:
    """Test ConcurrentExecutor functionality"""

    @pytest.fixture
    def executor(self):
        return ConcurrentExecutor(max_concurrent=5, timeout_seconds=2.0)

    @pytest.mark.asyncio
    async def test_should_execute_coroutines_concurrently(self, executor):
        """Test concurrent execution of coroutines"""
        async def task(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"task_completed_{delay}"

        # Execute tasks with different delays
        coros = [task(0.1), task(0.2), task(0.1)]

        start_time = time.time()
        results = await executor.execute_many(coros)
        end_time = time.time()

        # Should complete in ~0.2s (max delay) rather than 0.4s (sum of delays)
        assert end_time - start_time < 0.3
        assert len(results) == 3
        assert all("task_completed_" in str(result) for result in results if not isinstance(result, Exception))

    @pytest.mark.asyncio
    async def test_should_handle_timeout(self, executor):
        """Test timeout handling"""
        async def slow_task():
            await asyncio.sleep(5.0)  # Longer than timeout
            return "completed"

        with pytest.raises(asyncio.TimeoutError):
            await executor.execute(slow_task())

    @pytest.mark.asyncio
    async def test_should_implement_circuit_breaker(self, executor):
        """Test circuit breaker functionality"""
        async def failing_task():
            raise ValueError("Task failed")

        # Trigger enough failures to open circuit breaker
        for _ in range(15):  # More than threshold
            try:
                await executor.execute(failing_task())
            except ValueError:
                pass

        # Circuit breaker should now be open
        with pytest.raises(CircuitBreakerOpen):
            await executor.execute(failing_task())

    @pytest.mark.asyncio
    async def test_should_shutdown_gracefully(self, executor):
        """Test graceful shutdown"""
        async def long_task():
            await asyncio.sleep(1.0)
            return "completed"

        # Start some tasks
        task1 = asyncio.create_task(executor.execute(long_task()))
        task2 = asyncio.create_task(executor.execute(long_task()))

        # Shutdown executor
        await executor.shutdown(timeout=2.0)

        # Tasks should be cancelled or completed
        assert task1.done()
        assert task2.done()

    def test_should_track_metrics(self, executor):
        """Test metrics collection"""
        metrics = executor.get_metrics()

        assert 'total_executions' in metrics
        assert 'total_failures' in metrics
        assert 'success_rate' in metrics
        assert 'active_tasks' in metrics
        assert 'circuit_open' in metrics


class TestThrottledExecutor:
    """Test ThrottledExecutor functionality"""

    @pytest.fixture
    def throttled_executor(self):
        return ThrottledExecutor(rate_limit=10.0, burst_size=5)  # 10 requests/second

    @pytest.mark.asyncio
    async def test_should_throttle_requests(self, throttled_executor):
        """Test request throttling"""
        async def fast_task() -> str:
            return "completed"

        # Execute more requests than burst size
        start_time = time.time()
        results = []

        for i in range(8):  # More than burst size (5)
            result = await throttled_executor.execute(fast_task())
            results.append(result)

        end_time = time.time()

        # Should take some time due to throttling
        assert end_time - start_time > 0.1  # Some delay expected
        assert len(results) == 8
        assert all(result == "completed" for result in results)

    def test_should_track_throttling_metrics(self, throttled_executor):
        """Test throttling metrics"""
        metrics = throttled_executor.get_metrics()

        assert 'total_requests' in metrics
        assert 'throttled_requests' in metrics
        assert 'throttle_rate' in metrics
        assert 'current_tokens' in metrics
        assert 'rate_limit' in metrics


class TestConvenienceFunctions:
    """Test convenience functions"""

    @pytest.mark.asyncio
    async def test_process_concurrently(self):
        """Test process_concurrently convenience function"""
        items = [1, 2, 3, 4, 5]

        async def processor(item: int) -> int:
            await asyncio.sleep(0.01)
            return item * 2

        results = await process_concurrently(
            items=items,
            processor=processor,
            max_concurrent=3,
            timeout=1.0
        )

        assert len(results) == 5
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert successful_results == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_process_in_batches(self):
        """Test process_in_batches convenience function"""
        items = list(range(15))

        async def processor(item: int) -> str:
            await asyncio.sleep(0.01)
            return f"item_{item}"

        result = await process_in_batches(
            items=items,
            processor=processor,
            batch_size=5,
            max_concurrent_batches=2
        )

        assert result.total_items == 15
        assert result.success_count == 15
        assert result.failure_count == 0
        assert len(result.successful) == 15


class TestAsyncContextManagers:
    """Test async context managers"""

    @pytest.mark.asyncio
    async def test_managed_executor_context(self):
        """Test managed executor context manager"""
        from src.core.patterns.async_utils import managed_executor

        executor = ConcurrentExecutor(max_concurrent=5)

        async with managed_executor(executor) as exec_ctx:
            async def simple_task():
                return "completed"

            result = await exec_ctx.execute(simple_task())
            assert result == "completed"

        # Executor should be shutdown after context exit
        assert len(executor.active_tasks) == 0

    @pytest.mark.asyncio
    async def test_managed_semaphore_context(self):
        """Test managed semaphore context manager"""
        from src.core.patterns.async_utils import managed_semaphore

        semaphore = asyncio.Semaphore(1)

        async with managed_semaphore(semaphore, timeout=1.0):
            # Semaphore should be acquired
            assert semaphore.locked()

        # Semaphore should be released after context exit
        assert not semaphore.locked()

    @pytest.mark.asyncio
    async def test_managed_semaphore_timeout(self):
        """Test managed semaphore timeout"""
        from src.core.patterns.async_utils import managed_semaphore

        semaphore = asyncio.Semaphore(1)

        # Acquire semaphore manually
        await semaphore.acquire()

        try:
            # Should timeout when trying to acquire already locked semaphore
            with pytest.raises(SemaphoreTimeoutError):
                async with managed_semaphore(semaphore, timeout=0.1):
                    pass
        finally:
            semaphore.release()


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self):
        """Benchmark batch processing performance"""
        items = list(range(1000))

        async def processor(item: int) -> int:
            await asyncio.sleep(0.001)  # Simulate work
            return item * 2

        batch_processor = BatchProcessor(batch_size=50, max_concurrent_batches=5)

        start_time = time.time()
        result = await batch_processor.process_batch(items, processor)
        end_time = time.time()

        # Should complete much faster than sequential processing
        assert end_time - start_time < 2.0  # Should be well under 1 second
        assert result.success_count == 1000
        assert result.success_rate == 1.0

        # Check metrics
        metrics = batch_processor.get_metrics()
        assert metrics['total_processed'] == 1000
        assert metrics['overall_success_rate'] == 1.0

    @pytest.mark.asyncio
    async def test_concurrent_executor_performance(self):
        """Benchmark concurrent executor performance"""
        async def task(item: int) -> int:
            await asyncio.sleep(0.001)
            return item * 2

        executor = ConcurrentExecutor(max_concurrent=50)
        coros = [task(i) for i in range(500)]

        start_time = time.time()
        results = await executor.execute_many(coros)
        end_time = time.time()

        # Should complete much faster than sequential
        assert end_time - start_time < 1.0
        assert len(results) == 500
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 500

        await executor.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])