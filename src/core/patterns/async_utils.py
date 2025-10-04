# src/core/patterns/async_utils.py
"""
Async Utilities

High-performance async patterns for optimization:
- BatchProcessor: Efficient batch processing with configurable limits
- ConcurrentExecutor: Controlled concurrent execution with semaphores
- ThrottledExecutor: Rate-limited execution with backpressure
- AsyncContextManager: Reusable async context management patterns
"""

import asyncio
import time
import logging
from typing import (
    Any, Awaitable, Callable, List, Optional, TypeVar, Generic,
    Iterator, Dict, Set, Union
)
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime, timedelta

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchResult(Generic[T, R]):
    """Result of batch processing operation"""
    successful: List[R]
    failed: List[tuple[T, Exception]]
    total_items: int
    processing_time: float
    success_rate: float

    @property
    def success_count(self) -> int:
        return len(self.successful)

    @property
    def failure_count(self) -> int:
        return len(self.failed)


class BatchProcessor(Generic[T, R]):
    """
    High-performance batch processor with configurable limits

    Optimizes processing by batching items and processing them concurrently
    within configurable limits to prevent resource exhaustion.
    """

    def __init__(self,
                 batch_size: int = 50,
                 max_concurrent_batches: int = 5,
                 max_semaphore_size: int = 100,
                 timeout_seconds: float = 30.0):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.semaphore = asyncio.Semaphore(max_semaphore_size)
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(f"{__name__}.BatchProcessor")

        # Performance metrics
        self.total_processed = 0
        self.total_failures = 0
        self.total_processing_time = 0.0
        self.batch_count = 0

    async def process_batch(self,
                          items: List[T],
                          processor: Callable[[T], Awaitable[R]],
                          return_exceptions: bool = True) -> BatchResult[T, R]:
        """
        Process a list of items in optimized batches

        Args:
            items: Items to process
            processor: Async function to process each item
            return_exceptions: Whether to return exceptions or raise them

        Returns:
            BatchResult with successful/failed results and metrics
        """
        if not items:
            return BatchResult([], [], 0, 0.0, 1.0)

        start_time = time.time()
        successful: List[R] = []
        failed: List[tuple[T, Exception]] = []

        # Split into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        self.logger.debug(f"Processing {len(items)} items in {len(batches)} batches")

        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        async def process_single_batch(batch: List[T]) -> tuple[List[R], List[tuple[T, Exception]]]:
            async with semaphore:
                batch_successful = []
                batch_failed = []

                # Process items in batch concurrently
                tasks = []
                for item in batch:
                    async with self.semaphore:
                        task = asyncio.create_task(processor(item))
                        tasks.append((item, task))

                # Gather results with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*[task for _, task in tasks], return_exceptions=return_exceptions),
                        timeout=self.timeout_seconds
                    )

                    for (item, _), result in zip(tasks, results):
                        if isinstance(result, Exception):
                            batch_failed.append((item, result))
                        else:
                            batch_successful.append(result)

                except asyncio.TimeoutError as e:
                    # Handle timeout - mark all as failed
                    for item, task in tasks:
                        if not task.done():
                            task.cancel()
                        batch_failed.append((item, e))

                return batch_successful, batch_failed

        # Process all batches concurrently
        batch_results = await asyncio.gather(
            *[process_single_batch(batch) for batch in batches],
            return_exceptions=True
        )

        # Aggregate results
        for result in batch_results:
            if isinstance(result, Exception):
                # Batch processing failed entirely
                self.logger.error(f"Batch processing failed: {result}")
                continue

            batch_successful, batch_failed = result
            successful.extend(batch_successful)
            failed.extend(batch_failed)

        # Calculate metrics
        processing_time = time.time() - start_time
        success_rate = len(successful) / len(items) if items else 1.0

        # Update statistics
        self.total_processed += len(successful)
        self.total_failures += len(failed)
        self.total_processing_time += processing_time
        self.batch_count += 1

        self.logger.info(
            f"Batch processing complete: {len(successful)}/{len(items)} successful "
            f"({success_rate:.2%}) in {processing_time:.2f}s"
        )

        return BatchResult(
            successful=successful,
            failed=failed,
            total_items=len(items),
            processing_time=processing_time,
            success_rate=success_rate
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        avg_processing_time = (
            self.total_processing_time / self.batch_count
            if self.batch_count > 0 else 0.0
        )

        overall_success_rate = (
            self.total_processed / (self.total_processed + self.total_failures)
            if (self.total_processed + self.total_failures) > 0 else 1.0
        )

        return {
            'total_processed': self.total_processed,
            'total_failures': self.total_failures,
            'total_batches': self.batch_count,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': avg_processing_time,
            'overall_success_rate': overall_success_rate,
            'batch_size': self.batch_size,
            'max_concurrent_batches': self.max_concurrent_batches
        }


class ConcurrentExecutor:
    """
    Controlled concurrent execution with resource management

    Manages concurrent execution with configurable limits,
    circuit breaking, and automatic backpressure handling.
    """

    def __init__(self,
                 max_concurrent: int = 20,
                 timeout_seconds: float = 30.0,
                 circuit_breaker_threshold: int = 10,
                 circuit_breaker_timeout: timedelta = timedelta(minutes=1)):
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Circuit breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.consecutive_failures = 0
        self.circuit_open_until: Optional[datetime] = None

        # Metrics
        self.total_executions = 0
        self.total_failures = 0
        self.active_tasks: Set[asyncio.Task] = set()

        self.logger = logging.getLogger(f"{__name__}.ConcurrentExecutor")

    @property
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_open_until is None:
            return False
        return datetime.now() < self.circuit_open_until

    async def execute(self, coro: Awaitable[T]) -> T:
        """
        Execute coroutine with concurrency control

        Args:
            coro: Coroutine to execute

        Returns:
            Result of coroutine execution

        Raises:
            CircuitBreakerOpen: If circuit breaker is open
            asyncio.TimeoutError: If execution times out
        """
        # Check circuit breaker
        if self.is_circuit_open:
            raise CircuitBreakerOpen(
                f"Circuit breaker open until {self.circuit_open_until}"
            )

        async with self.semaphore:
            try:
                self.total_executions += 1

                # Create and track task
                task = asyncio.create_task(coro)
                self.active_tasks.add(task)

                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(task, timeout=self.timeout_seconds)

                    # Reset circuit breaker on success
                    self.consecutive_failures = 0

                    return result

                finally:
                    self.active_tasks.discard(task)

            except Exception as e:
                self.total_failures += 1
                self.consecutive_failures += 1

                # Check if we should open circuit breaker
                if self.consecutive_failures >= self.circuit_breaker_threshold:
                    self.circuit_open_until = datetime.now() + self.circuit_breaker_timeout
                    self.logger.warning(
                        f"Circuit breaker opened due to {self.consecutive_failures} consecutive failures"
                    )

                raise

    async def execute_many(self,
                          coros: List[Awaitable[T]],
                          return_exceptions: bool = True) -> List[Union[T, Exception]]:
        """
        Execute multiple coroutines concurrently

        Args:
            coros: List of coroutines to execute
            return_exceptions: Whether to return exceptions or raise them

        Returns:
            List of results (or exceptions if return_exceptions=True)
        """
        if not coros:
            return []

        # Execute all with concurrency control
        tasks = [self.execute(coro) for coro in coros]

        try:
            return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        except Exception:
            # Cancel remaining tasks on failure
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    async def shutdown(self, timeout: float = 10.0):
        """
        Gracefully shutdown executor

        Args:
            timeout: Maximum time to wait for active tasks
        """
        if not self.active_tasks:
            return

        self.logger.info(f"Shutting down executor with {len(self.active_tasks)} active tasks")

        # Wait for active tasks to complete
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.active_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()

            self.logger.warning(f"Cancelled {len(self.active_tasks)} tasks during shutdown")

    def get_metrics(self) -> Dict[str, Any]:
        """Get executor metrics"""
        success_rate = (
            (self.total_executions - self.total_failures) / self.total_executions
            if self.total_executions > 0 else 1.0
        )

        return {
            'total_executions': self.total_executions,
            'total_failures': self.total_failures,
            'success_rate': success_rate,
            'active_tasks': len(self.active_tasks),
            'consecutive_failures': self.consecutive_failures,
            'circuit_open': self.is_circuit_open,
            'max_concurrent': self.max_concurrent
        }


class ThrottledExecutor:
    """
    Rate-limited executor with backpressure handling

    Controls execution rate to prevent overwhelming downstream systems
    with configurable rate limiting and burst capacity.
    """

    def __init__(self,
                 rate_limit: float = 100.0,  # requests per second
                 burst_size: int = 10,
                 time_window: float = 1.0):
        self.rate_limit = rate_limit
        self.burst_size = burst_size
        self.time_window = time_window

        # Token bucket algorithm
        self.tokens = burst_size
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

        # Metrics
        self.total_requests = 0
        self.throttled_requests = 0
        self.request_times: deque = deque(maxlen=1000)

        self.logger = logging.getLogger(f"{__name__}.ThrottledExecutor")

    async def _refill_tokens(self):
        """Refill tokens based on rate limit"""
        now = time.time()
        time_passed = now - self.last_refill

        # Add tokens based on time passed
        tokens_to_add = time_passed * self.rate_limit
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_refill = now

    async def execute(self, coro: Awaitable[T]) -> T:
        """
        Execute coroutine with rate limiting

        Args:
            coro: Coroutine to execute

        Returns:
            Result of coroutine execution
        """
        async with self.lock:
            await self._refill_tokens()

            if self.tokens < 1:
                # Calculate wait time
                wait_time = (1 - self.tokens) / self.rate_limit
                self.throttled_requests += 1

                self.logger.debug(f"Rate limit exceeded, waiting {wait_time:.3f}s")
                await asyncio.sleep(wait_time)
                await self._refill_tokens()

            # Consume token
            self.tokens -= 1
            self.total_requests += 1

        # Execute coroutine
        start_time = time.time()
        try:
            result = await coro
            self.request_times.append(time.time() - start_time)
            return result
        except Exception:
            self.request_times.append(time.time() - start_time)
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get throttling metrics"""
        avg_request_time = (
            sum(self.request_times) / len(self.request_times)
            if self.request_times else 0.0
        )

        throttle_rate = (
            self.throttled_requests / self.total_requests
            if self.total_requests > 0 else 0.0
        )

        return {
            'total_requests': self.total_requests,
            'throttled_requests': self.throttled_requests,
            'throttle_rate': throttle_rate,
            'avg_request_time': avg_request_time,
            'current_tokens': self.tokens,
            'rate_limit': self.rate_limit,
            'burst_size': self.burst_size
        }


@asynccontextmanager
async def managed_semaphore(semaphore: asyncio.Semaphore, timeout: Optional[float] = None):
    """
    Context manager for semaphore with timeout

    Args:
        semaphore: Semaphore to acquire
        timeout: Optional timeout for acquisition
    """
    if timeout is None:
        async with semaphore:
            yield
    else:
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
            try:
                yield
            finally:
                semaphore.release()
        except asyncio.TimeoutError:
            raise SemaphoreTimeoutError(f"Failed to acquire semaphore within {timeout}s")


@asynccontextmanager
async def managed_executor(executor: ConcurrentExecutor):
    """
    Context manager for executor lifecycle

    Args:
        executor: Executor to manage
    """
    try:
        yield executor
    finally:
        await executor.shutdown()


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class SemaphoreTimeoutError(Exception):
    """Exception raised when semaphore acquisition times out"""
    pass


# Convenience functions for common patterns
async def process_concurrently(items: List[T],
                             processor: Callable[[T], Awaitable[R]],
                             max_concurrent: int = 20,
                             timeout: float = 30.0) -> List[Union[R, Exception]]:
    """
    Process items concurrently with simple interface

    Args:
        items: Items to process
        processor: Async processor function
        max_concurrent: Maximum concurrent executions
        timeout: Timeout for each item

    Returns:
        List of results or exceptions
    """
    executor = ConcurrentExecutor(max_concurrent=max_concurrent, timeout_seconds=timeout)
    async with managed_executor(executor):
        coros = [processor(item) for item in items]
        return await executor.execute_many(coros, return_exceptions=True)


async def process_in_batches(items: List[T],
                           processor: Callable[[T], Awaitable[R]],
                           batch_size: int = 50,
                           max_concurrent_batches: int = 5) -> BatchResult[T, R]:
    """
    Process items in batches with simple interface

    Args:
        items: Items to process
        processor: Async processor function
        batch_size: Size of each batch
        max_concurrent_batches: Maximum concurrent batches

    Returns:
        BatchResult with aggregated results
    """
    batch_processor = BatchProcessor(
        batch_size=batch_size,
        max_concurrent_batches=max_concurrent_batches
    )
    return await batch_processor.process_batch(items, processor)