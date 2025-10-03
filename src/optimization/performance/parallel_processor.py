"""
Parallel processing system with thread and process pool support.

This module provides optimized parallel execution for CPU and I/O bound tasks
with comprehensive performance monitoring and batch processing capabilities.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .models import OptimizationError

logger = logging.getLogger(__name__)


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