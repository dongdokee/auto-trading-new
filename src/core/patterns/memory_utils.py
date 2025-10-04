# src/core/patterns/memory_utils.py
"""
Memory Utilities

Memory-efficient patterns for high-performance data processing:
- StreamProcessor: Generator-based streaming with backpressure
- CircularBuffer: Fixed-size memory-efficient circular buffer
- LazyLoader: On-demand loading with caching
- ChunkedProcessor: Memory-efficient chunked data processing
"""

import asyncio
import gc
import sys
import weakref
from typing import (
    Any, AsyncIterator, Awaitable, Callable, Dict, Generic, Iterator, List,
    Optional, TypeVar, Union, Tuple, AsyncIterable, Iterable
)
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import psutil
import os

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory usage percentage
    available_mb: float  # Available memory in MB
    timestamp: datetime


class MemoryMonitor:
    """Monitor memory usage with alerts"""

    def __init__(self, alert_threshold_mb: float = 1000.0, check_interval: float = 5.0):
        self.alert_threshold_mb = alert_threshold_mb
        self.check_interval = check_interval
        self.process = psutil.Process(os.getpid())
        self.logger = logging.getLogger(f"{__name__}.MemoryMonitor")

        # Statistics
        self.peak_usage_mb = 0.0
        self.alert_count = 0
        self.history: deque[MemoryStats] = deque(maxlen=100)

    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()

        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024

        stats = MemoryStats(
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            percent=self.process.memory_percent(),
            available_mb=virtual_memory.available / 1024 / 1024,
            timestamp=datetime.now()
        )

        # Update peak usage
        if rss_mb > self.peak_usage_mb:
            self.peak_usage_mb = rss_mb

        # Check for alerts
        if rss_mb > self.alert_threshold_mb:
            self.alert_count += 1
            self.logger.warning(
                f"Memory usage alert: {rss_mb:.1f}MB exceeds threshold {self.alert_threshold_mb}MB"
            )

        self.history.append(stats)
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        current = self.get_current_stats()

        if len(self.history) > 1:
            avg_rss = sum(s.rss_mb for s in self.history) / len(self.history)
            min_rss = min(s.rss_mb for s in self.history)
            max_rss = max(s.rss_mb for s in self.history)
        else:
            avg_rss = min_rss = max_rss = current.rss_mb

        return {
            'current_rss_mb': current.rss_mb,
            'current_vms_mb': current.vms_mb,
            'current_percent': current.percent,
            'available_mb': current.available_mb,
            'peak_usage_mb': self.peak_usage_mb,
            'avg_rss_mb': avg_rss,
            'min_rss_mb': min_rss,
            'max_rss_mb': max_rss,
            'alert_count': self.alert_count,
            'history_count': len(self.history)
        }


class CircularBuffer(Generic[T]):
    """
    Memory-efficient circular buffer with fixed size

    Automatically overwrites oldest entries when capacity is reached,
    preventing unbounded memory growth.
    """

    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")

        self.maxsize = maxsize
        self._buffer: deque[T] = deque(maxlen=maxsize)
        self._total_added = 0
        self.logger = logging.getLogger(f"{__name__}.CircularBuffer")

    def append(self, item: T) -> Optional[T]:
        """
        Add item to buffer

        Returns:
            Item that was removed if buffer was full, None otherwise
        """
        removed_item = None
        if len(self._buffer) == self.maxsize:
            removed_item = self._buffer[0]  # Item that will be removed

        self._buffer.append(item)
        self._total_added += 1

        return removed_item

    def extend(self, items: Iterable[T]) -> List[T]:
        """
        Add multiple items to buffer

        Returns:
            List of items that were removed
        """
        removed_items = []
        for item in items:
            removed = self.append(item)
            if removed is not None:
                removed_items.append(removed)
        return removed_items

    def get_recent(self, n: int) -> List[T]:
        """Get n most recent items"""
        if n <= 0:
            return []
        return list(self._buffer)[-n:]

    def get_oldest(self, n: int) -> List[T]:
        """Get n oldest items"""
        if n <= 0:
            return []
        return list(self._buffer)[:n]

    def clear(self):
        """Clear the buffer"""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[T]:
        return iter(self._buffer)

    def __bool__(self) -> bool:
        return len(self._buffer) > 0

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.maxsize

    @property
    def utilization(self) -> float:
        """Buffer utilization as percentage"""
        return (len(self._buffer) / self.maxsize) * 100

    @property
    def total_added(self) -> int:
        """Total number of items added (including overwritten ones)"""
        return self._total_added

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            'current_size': len(self._buffer),
            'max_size': self.maxsize,
            'utilization_pct': self.utilization,
            'total_added': self.total_added,
            'is_full': self.is_full
        }


class StreamProcessor(Generic[T, R]):
    """
    Memory-efficient stream processor using generators

    Processes data streams without loading everything into memory,
    with configurable chunk sizes and backpressure handling.
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 max_buffer_size: int = 10000,
                 memory_threshold_mb: float = 500.0):
        self.chunk_size = chunk_size
        self.max_buffer_size = max_buffer_size
        self.memory_threshold_mb = memory_threshold_mb
        self.memory_monitor = MemoryMonitor(alert_threshold_mb=memory_threshold_mb)

        # Processing stats
        self.total_processed = 0
        self.total_chunks = 0
        self.processing_errors = 0

        self.logger = logging.getLogger(f"{__name__}.StreamProcessor")

    async def process_stream(self,
                           stream: AsyncIterable[T],
                           processor: Callable[[T], Union[R, Awaitable[R]]],
                           enable_backpressure: bool = True) -> AsyncIterator[R]:
        """
        Process async stream with memory efficiency

        Args:
            stream: Input async iterable
            processor: Function to process each item
            enable_backpressure: Whether to apply backpressure based on memory

        Yields:
            Processed results
        """
        buffer = []

        async for item in stream:
            # Check memory pressure for backpressure
            if enable_backpressure:
                memory_stats = self.memory_monitor.get_current_stats()
                if memory_stats.rss_mb > self.memory_threshold_mb:
                    self.logger.warning(
                        f"Memory pressure detected ({memory_stats.rss_mb:.1f}MB), "
                        "applying backpressure"
                    )
                    await asyncio.sleep(0.1)  # Brief pause to allow GC
                    gc.collect()  # Force garbage collection

            buffer.append(item)

            # Process chunk when buffer is full
            if len(buffer) >= self.chunk_size:
                async for result in self._process_chunk(buffer, processor):
                    yield result
                buffer.clear()
                self.total_chunks += 1

        # Process remaining items
        if buffer:
            async for result in self._process_chunk(buffer, processor):
                yield result
            self.total_chunks += 1

    async def _process_chunk(self,
                           chunk: List[T],
                           processor: Callable[[T], Union[R, Awaitable[R]]]) -> AsyncIterator[R]:
        """Process a chunk of items"""
        for item in chunk:
            try:
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(item)
                else:
                    result = processor(item)

                self.total_processed += 1
                yield result

            except Exception as e:
                self.processing_errors += 1
                self.logger.error(f"Error processing item: {e}")
                # Continue processing other items

    def process_iterable(self,
                        iterable: Iterable[T],
                        processor: Callable[[T], R]) -> Iterator[R]:
        """
        Process sync iterable with memory efficiency

        Args:
            iterable: Input iterable
            processor: Function to process each item

        Yields:
            Processed results
        """
        chunk = []

        for item in iterable:
            chunk.append(item)

            if len(chunk) >= self.chunk_size:
                yield from self._process_sync_chunk(chunk, processor)
                chunk.clear()
                self.total_chunks += 1

                # Check memory and force GC if needed
                memory_stats = self.memory_monitor.get_current_stats()
                if memory_stats.rss_mb > self.memory_threshold_mb:
                    gc.collect()

        # Process remaining items
        if chunk:
            yield from self._process_sync_chunk(chunk, processor)
            self.total_chunks += 1

    def _process_sync_chunk(self, chunk: List[T], processor: Callable[[T], R]) -> Iterator[R]:
        """Process a chunk of items synchronously"""
        for item in chunk:
            try:
                result = processor(item)
                self.total_processed += 1
                yield result
            except Exception as e:
                self.processing_errors += 1
                self.logger.error(f"Error processing item: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        memory_summary = self.memory_monitor.get_summary()

        return {
            'total_processed': self.total_processed,
            'total_chunks': self.total_chunks,
            'processing_errors': self.processing_errors,
            'chunk_size': self.chunk_size,
            'memory_stats': memory_summary,
            'error_rate': (
                self.processing_errors / self.total_processed
                if self.total_processed > 0 else 0.0
            )
        }


class LazyLoader(Generic[T]):
    """
    Lazy loading with weak reference caching

    Loads data on-demand and caches with weak references
    to allow automatic cleanup when no longer referenced.
    """

    def __init__(self,
                 loader: Callable[[str], Union[T, Awaitable[T]]],
                 cache_size: int = 100,
                 ttl_seconds: Optional[float] = None):
        self.loader = loader
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds

        # Use WeakValueDictionary for automatic cleanup
        self.cache: weakref.WeakValueDictionary[str, T] = weakref.WeakValueDictionary()
        self.access_times: Dict[str, datetime] = {}
        self.load_counts: Dict[str, int] = {}

        self.logger = logging.getLogger(f"{__name__}.LazyLoader")

    async def get(self, key: str) -> T:
        """
        Get item by key, loading if necessary

        Args:
            key: Item key

        Returns:
            Loaded item
        """
        # Check cache first
        if key in self.cache:
            # Check TTL if configured
            if self.ttl_seconds is not None:
                access_time = self.access_times.get(key)
                if access_time is not None:
                    age = (datetime.now() - access_time).total_seconds()
                    if age > self.ttl_seconds:
                        # Expired, remove from cache
                        del self.cache[key]
                        del self.access_times[key]
                    else:
                        # Still valid, return cached
                        self.access_times[key] = datetime.now()
                        return self.cache[key]

        # Load the item
        if asyncio.iscoroutinefunction(self.loader):
            item = await self.loader(key)
        else:
            item = self.loader(key)

        # Cache the item
        self.cache[key] = item
        self.access_times[key] = datetime.now()
        self.load_counts[key] = self.load_counts.get(key, 0) + 1

        # Cleanup old entries if cache is too large
        if len(self.cache) > self.cache_size:
            self._cleanup_cache()

        return item

    def _cleanup_cache(self):
        """Clean up old cache entries"""
        if not self.access_times:
            return

        # Sort by access time and remove oldest
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        items_to_remove = len(sorted_items) - self.cache_size + 1

        for key, _ in sorted_items[:items_to_remove]:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]

    def invalidate(self, key: str):
        """Invalidate specific cache entry"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]

    def clear_cache(self):
        """Clear entire cache"""
        self.cache.clear()
        self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics"""
        total_loads = sum(self.load_counts.values())

        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'total_loads': total_loads,
            'unique_keys': len(self.load_counts),
            'cache_utilization_pct': (len(self.cache) / self.cache_size) * 100,
            'ttl_seconds': self.ttl_seconds
        }


class ChunkedProcessor:
    """
    Memory-efficient processor for large datasets

    Processes data in chunks to minimize memory usage
    while maintaining processing efficiency.
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 overlap_size: int = 0,
                 memory_monitor: Optional[MemoryMonitor] = None):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.memory_monitor = memory_monitor or MemoryMonitor()

        # Statistics
        self.chunks_processed = 0
        self.total_items = 0
        self.processing_time = 0.0

        self.logger = logging.getLogger(f"{__name__}.ChunkedProcessor")

    async def process_async(self,
                          data: AsyncIterable[T],
                          processor: Callable[[List[T]], Union[List[R], Awaitable[List[R]]]]) -> AsyncIterator[R]:
        """
        Process async data in chunks

        Args:
            data: Async iterable of data
            processor: Function that processes chunks

        Yields:
            Processed results
        """
        chunk = []
        overlap_buffer = []

        async for item in data:
            chunk.append(item)
            self.total_items += 1

            if len(chunk) >= self.chunk_size:
                # Process chunk
                full_chunk = overlap_buffer + chunk

                if asyncio.iscoroutinefunction(processor):
                    results = await processor(full_chunk)
                else:
                    results = processor(full_chunk)

                for result in results:
                    yield result

                # Prepare overlap for next chunk
                if self.overlap_size > 0:
                    overlap_buffer = chunk[-self.overlap_size:]
                else:
                    overlap_buffer = []

                chunk.clear()
                self.chunks_processed += 1

                # Check memory
                self.memory_monitor.get_current_stats()

        # Process final chunk
        if chunk:
            full_chunk = overlap_buffer + chunk

            if asyncio.iscoroutinefunction(processor):
                results = await processor(full_chunk)
            else:
                results = processor(full_chunk)

            for result in results:
                yield result

            self.chunks_processed += 1

    def process_sync(self,
                    data: Iterable[T],
                    processor: Callable[[List[T]], List[R]]) -> Iterator[R]:
        """
        Process sync data in chunks

        Args:
            data: Iterable of data
            processor: Function that processes chunks

        Yields:
            Processed results
        """
        chunk = []
        overlap_buffer = []

        for item in data:
            chunk.append(item)
            self.total_items += 1

            if len(chunk) >= self.chunk_size:
                # Process chunk
                full_chunk = overlap_buffer + chunk
                results = processor(full_chunk)

                for result in results:
                    yield result

                # Prepare overlap for next chunk
                if self.overlap_size > 0:
                    overlap_buffer = chunk[-self.overlap_size:]
                else:
                    overlap_buffer = []

                chunk.clear()
                self.chunks_processed += 1

                # Check memory periodically
                if self.chunks_processed % 10 == 0:
                    self.memory_monitor.get_current_stats()

        # Process final chunk
        if chunk:
            full_chunk = overlap_buffer + chunk
            results = processor(full_chunk)

            for result in results:
                yield result

            self.chunks_processed += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        memory_summary = self.memory_monitor.get_summary()

        avg_chunk_size = (
            self.total_items / self.chunks_processed
            if self.chunks_processed > 0 else 0
        )

        return {
            'chunks_processed': self.chunks_processed,
            'total_items': self.total_items,
            'avg_chunk_size': avg_chunk_size,
            'configured_chunk_size': self.chunk_size,
            'overlap_size': self.overlap_size,
            'memory_stats': memory_summary
        }


# Context managers for memory management
@asynccontextmanager
async def memory_managed_processing(memory_threshold_mb: float = 1000.0):
    """
    Context manager for memory-managed processing

    Args:
        memory_threshold_mb: Memory threshold for alerts
    """
    monitor = MemoryMonitor(alert_threshold_mb=memory_threshold_mb)
    initial_stats = monitor.get_current_stats()

    try:
        yield monitor
    finally:
        final_stats = monitor.get_current_stats()
        memory_delta = final_stats.rss_mb - initial_stats.rss_mb

        logging.getLogger(__name__).info(
            f"Memory usage delta: {memory_delta:+.1f}MB "
            f"(from {initial_stats.rss_mb:.1f}MB to {final_stats.rss_mb:.1f}MB)"
        )

        # Force cleanup
        gc.collect()


@contextmanager
def circular_buffer_manager(maxsize: int):
    """
    Context manager for circular buffer

    Args:
        maxsize: Maximum buffer size
    """
    buffer = CircularBuffer(maxsize)
    try:
        yield buffer
    finally:
        buffer.clear()


# Convenience functions
def create_stream_processor(chunk_size: int = 1000,
                          memory_threshold_mb: float = 500.0) -> StreamProcessor:
    """Create stream processor with optimal defaults"""
    return StreamProcessor(
        chunk_size=chunk_size,
        memory_threshold_mb=memory_threshold_mb
    )


def create_lazy_loader(loader: Callable[[str], T],
                      cache_size: int = 100,
                      ttl_minutes: Optional[float] = None) -> LazyLoader[T]:
    """Create lazy loader with optimal defaults"""
    ttl_seconds = ttl_minutes * 60 if ttl_minutes is not None else None
    return LazyLoader(
        loader=loader,
        cache_size=cache_size,
        ttl_seconds=ttl_seconds
    )