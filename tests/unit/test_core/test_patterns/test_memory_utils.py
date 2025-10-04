# tests/unit/test_core/test_patterns/test_memory_utils.py
"""
Test suite for memory utility patterns (Phase 8 optimizations)
"""

import pytest
import asyncio
import gc
from typing import List, AsyncIterator, Iterator
from unittest.mock import patch, MagicMock

from src.core.patterns.memory_utils import (
    CircularBuffer, StreamProcessor, LazyLoader, ChunkedProcessor,
    MemoryMonitor, MemoryStats, create_stream_processor, create_lazy_loader,
    memory_managed_processing, circular_buffer_manager
)


class TestCircularBuffer:
    """Test CircularBuffer functionality"""

    def test_should_initialize_correctly(self):
        """Test buffer initialization"""
        buffer = CircularBuffer(maxsize=5)
        assert len(buffer) == 0
        assert buffer.maxsize == 5
        assert not buffer.is_full
        assert buffer.utilization == 0.0

    def test_should_append_items_until_full(self):
        """Test appending items until buffer is full"""
        buffer = CircularBuffer(maxsize=3)

        # Add items without overflow
        removed = buffer.append("item1")
        assert removed is None
        assert len(buffer) == 1

        removed = buffer.append("item2")
        assert removed is None
        assert len(buffer) == 2

        removed = buffer.append("item3")
        assert removed is None
        assert len(buffer) == 3
        assert buffer.is_full

    def test_should_overwrite_oldest_when_full(self):
        """Test circular buffer overflow behavior"""
        buffer = CircularBuffer(maxsize=3)

        # Fill buffer
        buffer.append("item1")
        buffer.append("item2")
        buffer.append("item3")

        # Add one more - should remove oldest
        removed = buffer.append("item4")
        assert removed == "item1"
        assert len(buffer) == 3
        assert list(buffer) == ["item2", "item3", "item4"]

    def test_should_get_recent_items(self):
        """Test getting recent items"""
        buffer = CircularBuffer(maxsize=5)
        for i in range(7):  # More than maxsize
            buffer.append(f"item{i}")

        recent = buffer.get_recent(3)
        assert recent == ["item4", "item5", "item6"]

        # Test edge cases
        assert buffer.get_recent(0) == []
        assert buffer.get_recent(10) == ["item2", "item3", "item4", "item5", "item6"]

    def test_should_get_oldest_items(self):
        """Test getting oldest items"""
        buffer = CircularBuffer(maxsize=5)
        for i in range(3):
            buffer.append(f"item{i}")

        oldest = buffer.get_oldest(2)
        assert oldest == ["item0", "item1"]

    def test_should_extend_with_multiple_items(self):
        """Test extending buffer with multiple items"""
        buffer = CircularBuffer(maxsize=3)
        items = ["a", "b", "c", "d", "e"]

        removed_items = buffer.extend(items)
        assert len(removed_items) == 2  # "a" and "b" were removed
        assert list(buffer) == ["c", "d", "e"]

    def test_should_track_statistics(self):
        """Test buffer statistics"""
        buffer = CircularBuffer(maxsize=5)
        for i in range(7):
            buffer.append(f"item{i}")

        stats = buffer.get_stats()
        assert stats['current_size'] == 5
        assert stats['max_size'] == 5
        assert stats['utilization_pct'] == 100.0
        assert stats['total_added'] == 7
        assert stats['is_full'] == True

    def test_should_clear_buffer(self):
        """Test clearing buffer"""
        buffer = CircularBuffer(maxsize=3)
        buffer.extend(["a", "b", "c"])

        buffer.clear()
        assert len(buffer) == 0
        assert not buffer.is_full


class TestStreamProcessor:
    """Test StreamProcessor functionality"""

    @pytest.fixture
    def stream_processor(self):
        return StreamProcessor(chunk_size=3, memory_threshold_mb=100.0)

    @pytest.mark.asyncio
    async def test_should_process_async_stream(self, stream_processor):
        """Test async stream processing"""
        async def data_generator():
            for i in range(10):
                yield i

        async def processor(item: int) -> str:
            return f"processed_{item}"

        results = []
        async for result in stream_processor.process_stream(
            data_generator(), processor, enable_backpressure=False
        ):
            results.append(result)

        assert len(results) == 10
        assert results == [f"processed_{i}" for i in range(10)]

    @pytest.mark.asyncio
    async def test_should_handle_async_processor(self, stream_processor):
        """Test with async processor function"""
        async def data_generator():
            for i in range(5):
                yield i

        async def async_processor(item: int) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            return f"async_processed_{item}"

        results = []
        async for result in stream_processor.process_stream(
            data_generator(), async_processor
        ):
            results.append(result)

        assert len(results) == 5
        assert all("async_processed_" in result for result in results)

    def test_should_process_sync_iterable(self, stream_processor):
        """Test sync iterable processing"""
        data = list(range(10))

        def processor(item: int) -> str:
            return f"sync_processed_{item}"

        results = list(stream_processor.process_iterable(data, processor))

        assert len(results) == 10
        assert results == [f"sync_processed_{i}" for i in range(10)]

    @pytest.mark.asyncio
    async def test_should_handle_processing_errors(self, stream_processor):
        """Test error handling in stream processing"""
        async def data_generator():
            for i in range(5):
                yield i

        async def failing_processor(item: int) -> str:
            if item == 2:
                raise ValueError("Processing failed")
            return f"processed_{item}"

        results = []
        async for result in stream_processor.process_stream(
            data_generator(), failing_processor
        ):
            results.append(result)

        # Should continue processing despite error
        assert len(results) == 4  # 5 items - 1 failed
        stats = stream_processor.get_stats()
        assert stats['processing_errors'] == 1

    def test_should_track_processing_stats(self, stream_processor):
        """Test processing statistics"""
        data = list(range(20))

        def processor(item: int) -> str:
            return f"item_{item}"

        list(stream_processor.process_iterable(data, processor))

        stats = stream_processor.get_stats()
        assert stats['total_processed'] == 20
        assert stats['total_chunks'] > 0
        assert stats['processing_errors'] == 0
        assert 'memory_stats' in stats


class TestLazyLoader:
    """Test LazyLoader functionality"""

    @pytest.fixture
    def simple_loader(self):
        def load_function(key: str) -> str:
            return f"loaded_{key}"

        return LazyLoader(loader=load_function, cache_size=3)

    @pytest.mark.asyncio
    async def test_should_load_items_on_demand(self, simple_loader):
        """Test lazy loading functionality"""
        result = await simple_loader.get("test_key")
        assert result == "loaded_test_key"

    @pytest.mark.asyncio
    async def test_should_cache_loaded_items(self, simple_loader):
        """Test caching behavior"""
        # Load item twice
        result1 = await simple_loader.get("test_key")
        result2 = await simple_loader.get("test_key")

        assert result1 == result2 == "loaded_test_key"

        stats = simple_loader.get_stats()
        assert stats['total_loads'] == 1  # Should only load once due to caching

    @pytest.mark.asyncio
    async def test_should_handle_cache_overflow(self, simple_loader):
        """Test cache size limits"""
        # Load more items than cache size
        for i in range(5):  # Cache size is 3
            await simple_loader.get(f"key_{i}")

        stats = simple_loader.get_stats()
        assert stats['cache_size'] <= 3  # Should not exceed cache size

    @pytest.mark.asyncio
    async def test_should_handle_async_loader(self):
        """Test with async loader function"""
        async def async_loader(key: str) -> str:
            await asyncio.sleep(0.01)
            return f"async_loaded_{key}"

        loader = LazyLoader(loader=async_loader, cache_size=5)
        result = await loader.get("async_key")
        assert result == "async_loaded_async_key"

    @pytest.mark.asyncio
    async def test_should_handle_ttl_expiration(self):
        """Test TTL (time-to-live) functionality"""
        def load_function(key: str) -> str:
            return f"loaded_{key}"

        loader = LazyLoader(loader=load_function, cache_size=5, ttl_seconds=0.1)

        # Load item
        result1 = await loader.get("ttl_key")

        # Wait for TTL to expire
        await asyncio.sleep(0.2)

        # Load again - should reload due to TTL expiration
        result2 = await loader.get("ttl_key")

        assert result1 == result2
        stats = loader.get_stats()
        assert stats['total_loads'] == 2  # Should have loaded twice

    def test_should_invalidate_cache_entries(self, simple_loader):
        """Test cache invalidation"""
        # This is a sync test since invalidate is sync
        simple_loader.cache["test_key"] = "cached_value"
        simple_loader.access_times["test_key"] = simple_loader.access_times.get("test_key", None)

        simple_loader.invalidate("test_key")
        assert "test_key" not in simple_loader.cache

    def test_should_clear_entire_cache(self, simple_loader):
        """Test clearing entire cache"""
        # Add some items to cache manually for testing
        simple_loader.cache["key1"] = "value1"
        simple_loader.cache["key2"] = "value2"

        simple_loader.clear_cache()
        assert len(simple_loader.cache) == 0


class TestChunkedProcessor:
    """Test ChunkedProcessor functionality"""

    @pytest.fixture
    def chunked_processor(self):
        return ChunkedProcessor(chunk_size=3, overlap_size=1)

    @pytest.mark.asyncio
    async def test_should_process_async_data_in_chunks(self, chunked_processor):
        """Test async chunked processing"""
        async def data_generator():
            for i in range(10):
                yield i

        async def chunk_processor(chunk: List[int]) -> List[str]:
            return [f"chunk_item_{item}" for item in chunk]

        results = []
        async for result in chunked_processor.process_async(
            data_generator(), chunk_processor
        ):
            results.append(result)

        assert len(results) > 0
        assert all("chunk_item_" in result for result in results)

    def test_should_process_sync_data_in_chunks(self, chunked_processor):
        """Test sync chunked processing"""
        data = list(range(10))

        def chunk_processor(chunk: List[int]) -> List[str]:
            return [f"chunk_{item}" for item in chunk]

        results = list(chunked_processor.process_sync(data, chunk_processor))

        assert len(results) > 0
        assert all("chunk_" in result for result in results)

    def test_should_handle_overlap_correctly(self):
        """Test chunk overlap functionality"""
        processor = ChunkedProcessor(chunk_size=3, overlap_size=1)

        # Test that overlap is working by checking the processing stats
        data = list(range(10))

        def chunk_processor(chunk: List[int]) -> List[str]:
            return [f"item_{item}" for item in chunk]

        results = list(processor.process_sync(data, chunk_processor))

        stats = processor.get_stats()
        assert stats['chunks_processed'] > 0
        assert stats['total_items'] == 10
        assert stats['overlap_size'] == 1

    def test_should_track_processing_statistics(self, chunked_processor):
        """Test processing statistics"""
        data = list(range(15))

        def processor(chunk: List[int]) -> List[int]:
            return [item * 2 for item in chunk]

        list(chunked_processor.process_sync(data, processor))

        stats = chunked_processor.get_stats()
        assert stats['chunks_processed'] > 0
        assert stats['total_items'] == 15
        assert stats['configured_chunk_size'] == 3
        assert 'memory_stats' in stats


class TestMemoryMonitor:
    """Test MemoryMonitor functionality"""

    @pytest.fixture
    def memory_monitor(self):
        return MemoryMonitor(alert_threshold_mb=100.0)

    def test_should_get_current_memory_stats(self, memory_monitor):
        """Test getting current memory statistics"""
        stats = memory_monitor.get_current_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.rss_mb > 0
        assert stats.vms_mb > 0
        assert stats.percent >= 0
        assert stats.available_mb > 0

    def test_should_track_peak_usage(self, memory_monitor):
        """Test peak usage tracking"""
        # Get initial stats
        initial_stats = memory_monitor.get_current_stats()
        initial_peak = memory_monitor.peak_usage_mb

        # Force some memory allocation
        large_list = [i for i in range(10000)]  # Small allocation for testing

        # Get stats again
        new_stats = memory_monitor.get_current_stats()

        # Peak should be updated if current usage is higher
        assert memory_monitor.peak_usage_mb >= initial_peak

    def test_should_generate_alerts_on_threshold_breach(self, memory_monitor):
        """Test alert generation on threshold breach"""
        # Set very low threshold to trigger alert
        monitor = MemoryMonitor(alert_threshold_mb=1.0)  # Very low threshold

        stats = monitor.get_current_stats()

        # Should have generated an alert if current usage > 1MB (very likely)
        if stats.rss_mb > 1.0:
            assert monitor.alert_count > 0

    def test_should_provide_memory_summary(self, memory_monitor):
        """Test memory usage summary"""
        # Generate some history
        for _ in range(5):
            memory_monitor.get_current_stats()

        summary = memory_monitor.get_summary()

        assert 'current_rss_mb' in summary
        assert 'peak_usage_mb' in summary
        assert 'avg_rss_mb' in summary
        assert 'min_rss_mb' in summary
        assert 'max_rss_mb' in summary
        assert 'alert_count' in summary


class TestContextManagers:
    """Test context managers"""

    @pytest.mark.asyncio
    async def test_memory_managed_processing_context(self):
        """Test memory managed processing context"""
        async with memory_managed_processing(memory_threshold_mb=1000.0) as monitor:
            assert isinstance(monitor, MemoryMonitor)

            # Perform some work
            initial_stats = monitor.get_current_stats()
            assert initial_stats.rss_mb > 0

    def test_circular_buffer_manager_context(self):
        """Test circular buffer manager context"""
        with circular_buffer_manager(maxsize=5) as buffer:
            assert isinstance(buffer, CircularBuffer)
            assert buffer.maxsize == 5

            # Use buffer
            buffer.append("test_item")
            assert len(buffer) == 1

        # Buffer should be cleared after context exit
        assert len(buffer) == 0


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_stream_processor(self):
        """Test stream processor creation"""
        processor = create_stream_processor(chunk_size=100, memory_threshold_mb=200.0)

        assert isinstance(processor, StreamProcessor)
        assert processor.chunk_size == 100
        assert processor.memory_threshold_mb == 200.0

    @pytest.mark.asyncio
    async def test_create_lazy_loader(self):
        """Test lazy loader creation"""
        def loader_func(key: str) -> str:
            return f"loaded_{key}"

        loader = create_lazy_loader(
            loader=loader_func,
            cache_size=10,
            ttl_minutes=5.0
        )

        assert isinstance(loader, LazyLoader)
        assert loader.cache_size == 10
        assert loader.ttl_seconds == 300.0  # 5 minutes in seconds

        # Test functionality
        result = await loader.get("test")
        assert result == "loaded_test"


class TestMemoryEfficiencyBenchmarks:
    """Memory efficiency benchmark tests"""

    def test_circular_buffer_memory_usage(self):
        """Test that circular buffer maintains constant memory usage"""
        buffer = CircularBuffer(maxsize=1000)

        # Add many items (more than maxsize)
        for i in range(5000):
            buffer.append(f"item_{i}")

        # Buffer should still be at maxsize
        assert len(buffer) == 1000
        assert buffer.total_added == 5000

    @pytest.mark.asyncio
    async def test_stream_processor_memory_efficiency(self):
        """Test stream processor memory efficiency"""
        processor = StreamProcessor(chunk_size=100, memory_threshold_mb=50.0)

        async def large_data_generator():
            for i in range(10000):  # Large dataset
                yield f"data_item_{i}"

        async def simple_processor(item: str) -> str:
            return item.upper()

        processed_count = 0
        async for result in processor.process_stream(
            large_data_generator(), simple_processor
        ):
            processed_count += 1

        assert processed_count == 10000

        # Check that processing was done in chunks
        stats = processor.get_stats()
        assert stats['total_chunks'] > 1
        assert stats['total_processed'] == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])