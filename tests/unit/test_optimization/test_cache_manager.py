"""
Tests for Caching Layer with Redis.

Following TDD methodology: Red -> Green -> Refactor
Tests for cache management, TTL handling, Redis integration, and performance optimization.
"""

import pytest
import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.optimization.cache_manager import (
    CacheManager,
    CacheConfig,
    CacheEntry,
    CacheStats,
    CacheError
)


class TestCacheConfig:
    """Test suite for CacheConfig class."""

    def test_should_initialize_with_default_values(self):
        """Test that CacheConfig initializes with default values."""
        config = CacheConfig()

        assert config.redis_host == 'localhost'
        assert config.redis_port == 6379
        assert config.redis_db == 0
        assert config.default_ttl == 300
        assert config.max_memory_mb == 100
        assert config.enable_compression is True
        assert config.enable_encryption is False

    def test_should_initialize_with_custom_values(self):
        """Test that CacheConfig accepts custom values."""
        config = CacheConfig(
            redis_host='cache.example.com',
            redis_port=6380,
            default_ttl=600,
            enable_compression=False
        )

        assert config.redis_host == 'cache.example.com'
        assert config.redis_port == 6380
        assert config.default_ttl == 600
        assert config.enable_compression is False

    def test_should_validate_configuration_values(self):
        """Test that CacheConfig validates configuration values."""
        # Test invalid port
        with pytest.raises(ValueError, match="redis_port must be between 1 and 65535"):
            CacheConfig(redis_port=0)

        with pytest.raises(ValueError, match="redis_port must be between 1 and 65535"):
            CacheConfig(redis_port=70000)

        # Test invalid TTL
        with pytest.raises(ValueError, match="default_ttl must be positive"):
            CacheConfig(default_ttl=0)

        # Test invalid memory limit
        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            CacheConfig(max_memory_mb=0)

    def test_should_convert_to_dictionary(self):
        """Test that CacheConfig can be converted to dictionary."""
        config = CacheConfig(redis_host='test.com', redis_port=6380)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['redis_host'] == 'test.com'
        assert config_dict['redis_port'] == 6380
        assert 'default_ttl' in config_dict


class TestCacheEntry:
    """Test suite for CacheEntry class."""

    def test_should_initialize_with_data(self):
        """Test that CacheEntry initializes with data."""
        data = {'key': 'value', 'number': 42}
        entry = CacheEntry(data=data, ttl=300)

        assert entry.data == data
        assert entry.ttl == 300
        assert entry.created_at is not None
        assert entry.access_count == 0

    def test_should_check_expiration(self):
        """Test that CacheEntry can check if it's expired."""
        # Non-expired entry
        entry = CacheEntry(data={'test': 'data'}, ttl=60)
        assert entry.is_expired() is False

        # Expired entry
        entry.created_at = datetime.utcnow() - timedelta(seconds=120)
        assert entry.is_expired() is True

    def test_should_track_access_count(self):
        """Test that CacheEntry tracks access count."""
        entry = CacheEntry(data={'test': 'data'}, ttl=300)

        assert entry.access_count == 0

        entry.mark_accessed()
        assert entry.access_count == 1

        entry.mark_accessed()
        assert entry.access_count == 2

    def test_should_calculate_size(self):
        """Test that CacheEntry can calculate its size."""
        small_data = {'key': 'value'}
        large_data = {'key': 'x' * 1000}

        small_entry = CacheEntry(data=small_data, ttl=300)
        large_entry = CacheEntry(data=large_data, ttl=300)

        small_size = small_entry.calculate_size()
        large_size = large_entry.calculate_size()

        assert isinstance(small_size, int)
        assert isinstance(large_size, int)
        assert large_size > small_size

    def test_should_convert_to_dictionary(self):
        """Test that CacheEntry can be converted to dictionary."""
        data = {'test': 'data'}
        entry = CacheEntry(data=data, ttl=300)
        entry.mark_accessed()

        entry_dict = entry.to_dict()

        assert isinstance(entry_dict, dict)
        assert entry_dict['data'] == data
        assert entry_dict['ttl'] == 300
        assert entry_dict['access_count'] == 1
        assert 'created_at' in entry_dict


class TestCacheStats:
    """Test suite for CacheStats class."""

    def test_should_initialize_with_default_values(self):
        """Test that CacheStats initializes with default values."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.deletes == 0
        assert stats.total_memory_mb == 0.0
        assert stats.entry_count == 0

    def test_should_calculate_hit_rate(self):
        """Test that CacheStats can calculate hit rate."""
        stats = CacheStats()

        # No requests yet
        assert stats.get_hit_rate() == 0.0

        # Add some hits and misses
        stats.hits = 80
        stats.misses = 20

        assert stats.get_hit_rate() == 0.8

    def test_should_convert_to_dictionary(self):
        """Test that CacheStats can be converted to dictionary."""
        stats = CacheStats()
        stats.hits = 100
        stats.misses = 25

        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert stats_dict['hits'] == 100
        assert stats_dict['misses'] == 25
        assert stats_dict['hit_rate'] == 0.8


class TestCacheManager:
    """Test suite for CacheManager class."""

    @pytest.fixture
    def cache_config(self):
        """Create cache configuration for testing."""
        return CacheConfig(
            redis_host='localhost',
            redis_port=6379,
            default_ttl=300,
            max_memory_mb=50
        )

    @pytest.fixture
    def cache_manager(self, cache_config):
        """Create CacheManager instance for testing."""
        return CacheManager(cache_config)

    def test_should_initialize_with_configuration(self, cache_manager, cache_config):
        """Test that CacheManager initializes with configuration."""
        assert cache_manager.config == cache_config
        assert cache_manager.redis_client is None
        assert cache_manager.is_connected is False
        assert isinstance(cache_manager.stats, CacheStats)

    @pytest.mark.asyncio
    async def test_should_connect_to_redis(self, cache_manager):
        """Test that CacheManager can connect to Redis."""
        # Mock Redis connection
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True

            await cache_manager.connect()

            assert cache_manager.redis_client is not None
            assert cache_manager.is_connected is True
            mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_handle_connection_failure(self, cache_manager):
        """Test that CacheManager handles Redis connection failures."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping.side_effect = ConnectionError("Connection failed")

            with pytest.raises(CacheError, match="Failed to connect to Redis"):
                await cache_manager.connect()

            assert cache_manager.is_connected is False

    @pytest.mark.asyncio
    async def test_should_disconnect_from_redis(self, cache_manager):
        """Test that CacheManager can disconnect from Redis."""
        # Mock connected state
        cache_manager.redis_client = AsyncMock()
        cache_manager.is_connected = True

        await cache_manager.disconnect()

        assert cache_manager.redis_client is None
        assert cache_manager.is_connected is False

    @pytest.mark.asyncio
    async def test_should_set_and_get_cache_entries(self, cache_manager):
        """Test that CacheManager can set and get cache entries."""
        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        test_data = {'message': 'Hello, Cache!', 'number': 42}
        key = 'test_key'

        # Test set operation
        mock_client.setex.return_value = True
        result = await cache_manager.set(key, test_data, ttl=600)

        assert result is True
        mock_client.setex.assert_called_once()
        assert cache_manager.stats.sets == 1

        # Test get operation
        serialized_data = json.dumps(test_data)
        mock_client.get.return_value = serialized_data.encode()

        retrieved_data = await cache_manager.get(key)

        assert retrieved_data == test_data
        mock_client.get.assert_called_once_with(key)
        assert cache_manager.stats.hits == 1

    @pytest.mark.asyncio
    async def test_should_handle_cache_miss(self, cache_manager):
        """Test that CacheManager handles cache misses correctly."""
        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        # Test cache miss
        mock_client.get.return_value = None

        result = await cache_manager.get('nonexistent_key')

        assert result is None
        assert cache_manager.stats.misses == 1

    @pytest.mark.asyncio
    async def test_should_delete_cache_entries(self, cache_manager):
        """Test that CacheManager can delete cache entries."""
        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        mock_client.delete.return_value = 1

        result = await cache_manager.delete('test_key')

        assert result is True
        mock_client.delete.assert_called_once_with('test_key')
        assert cache_manager.stats.deletes == 1

    @pytest.mark.asyncio
    async def test_should_check_key_existence(self, cache_manager):
        """Test that CacheManager can check if keys exist."""
        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        # Key exists
        mock_client.exists.return_value = 1
        exists = await cache_manager.exists('existing_key')
        assert exists is True

        # Key doesn't exist
        mock_client.exists.return_value = 0
        exists = await cache_manager.exists('nonexistent_key')
        assert exists is False

    @pytest.mark.asyncio
    async def test_should_set_expiration_for_keys(self, cache_manager):
        """Test that CacheManager can set expiration for keys."""
        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        mock_client.expire.return_value = True

        result = await cache_manager.expire('test_key', 300)

        assert result is True
        mock_client.expire.assert_called_once_with('test_key', 300)

    @pytest.mark.asyncio
    async def test_should_get_keys_by_pattern(self, cache_manager):
        """Test that CacheManager can get keys by pattern."""
        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        mock_keys = [b'user:1', b'user:2', b'user:3']
        mock_client.keys.return_value = mock_keys

        keys = await cache_manager.get_keys('user:*')

        expected_keys = ['user:1', 'user:2', 'user:3']
        assert keys == expected_keys
        mock_client.keys.assert_called_once_with('user:*')

    @pytest.mark.asyncio
    async def test_should_clear_all_cache(self, cache_manager):
        """Test that CacheManager can clear all cache."""
        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        mock_client.flushdb.return_value = True

        result = await cache_manager.clear()

        assert result is True
        mock_client.flushdb.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_handle_compression(self, cache_manager):
        """Test that CacheManager handles data compression."""
        cache_manager.config.enable_compression = True

        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        # Large data that should be compressed
        large_data = {'data': 'x' * 1000}

        # Mock successful set
        mock_client.setex.return_value = True

        await cache_manager.set('large_key', large_data)

        # Verify that setex was called (compression happens internally)
        mock_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_handle_batch_operations(self, cache_manager):
        """Test that CacheManager can handle batch operations."""
        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        # Mock pipeline with proper methods
        mock_pipeline = MagicMock()
        mock_pipeline.setex = MagicMock()
        mock_pipeline.set = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[True, True, True])
        mock_client.pipeline = MagicMock(return_value=mock_pipeline)

        # Test batch set
        data_dict = {
            'key1': {'value': 1},
            'key2': {'value': 2},
            'key3': {'value': 3}
        }

        result = await cache_manager.set_many(data_dict, ttl=300)

        assert result is True
        mock_client.pipeline.assert_called()
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_get_cache_statistics(self, cache_manager):
        """Test that CacheManager provides cache statistics."""
        # Mock some operations
        cache_manager.stats.hits = 80
        cache_manager.stats.misses = 20
        cache_manager.stats.sets = 50
        cache_manager.stats.deletes = 10

        stats = cache_manager.get_stats()

        assert isinstance(stats, CacheStats)
        assert stats.hits == 80
        assert stats.misses == 20
        assert stats.get_hit_rate() == 0.8

    @pytest.mark.asyncio
    async def test_should_implement_cache_warming(self, cache_manager):
        """Test that CacheManager can warm up cache with data."""
        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        # Mock pipeline for batch operations
        mock_pipeline = MagicMock()
        mock_pipeline.setex = MagicMock()
        mock_pipeline.set = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[True] * 3)
        mock_client.pipeline = MagicMock(return_value=mock_pipeline)

        warm_data = {
            'warm_key1': {'data': 'value1'},
            'warm_key2': {'data': 'value2'},
            'warm_key3': {'data': 'value3'}
        }

        result = await cache_manager.warm_cache(warm_data)

        assert result is True
        mock_client.pipeline.assert_called()

    @pytest.mark.asyncio
    async def test_should_monitor_memory_usage(self, cache_manager):
        """Test that CacheManager monitors memory usage."""
        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        # Mock memory info
        mock_client.info.return_value = {
            'used_memory': 10485760,  # 10MB in bytes
            'maxmemory': 52428800     # 50MB in bytes
        }

        memory_info = await cache_manager.get_memory_usage()

        assert isinstance(memory_info, dict)
        assert 'used_memory_mb' in memory_info
        assert 'max_memory_mb' in memory_info
        assert 'usage_percentage' in memory_info

        # Should be approximately 10MB used out of 50MB
        assert abs(memory_info['used_memory_mb'] - 10.0) < 1.0
        assert abs(memory_info['usage_percentage'] - 20.0) < 1.0

    @pytest.mark.asyncio
    async def test_should_implement_cache_eviction(self, cache_manager):
        """Test that CacheManager implements cache eviction strategies."""
        # Mock Redis client
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        # Mock keys for eviction
        mock_keys = [b'old_key1', b'old_key2', b'old_key3']
        mock_client.keys.return_value = mock_keys
        mock_client.delete.return_value = 3

        result = await cache_manager.evict_expired_keys()

        assert isinstance(result, int)
        assert result >= 0

    @pytest.mark.asyncio
    async def test_should_handle_connection_errors_gracefully(self, cache_manager):
        """Test that CacheManager handles connection errors gracefully."""
        # Mock Redis client with connection error
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        mock_client.get.side_effect = ConnectionError("Connection lost")

        # Should not raise error, but return None and set connection status
        result = await cache_manager.get('test_key')

        assert result is None
        assert cache_manager.is_connected is False

    @pytest.mark.asyncio
    async def test_should_implement_cache_decorators(self, cache_manager):
        """Test that CacheManager provides caching decorators."""
        # This test checks if the cache manager can be used as a decorator
        cache_hits = 0

        @cache_manager.cached(ttl=300, key_prefix='test_func')
        async def expensive_function(x, y):
            nonlocal cache_hits
            cache_hits += 1
            return x + y

        # Mock Redis operations
        mock_client = AsyncMock()
        cache_manager.redis_client = mock_client
        cache_manager.is_connected = True

        # First call - cache miss
        mock_client.get.return_value = None
        mock_client.setex.return_value = True

        result1 = await expensive_function(5, 3)
        assert result1 == 8
        assert cache_hits == 1

        # Second call - cache hit
        cached_result = json.dumps(8).encode()
        mock_client.get.return_value = cached_result

        result2 = await expensive_function(5, 3)
        assert result2 == 8
        assert cache_hits == 1  # Should not increment

    def test_should_export_cache_configuration(self, cache_manager):
        """Test that CacheManager can export its configuration."""
        config_dict = cache_manager.export_config()

        assert isinstance(config_dict, dict)
        assert 'redis_host' in config_dict
        assert 'redis_port' in config_dict
        assert 'default_ttl' in config_dict
        assert 'max_memory_mb' in config_dict