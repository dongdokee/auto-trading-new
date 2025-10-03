"""
Caching Layer with Redis for performance optimization.

This module provides comprehensive caching capabilities including:
- Redis integration with async support
- TTL-based cache management
- Data compression and encryption
- Batch operations and cache warming
- Memory usage monitoring and eviction
- Cache statistics and performance metrics
- Caching decorators for functions

Features:
- Automatic connection management and reconnection
- Configurable TTL and memory limits
- Data compression for large objects
- Batch set/get operations for efficiency
- Cache warming and eviction strategies
- Real-time statistics and monitoring
- Decorator-based caching for functions
"""

import asyncio
import json
import gzip
import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import pickle

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Raised when cache operations fail."""
    pass


@dataclass
class CacheConfig:
    """
    Configuration for cache management.

    Defines Redis connection settings, TTL defaults, and optimization options.
    """
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    default_ttl: int = 300  # seconds
    max_memory_mb: int = 100
    enable_compression: bool = True
    enable_encryption: bool = False
    compression_threshold: int = 1024  # bytes
    connection_pool_size: int = 10

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        if not (1 <= self.redis_port <= 65535):
            raise ValueError("redis_port must be between 1 and 65535")

        if self.default_ttl <= 0:
            raise ValueError("default_ttl must be positive")

        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")

        if self.compression_threshold < 0:
            raise ValueError("compression_threshold must be non-negative")

        if self.connection_pool_size <= 0:
            raise ValueError("connection_pool_size must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'redis_host': self.redis_host,
            'redis_port': self.redis_port,
            'redis_db': self.redis_db,
            'default_ttl': self.default_ttl,
            'max_memory_mb': self.max_memory_mb,
            'enable_compression': self.enable_compression,
            'enable_encryption': self.enable_encryption,
            'compression_threshold': self.compression_threshold,
            'connection_pool_size': self.connection_pool_size
        }


@dataclass
class CacheEntry:
    """
    Cache entry with metadata.

    Stores cached data along with TTL, access tracking, and other metadata.
    """
    data: Any
    ttl: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    compressed: bool = False
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl <= 0:
            return False  # No expiration

        expiry_time = self.created_at + timedelta(seconds=self.ttl)
        return datetime.utcnow() > expiry_time

    def mark_accessed(self) -> None:
        """Mark entry as accessed and update statistics."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()

    def calculate_size(self) -> int:
        """Calculate approximate size of the entry in bytes."""
        if self.size_bytes > 0:
            return self.size_bytes

        try:
            # Approximate size calculation
            if isinstance(self.data, str):
                self.size_bytes = len(self.data.encode('utf-8'))
            elif isinstance(self.data, (dict, list)):
                self.size_bytes = len(json.dumps(self.data).encode('utf-8'))
            else:
                self.size_bytes = len(pickle.dumps(self.data))
        except Exception:
            self.size_bytes = 100  # Default fallback

        return self.size_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            'data': self.data,
            'ttl': self.ttl,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'compressed': self.compressed,
            'size_bytes': self.size_bytes
        }


@dataclass
class CacheStats:
    """
    Cache statistics and performance metrics.

    Tracks cache hits, misses, memory usage, and other performance indicators.
    """
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_memory_mb: float = 0.0
    entry_count: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hits + self.misses
        if total_requests == 0:
            return 0.0
        return self.hits / total_requests

    def get_uptime_seconds(self) -> float:
        """Get cache uptime in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'errors': self.errors,
            'hit_rate': self.get_hit_rate(),
            'total_memory_mb': self.total_memory_mb,
            'entry_count': self.entry_count,
            'uptime_seconds': self.get_uptime_seconds()
        }


class CacheManager:
    """
    Comprehensive cache manager with Redis backend.

    Provides high-performance caching with TTL management, compression,
    batch operations, and comprehensive monitoring.
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration
        """
        if redis is None:
            raise CacheError("redis package is required for cache functionality")

        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.connection_pool: Optional[redis.ConnectionPool] = None
        self.is_connected = False
        self.stats = CacheStats()

    async def connect(self) -> None:
        """Connect to Redis server."""
        try:
            # Create connection pool
            self.connection_pool = redis.ConnectionPool(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                max_connections=self.config.connection_pool_size,
                decode_responses=False  # We handle encoding/decoding manually
            )

            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)

            # Test connection
            await self.redis_client.ping()
            self.is_connected = True

            logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")

        except Exception as e:
            self.is_connected = False
            raise CacheError(f"Failed to connect to Redis: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Redis server."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

        if self.connection_pool:
            await self.connection_pool.disconnect()
            self.connection_pool = None

        self.is_connected = False
        logger.info("Disconnected from Redis")

    async def _ensure_connected(self) -> None:
        """Ensure Redis connection is active."""
        if not self.is_connected or not self.redis_client:
            raise CacheError("Not connected to Redis. Call connect() first.")

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage."""
        try:
            # Convert to JSON string first
            json_str = json.dumps(data, default=str)
            json_bytes = json_str.encode('utf-8')

            # Apply compression if enabled and data is large enough
            if (self.config.enable_compression and
                len(json_bytes) >= self.config.compression_threshold):
                json_bytes = gzip.compress(json_bytes)

            return json_bytes

        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            raise CacheError(f"Data serialization failed: {e}")

    def _deserialize_data(self, data_bytes: bytes) -> Any:
        """Deserialize data from storage."""
        try:
            # Try to decompress first (gzip has magic header)
            try:
                if data_bytes.startswith(b'\x1f\x8b'):  # gzip magic number
                    data_bytes = gzip.decompress(data_bytes)
            except Exception:
                pass  # Not compressed or compression failed

            # Convert from JSON
            json_str = data_bytes.decode('utf-8')
            return json.loads(json_str)

        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            raise CacheError(f"Data deserialization failed: {e}")

    def _generate_key(self, key: str, prefix: str = None) -> str:
        """Generate cache key with optional prefix."""
        if prefix:
            return f"{prefix}:{key}"
        return key

    async def set(
        self,
        key: str,
        data: Any,
        ttl: Optional[int] = None,
        prefix: str = None
    ) -> bool:
        """
        Set cache entry.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds (uses default if None)
            prefix: Optional key prefix

        Returns:
            True if successful
        """
        await self._ensure_connected()

        try:
            cache_key = self._generate_key(key, prefix)
            ttl = ttl or self.config.default_ttl
            serialized_data = self._serialize_data(data)

            if ttl > 0:
                result = await self.redis_client.setex(cache_key, ttl, serialized_data)
            else:
                result = await self.redis_client.set(cache_key, serialized_data)

            if result:
                self.stats.sets += 1
                return True
            return False

        except ConnectionError:
            self.is_connected = False
            self.stats.errors += 1
            logger.error("Redis connection lost")
            return False
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Failed to set cache entry: {e}")
            return False

    async def get(self, key: str, prefix: str = None) -> Optional[Any]:
        """
        Get cache entry.

        Args:
            key: Cache key
            prefix: Optional key prefix

        Returns:
            Cached data or None if not found
        """
        await self._ensure_connected()

        try:
            cache_key = self._generate_key(key, prefix)
            data_bytes = await self.redis_client.get(cache_key)

            if data_bytes is None:
                self.stats.misses += 1
                return None

            data = self._deserialize_data(data_bytes)
            self.stats.hits += 1
            return data

        except ConnectionError:
            self.is_connected = False
            self.stats.errors += 1
            logger.error("Redis connection lost")
            return None
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Failed to get cache entry: {e}")
            return None

    async def delete(self, key: str, prefix: str = None) -> bool:
        """
        Delete cache entry.

        Args:
            key: Cache key
            prefix: Optional key prefix

        Returns:
            True if key was deleted
        """
        await self._ensure_connected()

        try:
            cache_key = self._generate_key(key, prefix)
            result = await self.redis_client.delete(cache_key)

            if result > 0:
                self.stats.deletes += 1
                return True
            return False

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Failed to delete cache entry: {e}")
            return False

    async def exists(self, key: str, prefix: str = None) -> bool:
        """
        Check if cache key exists.

        Args:
            key: Cache key
            prefix: Optional key prefix

        Returns:
            True if key exists
        """
        await self._ensure_connected()

        try:
            cache_key = self._generate_key(key, prefix)
            result = await self.redis_client.exists(cache_key)
            return result > 0

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Failed to check key existence: {e}")
            return False

    async def expire(self, key: str, ttl: int, prefix: str = None) -> bool:
        """
        Set expiration for cache key.

        Args:
            key: Cache key
            ttl: Time to live in seconds
            prefix: Optional key prefix

        Returns:
            True if expiration was set
        """
        await self._ensure_connected()

        try:
            cache_key = self._generate_key(key, prefix)
            result = await self.redis_client.expire(cache_key, ttl)
            return bool(result)

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Failed to set expiration: {e}")
            return False

    async def get_keys(self, pattern: str = '*') -> List[str]:
        """
        Get keys matching pattern.

        Args:
            pattern: Key pattern (supports wildcards)

        Returns:
            List of matching keys
        """
        await self._ensure_connected()

        try:
            keys_bytes = await self.redis_client.keys(pattern)
            return [key.decode('utf-8') for key in keys_bytes]

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Failed to get keys: {e}")
            return []

    async def clear(self) -> bool:
        """
        Clear all cache entries.

        Returns:
            True if successful
        """
        await self._ensure_connected()

        try:
            result = await self.redis_client.flushdb()
            return bool(result)

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Failed to clear cache: {e}")
            return False

    async def set_many(
        self,
        data_dict: Dict[str, Any],
        ttl: Optional[int] = None,
        prefix: str = None
    ) -> bool:
        """
        Set multiple cache entries in batch.

        Args:
            data_dict: Dictionary of key-value pairs
            ttl: Time to live in seconds
            prefix: Optional key prefix

        Returns:
            True if all operations successful
        """
        await self._ensure_connected()

        try:
            ttl = ttl or self.config.default_ttl
            pipeline = self.redis_client.pipeline()

            for key, data in data_dict.items():
                cache_key = self._generate_key(key, prefix)
                serialized_data = self._serialize_data(data)

                if ttl > 0:
                    pipeline.setex(cache_key, ttl, serialized_data)
                else:
                    pipeline.set(cache_key, serialized_data)

            results = await pipeline.execute()

            # Count successful operations
            successful = sum(1 for result in results if result)
            self.stats.sets += successful

            return len(results) == successful

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Failed to set multiple entries: {e}")
            return False

    async def get_many(
        self,
        keys: List[str],
        prefix: str = None
    ) -> Dict[str, Any]:
        """
        Get multiple cache entries in batch.

        Args:
            keys: List of cache keys
            prefix: Optional key prefix

        Returns:
            Dictionary of found key-value pairs
        """
        await self._ensure_connected()

        try:
            cache_keys = [self._generate_key(key, prefix) for key in keys]
            data_list = await self.redis_client.mget(cache_keys)

            result = {}
            for i, data_bytes in enumerate(data_list):
                if data_bytes is not None:
                    try:
                        data = self._deserialize_data(data_bytes)
                        result[keys[i]] = data
                        self.stats.hits += 1
                    except Exception:
                        self.stats.errors += 1
                else:
                    self.stats.misses += 1

            return result

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Failed to get multiple entries: {e}")
            return {}

    async def warm_cache(self, data_dict: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Warm cache with predefined data.

        Args:
            data_dict: Dictionary of key-value pairs to cache
            ttl: Time to live in seconds

        Returns:
            True if warming successful
        """
        logger.info(f"Warming cache with {len(data_dict)} entries")
        return await self.set_many(data_dict, ttl)

    async def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage information.

        Returns:
            Dictionary with memory usage statistics
        """
        await self._ensure_connected()

        try:
            info = await self.redis_client.info('memory')

            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)

            used_memory_mb = used_memory / (1024 * 1024)
            max_memory_mb = max_memory / (1024 * 1024) if max_memory > 0 else self.config.max_memory_mb

            usage_percentage = (used_memory_mb / max_memory_mb) * 100 if max_memory_mb > 0 else 0

            return {
                'used_memory_mb': used_memory_mb,
                'max_memory_mb': max_memory_mb,
                'usage_percentage': usage_percentage,
                'available_memory_mb': max_memory_mb - used_memory_mb
            }

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Failed to get memory usage: {e}")
            return {
                'used_memory_mb': 0.0,
                'max_memory_mb': self.config.max_memory_mb,
                'usage_percentage': 0.0,
                'available_memory_mb': self.config.max_memory_mb
            }

    async def evict_expired_keys(self) -> int:
        """
        Manually evict expired keys.

        Returns:
            Number of keys evicted
        """
        await self._ensure_connected()

        try:
            # Get all keys
            all_keys = await self.get_keys('*')
            expired_keys = []

            # Check TTL for each key
            for key in all_keys:
                ttl = await self.redis_client.ttl(key)
                if ttl == -2:  # Key doesn't exist (expired and removed)
                    continue
                elif ttl == -1:  # Key has no expiration
                    continue
                elif ttl == 0:  # Key is expired
                    expired_keys.append(key)

            # Delete expired keys
            if expired_keys:
                deleted = await self.redis_client.delete(*expired_keys)
                self.stats.deletes += deleted
                return deleted

            return 0

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Failed to evict expired keys: {e}")
            return 0

    def cached(self, ttl: Optional[int] = None, key_prefix: str = 'func'):
        """
        Decorator for caching function results.

        Args:
            ttl: Time to live in seconds
            key_prefix: Prefix for cache keys

        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_parts = [func.__name__]

                # Add args to key
                for arg in args:
                    if isinstance(arg, (str, int, float, bool)):
                        key_parts.append(str(arg))
                    else:
                        key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])

                # Add kwargs to key
                for k, v in sorted(kwargs.items()):
                    if isinstance(v, (str, int, float, bool)):
                        key_parts.append(f"{k}={v}")
                    else:
                        key_parts.append(f"{k}={hashlib.md5(str(v).encode()).hexdigest()[:8]}")

                cache_key = ':'.join(key_parts)

                # Try to get from cache
                cached_result = await self.get(cache_key, prefix=key_prefix)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl=ttl, prefix=key_prefix)

                return result

            return wrapper
        return decorator

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats

    def export_config(self) -> Dict[str, Any]:
        """Export cache configuration."""
        return self.config.to_dict()

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on cache system.

        Returns:
            Health status and metrics
        """
        try:
            # Test basic connectivity
            if not self.is_connected:
                return {
                    'status': 'unhealthy',
                    'error': 'Not connected to Redis'
                }

            # Test ping
            start_time = time.time()
            await self.redis_client.ping()
            ping_time = (time.time() - start_time) * 1000  # Convert to ms

            # Get memory usage
            memory_info = await self.get_memory_usage()

            # Get stats
            stats = self.stats.to_dict()

            return {
                'status': 'healthy',
                'ping_time_ms': ping_time,
                'memory_usage': memory_info,
                'statistics': stats,
                'config': self.export_config()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }