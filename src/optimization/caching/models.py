"""
Data models and configurations for the caching system.

This module contains all the data classes and configuration models used
throughout the caching package for Redis integration and cache management.
"""

import json
import pickle
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field


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