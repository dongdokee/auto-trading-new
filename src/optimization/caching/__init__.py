"""
Caching package for performance optimization with Redis support.

This package provides comprehensive caching capabilities including:
- Redis integration with async support
- TTL-based cache management
- Data compression and encryption
- Batch operations and cache warming
- Memory usage monitoring and eviction
- Cache statistics and performance metrics
- Caching decorators for functions
"""

from .models import (
    CacheError,
    CacheConfig,
    CacheEntry,
    CacheStats
)

from .cache_manager import CacheManager

__all__ = [
    "CacheError",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "CacheManager"
]