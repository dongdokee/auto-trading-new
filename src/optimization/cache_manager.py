"""
Caching Layer with Redis - Refactored Module

This module now serves as a backward-compatible wrapper around the refactored caching package.
All original functionality is preserved through imports from submodules.

DEPRECATION NOTICE:
This file is maintained for backward compatibility only.
New code should import directly from the caching package:
    from .caching import CacheManager, CacheConfig, etc.

Original file has been split into:
- caching/models.py: Data models and configurations
- caching/cache_manager.py: Main cache manager with Redis support
"""

# Backward compatibility imports
from .caching import (
    CacheError,
    CacheConfig,
    CacheEntry,
    CacheStats,
    CacheManager
)

# Re-export all classes for backward compatibility
__all__ = [
    "CacheError",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "CacheManager"
]

# Deprecation warning for direct imports
import warnings
warnings.warn(
    "Importing from cache_manager.py is deprecated. "
    "Use 'from .caching import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)