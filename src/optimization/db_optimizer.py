"""
Database Optimizer - Refactored Module

This module now serves as a backward-compatible wrapper around the refactored database package.
All original functionality is preserved through imports from submodules.

DEPRECATION NOTICE:
This file is maintained for backward compatibility only.
New code should import directly from the database package:
    from .database import DatabaseOptimizer, QueryOptimizer, etc.

Original file has been split into:
- database/models.py: Data models and exceptions
- database/connection_pool.py: Connection pool management
- database/query_optimizer.py: Query optimization and caching
- database/optimizer.py: Main database optimizer system
"""

# Backward compatibility imports
from .database import (
    OptimizationError,
    QueryPlan,
    QueryStats,
    IndexRecommendation,
    ConnectionPoolManager,
    QueryOptimizer,
    DatabaseOptimizer
)

# Re-export all classes for backward compatibility
__all__ = [
    "OptimizationError",
    "QueryPlan",
    "QueryStats",
    "IndexRecommendation",
    "ConnectionPoolManager",
    "QueryOptimizer",
    "DatabaseOptimizer"
]

# Deprecation warning for direct imports
import warnings
warnings.warn(
    "Importing from db_optimizer.py is deprecated. "
    "Use 'from .database import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)