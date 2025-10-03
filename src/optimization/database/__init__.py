"""
Database optimization module for comprehensive database performance optimization.

This module provides advanced database optimization capabilities including:
- Query plan analysis and optimization
- Connection pool management with auto-scaling
- Index recommendation system
- Query caching and result optimization
- Performance monitoring and metrics collection

Components:
- OptimizationError: Exception for database optimization operations
- QueryPlan: Query execution plan with performance metrics
- QueryStats: Query performance statistics
- IndexRecommendation: Index recommendation data
- ConnectionPoolManager: Connection pool management
- QueryOptimizer: Query analysis and optimization
- DatabaseOptimizer: Main database optimizer system
"""

from .models import OptimizationError, QueryPlan, QueryStats, IndexRecommendation
from .connection_pool import ConnectionPoolManager
from .query_optimizer import QueryOptimizer
from .optimizer import DatabaseOptimizer

__all__ = [
    "OptimizationError",
    "QueryPlan",
    "QueryStats",
    "IndexRecommendation",
    "ConnectionPoolManager",
    "QueryOptimizer",
    "DatabaseOptimizer"
]