"""
Portfolio Allocation Module

Specialized modules for adaptive portfolio allocation and rebalancing.
Extracted from large adaptive_allocator.py for better maintainability.
"""

from .models import (
    AdaptiveConfig,
    PerformanceWindow,
    AllocationUpdate,
    RebalanceRecommendation
)
from .performance_analyzer import PerformanceAnalyzer
from .weight_optimizer import WeightOptimizer
from .rebalance_engine import RebalanceEngine
from .adaptive_allocator import AdaptiveAllocator

__all__ = [
    "AdaptiveConfig",
    "PerformanceWindow",
    "AllocationUpdate",
    "RebalanceRecommendation",
    "PerformanceAnalyzer",
    "WeightOptimizer",
    "RebalanceEngine",
    "AdaptiveAllocator"
]