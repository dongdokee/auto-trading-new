"""
Performance Enhancement System - Refactored Module

This module now serves as a backward-compatible wrapper around the refactored performance package.
All original functionality is preserved through imports from submodules.

DEPRECATION NOTICE:
This file is maintained for backward compatibility only.
New code should import directly from the performance package:
    from .performance import PerformanceEnhancer, ResourceManager, etc.

Original file has been split into:
- performance/models.py: Performance metrics and data models
- performance/resource_manager.py: System resource monitoring and management
- performance/parallel_processor.py: Parallel processing with thread/process pools
- performance/performance_enhancer.py: Main performance enhancement system
"""

# Backward compatibility imports
from .performance import (
    OptimizationError,
    PerformanceMetrics,
    ResourceManager,
    ParallelProcessor,
    OptimizationStrategy,
    PerformanceEnhancer
)

# Re-export all classes for backward compatibility
__all__ = [
    "OptimizationError",
    "PerformanceMetrics",
    "ResourceManager",
    "ParallelProcessor",
    "OptimizationStrategy",
    "PerformanceEnhancer"
]

# Deprecation warning for direct imports
import warnings
warnings.warn(
    "Importing from performance_enhancer.py is deprecated. "
    "Use 'from .performance import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)