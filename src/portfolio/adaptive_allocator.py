"""
Adaptive Allocator - Backward Compatibility Wrapper

This file maintains backward compatibility while redirecting to the new modular structure.
The actual implementation is now split across specialized modules in allocation/.

DEPRECATED: This file will be removed in a future version.
Use: from src.portfolio.allocation import AdaptiveAllocator
"""

import warnings
from .allocation import (
    AdaptiveAllocator,
    AdaptiveConfig,
    PerformanceWindow,
    AllocationUpdate,
    RebalanceRecommendation
)

# Issue deprecation warning
warnings.warn(
    "Importing from 'src.portfolio.adaptive_allocator' is deprecated. "
    "The module has been refactored into specialized components. "
    "Use 'from src.portfolio.allocation import AdaptiveAllocator' instead. "
    "See src/portfolio/allocation/ for the new modular structure.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
__all__ = [
    'AdaptiveAllocator',
    'AdaptiveConfig',
    'PerformanceWindow',
    'AllocationUpdate',
    'RebalanceRecommendation'
]