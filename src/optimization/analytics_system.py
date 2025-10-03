"""
Analytics System - Refactored Module

This module now serves as a backward-compatible wrapper around the refactored analytics package.
All original functionality is preserved through imports from submodules.

DEPRECATION NOTICE:
This file is maintained for backward compatibility only.
New code should import directly from the analytics package:
    from .analytics import AdvancedAnalyticsSystem, StatisticalAnalyzer, etc.

Original file has been split into:
- analytics/core.py: Base classes and data models
- analytics/statistical.py: Statistical analysis
- analytics/timeseries.py: Time series analysis
- analytics/ml_analyzer.py: Machine learning analysis
- analytics/system.py: Main orchestrator system
"""

# Backward compatibility imports
from .analytics import (
    AnalyticsError,
    AnalyticsResult,
    TimeSeriesData,
    BaseAnalyzer,
    StatisticalAnalyzer,
    TimeSeriesAnalyzer,
    MachineLearningAnalyzer,
    AdvancedAnalyticsSystem
)

# Re-export all classes for backward compatibility
__all__ = [
    "AnalyticsError",
    "AnalyticsResult",
    "TimeSeriesData",
    "BaseAnalyzer",
    "StatisticalAnalyzer",
    "TimeSeriesAnalyzer",
    "MachineLearningAnalyzer",
    "AdvancedAnalyticsSystem"
]

# Deprecation warning for direct imports
import warnings
warnings.warn(
    "Importing from analytics_system.py is deprecated. "
    "Use 'from .analytics import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)