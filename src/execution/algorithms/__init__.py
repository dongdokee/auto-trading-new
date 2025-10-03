"""
Execution Algorithms Package

This package contains the refactored execution algorithms from execution_algorithms.py.
Each algorithm type is implemented in its own specialized module for better maintainability
and separation of concerns.

Modules:
- base: Common base classes and utilities
- twap_algorithms: Time-Weighted Average Price algorithms
- vwap_algorithms: Volume-Weighted Average Price algorithms
- adaptive_algorithms: Multi-signal adaptive execution algorithms
- participation_algorithms: Market participation rate control algorithms
- analytics: Performance analysis and metrics calculation

Usage:
    from .base import BaseExecutionAlgorithm
    from .twap_algorithms import TWAPAlgorithm
    from .vwap_algorithms import VWAPAlgorithm
    from .adaptive_algorithms import AdaptiveAlgorithm
    from .analytics import ExecutionAnalytics
"""

from .base import BaseExecutionAlgorithm, ExecutionMetrics
from .twap_algorithms import TWAPAlgorithm, DynamicTWAPAlgorithm, TWAPWithFallback
from .vwap_algorithms import VWAPAlgorithm, AdaptiveVWAPAlgorithm
from .adaptive_algorithms import AdaptiveAlgorithm, SignalCalculator
from .participation_algorithms import ParticipationRateAlgorithm
from .analytics import ExecutionAnalytics, ImplementationShortfall, PerformanceMetrics

__all__ = [
    # Base classes
    "BaseExecutionAlgorithm",
    "ExecutionMetrics",

    # TWAP algorithms
    "TWAPAlgorithm",
    "DynamicTWAPAlgorithm",
    "TWAPWithFallback",

    # VWAP algorithms
    "VWAPAlgorithm",
    "AdaptiveVWAPAlgorithm",

    # Adaptive algorithms
    "AdaptiveAlgorithm",
    "SignalCalculator",

    # Participation algorithms
    "ParticipationRateAlgorithm",

    # Analytics
    "ExecutionAnalytics",
    "ImplementationShortfall",
    "PerformanceMetrics"
]