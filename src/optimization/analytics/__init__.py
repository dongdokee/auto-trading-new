"""
Analytics module for advanced data analysis.

This module provides comprehensive analytics capabilities including:
- Statistical analysis and data modeling
- Machine learning model training and evaluation
- Time series analysis and forecasting
- Performance attribution and risk analysis

Components:
- AnalyticsResult: Result container for analysis operations
- TimeSeriesData: Time series data container
- StatisticalAnalyzer: Statistical analysis and hypothesis testing
- TimeSeriesAnalyzer: Time series analysis and forecasting
- MachineLearningAnalyzer: ML model training and evaluation
- AdvancedAnalyticsSystem: Comprehensive analytics system
"""

from .core import AnalyticsError, AnalyticsResult, TimeSeriesData, BaseAnalyzer
from .statistical import StatisticalAnalyzer
from .timeseries import TimeSeriesAnalyzer
from .ml_analyzer import MachineLearningAnalyzer
from .system import AdvancedAnalyticsSystem

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