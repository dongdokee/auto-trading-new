"""
Main analytics system orchestrator.

This module provides the unified interface for all analytics capabilities:
- Statistical analysis orchestration
- Time series analysis coordination
- Machine learning pipeline management
- Analysis history tracking
- Comprehensive reporting
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from .core import AnalyticsResult, AnalyticsError, TimeSeriesData
from .statistical import StatisticalAnalyzer
from .timeseries import TimeSeriesAnalyzer
from .ml_analyzer import MachineLearningAnalyzer

logger = logging.getLogger(__name__)


class AdvancedAnalyticsSystem:
    """
    Main analytics system orchestrator.

    Integrates all analytics components and provides unified interface.
    """

    def __init__(self):
        """Initialize advanced analytics system."""
        self.statistical_analyzer = StatisticalAnalyzer()
        self.timeseries_analyzer = TimeSeriesAnalyzer()
        self.ml_analyzer = MachineLearningAnalyzer()
        self.analysis_history: List[AnalyticsResult] = []
        self.is_initialized = False

    async def initialize(self):
        """Initialize analytics system."""
        try:
            self.is_initialized = True
            logger.info("Advanced analytics system initialized successfully")
        except Exception as e:
            self.is_initialized = False
            raise AnalyticsError(f"Failed to initialize analytics system: {e}")

    def run_statistical_analysis(self, data: Union[TimeSeriesData, List[float]], **kwargs) -> AnalyticsResult:
        """Run comprehensive statistical analysis."""
        if not self.is_initialized:
            raise AnalyticsError("Analytics system not initialized")

        result = self.statistical_analyzer.analyze(data, **kwargs)
        self.analysis_history.append(result)
        return result

    def run_correlation_analysis(self, data1: List[float], data2: List[float]) -> AnalyticsResult:
        """Run correlation analysis between two datasets."""
        if not self.is_initialized:
            raise AnalyticsError("Analytics system not initialized")

        result = self.statistical_analyzer.correlation_analysis(data1, data2)
        self.analysis_history.append(result)
        return result

    def run_timeseries_analysis(self, data: TimeSeriesData, **kwargs) -> AnalyticsResult:
        """Run comprehensive time series analysis."""
        if not self.is_initialized:
            raise AnalyticsError("Analytics system not initialized")

        result = self.timeseries_analyzer.analyze(data, **kwargs)
        self.analysis_history.append(result)
        return result

    def run_forecast(self, data: TimeSeriesData, periods: int = 10, method: str = "linear") -> AnalyticsResult:
        """Generate forecasts for time series data."""
        if not self.is_initialized:
            raise AnalyticsError("Analytics system not initialized")

        result = self.timeseries_analyzer.forecast(data, periods, method)
        self.analysis_history.append(result)
        return result

    def run_anomaly_detection(self, data: TimeSeriesData, threshold: float = 2.0) -> AnalyticsResult:
        """Detect anomalies in time series data."""
        if not self.is_initialized:
            raise AnalyticsError("Analytics system not initialized")

        result = self.timeseries_analyzer.detect_anomalies(data, threshold)
        self.analysis_history.append(result)
        return result

    def run_ml_analysis(self, features: List[List[float]], targets: List[float], **kwargs) -> AnalyticsResult:
        """Run machine learning analysis."""
        if not self.is_initialized:
            raise AnalyticsError("Analytics system not initialized")

        result = self.ml_analyzer.analyze(features, targets, **kwargs)
        self.analysis_history.append(result)
        return result

    def get_analysis_history(self, analysis_type: Optional[str] = None) -> List[AnalyticsResult]:
        """Get history of analysis results."""
        if analysis_type:
            return [result for result in self.analysis_history if result.analysis_type == analysis_type]
        return self.analysis_history.copy()

    def export_analysis_report(self, hours: int = 24) -> Dict[str, Any]:
        """Export comprehensive analysis report."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_analyses = [
            result for result in self.analysis_history
            if result.timestamp >= cutoff_time
        ]

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'report_period_hours': hours,
            'total_analyses': len(recent_analyses),
            'analysis_types': list(set(result.analysis_type for result in recent_analyses)),
            'analyses': [result.to_dict() for result in recent_analyses],
            'summary': {
                'avg_confidence': statistics.mean([r.confidence for r in recent_analyses]) if recent_analyses else 0,
                'total_insights': sum(len(r.insights) for r in recent_analyses),
                'total_data_points': sum(r.data_points for r in recent_analyses)
            }
        }

        return report

    def clear_history(self):
        """Clear analysis history."""
        self.analysis_history.clear()
        logger.info("Analysis history cleared")