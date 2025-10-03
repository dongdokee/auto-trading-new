"""
Core analytics components and base classes.

This module contains the fundamental building blocks for the analytics system:
- Exception classes
- Data containers
- Base analyzer interface
"""

import logging
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List

try:
    import pandas as pd
except ImportError:
    pd = None


class AnalyticsError(Exception):
    """Raised when analytics operations fail."""
    pass


@dataclass
class AnalyticsResult:
    """
    Result of an analytics operation.

    Contains analysis results, metrics, and metadata.
    """
    analysis_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    confidence: float = 0.0
    data_points: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    def add_insight(self, insight: str):
        """Add an insight to the results."""
        self.insights.append(insight)

    def get_summary(self) -> str:
        """Get a summary of the analysis results."""
        summary = f"Analysis Type: {self.analysis_type}\n"
        summary += f"Data Points: {self.data_points}\n"
        summary += f"Confidence: {self.confidence:.2%}\n"
        summary += f"Key Metrics: {len(self.metrics)}\n"
        summary += f"Insights: {len(self.insights)}\n"
        return summary


@dataclass
class TimeSeriesData:
    """
    Time series data container.

    Holds time series data with timestamps and values.
    """
    timestamps: List[datetime]
    values: List[float]
    name: str = "time_series"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data after initialization."""
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have the same length")

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        if pd is None:
            raise ImportError("pandas is required for DataFrame conversion")

        return pd.DataFrame({
            'timestamp': self.timestamps,
            'value': self.values
        })

    def get_statistics(self) -> Dict[str, float]:
        """Get basic statistics of the time series."""
        if not self.values:
            return {}

        return {
            'count': len(self.values),
            'mean': statistics.mean(self.values),
            'median': statistics.median(self.values),
            'min': min(self.values),
            'max': max(self.values),
            'std': statistics.stdev(self.values) if len(self.values) > 1 else 0.0,
            'range': max(self.values) - min(self.values)
        }


class BaseAnalyzer(ABC):
    """
    Base class for analytics components.

    Provides common interface for all analyzers.
    """

    def __init__(self, name: str):
        """Initialize analyzer with name."""
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def analyze(self, data: Any, **kwargs) -> AnalyticsResult:
        """Perform analysis on the provided data."""
        pass

    def validate_data(self, data: Any) -> bool:
        """Validate input data."""
        return data is not None