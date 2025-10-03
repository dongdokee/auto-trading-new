"""
Advanced Analytics System for comprehensive data analysis.

This module provides sophisticated analytics capabilities including:
- Statistical analysis and data modeling
- Machine learning model training and evaluation
- Performance attribution and risk analysis
- Predictive analytics and forecasting
- Pattern recognition and anomaly detection
- Automated report generation and insights

Features:
- Time series analysis and forecasting
- Statistical hypothesis testing
- Machine learning pipeline automation
- Performance benchmarking and attribution
- Risk factor analysis and decomposition
- Automated insights and recommendations
- Custom analytics framework extension
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json
import statistics
from collections import defaultdict
from abc import ABC, abstractmethod

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.decomposition import PCA
    import scipy.stats as stats
except ImportError:
    # Graceful degradation if sklearn/scipy not available
    train_test_split = None
    cross_val_score = None
    mean_squared_error = None
    mean_absolute_error = None
    r2_score = None
    StandardScaler = None
    LinearRegression = None
    RandomForestRegressor = None
    PCA = None
    stats = None

logger = logging.getLogger(__name__)


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

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
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


class StatisticalAnalyzer(BaseAnalyzer):
    """
    Statistical analysis and hypothesis testing.

    Performs descriptive statistics, correlation analysis, and hypothesis tests.
    """

    def __init__(self):
        """Initialize statistical analyzer."""
        super().__init__("StatisticalAnalyzer")

    def analyze(self, data: Union[TimeSeriesData, List[float]], **kwargs) -> AnalyticsResult:
        """Perform statistical analysis."""
        if not self.validate_data(data):
            raise AnalyticsError("Invalid data for statistical analysis")

        result = AnalyticsResult(
            analysis_type="statistical_analysis",
            data_points=len(data.values) if isinstance(data, TimeSeriesData) else len(data)
        )

        values = data.values if isinstance(data, TimeSeriesData) else data

        # Basic statistics
        basic_stats = self._calculate_basic_statistics(values)
        result.results['basic_statistics'] = basic_stats
        result.metrics.update(basic_stats)

        # Distribution analysis
        distribution_stats = self._analyze_distribution(values)
        result.results['distribution'] = distribution_stats
        result.metrics.update(distribution_stats)

        # Generate insights
        result.insights = self._generate_statistical_insights(basic_stats, distribution_stats)
        result.confidence = 0.95  # Statistical confidence

        return result

    def correlation_analysis(self, data1: List[float], data2: List[float]) -> AnalyticsResult:
        """Perform correlation analysis between two datasets."""
        if len(data1) != len(data2):
            raise AnalyticsError("Data series must have the same length")

        result = AnalyticsResult(
            analysis_type="correlation_analysis",
            data_points=len(data1)
        )

        # Calculate correlation coefficient
        correlation = self._calculate_correlation(data1, data2)
        result.metrics['correlation'] = correlation
        result.results['correlation_coefficient'] = correlation

        # Correlation strength interpretation
        strength = self._interpret_correlation_strength(correlation)
        result.results['correlation_strength'] = strength

        # Generate insights
        result.add_insight(f"Correlation coefficient: {correlation:.4f}")
        result.add_insight(f"Correlation strength: {strength}")

        if abs(correlation) > 0.7:
            result.add_insight("Strong correlation detected between the two datasets")
        elif abs(correlation) > 0.3:
            result.add_insight("Moderate correlation detected between the two datasets")
        else:
            result.add_insight("Weak or no correlation detected between the two datasets")

        result.confidence = 0.90

        return result

    def hypothesis_test(self, sample1: List[float], sample2: List[float], test_type: str = "t_test") -> AnalyticsResult:
        """Perform hypothesis testing."""
        if stats is None:
            raise AnalyticsError("scipy.stats is required for hypothesis testing")

        result = AnalyticsResult(
            analysis_type=f"hypothesis_test_{test_type}",
            data_points=len(sample1) + len(sample2)
        )

        if test_type == "t_test":
            statistic, p_value = stats.ttest_ind(sample1, sample2)
            result.results['test_statistic'] = statistic
            result.results['p_value'] = p_value
            result.metrics['t_statistic'] = statistic
            result.metrics['p_value'] = p_value

            # Interpret results
            alpha = 0.05
            is_significant = p_value < alpha
            result.results['is_significant'] = is_significant
            result.results['alpha'] = alpha

            if is_significant:
                result.add_insight(f"Statistically significant difference detected (p={p_value:.4f})")
            else:
                result.add_insight(f"No statistically significant difference (p={p_value:.4f})")

        result.confidence = 1 - result.metrics.get('p_value', 0.5)

        return result

    def _calculate_basic_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic descriptive statistics."""
        if not values:
            return {}

        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'mode': statistics.mode(values) if len(set(values)) < len(values) else values[0],
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'variance': statistics.variance(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values),
            'skewness': self._calculate_skewness(values),
            'kurtosis': self._calculate_kurtosis(values)
        }

    def _analyze_distribution(self, values: List[float]) -> Dict[str, Any]:
        """Analyze the distribution of values."""
        if len(values) < 2:
            return {}

        # Quartiles and percentiles
        sorted_values = sorted(values)
        n = len(sorted_values)

        q1_idx = n // 4
        q2_idx = n // 2
        q3_idx = 3 * n // 4

        return {
            'q1': sorted_values[q1_idx],
            'q2': sorted_values[q2_idx],
            'q3': sorted_values[q3_idx],
            'iqr': sorted_values[q3_idx] - sorted_values[q1_idx],
            'p5': sorted_values[n // 20],
            'p95': sorted_values[19 * n // 20],
            'outliers_count': self._count_outliers(values)
        }

    def _calculate_correlation(self, data1: List[float], data2: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(data1) != len(data2) or len(data1) < 2:
            return 0.0

        n = len(data1)
        sum1 = sum(data1)
        sum2 = sum(data2)
        sum1_sq = sum(x * x for x in data1)
        sum2_sq = sum(x * x for x in data2)
        sum_prod = sum(x1 * x2 for x1, x2 in zip(data1, data2))

        numerator = n * sum_prod - sum1 * sum2
        denominator = ((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2)) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.9:
            return "Very Strong"
        elif abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        else:
            return "Very Weak"

    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of the distribution."""
        if len(values) < 3:
            return 0.0

        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)

        if std_val == 0:
            return 0.0

        n = len(values)
        skewness = sum((x - mean_val) ** 3 for x in values) / (n * std_val ** 3)
        return skewness

    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of the distribution."""
        if len(values) < 4:
            return 0.0

        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)

        if std_val == 0:
            return 0.0

        n = len(values)
        kurtosis = sum((x - mean_val) ** 4 for x in values) / (n * std_val ** 4) - 3
        return kurtosis

    def _count_outliers(self, values: List[float]) -> int:
        """Count outliers using IQR method."""
        if len(values) < 4:
            return 0

        sorted_values = sorted(values)
        n = len(sorted_values)
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return sum(1 for x in values if x < lower_bound or x > upper_bound)

    def _generate_statistical_insights(self, basic_stats: Dict[str, float], distribution_stats: Dict[str, Any]) -> List[str]:
        """Generate insights from statistical analysis."""
        insights = []

        # Mean vs Median comparison
        if basic_stats.get('mean', 0) != basic_stats.get('median', 0):
            diff_pct = abs(basic_stats['mean'] - basic_stats['median']) / basic_stats['median'] * 100
            if diff_pct > 10:
                insights.append(f"Distribution is skewed (mean-median difference: {diff_pct:.1f}%)")

        # Variability assessment
        cv = basic_stats.get('std', 0) / basic_stats.get('mean', 1) * 100
        if cv < 15:
            insights.append("Low variability detected in the data")
        elif cv > 50:
            insights.append("High variability detected in the data")

        # Outliers
        outliers = distribution_stats.get('outliers_count', 0)
        if outliers > 0:
            insights.append(f"Detected {outliers} outliers in the dataset")

        # Skewness interpretation
        skewness = basic_stats.get('skewness', 0)
        if abs(skewness) > 1:
            direction = "right" if skewness > 0 else "left"
            insights.append(f"Distribution is significantly skewed to the {direction}")

        return insights


class TimeSeriesAnalyzer(BaseAnalyzer):
    """
    Time series analysis and forecasting.

    Performs trend analysis, seasonality detection, and forecasting.
    """

    def __init__(self):
        """Initialize time series analyzer."""
        super().__init__("TimeSeriesAnalyzer")

    def analyze(self, data: TimeSeriesData, **kwargs) -> AnalyticsResult:
        """Perform time series analysis."""
        if not self.validate_data(data):
            raise AnalyticsError("Invalid time series data")

        result = AnalyticsResult(
            analysis_type="time_series_analysis",
            data_points=len(data.values)
        )

        # Trend analysis
        trend_stats = self._analyze_trend(data.values)
        result.results['trend'] = trend_stats
        result.metrics.update(trend_stats)

        # Volatility analysis
        volatility_stats = self._analyze_volatility(data.values)
        result.results['volatility'] = volatility_stats
        result.metrics.update(volatility_stats)

        # Stationarity test
        stationarity = self._test_stationarity(data.values)
        result.results['stationarity'] = stationarity

        # Generate insights
        result.insights = self._generate_timeseries_insights(trend_stats, volatility_stats, stationarity)
        result.confidence = 0.85

        return result

    def forecast(self, data: TimeSeriesData, periods: int = 10, method: str = "linear") -> AnalyticsResult:
        """Generate forecasts for the time series."""
        if len(data.values) < 3:
            raise AnalyticsError("Insufficient data for forecasting")

        result = AnalyticsResult(
            analysis_type=f"forecast_{method}",
            data_points=len(data.values)
        )

        if method == "linear":
            forecasts = self._linear_forecast(data.values, periods)
        elif method == "moving_average":
            forecasts = self._moving_average_forecast(data.values, periods)
        else:
            raise AnalyticsError(f"Unknown forecasting method: {method}")

        result.results['forecasts'] = forecasts
        result.results['forecast_periods'] = periods
        result.results['method'] = method

        # Calculate forecast confidence based on historical accuracy
        accuracy = self._calculate_forecast_accuracy(data.values, method)
        result.confidence = accuracy
        result.metrics['forecast_accuracy'] = accuracy

        result.add_insight(f"Generated {periods} period forecast using {method} method")
        result.add_insight(f"Forecast accuracy: {accuracy:.2%}")

        return result

    def detect_anomalies(self, data: TimeSeriesData, threshold: float = 2.0) -> AnalyticsResult:
        """Detect anomalies in time series data."""
        result = AnalyticsResult(
            analysis_type="anomaly_detection",
            data_points=len(data.values)
        )

        anomalies = self._detect_statistical_anomalies(data.values, threshold)
        result.results['anomalies'] = anomalies
        result.results['anomaly_threshold'] = threshold
        result.metrics['anomaly_count'] = len(anomalies)
        result.metrics['anomaly_rate'] = len(anomalies) / len(data.values) * 100

        if anomalies:
            result.add_insight(f"Detected {len(anomalies)} anomalies ({result.metrics['anomaly_rate']:.1f}% of data)")
        else:
            result.add_insight("No anomalies detected in the time series")

        result.confidence = 0.80

        return result

    def _analyze_trend(self, values: List[float]) -> Dict[str, float]:
        """Analyze trend in time series."""
        if len(values) < 2:
            return {}

        # Simple linear trend
        x = list(range(len(values)))
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_x2 = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in values)
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, values))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'trend_slope': slope,
            'trend_intercept': intercept,
            'trend_r_squared': r_squared,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat'
        }

    def _analyze_volatility(self, values: List[float]) -> Dict[str, float]:
        """Analyze volatility in time series."""
        if len(values) < 2:
            return {}

        # Calculate returns
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values)) if values[i-1] != 0]

        if not returns:
            return {}

        volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
        mean_return = statistics.mean(returns)

        # Rolling volatility (if enough data)
        rolling_volatility = []
        window = min(10, len(returns) // 2)
        if window >= 2:
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                rolling_volatility.append(statistics.stdev(window_returns))

        return {
            'volatility': volatility,
            'mean_return': mean_return,
            'volatility_trend': 'increasing' if len(rolling_volatility) > 1 and rolling_volatility[-1] > rolling_volatility[0] else 'decreasing'
        }

    def _test_stationarity(self, values: List[float]) -> Dict[str, Any]:
        """Simple stationarity test based on rolling statistics."""
        if len(values) < 10:
            return {'is_stationary': True, 'method': 'insufficient_data'}

        # Split data into chunks and compare means/variances
        chunk_size = len(values) // 3
        chunks = [values[i:i+chunk_size] for i in range(0, len(values), chunk_size)]

        if len(chunks) < 3:
            return {'is_stationary': True, 'method': 'insufficient_chunks'}

        chunk_means = [statistics.mean(chunk) for chunk in chunks if chunk]
        chunk_vars = [statistics.variance(chunk) for chunk in chunks if len(chunk) > 1]

        # Simple test: check if means and variances are relatively stable
        mean_stability = max(chunk_means) - min(chunk_means) < statistics.stdev(values) * 0.5
        var_stability = len(chunk_vars) < 2 or (max(chunk_vars) - min(chunk_vars) < statistics.variance(values) * 0.5)

        is_stationary = mean_stability and var_stability

        return {
            'is_stationary': is_stationary,
            'method': 'rolling_statistics',
            'mean_stability': mean_stability,
            'variance_stability': var_stability
        }

    def _linear_forecast(self, values: List[float], periods: int) -> List[float]:
        """Generate linear trend forecast."""
        if len(values) < 2:
            return [values[-1]] * periods if values else [0.0] * periods

        # Calculate linear trend
        x = list(range(len(values)))
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_x2 = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        # Generate forecasts
        forecasts = []
        for i in range(periods):
            forecast_x = len(values) + i
            forecast_y = slope * forecast_x + intercept
            forecasts.append(forecast_y)

        return forecasts

    def _moving_average_forecast(self, values: List[float], periods: int, window: int = 3) -> List[float]:
        """Generate moving average forecast."""
        window = min(window, len(values))
        if window == 0:
            return [0.0] * periods

        recent_avg = statistics.mean(values[-window:])
        return [recent_avg] * periods

    def _calculate_forecast_accuracy(self, values: List[float], method: str) -> float:
        """Calculate forecast accuracy using cross-validation."""
        if len(values) < 6:
            return 0.5

        # Use last 20% for validation
        split_idx = int(len(values) * 0.8)
        train_data = values[:split_idx]
        test_data = values[split_idx:]

        if method == "linear":
            forecasts = self._linear_forecast(train_data, len(test_data))
        elif method == "moving_average":
            forecasts = self._moving_average_forecast(train_data, len(test_data))
        else:
            return 0.5

        # Calculate MAPE (Mean Absolute Percentage Error)
        errors = []
        for actual, forecast in zip(test_data, forecasts):
            if actual != 0:
                error = abs((actual - forecast) / actual)
                errors.append(error)

        if not errors:
            return 0.5

        mape = statistics.mean(errors)
        accuracy = max(0, 1 - mape)
        return min(1.0, accuracy)

    def _detect_statistical_anomalies(self, values: List[float], threshold: float) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical methods."""
        if len(values) < 3:
            return []

        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)

        anomalies = []
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
            if z_score > threshold:
                anomalies.append({
                    'index': i,
                    'value': value,
                    'z_score': z_score,
                    'type': 'statistical_outlier'
                })

        return anomalies

    def _generate_timeseries_insights(self, trend_stats: Dict, volatility_stats: Dict, stationarity: Dict) -> List[str]:
        """Generate insights from time series analysis."""
        insights = []

        # Trend insights
        if trend_stats.get('trend_r_squared', 0) > 0.5:
            direction = trend_stats.get('trend_direction', 'unknown')
            insights.append(f"Strong {direction} trend detected (R² = {trend_stats['trend_r_squared']:.3f})")

        # Volatility insights
        if volatility_stats.get('volatility', 0) > 0.1:
            insights.append("High volatility detected in the time series")

        # Stationarity insights
        if not stationarity.get('is_stationary', True):
            insights.append("Time series appears to be non-stationary")

        return insights


class MachineLearningAnalyzer(BaseAnalyzer):
    """
    Machine learning model training and evaluation.

    Provides automated ML pipeline for predictive analytics.
    """

    def __init__(self):
        """Initialize ML analyzer."""
        super().__init__("MachineLearningAnalyzer")
        self.models = {}
        self.scalers = {}

    def analyze(self, features: List[List[float]], targets: List[float], **kwargs) -> AnalyticsResult:
        """Train and evaluate ML models."""
        if LinearRegression is None:
            raise AnalyticsError("scikit-learn is required for machine learning analysis")

        if len(features) != len(targets) or len(features) < 3:
            raise AnalyticsError("Insufficient or mismatched data for ML analysis")

        result = AnalyticsResult(
            analysis_type="machine_learning_analysis",
            data_points=len(features)
        )

        # Train multiple models
        models_performance = {}

        # Linear Regression
        lr_performance = self._train_and_evaluate_model(
            LinearRegression(), features, targets, "linear_regression"
        )
        models_performance['linear_regression'] = lr_performance

        # Random Forest (if available)
        if RandomForestRegressor is not None:
            rf_performance = self._train_and_evaluate_model(
                RandomForestRegressor(n_estimators=50, random_state=42),
                features, targets, "random_forest"
            )
            models_performance['random_forest'] = rf_performance

        result.results['models'] = models_performance

        # Select best model
        best_model = max(models_performance.items(), key=lambda x: x[1]['r2_score'])
        result.results['best_model'] = {
            'name': best_model[0],
            'performance': best_model[1]
        }
        result.metrics.update(best_model[1])

        # Generate insights
        result.insights = self._generate_ml_insights(models_performance, best_model)
        result.confidence = best_model[1]['r2_score']

        return result

    def predict(self, model_name: str, features: List[List[float]]) -> AnalyticsResult:
        """Make predictions using trained model."""
        if model_name not in self.models:
            raise AnalyticsError(f"Model {model_name} not found")

        model = self.models[model_name]
        scaler = self.scalers.get(model_name)

        result = AnalyticsResult(
            analysis_type=f"prediction_{model_name}",
            data_points=len(features)
        )

        # Scale features if scaler available
        if scaler:
            features = scaler.transform(features)

        # Make predictions
        predictions = model.predict(features).tolist()
        result.results['predictions'] = predictions
        result.metrics['prediction_count'] = len(predictions)

        result.add_insight(f"Generated {len(predictions)} predictions using {model_name} model")
        result.confidence = 0.85

        return result

    def feature_importance(self, model_name: str) -> AnalyticsResult:
        """Analyze feature importance for the specified model."""
        if model_name not in self.models:
            raise AnalyticsError(f"Model {model_name} not found")

        model = self.models[model_name]
        result = AnalyticsResult(
            analysis_type=f"feature_importance_{model_name}"
        )

        # Get feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_.tolist()
            result.results['feature_importances'] = importances
            result.metrics['top_feature_importance'] = max(importances)

            # Generate insights
            top_feature_idx = importances.index(max(importances))
            result.add_insight(f"Most important feature: Feature {top_feature_idx} (importance: {max(importances):.3f})")

        elif hasattr(model, 'coef_'):
            coefficients = model.coef_.tolist()
            result.results['coefficients'] = coefficients
            result.metrics['max_coefficient'] = max(abs(c) for c in coefficients)

            # Generate insights
            max_coef_idx = coefficients.index(max(coefficients, key=abs))
            result.add_insight(f"Highest coefficient: Feature {max_coef_idx} (coefficient: {coefficients[max_coef_idx]:.3f})")

        result.confidence = 0.90

        return result

    def _train_and_evaluate_model(self, model, features: List[List[float]], targets: List[float], model_name: str) -> Dict[str, float]:
        """Train and evaluate a single model."""
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler

        return {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'rmse': np.sqrt(mse)
        }

    def _generate_ml_insights(self, models_performance: Dict, best_model: Tuple) -> List[str]:
        """Generate insights from ML analysis."""
        insights = []

        best_name, best_perf = best_model
        insights.append(f"Best performing model: {best_name} (R² = {best_perf['r2_score']:.3f})")

        if best_perf['r2_score'] > 0.8:
            insights.append("Excellent model performance achieved")
        elif best_perf['r2_score'] > 0.6:
            insights.append("Good model performance achieved")
        elif best_perf['r2_score'] > 0.3:
            insights.append("Moderate model performance - consider feature engineering")
        else:
            insights.append("Poor model performance - data may not be suitable for prediction")

        # Compare models if multiple available
        if len(models_performance) > 1:
            performances = [perf['r2_score'] for perf in models_performance.values()]
            if max(performances) - min(performances) > 0.1:
                insights.append("Significant performance differences between models detected")

        return insights


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