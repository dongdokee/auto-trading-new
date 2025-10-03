"""
Time series analysis and forecasting.

This module provides comprehensive time series analysis capabilities including:
- Trend analysis and detection
- Volatility analysis and measurement
- Stationarity testing
- Forecasting using various methods
- Anomaly detection in time series
"""

import statistics
from typing import Dict, Any, List

from .core import BaseAnalyzer, AnalyticsResult, AnalyticsError, TimeSeriesData


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

        if n * sum_x2 - sum_x * sum_x == 0:
            return {'trend_slope': 0.0, 'trend_intercept': sum_y / n, 'trend_r_squared': 0.0, 'trend_direction': 'flat'}

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

        if not chunk_means:
            return {'is_stationary': True, 'method': 'no_valid_chunks'}

        # Simple test: check if means and variances are relatively stable
        std_values = statistics.stdev(values) if len(values) > 1 else 0
        mean_stability = max(chunk_means) - min(chunk_means) < std_values * 0.5

        var_values = statistics.variance(values) if len(values) > 1 else 0
        var_stability = len(chunk_vars) < 2 or (max(chunk_vars) - min(chunk_vars) < var_values * 0.5)

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

        if n * sum_x2 - sum_x * sum_x == 0:
            # No trend, use last value
            return [values[-1]] * periods

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
        std_val = statistics.stdev(values) if len(values) > 1 else 0

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