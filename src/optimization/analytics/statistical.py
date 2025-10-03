"""
Statistical analysis and hypothesis testing.

This module provides comprehensive statistical analysis capabilities including:
- Descriptive statistics calculation
- Correlation analysis between datasets
- Hypothesis testing
- Distribution analysis
- Outlier detection
"""

import statistics
from typing import Dict, Any, List, Union

from .core import BaseAnalyzer, AnalyticsResult, AnalyticsError, TimeSeriesData

try:
    import scipy.stats as stats
except ImportError:
    stats = None


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
            'p5': sorted_values[n // 20] if n >= 20 else sorted_values[0],
            'p95': sorted_values[19 * n // 20] if n >= 20 else sorted_values[-1],
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
        mean_val = basic_stats.get('mean', 0)
        median_val = basic_stats.get('median', 0)
        if mean_val != median_val and median_val != 0:
            diff_pct = abs(mean_val - median_val) / median_val * 100
            if diff_pct > 10:
                insights.append(f"Distribution is skewed (mean-median difference: {diff_pct:.1f}%)")

        # Variability assessment
        std_val = basic_stats.get('std', 0)
        mean_val = basic_stats.get('mean', 0)
        if mean_val != 0:
            cv = std_val / mean_val * 100
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