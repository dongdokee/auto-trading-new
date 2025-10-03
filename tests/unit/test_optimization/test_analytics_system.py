"""
Comprehensive tests for the Advanced Analytics System.

Tests cover:
- Statistical analysis and hypothesis testing
- Time series analysis and forecasting
- Machine learning model training and evaluation
- Analytics system integration
- Data validation and error handling
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.optimization.analytics_system import (
    AnalyticsResult,
    TimeSeriesData,
    StatisticalAnalyzer,
    TimeSeriesAnalyzer,
    MachineLearningAnalyzer,
    AdvancedAnalyticsSystem,
    AnalyticsError
)


class TestAnalyticsResult:
    """Test AnalyticsResult data structure."""

    def test_should_initialize_with_required_fields(self):
        """Test that AnalyticsResult initializes with required fields."""
        result = AnalyticsResult(analysis_type="test_analysis")

        assert result.analysis_type == "test_analysis"
        assert isinstance(result.timestamp, datetime)
        assert result.results == {}
        assert result.metrics == {}
        assert result.insights == []
        assert result.confidence == 0.0
        assert result.data_points == 0

    def test_should_convert_to_dictionary(self):
        """Test that AnalyticsResult can convert to dictionary."""
        result = AnalyticsResult(
            analysis_type="test_analysis",
            data_points=100,
            confidence=0.95
        )
        result.add_insight("Test insight")

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict['analysis_type'] == "test_analysis"
        assert result_dict['data_points'] == 100
        assert result_dict['confidence'] == 0.95
        assert len(result_dict['insights']) == 1

    def test_should_add_insights(self):
        """Test that AnalyticsResult can add insights."""
        result = AnalyticsResult(analysis_type="test_analysis")

        result.add_insight("First insight")
        result.add_insight("Second insight")

        assert len(result.insights) == 2
        assert "First insight" in result.insights
        assert "Second insight" in result.insights

    def test_should_generate_summary(self):
        """Test that AnalyticsResult can generate summary."""
        result = AnalyticsResult(
            analysis_type="test_analysis",
            data_points=100,
            confidence=0.85
        )
        result.metrics["accuracy"] = 0.95
        result.add_insight("Test insight")

        summary = result.get_summary()

        assert "test_analysis" in summary
        assert "100" in summary
        assert "85" in summary and "%" in summary


class TestTimeSeriesData:
    """Test TimeSeriesData container."""

    def test_should_initialize_with_valid_data(self):
        """Test that TimeSeriesData initializes with valid data."""
        timestamps = [datetime.utcnow() + timedelta(hours=i) for i in range(5)]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        ts_data = TimeSeriesData(
            timestamps=timestamps,
            values=values,
            name="test_series"
        )

        assert ts_data.timestamps == timestamps
        assert ts_data.values == values
        assert ts_data.name == "test_series"
        assert ts_data.metadata == {}

    def test_should_validate_data_length(self):
        """Test that TimeSeriesData validates data length."""
        timestamps = [datetime.utcnow()]
        values = [1.0, 2.0]  # Different length

        with pytest.raises(ValueError, match="Timestamps and values must have the same length"):
            TimeSeriesData(timestamps=timestamps, values=values)

    def test_should_convert_to_dataframe(self):
        """Test that TimeSeriesData can convert to DataFrame."""
        import pandas as pd

        timestamps = [datetime.utcnow() + timedelta(hours=i) for i in range(3)]
        values = [1.0, 2.0, 3.0]

        ts_data = TimeSeriesData(timestamps=timestamps, values=values)
        df = ts_data.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'timestamp' in df.columns
        assert 'value' in df.columns

    def test_should_calculate_statistics(self):
        """Test that TimeSeriesData can calculate statistics."""
        timestamps = [datetime.utcnow() + timedelta(hours=i) for i in range(5)]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        ts_data = TimeSeriesData(timestamps=timestamps, values=values)
        stats = ts_data.get_statistics()

        assert stats['count'] == 5
        assert stats['mean'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['range'] == 4.0

    def test_should_handle_empty_data(self):
        """Test that TimeSeriesData handles empty data gracefully."""
        ts_data = TimeSeriesData(timestamps=[], values=[])
        stats = ts_data.get_statistics()

        assert stats == {}


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer functionality."""

    @pytest.fixture
    def statistical_analyzer(self):
        """Create statistical analyzer for testing."""
        return StatisticalAnalyzer()

    def test_should_initialize_correctly(self, statistical_analyzer):
        """Test that StatisticalAnalyzer initializes correctly."""
        assert statistical_analyzer.name == "StatisticalAnalyzer"

    def test_should_analyze_list_data(self, statistical_analyzer):
        """Test that StatisticalAnalyzer can analyze list data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        result = statistical_analyzer.analyze(data)

        assert result.analysis_type == "statistical_analysis"
        assert result.data_points == 10
        assert 'basic_statistics' in result.results
        assert 'distribution' in result.results
        assert result.confidence == 0.95

    def test_should_analyze_timeseries_data(self, statistical_analyzer):
        """Test that StatisticalAnalyzer can analyze TimeSeriesData."""
        timestamps = [datetime.utcnow() + timedelta(hours=i) for i in range(10)]
        values = [float(i) for i in range(1, 11)]
        ts_data = TimeSeriesData(timestamps=timestamps, values=values)

        result = statistical_analyzer.analyze(ts_data)

        assert result.analysis_type == "statistical_analysis"
        assert result.data_points == 10
        assert result.metrics['mean'] == 5.5
        assert result.metrics['min'] == 1.0
        assert result.metrics['max'] == 10.0

    def test_should_calculate_basic_statistics(self, statistical_analyzer):
        """Test that StatisticalAnalyzer calculates basic statistics correctly."""
        data = [2.0, 4.0, 6.0, 8.0, 10.0]

        result = statistical_analyzer.analyze(data)
        stats = result.results['basic_statistics']

        assert stats['count'] == 5
        assert stats['mean'] == 6.0
        assert stats['median'] == 6.0
        assert stats['min'] == 2.0
        assert stats['max'] == 10.0

    def test_should_perform_correlation_analysis(self, statistical_analyzer):
        """Test that StatisticalAnalyzer can perform correlation analysis."""
        data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        data2 = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect positive correlation

        result = statistical_analyzer.correlation_analysis(data1, data2)

        assert result.analysis_type == "correlation_analysis"
        assert result.data_points == 5
        assert abs(result.metrics['correlation'] - 1.0) < 0.001  # Should be close to 1
        assert result.results['correlation_strength'] == "Very Strong"

    def test_should_detect_no_correlation(self, statistical_analyzer):
        """Test that StatisticalAnalyzer detects no correlation."""
        data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        data2 = [5.0, 1.0, 3.0, 2.0, 4.0]  # No clear correlation

        result = statistical_analyzer.correlation_analysis(data1, data2)

        assert abs(result.metrics['correlation']) < 0.5
        assert result.results['correlation_strength'] in ["Very Weak", "Weak"]

    @pytest.mark.skipif(True, reason="scipy.stats not available in test environment")
    def test_should_perform_hypothesis_test(self, statistical_analyzer):
        """Test that StatisticalAnalyzer can perform hypothesis testing."""
        sample1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample2 = [6.0, 7.0, 8.0, 9.0, 10.0]

        result = statistical_analyzer.hypothesis_test(sample1, sample2)

        assert result.analysis_type == "hypothesis_test_t_test"
        assert 'test_statistic' in result.results
        assert 'p_value' in result.results
        assert 'is_significant' in result.results

    def test_should_handle_invalid_data(self, statistical_analyzer):
        """Test that StatisticalAnalyzer handles invalid data."""
        with pytest.raises(AnalyticsError, match="Invalid data for statistical analysis"):
            statistical_analyzer.analyze(None)

    def test_should_handle_mismatched_correlation_data(self, statistical_analyzer):
        """Test that StatisticalAnalyzer handles mismatched correlation data."""
        data1 = [1.0, 2.0, 3.0]
        data2 = [1.0, 2.0]  # Different length

        with pytest.raises(AnalyticsError, match="Data series must have the same length"):
            statistical_analyzer.correlation_analysis(data1, data2)

    def test_should_generate_statistical_insights(self, statistical_analyzer):
        """Test that StatisticalAnalyzer generates meaningful insights."""
        # Data with outliers
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]

        result = statistical_analyzer.analyze(data)

        assert len(result.insights) > 0
        # Should detect high variability or outliers
        insight_text = " ".join(result.insights).lower()
        assert any(keyword in insight_text for keyword in ["outlier", "variability", "skewed"])


class TestTimeSeriesAnalyzer:
    """Test TimeSeriesAnalyzer functionality."""

    @pytest.fixture
    def timeseries_analyzer(self):
        """Create time series analyzer for testing."""
        return TimeSeriesAnalyzer()

    @pytest.fixture
    def sample_timeseries(self):
        """Create sample time series data for testing."""
        timestamps = [datetime.utcnow() + timedelta(hours=i) for i in range(20)]
        values = [float(i) + np.sin(i * 0.5) for i in range(20)]  # Trend + noise
        return TimeSeriesData(timestamps=timestamps, values=values, name="test_series")

    def test_should_initialize_correctly(self, timeseries_analyzer):
        """Test that TimeSeriesAnalyzer initializes correctly."""
        assert timeseries_analyzer.name == "TimeSeriesAnalyzer"

    def test_should_analyze_timeseries(self, timeseries_analyzer, sample_timeseries):
        """Test that TimeSeriesAnalyzer can analyze time series."""
        result = timeseries_analyzer.analyze(sample_timeseries)

        assert result.analysis_type == "time_series_analysis"
        assert result.data_points == 20
        assert 'trend' in result.results
        assert 'volatility' in result.results
        assert 'stationarity' in result.results
        assert result.confidence == 0.85

    def test_should_analyze_trend(self, timeseries_analyzer):
        """Test that TimeSeriesAnalyzer can analyze trend."""
        # Create upward trending data
        timestamps = [datetime.utcnow() + timedelta(hours=i) for i in range(10)]
        values = [float(i) * 2 for i in range(10)]  # Clear upward trend
        ts_data = TimeSeriesData(timestamps=timestamps, values=values)

        result = timeseries_analyzer.analyze(ts_data)
        trend_stats = result.results['trend']

        assert trend_stats['trend_slope'] > 0
        assert trend_stats['trend_direction'] == 'increasing'
        assert trend_stats['trend_r_squared'] > 0.9  # Should be strong trend

    def test_should_generate_linear_forecast(self, timeseries_analyzer, sample_timeseries):
        """Test that TimeSeriesAnalyzer can generate linear forecasts."""
        result = timeseries_analyzer.forecast(sample_timeseries, periods=5, method="linear")

        assert result.analysis_type == "forecast_linear"
        assert result.data_points == 20
        assert len(result.results['forecasts']) == 5
        assert result.results['forecast_periods'] == 5
        assert result.results['method'] == "linear"

    def test_should_generate_moving_average_forecast(self, timeseries_analyzer, sample_timeseries):
        """Test that TimeSeriesAnalyzer can generate moving average forecasts."""
        result = timeseries_analyzer.forecast(sample_timeseries, periods=3, method="moving_average")

        assert result.analysis_type == "forecast_moving_average"
        assert len(result.results['forecasts']) == 3
        assert result.results['method'] == "moving_average"

    def test_should_detect_anomalies(self, timeseries_analyzer):
        """Test that TimeSeriesAnalyzer can detect anomalies."""
        # Create data with clear anomaly
        timestamps = [datetime.utcnow() + timedelta(hours=i) for i in range(10)]
        values = [1.0, 2.0, 3.0, 4.0, 100.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # 100.0 is anomaly
        ts_data = TimeSeriesData(timestamps=timestamps, values=values)

        result = timeseries_analyzer.detect_anomalies(ts_data, threshold=2.0)

        assert result.analysis_type == "anomaly_detection"
        assert result.metrics['anomaly_count'] > 0
        assert len(result.results['anomalies']) > 0

    def test_should_handle_insufficient_forecast_data(self, timeseries_analyzer):
        """Test that TimeSeriesAnalyzer handles insufficient data for forecasting."""
        timestamps = [datetime.utcnow()]
        values = [1.0]
        ts_data = TimeSeriesData(timestamps=timestamps, values=values)

        with pytest.raises(AnalyticsError, match="Insufficient data for forecasting"):
            timeseries_analyzer.forecast(ts_data, periods=5)

    def test_should_handle_unknown_forecast_method(self, timeseries_analyzer, sample_timeseries):
        """Test that TimeSeriesAnalyzer handles unknown forecast methods."""
        with pytest.raises(AnalyticsError, match="Unknown forecasting method"):
            timeseries_analyzer.forecast(sample_timeseries, periods=5, method="unknown_method")

    def test_should_validate_input_data(self, timeseries_analyzer):
        """Test that TimeSeriesAnalyzer validates input data."""
        with pytest.raises(AnalyticsError, match="Invalid time series data"):
            timeseries_analyzer.analyze(None)


class TestMachineLearningAnalyzer:
    """Test MachineLearningAnalyzer functionality."""

    @pytest.fixture
    def ml_analyzer(self):
        """Create ML analyzer for testing."""
        return MachineLearningAnalyzer()

    @pytest.fixture
    def sample_ml_data(self):
        """Create sample ML data for testing."""
        np.random.seed(42)
        features = [[float(i), float(i**2)] for i in range(1, 21)]  # 20 samples, 2 features
        targets = [2*i + 3*i**2 + np.random.normal(0, 0.1) for i in range(1, 21)]  # Linear relationship with noise
        return features, targets

    def test_should_initialize_correctly(self, ml_analyzer):
        """Test that MachineLearningAnalyzer initializes correctly."""
        assert ml_analyzer.name == "MachineLearningAnalyzer"
        assert ml_analyzer.models == {}
        assert ml_analyzer.scalers == {}

    @pytest.mark.skipif(True, reason="scikit-learn not available in test environment")
    def test_should_train_and_evaluate_models(self, ml_analyzer, sample_ml_data):
        """Test that MachineLearningAnalyzer can train and evaluate models."""
        features, targets = sample_ml_data

        result = ml_analyzer.analyze(features, targets)

        assert result.analysis_type == "machine_learning_analysis"
        assert result.data_points == 20
        assert 'models' in result.results
        assert 'best_model' in result.results
        assert 'linear_regression' in result.results['models']

    @pytest.mark.skipif(True, reason="scikit-learn not available in test environment")
    def test_should_make_predictions(self, ml_analyzer, sample_ml_data):
        """Test that MachineLearningAnalyzer can make predictions."""
        features, targets = sample_ml_data

        # Train model first
        ml_analyzer.analyze(features, targets)

        # Make predictions
        test_features = [[21.0, 441.0], [22.0, 484.0]]
        result = ml_analyzer.predict("linear_regression", test_features)

        assert result.analysis_type == "prediction_linear_regression"
        assert len(result.results['predictions']) == 2

    @pytest.mark.skipif(True, reason="scikit-learn not available in test environment")
    def test_should_analyze_feature_importance(self, ml_analyzer, sample_ml_data):
        """Test that MachineLearningAnalyzer can analyze feature importance."""
        features, targets = sample_ml_data

        # Train model first
        ml_analyzer.analyze(features, targets)

        # Analyze feature importance
        result = ml_analyzer.feature_importance("linear_regression")

        assert result.analysis_type == "feature_importance_linear_regression"
        assert 'coefficients' in result.results or 'feature_importances' in result.results

    def test_should_handle_insufficient_data(self, ml_analyzer):
        """Test that MachineLearningAnalyzer handles insufficient data."""
        features = [[1.0], [2.0]]  # Only 2 samples
        targets = [1.0, 2.0]

        with pytest.raises(AnalyticsError, match="Insufficient or mismatched data"):
            ml_analyzer.analyze(features, targets)

    def test_should_handle_mismatched_data(self, ml_analyzer):
        """Test that MachineLearningAnalyzer handles mismatched data."""
        features = [[1.0], [2.0], [3.0]]
        targets = [1.0, 2.0]  # Different length

        with pytest.raises(AnalyticsError, match="Insufficient or mismatched data"):
            ml_analyzer.analyze(features, targets)

    def test_should_handle_missing_model_prediction(self, ml_analyzer):
        """Test that MachineLearningAnalyzer handles missing model for prediction."""
        test_features = [[1.0, 2.0]]

        with pytest.raises(AnalyticsError, match="Model nonexistent_model not found"):
            ml_analyzer.predict("nonexistent_model", test_features)

    def test_should_handle_missing_model_feature_importance(self, ml_analyzer):
        """Test that MachineLearningAnalyzer handles missing model for feature importance."""
        with pytest.raises(AnalyticsError, match="Model nonexistent_model not found"):
            ml_analyzer.feature_importance("nonexistent_model")

    @pytest.mark.skipif(True, reason="scikit-learn not available in test environment")
    def test_should_require_sklearn(self, ml_analyzer, sample_ml_data):
        """Test that MachineLearningAnalyzer requires scikit-learn."""
        features, targets = sample_ml_data

        with patch('src.optimization.analytics_system.LinearRegression', None):
            with pytest.raises(AnalyticsError, match="scikit-learn is required"):
                ml_analyzer.analyze(features, targets)


class TestAdvancedAnalyticsSystem:
    """Test AdvancedAnalyticsSystem integration."""

    @pytest.fixture
    def analytics_system(self):
        """Create analytics system for testing."""
        return AdvancedAnalyticsSystem()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        timestamps = [datetime.utcnow() + timedelta(hours=i) for i in range(10)]
        values = [float(i) for i in range(1, 11)]
        return TimeSeriesData(timestamps=timestamps, values=values, name="test_data")

    @pytest.mark.asyncio
    async def test_should_initialize_successfully(self, analytics_system):
        """Test that AdvancedAnalyticsSystem initializes successfully."""
        await analytics_system.initialize()

        assert analytics_system.is_initialized is True
        assert isinstance(analytics_system.statistical_analyzer, StatisticalAnalyzer)
        assert isinstance(analytics_system.timeseries_analyzer, TimeSeriesAnalyzer)
        assert isinstance(analytics_system.ml_analyzer, MachineLearningAnalyzer)

    def test_should_require_initialization(self, analytics_system):
        """Test that AdvancedAnalyticsSystem requires initialization."""
        data = [1.0, 2.0, 3.0]

        with pytest.raises(AnalyticsError, match="Analytics system not initialized"):
            analytics_system.run_statistical_analysis(data)

    @pytest.mark.asyncio
    async def test_should_run_statistical_analysis(self, analytics_system, sample_data):
        """Test that AdvancedAnalyticsSystem can run statistical analysis."""
        await analytics_system.initialize()

        result = analytics_system.run_statistical_analysis(sample_data.values)

        assert result.analysis_type == "statistical_analysis"
        assert len(analytics_system.analysis_history) == 1

    @pytest.mark.asyncio
    async def test_should_run_correlation_analysis(self, analytics_system):
        """Test that AdvancedAnalyticsSystem can run correlation analysis."""
        await analytics_system.initialize()

        data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        data2 = [2.0, 4.0, 6.0, 8.0, 10.0]

        result = analytics_system.run_correlation_analysis(data1, data2)

        assert result.analysis_type == "correlation_analysis"
        assert len(analytics_system.analysis_history) == 1

    @pytest.mark.asyncio
    async def test_should_run_timeseries_analysis(self, analytics_system, sample_data):
        """Test that AdvancedAnalyticsSystem can run time series analysis."""
        await analytics_system.initialize()

        result = analytics_system.run_timeseries_analysis(sample_data)

        assert result.analysis_type == "time_series_analysis"
        assert len(analytics_system.analysis_history) == 1

    @pytest.mark.asyncio
    async def test_should_run_forecast(self, analytics_system, sample_data):
        """Test that AdvancedAnalyticsSystem can run forecasting."""
        await analytics_system.initialize()

        result = analytics_system.run_forecast(sample_data, periods=5, method="linear")

        assert result.analysis_type == "forecast_linear"
        assert len(analytics_system.analysis_history) == 1

    @pytest.mark.asyncio
    async def test_should_run_anomaly_detection(self, analytics_system, sample_data):
        """Test that AdvancedAnalyticsSystem can run anomaly detection."""
        await analytics_system.initialize()

        result = analytics_system.run_anomaly_detection(sample_data, threshold=2.0)

        assert result.analysis_type == "anomaly_detection"
        assert len(analytics_system.analysis_history) == 1

    @pytest.mark.asyncio
    async def test_should_run_ml_analysis(self, analytics_system):
        """Test that AdvancedAnalyticsSystem can run ML analysis."""
        await analytics_system.initialize()

        features = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]
        targets = [3.0, 6.0, 9.0, 12.0, 15.0]

        # Mock sklearn to avoid dependency
        with patch('src.optimization.analytics_system.LinearRegression') as mock_lr:
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([3.0, 6.0])
            mock_lr.return_value = mock_model

            with patch('src.optimization.analytics_system.train_test_split') as mock_split:
                mock_split.return_value = (features[:3], features[3:], targets[:3], targets[3:])

                with patch('src.optimization.analytics_system.mean_squared_error', return_value=0.1), \
                     patch('src.optimization.analytics_system.mean_absolute_error', return_value=0.2), \
                     patch('src.optimization.analytics_system.r2_score', return_value=0.9), \
                     patch('src.optimization.analytics_system.StandardScaler') as mock_scaler:

                    mock_scaler_instance = Mock()
                    mock_scaler_instance.fit_transform.return_value = np.array(features)
                    mock_scaler_instance.transform.return_value = np.array(features)
                    mock_scaler.return_value = mock_scaler_instance

                    result = analytics_system.run_ml_analysis(features, targets)

                    assert result.analysis_type == "machine_learning_analysis"
                    assert len(analytics_system.analysis_history) == 1

    @pytest.mark.asyncio
    async def test_should_get_analysis_history(self, analytics_system, sample_data):
        """Test that AdvancedAnalyticsSystem can get analysis history."""
        await analytics_system.initialize()

        # Run multiple analyses
        analytics_system.run_statistical_analysis(sample_data.values)
        analytics_system.run_timeseries_analysis(sample_data)

        # Get all history
        all_history = analytics_system.get_analysis_history()
        assert len(all_history) == 2

        # Get filtered history
        stat_history = analytics_system.get_analysis_history("statistical_analysis")
        assert len(stat_history) == 1
        assert stat_history[0].analysis_type == "statistical_analysis"

    @pytest.mark.asyncio
    async def test_should_export_analysis_report(self, analytics_system, sample_data):
        """Test that AdvancedAnalyticsSystem can export analysis report."""
        await analytics_system.initialize()

        # Run some analyses
        analytics_system.run_statistical_analysis(sample_data.values)
        analytics_system.run_timeseries_analysis(sample_data)

        report = analytics_system.export_analysis_report(hours=24)

        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'total_analyses' in report
        assert 'analysis_types' in report
        assert 'analyses' in report
        assert 'summary' in report
        assert report['total_analyses'] == 2

    @pytest.mark.asyncio
    async def test_should_clear_history(self, analytics_system, sample_data):
        """Test that AdvancedAnalyticsSystem can clear history."""
        await analytics_system.initialize()

        # Add some analyses
        analytics_system.run_statistical_analysis(sample_data.values)
        assert len(analytics_system.analysis_history) == 1

        # Clear history
        analytics_system.clear_history()
        assert len(analytics_system.analysis_history) == 0

    @pytest.mark.asyncio
    async def test_should_handle_initialization_errors(self):
        """Test that AdvancedAnalyticsSystem handles initialization errors."""
        analytics_system = AdvancedAnalyticsSystem()

        # Simulate error during initialization
        with patch('src.optimization.analytics_system.logger.info', side_effect=Exception("Test error")):
            with pytest.raises(AnalyticsError, match="Failed to initialize analytics system"):
                await analytics_system.initialize()

            assert analytics_system.is_initialized is False

    @pytest.mark.asyncio
    async def test_should_integrate_all_components(self, analytics_system):
        """Test that AdvancedAnalyticsSystem integrates all components correctly."""
        await analytics_system.initialize()

        # Create comprehensive test data
        timestamps = [datetime.utcnow() + timedelta(hours=i) for i in range(20)]
        values = [float(i) + np.sin(i * 0.3) for i in range(20)]
        ts_data = TimeSeriesData(timestamps=timestamps, values=values, name="integration_test")

        # Run statistical analysis
        stat_result = analytics_system.run_statistical_analysis(ts_data)
        assert stat_result.analysis_type == "statistical_analysis"

        # Run time series analysis
        ts_result = analytics_system.run_timeseries_analysis(ts_data)
        assert ts_result.analysis_type == "time_series_analysis"

        # Run forecasting
        forecast_result = analytics_system.run_forecast(ts_data, periods=3)
        assert forecast_result.analysis_type == "forecast_linear"

        # Run anomaly detection
        anomaly_result = analytics_system.run_anomaly_detection(ts_data)
        assert anomaly_result.analysis_type == "anomaly_detection"

        # Check that all analyses are stored
        assert len(analytics_system.analysis_history) == 4

        # Export comprehensive report
        report = analytics_system.export_analysis_report()
        assert report['total_analyses'] == 4
        assert len(report['analysis_types']) == 4