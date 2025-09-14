"""
Tests for Correlation Analyzer

Tests the cross-strategy correlation analysis and risk decomposition functionality.
Follows TDD methodology with comprehensive test coverage.
"""

import pytest
import numpy as np
import pandas as pd
from dataclasses import asdict
from unittest.mock import Mock, patch

from src.portfolio.correlation_analyzer import (
    CorrelationAnalyzer,
    CorrelationMatrix,
    CorrelationConfig,
    RiskDecomposition,
    DiversificationMetrics
)


class TestCorrelationAnalyzerInitialization:
    """Test correlation analyzer initialization and configuration"""

    def test_should_create_analyzer_with_default_config(self):
        """Test correlation analyzer creation with default parameters"""
        analyzer = CorrelationAnalyzer()

        assert analyzer.window_size == 60  # 60-day rolling window
        assert analyzer.min_periods == 30
        assert analyzer.method == 'pearson'
        assert analyzer.decay_factor == 0.94
        assert len(analyzer.strategy_returns) == 0

    def test_should_create_analyzer_with_custom_config(self):
        """Test correlation analyzer with custom configuration"""
        config = CorrelationConfig(
            window_size=120,
            min_periods=60,
            method='spearman',
            decay_factor=0.9
        )

        analyzer = CorrelationAnalyzer(config)

        assert analyzer.window_size == 120
        assert analyzer.min_periods == 60
        assert analyzer.method == 'spearman'
        assert analyzer.decay_factor == 0.9

    def test_should_validate_config_parameters(self):
        """Test configuration parameter validation"""
        with pytest.raises(ValueError, match="Window size must be positive"):
            CorrelationConfig(window_size=0)

        with pytest.raises(ValueError, match="Min periods must be positive"):
            CorrelationConfig(min_periods=0)

        with pytest.raises(ValueError, match="Invalid correlation method"):
            CorrelationConfig(method='invalid_method')

        with pytest.raises(ValueError, match="Decay factor must be between 0 and 1"):
            CorrelationConfig(decay_factor=1.5)


class TestDataManagement:
    """Test strategy data management"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        return CorrelationAnalyzer()

    @pytest.fixture
    def sample_strategy_returns(self):
        """Create sample strategy return data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Create correlated strategies
        base_factor = np.random.normal(0, 1, 100)

        strategies = {
            'TrendFollowing': base_factor * 0.6 + np.random.normal(0, 0.8, 100),
            'MeanReversion': -base_factor * 0.4 + np.random.normal(0, 0.9, 100),  # Negatively correlated
            'RangeTrading': base_factor * 0.3 + np.random.normal(0, 0.7, 100),
            'FundingArbitrage': np.random.normal(0, 0.5, 100)  # Independent
        }

        # Convert to returns (normalize)
        strategy_returns = {}
        for name, values in strategies.items():
            returns = values * 0.001  # Scale to reasonable return magnitudes
            strategy_returns[name] = pd.Series(returns, index=dates)

        return strategy_returns

    def test_should_add_strategy_data(self, analyzer, sample_strategy_returns):
        """Test adding strategy return data"""
        for strategy_name, returns in sample_strategy_returns.items():
            analyzer.add_strategy_returns(strategy_name, returns)

        assert len(analyzer.strategy_returns) == 4
        assert 'TrendFollowing' in analyzer.strategy_returns
        assert 'MeanReversion' in analyzer.strategy_returns
        assert 'RangeTrading' in analyzer.strategy_returns
        assert 'FundingArbitrage' in analyzer.strategy_returns

        # Check data integrity
        for strategy_name, returns in sample_strategy_returns.items():
            stored_returns = analyzer.strategy_returns[strategy_name]
            pd.testing.assert_series_equal(stored_returns, returns)

    def test_should_validate_strategy_data_format(self, analyzer):
        """Test strategy data format validation"""
        # Invalid returns type
        with pytest.raises(TypeError, match="Returns must be a pandas Series"):
            analyzer.add_strategy_returns('TestStrategy', [1, 2, 3])

        # Empty returns
        empty_series = pd.Series([])
        with pytest.raises(ValueError, match="Returns series cannot be empty"):
            analyzer.add_strategy_returns('TestStrategy', empty_series)

    def test_should_handle_strategy_data_update(self, analyzer, sample_strategy_returns):
        """Test updating existing strategy data"""
        # Add initial data
        initial_returns = sample_strategy_returns['TrendFollowing']
        analyzer.add_strategy_returns('TrendFollowing', initial_returns)

        # Update with new data
        new_dates = pd.date_range('2023-04-11', periods=50, freq='D')
        new_returns = pd.Series(np.random.normal(0.001, 0.02, 50), index=new_dates)

        analyzer.add_strategy_returns('TrendFollowing', new_returns, replace=False)

        # Should have combined data
        combined_returns = analyzer.strategy_returns['TrendFollowing']
        assert len(combined_returns) == len(initial_returns) + len(new_returns)

    def test_should_replace_strategy_data_when_specified(self, analyzer, sample_strategy_returns):
        """Test replacing strategy data entirely"""
        # Add initial data
        initial_returns = sample_strategy_returns['TrendFollowing']
        analyzer.add_strategy_returns('TrendFollowing', initial_returns)

        # Replace with new data
        new_dates = pd.date_range('2023-06-01', periods=30, freq='D')
        new_returns = pd.Series(np.random.normal(0.002, 0.015, 30), index=new_dates)

        analyzer.add_strategy_returns('TrendFollowing', new_returns, replace=True)

        # Should have only new data
        stored_returns = analyzer.strategy_returns['TrendFollowing']
        pd.testing.assert_series_equal(stored_returns, new_returns)


class TestCorrelationCalculation:
    """Test correlation matrix calculation"""

    @pytest.fixture
    def analyzer_with_data(self):
        """Create analyzer with sample data"""
        analyzer = CorrelationAnalyzer()

        # Create controlled correlation structure
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Strategy 1 and 2 are positively correlated
        base_returns = np.random.normal(0.001, 0.02, 100)
        strategy1 = base_returns + np.random.normal(0, 0.005, 100)
        strategy2 = base_returns + np.random.normal(0, 0.008, 100)

        # Strategy 3 is negatively correlated with Strategy 1
        strategy3 = -0.7 * base_returns + np.random.normal(0, 0.01, 100)

        # Strategy 4 is independent
        strategy4 = np.random.normal(0.0005, 0.015, 100)

        analyzer.add_strategy_returns('Strategy1', pd.Series(strategy1, index=dates))
        analyzer.add_strategy_returns('Strategy2', pd.Series(strategy2, index=dates))
        analyzer.add_strategy_returns('Strategy3', pd.Series(strategy3, index=dates))
        analyzer.add_strategy_returns('Strategy4', pd.Series(strategy4, index=dates))

        return analyzer

    def test_should_calculate_static_correlation_matrix(self, analyzer_with_data):
        """Test static correlation matrix calculation"""
        correlation_result = analyzer_with_data.calculate_correlation_matrix()

        assert isinstance(correlation_result, CorrelationMatrix)
        assert correlation_result.matrix.shape == (4, 4)

        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(correlation_result.matrix), 1.0)

        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(correlation_result.matrix, correlation_result.matrix.T)

        # Strategy1 and Strategy2 should be positively correlated
        corr_1_2 = correlation_result.matrix.loc['Strategy1', 'Strategy2']
        assert corr_1_2 > 0.5

        # Strategy1 and Strategy3 should be negatively correlated
        corr_1_3 = correlation_result.matrix.loc['Strategy1', 'Strategy3']
        assert corr_1_3 < -0.3

    def test_should_calculate_rolling_correlation_matrix(self, analyzer_with_data):
        """Test rolling correlation matrix calculation"""
        rolling_correlations = analyzer_with_data.calculate_rolling_correlation_matrix(window=30)

        assert isinstance(rolling_correlations, list)
        assert len(rolling_correlations) > 0

        # Each result should be a CorrelationMatrix
        for result in rolling_correlations:
            assert isinstance(result, CorrelationMatrix)
            assert result.matrix.shape == (4, 4)

        # Should have correct number of rolling windows
        expected_windows = 100 - 30 + 1  # 71 windows
        assert len(rolling_correlations) == expected_windows

    def test_should_calculate_exponentially_weighted_correlation(self, analyzer_with_data):
        """Test exponentially weighted correlation calculation"""
        correlation_result = analyzer_with_data.calculate_correlation_matrix(
            method='exponential',
            decay_factor=0.94
        )

        assert isinstance(correlation_result, CorrelationMatrix)
        assert correlation_result.method == 'exponential'
        assert correlation_result.decay_factor == 0.94
        assert correlation_result.matrix.shape == (4, 4)

    def test_should_handle_insufficient_data_gracefully(self):
        """Test handling of insufficient data for correlation calculation"""
        analyzer = CorrelationAnalyzer()

        # Add single strategy
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
        analyzer.add_strategy_returns('OnlyStrategy', returns)

        # Should handle gracefully
        correlation_result = analyzer.calculate_correlation_matrix()
        assert correlation_result.matrix.shape == (1, 1)
        assert correlation_result.matrix.iloc[0, 0] == 1.0

    def test_should_handle_different_correlation_methods(self, analyzer_with_data):
        """Test different correlation calculation methods"""
        # Pearson correlation
        pearson_result = analyzer_with_data.calculate_correlation_matrix(method='pearson')
        assert pearson_result.method == 'pearson'

        # Spearman correlation
        spearman_result = analyzer_with_data.calculate_correlation_matrix(method='spearman')
        assert spearman_result.method == 'spearman'

        # Kendall correlation
        kendall_result = analyzer_with_data.calculate_correlation_matrix(method='kendall')
        assert kendall_result.method == 'kendall'

        # Results should be different but similar in structure
        assert pearson_result.matrix.shape == spearman_result.matrix.shape
        assert not np.allclose(pearson_result.matrix.values, spearman_result.matrix.values, rtol=0.1)


class TestRiskDecomposition:
    """Test portfolio risk decomposition"""

    @pytest.fixture
    def analyzer_with_weights(self):
        """Create analyzer with strategy returns and weights"""
        analyzer = CorrelationAnalyzer()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Create strategies with known risk characteristics
        low_vol_returns = np.random.normal(0.0003, 0.01, 100)
        medium_vol_returns = np.random.normal(0.0008, 0.02, 100)
        high_vol_returns = np.random.normal(0.0012, 0.03, 100)

        analyzer.add_strategy_returns('LowVol', pd.Series(low_vol_returns, index=dates))
        analyzer.add_strategy_returns('MediumVol', pd.Series(medium_vol_returns, index=dates))
        analyzer.add_strategy_returns('HighVol', pd.Series(high_vol_returns, index=dates))

        return analyzer

    def test_should_decompose_portfolio_risk(self, analyzer_with_weights):
        """Test portfolio risk decomposition calculation"""
        # Define portfolio weights
        weights = {'LowVol': 0.4, 'MediumVol': 0.4, 'HighVol': 0.2}

        risk_decomp = analyzer_with_weights.calculate_risk_decomposition(weights)

        assert isinstance(risk_decomp, RiskDecomposition)
        assert len(risk_decomp.individual_risks) == 3
        assert len(risk_decomp.marginal_contributions) == 3
        assert len(risk_decomp.component_contributions) == 3

        # Total risk should be positive
        assert risk_decomp.total_portfolio_risk > 0

        # Component contributions should sum to total risk (approximately)
        total_contrib = sum(risk_decomp.component_contributions.values())
        assert abs(total_contrib - risk_decomp.total_portfolio_risk) < 1e-10

    def test_should_calculate_marginal_risk_contributions(self, analyzer_with_weights):
        """Test marginal risk contribution calculations"""
        weights = {'LowVol': 0.5, 'MediumVol': 0.3, 'HighVol': 0.2}

        risk_decomp = analyzer_with_weights.calculate_risk_decomposition(weights)

        # Marginal contributions should reflect individual strategy risks
        # HighVol should have higher marginal contribution than LowVol
        assert risk_decomp.marginal_contributions['HighVol'] > risk_decomp.marginal_contributions['LowVol']

        # All marginal contributions should be positive (for positive weights)
        for strategy, marginal in risk_decomp.marginal_contributions.items():
            assert marginal >= 0

    def test_should_handle_different_weight_scenarios(self, analyzer_with_weights):
        """Test risk decomposition under different weight scenarios"""
        # Equal weights
        equal_weights = {'LowVol': 1/3, 'MediumVol': 1/3, 'HighVol': 1/3}
        equal_risk_decomp = analyzer_with_weights.calculate_risk_decomposition(equal_weights)

        # Concentrated in low volatility
        low_vol_weights = {'LowVol': 0.8, 'MediumVol': 0.1, 'HighVol': 0.1}
        low_vol_risk_decomp = analyzer_with_weights.calculate_risk_decomposition(low_vol_weights)

        # Low volatility concentration should result in lower total risk
        assert low_vol_risk_decomp.total_portfolio_risk < equal_risk_decomp.total_portfolio_risk

    def test_should_validate_weights_input(self, analyzer_with_weights):
        """Test weight input validation"""
        # Weights don't sum to 1
        invalid_weights = {'LowVol': 0.3, 'MediumVol': 0.3, 'HighVol': 0.3}
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            analyzer_with_weights.calculate_risk_decomposition(invalid_weights)

        # Missing strategy
        incomplete_weights = {'LowVol': 0.5, 'MediumVol': 0.5}
        with pytest.raises(ValueError, match="Weights must be provided for all strategies"):
            analyzer_with_weights.calculate_risk_decomposition(incomplete_weights)

        # Negative weights
        negative_weights = {'LowVol': 0.6, 'MediumVol': 0.6, 'HighVol': -0.2}
        # Should handle gracefully (short positions allowed)
        risk_decomp = analyzer_with_weights.calculate_risk_decomposition(negative_weights)
        assert isinstance(risk_decomp, RiskDecomposition)


class TestDiversificationAnalysis:
    """Test diversification metrics calculation"""

    @pytest.fixture
    def diversification_analyzer(self):
        """Create analyzer for diversification testing"""
        analyzer = CorrelationAnalyzer()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Create strategies with varying correlation
        base_factor = np.random.normal(0, 1, 100)

        # Highly correlated strategies
        corr_strategy1 = base_factor * 0.8 + np.random.normal(0, 0.2, 100)
        corr_strategy2 = base_factor * 0.8 + np.random.normal(0, 0.2, 100)

        # Uncorrelated strategy
        uncorr_strategy = np.random.normal(0, 1, 100)

        analyzer.add_strategy_returns('Corr1', pd.Series(corr_strategy1 * 0.02, index=dates))
        analyzer.add_strategy_returns('Corr2', pd.Series(corr_strategy2 * 0.02, index=dates))
        analyzer.add_strategy_returns('Uncorr', pd.Series(uncorr_strategy * 0.02, index=dates))

        return analyzer

    def test_should_calculate_diversification_metrics(self, diversification_analyzer):
        """Test diversification metrics calculation"""
        # Equal weights
        weights = {'Corr1': 1/3, 'Corr2': 1/3, 'Uncorr': 1/3}

        diversification_metrics = diversification_analyzer.calculate_diversification_metrics(weights)

        assert isinstance(diversification_metrics, DiversificationMetrics)
        assert diversification_metrics.diversification_ratio >= 1.0  # Should be >= 1
        assert 0 <= diversification_metrics.effective_number_of_strategies <= 3
        assert diversification_metrics.concentration_ratio >= 0

    def test_should_calculate_diversification_ratio(self, diversification_analyzer):
        """Test diversification ratio calculation"""
        # Highly concentrated portfolio (should have lower diversification)
        concentrated_weights = {'Corr1': 0.8, 'Corr2': 0.1, 'Uncorr': 0.1}
        concentrated_metrics = diversification_analyzer.calculate_diversification_metrics(concentrated_weights)

        # Diversified portfolio
        diversified_weights = {'Corr1': 1/3, 'Corr2': 1/3, 'Uncorr': 1/3}
        diversified_metrics = diversification_analyzer.calculate_diversification_metrics(diversified_weights)

        # Diversified portfolio should have better diversification ratio
        assert diversified_metrics.diversification_ratio >= concentrated_metrics.diversification_ratio

    def test_should_calculate_effective_number_of_strategies(self, diversification_analyzer):
        """Test effective number of strategies calculation"""
        # Single strategy portfolio
        single_weights = {'Corr1': 1.0, 'Corr2': 0.0, 'Uncorr': 0.0}
        single_metrics = diversification_analyzer.calculate_diversification_metrics(single_weights)

        # Equal weight portfolio
        equal_weights = {'Corr1': 1/3, 'Corr2': 1/3, 'Uncorr': 1/3}
        equal_metrics = diversification_analyzer.calculate_diversification_metrics(equal_weights)

        # Equal weight should have higher effective number of strategies
        assert equal_metrics.effective_number_of_strategies > single_metrics.effective_number_of_strategies
        assert single_metrics.effective_number_of_strategies <= 1.1  # Should be close to 1


class TestCorrelationMatrix:
    """Test CorrelationMatrix data structure"""

    def test_should_create_correlation_matrix_with_required_fields(self):
        """Test CorrelationMatrix creation"""
        # Create sample correlation data
        strategies = ['Strategy1', 'Strategy2', 'Strategy3']
        correlation_data = np.array([
            [1.0, 0.6, -0.2],
            [0.6, 1.0, 0.3],
            [-0.2, 0.3, 1.0]
        ])

        correlation_df = pd.DataFrame(
            correlation_data,
            index=strategies,
            columns=strategies
        )

        correlation_matrix = CorrelationMatrix(
            matrix=correlation_df,
            method='pearson',
            timestamp=pd.Timestamp('2023-12-31'),
            window_size=60
        )

        assert correlation_matrix.matrix.shape == (3, 3)
        assert correlation_matrix.method == 'pearson'
        assert correlation_matrix.window_size == 60
        assert correlation_matrix.timestamp == pd.Timestamp('2023-12-31')

    def test_should_calculate_matrix_properties(self):
        """Test derived properties of correlation matrix"""
        strategies = ['A', 'B', 'C']
        correlation_data = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.1],
            [0.2, 0.1, 1.0]
        ])

        correlation_df = pd.DataFrame(
            correlation_data,
            index=strategies,
            columns=strategies
        )

        correlation_matrix = CorrelationMatrix(
            matrix=correlation_df,
            method='pearson'
        )

        # Test average correlation (excluding diagonal)
        expected_avg = (0.8 + 0.2 + 0.1) / 3
        assert abs(correlation_matrix.average_correlation - expected_avg) < 1e-10

        # Test max correlation (excluding diagonal)
        assert correlation_matrix.max_correlation == 0.8

        # Test min correlation
        assert correlation_matrix.min_correlation == 0.1


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_should_handle_single_strategy_portfolio(self):
        """Test correlation analysis with single strategy"""
        analyzer = CorrelationAnalyzer()

        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 50), index=dates)

        analyzer.add_strategy_returns('OnlyStrategy', returns)

        correlation_result = analyzer.calculate_correlation_matrix()
        assert correlation_result.matrix.shape == (1, 1)
        assert correlation_result.matrix.iloc[0, 0] == 1.0

        # Risk decomposition should work
        weights = {'OnlyStrategy': 1.0}
        risk_decomp = analyzer.calculate_risk_decomposition(weights)
        assert risk_decomp.total_portfolio_risk > 0

    def test_should_handle_perfectly_correlated_strategies(self):
        """Test handling of perfectly correlated strategies"""
        analyzer = CorrelationAnalyzer()

        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        base_returns = np.random.normal(0.001, 0.02, 50)

        # Perfectly correlated strategies
        analyzer.add_strategy_returns('Strategy1', pd.Series(base_returns, index=dates))
        analyzer.add_strategy_returns('Strategy2', pd.Series(base_returns, index=dates))

        correlation_result = analyzer.calculate_correlation_matrix()

        # Should handle perfect correlation
        assert abs(correlation_result.matrix.loc['Strategy1', 'Strategy2'] - 1.0) < 1e-10

    def test_should_handle_constant_returns(self):
        """Test handling of constant returns (zero variance)"""
        analyzer = CorrelationAnalyzer()

        dates = pd.date_range('2023-01-01', periods=50, freq='D')

        # One strategy with constant returns
        constant_returns = pd.Series([0.001] * 50, index=dates)
        variable_returns = pd.Series(np.random.normal(0.001, 0.02, 50), index=dates)

        analyzer.add_strategy_returns('Constant', constant_returns)
        analyzer.add_strategy_returns('Variable', variable_returns)

        # Should handle gracefully
        correlation_result = analyzer.calculate_correlation_matrix()
        assert isinstance(correlation_result, CorrelationMatrix)

        # Correlation with constant series should be NaN or 0
        corr_value = correlation_result.matrix.loc['Constant', 'Variable']
        assert np.isnan(corr_value) or abs(corr_value) < 1e-10

    def test_should_handle_misaligned_dates(self):
        """Test handling of strategies with different date ranges"""
        analyzer = CorrelationAnalyzer()

        # Strategy 1: Jan-Mar
        dates1 = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        returns1 = pd.Series(np.random.normal(0.001, 0.02, len(dates1)), index=dates1)

        # Strategy 2: Feb-Apr (overlapping)
        dates2 = pd.date_range('2023-02-01', '2023-04-30', freq='D')
        returns2 = pd.Series(np.random.normal(0.001, 0.02, len(dates2)), index=dates2)

        analyzer.add_strategy_returns('Strategy1', returns1)
        analyzer.add_strategy_returns('Strategy2', returns2)

        # Should use overlapping period for correlation
        correlation_result = analyzer.calculate_correlation_matrix()
        assert correlation_result.matrix.shape == (2, 2)
        assert not np.isnan(correlation_result.matrix.iloc[0, 1])

    def test_should_validate_rolling_window_parameters(self):
        """Test validation of rolling window parameters"""
        analyzer = CorrelationAnalyzer()

        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 30), index=dates)
        analyzer.add_strategy_returns('Strategy1', returns)

        # Window larger than data
        rolling_result = analyzer.calculate_rolling_correlation_matrix(window=50)
        assert len(rolling_result) == 0  # Should return empty list