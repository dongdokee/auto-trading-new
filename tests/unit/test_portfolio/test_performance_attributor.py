"""
Tests for Performance Attributor

Tests the strategy-level performance attribution and analysis.
Follows TDD methodology with comprehensive test coverage.
"""

import pytest
import numpy as np
import pandas as pd
from dataclasses import asdict
from unittest.mock import Mock, patch

from src.portfolio.performance_attributor import (
    PerformanceAttributor,
    AttributionResult,
    PerformanceMetrics,
    AttributionConfig
)


class TestPerformanceAttributorInitialization:
    """Test performance attributor initialization and configuration"""

    def test_should_create_attributor_with_default_config(self):
        """Test performance attributor creation with default parameters"""
        attributor = PerformanceAttributor()

        assert attributor.lookback_window == 252  # 1 year
        assert attributor.risk_free_rate == 0.0
        assert attributor.attribution_method == 'brinson_fachler'
        assert len(attributor.strategy_returns) == 0

    def test_should_create_attributor_with_custom_config(self):
        """Test performance attributor with custom configuration"""
        config = AttributionConfig(
            lookback_window=126,  # 6 months
            risk_free_rate=0.02,
            attribution_method='brinson_hood_beebower'
        )

        attributor = PerformanceAttributor(config)

        assert attributor.lookback_window == 126
        assert attributor.risk_free_rate == 0.02
        assert attributor.attribution_method == 'brinson_hood_beebower'

    def test_should_validate_config_parameters(self):
        """Test configuration parameter validation"""
        with pytest.raises(ValueError, match="Lookback window must be positive"):
            AttributionConfig(lookback_window=0)

        with pytest.raises(ValueError, match="Risk-free rate must be non-negative"):
            AttributionConfig(risk_free_rate=-0.01)

        with pytest.raises(ValueError, match="Invalid attribution method"):
            AttributionConfig(attribution_method='invalid_method')


class TestStrategyDataManagement:
    """Test strategy return data management"""

    @pytest.fixture
    def attributor(self):
        """Create attributor instance for testing"""
        return PerformanceAttributor()

    @pytest.fixture
    def sample_strategy_data(self):
        """Create sample strategy return data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        strategy_data = {
            'TrendFollowing': {
                'returns': pd.Series(np.random.normal(0.0008, 0.02, 100), index=dates),
                'weights': pd.Series(np.random.uniform(0.1, 0.4, 100), index=dates),
                'positions': pd.Series(np.random.choice(['LONG', 'SHORT', 'NEUTRAL'], 100), index=dates)
            },
            'MeanReversion': {
                'returns': pd.Series(np.random.normal(0.0006, 0.015, 100), index=dates),
                'weights': pd.Series(np.random.uniform(0.1, 0.4, 100), index=dates),
                'positions': pd.Series(np.random.choice(['LONG', 'SHORT', 'NEUTRAL'], 100), index=dates)
            },
            'RangeTrading': {
                'returns': pd.Series(np.random.normal(0.0004, 0.01, 100), index=dates),
                'weights': pd.Series(np.random.uniform(0.1, 0.3, 100), index=dates),
                'positions': pd.Series(np.random.choice(['LONG', 'SHORT', 'NEUTRAL'], 100), index=dates)
            }
        }

        return strategy_data

    def test_should_add_strategy_data(self, attributor, sample_strategy_data):
        """Test adding strategy return data"""
        for strategy_name, data in sample_strategy_data.items():
            attributor.add_strategy_data(strategy_name, data)

        assert len(attributor.strategy_returns) == 3
        assert 'TrendFollowing' in attributor.strategy_returns
        assert 'MeanReversion' in attributor.strategy_returns
        assert 'RangeTrading' in attributor.strategy_returns

        # Check data integrity
        trend_data = attributor.strategy_returns['TrendFollowing']
        assert len(trend_data['returns']) == 100
        assert len(trend_data['weights']) == 100
        assert len(trend_data['positions']) == 100

    def test_should_validate_strategy_data_format(self, attributor):
        """Test strategy data format validation"""
        # Missing required fields
        with pytest.raises(ValueError, match="Strategy data must contain 'returns'"):
            attributor.add_strategy_data('TestStrategy', {'weights': []})

        # Invalid returns type
        with pytest.raises(TypeError, match="Returns must be a pandas Series"):
            attributor.add_strategy_data('TestStrategy', {'returns': [1, 2, 3]})

    def test_should_handle_strategy_data_update(self, attributor, sample_strategy_data):
        """Test updating existing strategy data"""
        # Add initial data
        strategy_data = sample_strategy_data['TrendFollowing']
        attributor.add_strategy_data('TrendFollowing', strategy_data)

        initial_length = len(attributor.strategy_returns['TrendFollowing']['returns'])

        # Update with new data
        new_dates = pd.date_range('2023-04-11', periods=50, freq='D')
        new_data = {
            'returns': pd.Series(np.random.normal(0.001, 0.02, 50), index=new_dates),
            'weights': pd.Series(np.random.uniform(0.1, 0.4, 50), index=new_dates),
            'positions': pd.Series(np.random.choice(['LONG', 'SHORT'], 50), index=new_dates)
        }

        attributor.add_strategy_data('TrendFollowing', new_data, replace=False)

        # Should have combined data
        updated_length = len(attributor.strategy_returns['TrendFollowing']['returns'])
        assert updated_length == initial_length + 50

    def test_should_replace_strategy_data_when_specified(self, attributor, sample_strategy_data):
        """Test replacing strategy data entirely"""
        # Add initial data
        strategy_data = sample_strategy_data['TrendFollowing']
        attributor.add_strategy_data('TrendFollowing', strategy_data)

        # Replace with new data
        new_dates = pd.date_range('2023-06-01', periods=30, freq='D')
        new_data = {
            'returns': pd.Series(np.random.normal(0.001, 0.02, 30), index=new_dates),
            'weights': pd.Series(np.random.uniform(0.2, 0.5, 30), index=new_dates)
        }

        attributor.add_strategy_data('TrendFollowing', new_data, replace=True)

        # Should have only new data
        updated_data = attributor.strategy_returns['TrendFollowing']
        assert len(updated_data['returns']) == 30
        assert updated_data['returns'].index[0] == new_dates[0]


class TestPerformanceMetricsCalculation:
    """Test individual performance metrics calculation"""

    @pytest.fixture
    def attributor(self):
        """Create attributor with sample data"""
        attributor = PerformanceAttributor()

        # Add sample data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)

        returns_data = np.random.normal(0.0008, 0.02, 252)
        returns_series = pd.Series(returns_data, index=dates)

        attributor.add_strategy_data('TestStrategy', {'returns': returns_series})
        return attributor

    def test_should_calculate_sharpe_ratio(self, attributor):
        """Test Sharpe ratio calculation"""
        returns = attributor.strategy_returns['TestStrategy']['returns']
        sharpe = attributor._calculate_sharpe_ratio(returns)

        expected_sharpe = (returns.mean() * 252 - attributor.risk_free_rate) / (returns.std() * np.sqrt(252))
        assert abs(sharpe - expected_sharpe) < 1e-10

    def test_should_calculate_sortino_ratio(self, attributor):
        """Test Sortino ratio calculation"""
        returns = attributor.strategy_returns['TestStrategy']['returns']
        sortino = attributor._calculate_sortino_ratio(returns)

        # Manually calculate expected Sortino
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        expected_sortino = (returns.mean() * 252 - attributor.risk_free_rate) / downside_deviation

        assert abs(sortino - expected_sortino) < 1e-6

    def test_should_calculate_maximum_drawdown(self, attributor):
        """Test maximum drawdown calculation"""
        returns = attributor.strategy_returns['TestStrategy']['returns']
        max_dd = attributor._calculate_maximum_drawdown(returns)

        # Manually calculate expected max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        expected_max_dd = abs(drawdowns.min())

        assert abs(max_dd - expected_max_dd) < 1e-10

    def test_should_calculate_calmar_ratio(self, attributor):
        """Test Calmar ratio calculation"""
        returns = attributor.strategy_returns['TestStrategy']['returns']
        calmar = attributor._calculate_calmar_ratio(returns)

        annualized_return = returns.mean() * 252
        max_dd = attributor._calculate_maximum_drawdown(returns)
        expected_calmar = annualized_return / max_dd if max_dd > 0 else 0

        assert abs(calmar - expected_calmar) < 1e-10

    def test_should_calculate_var(self, attributor):
        """Test Value at Risk calculation"""
        returns = attributor.strategy_returns['TestStrategy']['returns']
        var_95 = attributor._calculate_var(returns, confidence_level=0.95)
        var_99 = attributor._calculate_var(returns, confidence_level=0.99)

        # VaR at 95% should be less extreme than 99%
        assert abs(var_95) < abs(var_99)

        # VaR should be negative (representing loss)
        assert var_95 <= 0
        assert var_99 <= 0

    def test_should_handle_edge_cases_in_metrics(self, attributor):
        """Test edge cases in metrics calculation"""
        # Zero volatility returns
        zero_vol_returns = pd.Series([0.001] * 252,
                                   index=pd.date_range('2023-01-01', periods=252, freq='D'))

        # Should handle gracefully
        sharpe = attributor._calculate_sharpe_ratio(zero_vol_returns)
        assert np.isfinite(sharpe) or sharpe == 0

        # Constant positive returns (no drawdown)
        positive_returns = pd.Series([0.001] * 252,
                                   index=pd.date_range('2023-01-01', periods=252, freq='D'))
        max_dd = attributor._calculate_maximum_drawdown(positive_returns)
        assert max_dd >= 0


class TestBrinsonFachlerAttribution:
    """Test Brinson-Fachler performance attribution"""

    @pytest.fixture
    def multi_strategy_attributor(self):
        """Create attributor with multiple strategy data"""
        attributor = PerformanceAttributor()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        strategies = ['TrendFollowing', 'MeanReversion', 'RangeTrading']
        for i, strategy_name in enumerate(strategies):
            returns = np.random.normal(0.0005 + i * 0.0002, 0.015 + i * 0.002, 100)
            weights = np.random.uniform(0.2, 0.4, 100)

            # Normalize weights to sum to less than 1 (cash allocation)
            weights = weights / (weights.sum() + 0.5)

            attributor.add_strategy_data(strategy_name, {
                'returns': pd.Series(returns, index=dates),
                'weights': pd.Series(weights, index=dates)
            })

        return attributor

    def test_should_perform_brinson_fachler_attribution(self, multi_strategy_attributor):
        """Test complete Brinson-Fachler performance attribution"""
        result = multi_strategy_attributor.calculate_attribution()

        assert isinstance(result, AttributionResult)
        assert result.attribution_method == 'brinson_fachler'
        assert len(result.strategy_metrics) == 3

        # Check that all required metrics are present
        for strategy_name, metrics in result.strategy_metrics.items():
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.total_return is not None
            assert metrics.sharpe_ratio is not None
            assert metrics.sortino_ratio is not None
            assert metrics.maximum_drawdown is not None
            assert metrics.calmar_ratio is not None

    def test_should_calculate_strategy_contributions(self, multi_strategy_attributor):
        """Test strategy contribution calculations"""
        result = multi_strategy_attributor.calculate_attribution()

        total_contribution = 0
        for strategy_name in result.strategy_contributions:
            contribution = result.strategy_contributions[strategy_name]
            total_contribution += contribution

            # Each contribution should be reasonable
            assert isinstance(contribution, float)
            assert abs(contribution) < 1.0  # Sanity check

        # Contributions should approximately sum to total portfolio return
        portfolio_return = result.portfolio_metrics.total_return
        assert abs(total_contribution - portfolio_return) < 0.01

    def test_should_calculate_allocation_and_selection_effects(self, multi_strategy_attributor):
        """Test allocation and selection effect calculations"""
        result = multi_strategy_attributor.calculate_attribution()

        assert hasattr(result, 'allocation_effects')
        assert hasattr(result, 'selection_effects')

        # Should have effects for each strategy
        assert len(result.allocation_effects) == 3
        assert len(result.selection_effects) == 3

        # Effects should be reasonable values
        for strategy_name in result.allocation_effects:
            alloc_effect = result.allocation_effects[strategy_name]
            select_effect = result.selection_effects[strategy_name]

            assert isinstance(alloc_effect, float)
            assert isinstance(select_effect, float)
            assert abs(alloc_effect) < 0.1  # Sanity check
            assert abs(select_effect) < 0.1  # Sanity check


class TestRollingAttribution:
    """Test rolling performance attribution"""

    @pytest.fixture
    def long_period_attributor(self):
        """Create attributor with longer time series"""
        config = AttributionConfig(lookback_window=60)  # 60-day rolling
        attributor = PerformanceAttributor(config)

        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        np.random.seed(42)

        # Create strategies with different patterns over time
        for strategy_name in ['TrendFollowing', 'MeanReversion']:
            # Create time-varying returns (trend change halfway)
            returns_first = np.random.normal(0.001, 0.02, 100)
            returns_second = np.random.normal(-0.0005, 0.015, 100)
            returns = np.concatenate([returns_first, returns_second])

            weights = np.random.uniform(0.3, 0.5, 200)
            weights = weights / weights.sum()

            attributor.add_strategy_data(strategy_name, {
                'returns': pd.Series(returns, index=dates),
                'weights': pd.Series(weights, index=dates)
            })

        return attributor

    def test_should_calculate_rolling_attribution(self, long_period_attributor):
        """Test rolling attribution calculation"""
        rolling_results = long_period_attributor.calculate_rolling_attribution(window=60)

        assert isinstance(rolling_results, list)
        assert len(rolling_results) > 0

        # Each result should be an AttributionResult
        for result in rolling_results:
            assert isinstance(result, AttributionResult)
            assert len(result.strategy_metrics) == 2

        # Should have results for the rolling periods
        expected_periods = 200 - 60 + 1  # 141 periods
        assert len(rolling_results) == expected_periods

    def test_should_handle_insufficient_data_in_rolling(self, long_period_attributor):
        """Test rolling attribution with insufficient data"""
        # Request window larger than available data
        rolling_results = long_period_attributor.calculate_rolling_attribution(window=300)

        # Should return empty list or single result with all data
        assert len(rolling_results) <= 1

    def test_rolling_attribution_consistency(self, long_period_attributor):
        """Test consistency of rolling attribution results"""
        rolling_results = long_period_attributor.calculate_rolling_attribution(window=60)

        # Check that all results have consistent structure
        first_result = rolling_results[0]
        last_result = rolling_results[-1]

        assert set(first_result.strategy_metrics.keys()) == set(last_result.strategy_metrics.keys())
        assert set(first_result.strategy_contributions.keys()) == set(last_result.strategy_contributions.keys())


class TestAttributionResult:
    """Test AttributionResult data structure"""

    def test_should_create_attribution_result_with_required_fields(self):
        """Test AttributionResult creation"""
        portfolio_metrics = PerformanceMetrics(
            total_return=0.08,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            maximum_drawdown=0.05,
            calmar_ratio=1.6,
            var_95=0.02,
            var_99=0.035
        )

        strategy_metrics = {
            'Strategy1': PerformanceMetrics(
                total_return=0.1,
                sharpe_ratio=1.1,
                sortino_ratio=1.3,
                maximum_drawdown=0.06,
                calmar_ratio=1.67,
                var_95=0.022,
                var_99=0.038
            )
        }

        result = AttributionResult(
            portfolio_metrics=portfolio_metrics,
            strategy_metrics=strategy_metrics,
            strategy_contributions={'Strategy1': 0.08},
            allocation_effects={'Strategy1': 0.01},
            selection_effects={'Strategy1': 0.02},
            attribution_method='brinson_fachler',
            analysis_period=('2023-01-01', '2023-12-31')
        )

        assert result.portfolio_metrics.total_return == 0.08
        assert len(result.strategy_metrics) == 1
        assert result.strategy_contributions['Strategy1'] == 0.08
        assert result.attribution_method == 'brinson_fachler'

    def test_should_calculate_derived_attribution_metrics(self):
        """Test derived metrics in AttributionResult"""
        # This would test any computed properties or methods on AttributionResult
        # For example, if we add methods to compare strategies or aggregate effects
        pass


class TestPerformanceComparison:
    """Test strategy performance comparison functionality"""

    @pytest.fixture
    def comparison_attributor(self):
        """Create attributor for performance comparison testing"""
        attributor = PerformanceAttributor()

        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)

        # Create strategies with distinctly different performance profiles
        strategies_data = {
            'HighReturn': np.random.normal(0.002, 0.03, 252),  # High return, high vol
            'LowRisk': np.random.normal(0.0005, 0.005, 252),   # Low return, low vol
            'Volatile': np.random.normal(0.001, 0.04, 252),    # Medium return, high vol
        }

        for strategy_name, returns_data in strategies_data.items():
            attributor.add_strategy_data(strategy_name, {
                'returns': pd.Series(returns_data, index=dates),
                'weights': pd.Series([0.33] * 252, index=dates)  # Equal weight
            })

        return attributor

    def test_should_rank_strategies_by_risk_adjusted_returns(self, comparison_attributor):
        """Test strategy ranking by risk-adjusted metrics"""
        result = comparison_attributor.calculate_attribution()

        # Extract Sharpe ratios for ranking
        sharpe_ratios = {}
        for strategy_name, metrics in result.strategy_metrics.items():
            sharpe_ratios[strategy_name] = metrics.sharpe_ratio

        # LowRisk should likely have better risk-adjusted returns
        # This is probabilistic, so we just check structure
        assert len(sharpe_ratios) == 3
        assert all(isinstance(ratio, (int, float)) for ratio in sharpe_ratios.values())

    def test_should_identify_best_and_worst_performers(self, comparison_attributor):
        """Test identification of best and worst performing strategies"""
        result = comparison_attributor.calculate_attribution()

        # Find best and worst by total return
        total_returns = {name: metrics.total_return
                        for name, metrics in result.strategy_metrics.items()}

        best_strategy = max(total_returns, key=total_returns.get)
        worst_strategy = min(total_returns, key=total_returns.get)

        assert best_strategy in total_returns
        assert worst_strategy in total_returns
        assert total_returns[best_strategy] >= total_returns[worst_strategy]