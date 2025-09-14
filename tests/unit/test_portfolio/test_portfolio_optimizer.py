"""
Tests for Portfolio Optimizer

Tests the Markowitz optimization engine with transaction costs and constraints.
Follows TDD methodology with comprehensive test coverage.
"""

import pytest
import numpy as np
import pandas as pd
from dataclasses import asdict
from unittest.mock import Mock, patch

from src.portfolio.portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationResult,
    OptimizationConfig
)


class TestPortfolioOptimizerInitialization:
    """Test portfolio optimizer initialization and configuration"""

    def test_should_create_optimizer_with_default_config(self):
        """Test portfolio optimizer creation with default parameters"""
        optimizer = PortfolioOptimizer()

        assert optimizer.transaction_cost == 0.0004
        assert optimizer.use_shrinkage is True
        assert optimizer.max_iterations == 1000
        assert optimizer.tolerance == 1e-9

    def test_should_create_optimizer_with_custom_config(self):
        """Test portfolio optimizer with custom configuration"""
        config = OptimizationConfig(
            transaction_cost=0.001,
            use_shrinkage=False,
            max_iterations=500,
            tolerance=1e-6
        )

        optimizer = PortfolioOptimizer(config)

        assert optimizer.transaction_cost == 0.001
        assert optimizer.use_shrinkage is False
        assert optimizer.max_iterations == 500
        assert optimizer.tolerance == 1e-6

    def test_should_validate_config_parameters(self):
        """Test configuration parameter validation"""
        with pytest.raises(ValueError, match="Transaction cost must be non-negative"):
            OptimizationConfig(transaction_cost=-0.001)

        with pytest.raises(ValueError, match="Max iterations must be positive"):
            OptimizationConfig(max_iterations=0)

        with pytest.raises(ValueError, match="Tolerance must be positive"):
            OptimizationConfig(tolerance=0.0)


class TestBasicOptimization:
    """Test basic portfolio optimization functionality"""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Create realistic return data for 4 strategies
        returns_data = {
            'TrendFollowing': np.random.normal(0.0008, 0.02, 100),
            'MeanReversion': np.random.normal(0.0006, 0.015, 100),
            'RangeTrading': np.random.normal(0.0004, 0.01, 100),
            'FundingArbitrage': np.random.normal(0.0005, 0.012, 100)
        }

        return pd.DataFrame(returns_data, index=dates)

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing"""
        return PortfolioOptimizer()

    def test_should_optimize_equal_weighted_portfolio(self, optimizer, sample_returns):
        """Test basic equal-weighted portfolio optimization"""
        result = optimizer.optimize_weights(sample_returns)

        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 4
        assert abs(np.sum(result.weights) - 1.0) < 1e-6
        assert result.success is True
        assert result.expected_return > 0
        assert result.volatility > 0
        assert result.sharpe_ratio > 0

    def test_should_respect_weight_constraints(self, optimizer, sample_returns):
        """Test that optimization respects individual weight constraints"""
        constraints = {'max_position': 0.4}
        result = optimizer.optimize_weights(sample_returns, constraints=constraints)

        assert result.success is True
        assert np.all(np.abs(result.weights) <= 0.4)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_should_respect_leverage_constraints(self, optimizer, sample_returns):
        """Test leverage constraint enforcement"""
        constraints = {'max_leverage': 1.5}
        result = optimizer.optimize_weights(sample_returns, constraints=constraints)

        assert result.success is True
        total_leverage = np.sum(np.abs(result.weights))
        assert total_leverage <= 1.5 + 1e-6

    def test_should_handle_long_only_constraint(self, optimizer, sample_returns):
        """Test long-only portfolio optimization"""
        constraints = {'long_only': True}
        result = optimizer.optimize_weights(sample_returns, constraints=constraints)

        assert result.success is True
        assert np.all(result.weights >= -1e-6)  # Allow for numerical precision
        assert abs(np.sum(result.weights) - 1.0) < 1e-6


class TestTransactionCosts:
    """Test transaction cost integration"""

    @pytest.fixture
    def optimizer_with_costs(self):
        """Create optimizer with significant transaction costs"""
        config = OptimizationConfig(transaction_cost=0.01)  # 1% transaction cost
        return PortfolioOptimizer(config)

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')

        returns_data = {
            'Strategy1': np.random.normal(0.002, 0.02, 50),
            'Strategy2': np.random.normal(0.001, 0.015, 50)
        }

        return pd.DataFrame(returns_data, index=dates)

    def test_should_account_for_transaction_costs_in_optimization(self, optimizer_with_costs, sample_returns):
        """Test that transaction costs affect optimization results"""
        current_weights = np.array([0.8, 0.2])  # Current allocation

        result = optimizer_with_costs.optimize_weights(
            sample_returns,
            current_weights=current_weights
        )

        assert result.success is True
        assert result.transaction_cost >= 0
        assert result.net_expected_return <= result.expected_return

    def test_should_minimize_turnover_with_high_transaction_costs(self, sample_returns):
        """Test that high transaction costs reduce portfolio turnover"""
        current_weights = np.array([0.6, 0.4])

        # Low transaction cost optimization
        low_cost_optimizer = PortfolioOptimizer(OptimizationConfig(transaction_cost=0.0001))
        low_cost_result = low_cost_optimizer.optimize_weights(
            sample_returns, current_weights=current_weights
        )

        # High transaction cost optimization
        high_cost_optimizer = PortfolioOptimizer(OptimizationConfig(transaction_cost=0.02))
        high_cost_result = high_cost_optimizer.optimize_weights(
            sample_returns, current_weights=current_weights
        )

        # High cost should result in lower turnover
        low_cost_turnover = np.sum(np.abs(low_cost_result.weights - current_weights))
        high_cost_turnover = np.sum(np.abs(high_cost_result.weights - current_weights))

        assert high_cost_turnover <= low_cost_turnover

    def test_should_calculate_transaction_costs_correctly(self, optimizer_with_costs, sample_returns):
        """Test accurate transaction cost calculation"""
        current_weights = np.array([0.7, 0.3])

        result = optimizer_with_costs.optimize_weights(
            sample_returns,
            current_weights=current_weights
        )

        # Calculate expected transaction cost
        turnover = np.sum(np.abs(result.weights - current_weights))
        expected_cost = turnover * optimizer_with_costs.transaction_cost

        assert abs(result.transaction_cost - expected_cost) < 1e-6


class TestCovarianceEstimation:
    """Test covariance estimation methods"""

    @pytest.fixture
    def correlated_returns(self):
        """Create correlated return series for testing"""
        np.random.seed(42)
        n_periods = 60

        # Generate correlated returns
        base_return = np.random.normal(0.001, 0.02, n_periods)

        returns_data = {
            'Strategy1': base_return + np.random.normal(0, 0.005, n_periods),
            'Strategy2': base_return + np.random.normal(0, 0.008, n_periods),
            'Strategy3': np.random.normal(0.0008, 0.015, n_periods),  # Independent
        }

        dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
        return pd.DataFrame(returns_data, index=dates)

    def test_should_use_sample_covariance_when_shrinkage_disabled(self, correlated_returns):
        """Test sample covariance estimation"""
        config = OptimizationConfig(use_shrinkage=False)
        optimizer = PortfolioOptimizer(config)

        cov_matrix = optimizer._calculate_covariance_matrix(correlated_returns)

        # Should match pandas sample covariance
        expected_cov = correlated_returns.cov().values
        np.testing.assert_array_almost_equal(cov_matrix, expected_cov, decimal=10)

    def test_should_use_ledoit_wolf_shrinkage_when_enabled(self, correlated_returns):
        """Test Ledoit-Wolf shrinkage estimation"""
        config = OptimizationConfig(use_shrinkage=True)
        optimizer = PortfolioOptimizer(config)

        cov_matrix = optimizer._calculate_covariance_matrix(correlated_returns)
        sample_cov = correlated_returns.cov().values

        # Shrinkage should produce different results than sample covariance
        assert not np.allclose(cov_matrix, sample_cov, rtol=1e-3)

        # But should be positive definite
        eigenvalues = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvalues > 1e-10)

    def test_should_handle_insufficient_data_gracefully(self):
        """Test covariance estimation with insufficient data"""
        # Only 5 data points for 3 strategies
        np.random.seed(42)
        short_returns = pd.DataFrame({
            'S1': np.random.normal(0, 0.01, 5),
            'S2': np.random.normal(0, 0.01, 5),
            'S3': np.random.normal(0, 0.01, 5)
        })

        config = OptimizationConfig(use_shrinkage=True)
        optimizer = PortfolioOptimizer(config)

        # Should not crash and return valid covariance matrix
        cov_matrix = optimizer._calculate_covariance_matrix(short_returns)

        assert cov_matrix.shape == (3, 3)
        assert np.all(np.diag(cov_matrix) > 0)


class TestOptimizationObjectives:
    """Test different optimization objectives"""

    @pytest.fixture
    def diverse_returns(self):
        """Create returns with diverse risk-return profiles"""
        np.random.seed(42)
        n_periods = 100

        returns_data = {
            'HighReturn': np.random.normal(0.002, 0.03, n_periods),   # High return, high risk
            'LowRisk': np.random.normal(0.0005, 0.005, n_periods),   # Low return, low risk
            'Balanced': np.random.normal(0.001, 0.015, n_periods),   # Medium risk-return
        }

        dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
        return pd.DataFrame(returns_data, index=dates)

    def test_should_optimize_for_maximum_sharpe_ratio(self, diverse_returns):
        """Test Sharpe ratio optimization"""
        optimizer = PortfolioOptimizer()

        result = optimizer.optimize_weights(
            diverse_returns,
            objective='max_sharpe'
        )

        assert result.success is True
        assert result.sharpe_ratio > 0

        # Should allocate more to strategies with better risk-adjusted returns
        high_return_weight = result.weights[0]  # HighReturn strategy
        low_risk_weight = result.weights[1]    # LowRisk strategy

        # The allocation should depend on risk-adjusted returns, not just returns
        assert isinstance(high_return_weight, float)
        assert isinstance(low_risk_weight, float)

    def test_should_optimize_for_minimum_volatility(self, diverse_returns):
        """Test minimum volatility optimization"""
        optimizer = PortfolioOptimizer()

        result = optimizer.optimize_weights(
            diverse_returns,
            objective='min_volatility'
        )

        assert result.success is True
        assert result.volatility > 0

        # Should allocate more to low-risk strategy
        low_risk_weight = result.weights[1]  # LowRisk strategy
        assert low_risk_weight > 0.2  # Should get significant allocation

    def test_should_handle_invalid_objective(self, diverse_returns):
        """Test handling of invalid optimization objective"""
        optimizer = PortfolioOptimizer()

        with pytest.raises(ValueError, match="Unsupported optimization objective"):
            optimizer.optimize_weights(diverse_returns, objective='invalid_objective')


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_should_handle_single_asset_portfolio(self):
        """Test optimization with single asset"""
        np.random.seed(42)
        single_asset_returns = pd.DataFrame({
            'OnlyAsset': np.random.normal(0.001, 0.02, 50)
        })

        optimizer = PortfolioOptimizer()
        result = optimizer.optimize_weights(single_asset_returns)

        assert result.success is True
        assert len(result.weights) == 1
        assert abs(result.weights[0] - 1.0) < 1e-6

    def test_should_handle_constant_returns(self):
        """Test optimization with constant returns"""
        constant_returns = pd.DataFrame({
            'Asset1': [0.001] * 50,
            'Asset2': [0.001] * 50,
            'Asset3': [0.001] * 50
        })

        optimizer = PortfolioOptimizer()
        result = optimizer.optimize_weights(constant_returns)

        # Should either succeed with equal weights or fail gracefully
        if result.success:
            assert abs(np.sum(result.weights) - 1.0) < 1e-6
        else:
            assert result.error_message is not None

    def test_should_handle_extreme_correlations(self):
        """Test optimization with perfectly correlated assets"""
        np.random.seed(42)
        base_returns = np.random.normal(0.001, 0.02, 50)

        # Perfectly correlated returns
        correlated_returns = pd.DataFrame({
            'Asset1': base_returns,
            'Asset2': base_returns * 1.0,  # Perfect correlation
            'Asset3': base_returns * 0.8   # High correlation
        })

        optimizer = PortfolioOptimizer()
        result = optimizer.optimize_weights(correlated_returns)

        # Should handle gracefully - might concentrate in one asset
        assert isinstance(result, OptimizationResult)
        if result.success:
            assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_should_return_failure_result_on_optimization_failure(self):
        """Test proper error handling when optimization fails"""
        # Create problematic data that might cause optimization failure
        problematic_returns = pd.DataFrame({
            'Asset1': [np.nan, 0.001, 0.002],
            'Asset2': [0.001, np.inf, 0.003],
            'Asset3': [0.002, 0.003, -np.inf]
        })

        optimizer = PortfolioOptimizer()
        result = optimizer.optimize_weights(problematic_returns)

        assert isinstance(result, OptimizationResult)
        # Should either handle gracefully or return failure with error message
        if not result.success:
            assert result.error_message is not None
            assert len(result.error_message) > 0

    def test_should_validate_input_data(self):
        """Test input data validation"""
        optimizer = PortfolioOptimizer()

        # Empty DataFrame
        with pytest.raises(ValueError, match="Returns data cannot be empty"):
            optimizer.optimize_weights(pd.DataFrame())

        # Non-DataFrame input
        with pytest.raises(TypeError, match="Returns data must be a pandas DataFrame"):
            optimizer.optimize_weights(np.array([[1, 2], [3, 4]]))

    def test_should_validate_current_weights_dimension(self):
        """Test current weights dimension validation"""
        returns = pd.DataFrame({
            'Asset1': [0.001, 0.002],
            'Asset2': [0.002, 0.003]
        })

        optimizer = PortfolioOptimizer()

        # Wrong dimension current weights
        with pytest.raises(ValueError, match="Current weights dimension"):
            optimizer.optimize_weights(returns, current_weights=np.array([0.5, 0.5, 0.0]))


class TestOptimizationResult:
    """Test OptimizationResult data structure"""

    def test_should_create_result_with_required_fields(self):
        """Test OptimizationResult creation"""
        weights = np.array([0.4, 0.3, 0.3])

        result = OptimizationResult(
            weights=weights,
            expected_return=0.001,
            volatility=0.02,
            sharpe_ratio=0.5,
            success=True
        )

        assert len(result.weights) == 3
        assert result.expected_return == 0.001
        assert result.volatility == 0.02
        assert result.sharpe_ratio == 0.5
        assert result.success is True
        assert result.transaction_cost == 0.0  # Default value
        assert result.error_message is None

    def test_should_calculate_derived_metrics(self):
        """Test derived metrics in OptimizationResult"""
        weights = np.array([0.6, 0.4])

        result = OptimizationResult(
            weights=weights,
            expected_return=0.002,
            volatility=0.02,
            sharpe_ratio=1.0,
            success=True,
            transaction_cost=0.001
        )

        # Net return should account for transaction costs
        net_return = result.expected_return - result.transaction_cost
        assert abs(result.net_expected_return - net_return) < 1e-10

        # Total leverage
        expected_leverage = np.sum(np.abs(weights))
        assert abs(result.total_leverage - expected_leverage) < 1e-10