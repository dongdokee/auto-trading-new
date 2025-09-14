"""
Tests for financial mathematics utilities.
Following TDD methodology: Red -> Green -> Refactor
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta

# These imports will fail initially (Red phase)
# We'll implement them to make tests pass (Green phase)
try:
    from src.utils.financial_math import (
        calculate_returns, calculate_log_returns, calculate_volatility,
        calculate_sharpe_ratio, calculate_sortino_ratio, calculate_calmar_ratio,
        calculate_max_drawdown, calculate_var, calculate_cvar,
        calculate_beta, calculate_correlation_matrix,
        calculate_compound_return, calculate_annualized_return,
        normalize_prices, calculate_rolling_correlation,
        calculate_information_ratio, calculate_treynor_ratio
    )
except ImportError:
    pytest.skip("Financial math utilities not yet implemented", allow_module_level=True)


class TestBasicReturnsCalculation:
    """Test basic returns calculation functions"""

    def test_should_calculate_simple_returns_from_prices(self):
        """Should calculate simple returns from price series"""
        prices = pd.Series([100, 105, 102, 108, 110])

        returns = calculate_returns(prices)

        expected = pd.Series([np.nan, 0.05, -0.0285714, 0.0588235, 0.0185185])
        pd.testing.assert_series_equal(returns, expected, rtol=1e-5)

    def test_should_calculate_log_returns_from_prices(self):
        """Should calculate logarithmic returns from price series"""
        prices = pd.Series([100, 105, 102, 108, 110])

        log_returns = calculate_log_returns(prices)

        assert not log_returns.iloc[0] == log_returns.iloc[0]  # First value should be NaN
        assert abs(log_returns.iloc[1] - np.log(105/100)) < 1e-10
        assert abs(log_returns.iloc[2] - np.log(102/105)) < 1e-10

    def test_should_handle_empty_series(self):
        """Should handle empty price series gracefully"""
        empty_series = pd.Series([], dtype=float)

        returns = calculate_returns(empty_series)
        log_returns = calculate_log_returns(empty_series)

        assert len(returns) == 0
        assert len(log_returns) == 0

    def test_should_handle_single_price(self):
        """Should handle single price gracefully"""
        single_price = pd.Series([100])

        returns = calculate_returns(single_price)
        log_returns = calculate_log_returns(single_price)

        assert len(returns) == 1
        assert pd.isna(returns.iloc[0])
        assert len(log_returns) == 1
        assert pd.isna(log_returns.iloc[0])


class TestVolatilityCalculation:
    """Test volatility calculation functions"""

    def test_should_calculate_historical_volatility(self):
        """Should calculate historical volatility from returns"""
        # Create known returns data
        returns = pd.Series([0.01, -0.005, 0.015, -0.02, 0.008])

        volatility = calculate_volatility(returns, periods=252)

        # Volatility should be positive
        assert volatility > 0
        # Check it's annualized (approximately)
        daily_vol = returns.std()
        expected_annual_vol = daily_vol * np.sqrt(252)
        assert abs(volatility - expected_annual_vol) < 1e-10

    def test_should_handle_zero_volatility(self):
        """Should handle zero volatility case"""
        constant_returns = pd.Series([0.01, 0.01, 0.01, 0.01])

        volatility = calculate_volatility(constant_returns)

        assert volatility == 0.0

    def test_should_calculate_rolling_volatility(self):
        """Should calculate rolling volatility with window"""
        returns = pd.Series(np.random.normal(0, 0.02, 50))

        rolling_vol = calculate_volatility(returns, window=20, periods=252)

        assert isinstance(rolling_vol, pd.Series)
        assert len(rolling_vol) == len(returns)
        # First 19 values should be NaN
        assert pd.isna(rolling_vol.iloc[:19]).all()


class TestRiskAdjustedReturns:
    """Test risk-adjusted return metrics"""

    def test_should_calculate_sharpe_ratio(self):
        """Should calculate Sharpe ratio correctly"""
        returns = pd.Series([0.01, -0.005, 0.015, -0.02, 0.008, 0.012])
        risk_free_rate = 0.02  # 2% annual

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods=252)

        # Should be a finite number
        assert np.isfinite(sharpe)
        # For positive average excess return, Sharpe should be positive
        excess_returns = returns - (risk_free_rate / 252)
        if excess_returns.mean() > 0:
            assert sharpe > 0

    def test_should_calculate_sortino_ratio(self):
        """Should calculate Sortino ratio (downside deviation)"""
        returns = pd.Series([0.02, -0.01, 0.015, -0.025, 0.01, 0.005])
        risk_free_rate = 0.02

        sortino = calculate_sortino_ratio(returns, risk_free_rate, periods=252)

        assert np.isfinite(sortino)
        # Sortino should generally be higher than Sharpe for same data
        # (since it only considers downside volatility)

    def test_should_calculate_calmar_ratio(self):
        """Should calculate Calmar ratio (return/max drawdown)"""
        # Create price series with known drawdown
        prices = pd.Series([100, 110, 105, 90, 95, 115])
        returns = calculate_returns(prices)

        calmar = calculate_calmar_ratio(returns, periods=252)

        assert np.isfinite(calmar)


class TestDrawdownCalculation:
    """Test drawdown calculation functions"""

    def test_should_calculate_max_drawdown(self):
        """Should calculate maximum drawdown correctly"""
        # Create price series with known drawdown
        prices = pd.Series([100, 110, 105, 90, 95, 115, 120])

        max_dd, dd_series = calculate_max_drawdown(prices)

        # Max drawdown should be from peak (110) to trough (90)
        expected_max_dd = (90 - 110) / 110  # -18.18%
        assert abs(max_dd - expected_max_dd) < 1e-4

        # Drawdown series should have same length as prices
        assert len(dd_series) == len(prices)
        # All drawdown values should be <= 0
        assert (dd_series <= 0).all()

    def test_should_handle_no_drawdown_case(self):
        """Should handle case with no drawdown (monotonically increasing)"""
        increasing_prices = pd.Series([100, 105, 110, 115, 120])

        max_dd, dd_series = calculate_max_drawdown(increasing_prices)

        assert max_dd == 0.0
        assert (dd_series == 0.0).all()


class TestVaRCalculation:
    """Test Value at Risk calculation"""

    def test_should_calculate_var_historical(self):
        """Should calculate VaR using historical method"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

        var_95 = calculate_var(returns, confidence=0.95, method='historical')
        var_99 = calculate_var(returns, confidence=0.99, method='historical')

        # VaR should be negative (representing loss)
        assert var_95 < 0
        assert var_99 < 0
        # 99% VaR should be more negative (larger loss) than 95% VaR
        assert var_99 < var_95

    def test_should_calculate_var_parametric(self):
        """Should calculate VaR using parametric method"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

        var_95 = calculate_var(returns, confidence=0.95, method='parametric')

        assert var_95 < 0
        assert np.isfinite(var_95)

    def test_should_calculate_cvar(self):
        """Should calculate Conditional VaR (Expected Shortfall)"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

        var_95 = calculate_var(returns, confidence=0.95)
        cvar_95 = calculate_cvar(returns, confidence=0.95)

        # CVaR should be more negative than VaR
        assert cvar_95 < var_95
        assert cvar_95 < 0


class TestCorrelationAnalysis:
    """Test correlation and beta calculations"""

    def test_should_calculate_beta(self):
        """Should calculate beta vs market benchmark"""
        # Create correlated returns
        market_returns = pd.Series(np.random.normal(0.001, 0.015, 100))
        asset_returns = 0.8 * market_returns + pd.Series(np.random.normal(0, 0.01, 100))

        beta = calculate_beta(asset_returns, market_returns)

        # Beta should be close to 0.8 (our synthetic relationship)
        assert 0.6 < beta < 1.0  # Allow for some noise
        assert np.isfinite(beta)

    def test_should_calculate_correlation_matrix(self):
        """Should calculate correlation matrix for multiple assets"""
        returns_data = pd.DataFrame({
            'Asset_A': np.random.normal(0.001, 0.02, 100),
            'Asset_B': np.random.normal(0.0005, 0.015, 100),
            'Asset_C': np.random.normal(0.002, 0.025, 100)
        })

        corr_matrix = calculate_correlation_matrix(returns_data)

        assert corr_matrix.shape == (3, 3)
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), [1.0, 1.0, 1.0])
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)

    def test_should_calculate_rolling_correlation(self):
        """Should calculate rolling correlation between two series"""
        series_a = pd.Series(np.random.normal(0.001, 0.02, 100))
        series_b = pd.Series(np.random.normal(0.001, 0.02, 100))

        rolling_corr = calculate_rolling_correlation(series_a, series_b, window=30)

        assert isinstance(rolling_corr, pd.Series)
        assert len(rolling_corr) == 100
        # First 29 values should be NaN
        assert pd.isna(rolling_corr.iloc[:29]).all()
        # Correlation values should be between -1 and 1
        valid_corr = rolling_corr.dropna()
        assert (valid_corr >= -1).all()
        assert (valid_corr <= 1).all()


class TestUtilityFunctions:
    """Test utility and helper functions"""

    def test_should_calculate_compound_return(self):
        """Should calculate compound return from return series"""
        returns = pd.Series([0.1, -0.05, 0.08, 0.03])

        compound_return = calculate_compound_return(returns)

        # Compound return = (1+r1)*(1+r2)*(1+r3)*(1+r4) - 1
        expected = (1.1 * 0.95 * 1.08 * 1.03) - 1
        assert abs(compound_return - expected) < 1e-10

    def test_should_calculate_annualized_return(self):
        """Should annualize returns correctly"""
        returns = pd.Series([0.01, -0.005, 0.015, -0.02, 0.008])

        annual_return = calculate_annualized_return(returns, periods=252)

        # Should be finite and reasonable
        assert np.isfinite(annual_return)
        # Should be approximately mean * periods for small returns
        expected_approx = returns.mean() * 252
        assert abs(annual_return - expected_approx) < 0.1  # Allow for compounding effect

    def test_should_normalize_prices(self):
        """Should normalize price series to start at 100"""
        prices = pd.Series([50, 55, 48, 60, 65])

        normalized = normalize_prices(prices, base=100)

        assert normalized.iloc[0] == 100
        assert abs(normalized.iloc[1] - 110) < 1e-10  # 55/50 * 100 = 110
        assert len(normalized) == len(prices)

    def test_should_calculate_information_ratio(self):
        """Should calculate information ratio vs benchmark"""
        asset_returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01])
        benchmark_returns = pd.Series([0.015, -0.005, 0.01, 0.0, 0.008])

        info_ratio = calculate_information_ratio(asset_returns, benchmark_returns)

        assert np.isfinite(info_ratio)

    def test_should_calculate_treynor_ratio(self):
        """Should calculate Treynor ratio (return per unit of systematic risk)"""
        returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01])
        market_returns = pd.Series([0.015, -0.005, 0.01, 0.0, 0.008])
        risk_free_rate = 0.02

        treynor = calculate_treynor_ratio(returns, market_returns, risk_free_rate, periods=252)

        assert np.isfinite(treynor)