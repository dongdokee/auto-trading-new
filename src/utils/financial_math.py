"""
Financial mathematics utilities for the AutoTrading system.
Provides comprehensive financial calculations for returns, risk metrics, and performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from scipy import stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns from price series.

    Args:
        prices: Price time series

    Returns:
        Simple returns series
    """
    if len(prices) == 0:
        return pd.Series([], dtype=float)

    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate logarithmic returns from price series.

    Args:
        prices: Price time series

    Returns:
        Log returns series
    """
    if len(prices) == 0:
        return pd.Series([], dtype=float)

    return np.log(prices / prices.shift(1))


def calculate_volatility(returns: pd.Series, window: Optional[int] = None,
                        periods: int = 252) -> Union[float, pd.Series]:
    """
    Calculate volatility (annualized standard deviation) from returns.

    Args:
        returns: Return time series
        window: Rolling window size (if None, calculates overall volatility)
        periods: Number of periods per year for annualization

    Returns:
        Annualized volatility (scalar or series if window specified)
    """
    if len(returns) == 0:
        return 0.0 if window is None else pd.Series([], dtype=float)

    if window is None:
        # Overall volatility
        return returns.std() * np.sqrt(periods)
    else:
        # Rolling volatility
        rolling_std = returns.rolling(window=window).std()
        return rolling_std * np.sqrt(periods)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                          periods: int = 252) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Return time series
        risk_free_rate: Risk-free rate (annual)
        periods: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    # Convert annual risk-free rate to period rate
    rf_period = risk_free_rate / periods

    # Calculate excess returns
    excess_returns = returns - rf_period

    if excess_returns.std() == 0:
        return 0.0

    # Annualized Sharpe ratio
    return (excess_returns.mean() * periods) / (excess_returns.std() * np.sqrt(periods))


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                           periods: int = 252) -> float:
    """
    Calculate Sortino ratio (uses downside deviation instead of total volatility).

    Args:
        returns: Return time series
        risk_free_rate: Risk-free rate (annual)
        periods: Number of periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    rf_period = risk_free_rate / periods
    excess_returns = returns - rf_period

    # Downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0

    downside_deviation = negative_returns.std() * np.sqrt(periods)

    if downside_deviation == 0:
        return 0.0

    return (excess_returns.mean() * periods) / downside_deviation


def calculate_calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        returns: Return time series
        periods: Number of periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    # Calculate cumulative returns to get price series
    cumulative = (1 + returns).cumprod()

    # Calculate max drawdown
    max_dd, _ = calculate_max_drawdown(cumulative)

    if abs(max_dd) < 1e-10:  # No drawdown
        return float('inf') if returns.mean() > 0 else 0.0

    annual_return = calculate_annualized_return(returns, periods)

    return annual_return / abs(max_dd)


def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, pd.Series]:
    """
    Calculate maximum drawdown and drawdown series.

    Args:
        prices: Price time series or cumulative return series

    Returns:
        Tuple of (max_drawdown, drawdown_series)
    """
    if len(prices) == 0:
        return 0.0, pd.Series([], dtype=float)

    # Calculate running maximum (peak)
    peak = prices.expanding().max()

    # Calculate drawdown series
    drawdown = (prices - peak) / peak

    # Maximum drawdown is the minimum value in drawdown series
    max_drawdown = drawdown.min()

    return max_drawdown, drawdown


def calculate_var(returns: pd.Series, confidence: float = 0.95,
                 method: str = 'historical') -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Return time series
        confidence: Confidence level (e.g., 0.95 for 95% VaR)
        method: Method to use ('historical' or 'parametric')

    Returns:
        VaR value (negative number representing loss)
    """
    if len(returns) == 0:
        return 0.0

    if method == 'historical':
        # Historical VaR
        return np.percentile(returns, (1 - confidence) * 100)

    elif method == 'parametric':
        # Parametric VaR (assuming normal distribution)
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        return mean + z_score * std

    else:
        raise ValueError(f"Unknown VaR method: {method}")


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Return time series
        confidence: Confidence level

    Returns:
        CVaR value
    """
    if len(returns) == 0:
        return 0.0

    var = calculate_var(returns, confidence, method='historical')

    # CVaR is the expected value of returns below VaR
    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        return var

    return tail_returns.mean()


def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta coefficient vs market benchmark.

    Args:
        asset_returns: Asset return time series
        market_returns: Market benchmark return time series

    Returns:
        Beta coefficient
    """
    if len(asset_returns) == 0 or len(market_returns) == 0:
        return 0.0

    # Align the series
    aligned_data = pd.DataFrame({
        'asset': asset_returns,
        'market': market_returns
    }).dropna()

    if len(aligned_data) < 2:
        return 0.0

    asset_aligned = aligned_data['asset']
    market_aligned = aligned_data['market']

    # Beta = Covariance(asset, market) / Variance(market)
    covariance = np.cov(asset_aligned, market_aligned)[0, 1]
    market_variance = np.var(market_aligned, ddof=1)

    if market_variance == 0:
        return 0.0

    return covariance / market_variance


def calculate_correlation_matrix(returns_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple return series.

    Args:
        returns_data: DataFrame with return series as columns

    Returns:
        Correlation matrix
    """
    return returns_data.corr()


def calculate_rolling_correlation(series_a: pd.Series, series_b: pd.Series,
                                 window: int) -> pd.Series:
    """
    Calculate rolling correlation between two series.

    Args:
        series_a: First time series
        series_b: Second time series
        window: Rolling window size

    Returns:
        Rolling correlation series
    """
    return series_a.rolling(window).corr(series_b)


def calculate_compound_return(returns: pd.Series) -> float:
    """
    Calculate compound return from return series.

    Args:
        returns: Return time series

    Returns:
        Total compound return
    """
    if len(returns) == 0:
        return 0.0

    return (1 + returns).prod() - 1


def calculate_annualized_return(returns: pd.Series, periods: int = 252) -> float:
    """
    Calculate annualized return from return series.

    Args:
        returns: Return time series
        periods: Number of periods per year

    Returns:
        Annualized return
    """
    if len(returns) == 0:
        return 0.0

    # Compound return over the period
    total_return = calculate_compound_return(returns)

    # Annualize using geometric mean
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0

    return (1 + total_return) ** (periods / n_periods) - 1


def normalize_prices(prices: pd.Series, base: float = 100) -> pd.Series:
    """
    Normalize price series to start at a base value.

    Args:
        prices: Price time series
        base: Base value for normalization

    Returns:
        Normalized price series
    """
    if len(prices) == 0:
        return pd.Series([], dtype=float)

    first_price = prices.iloc[0]
    if first_price == 0:
        return prices

    return prices * (base / first_price)


def calculate_information_ratio(asset_returns: pd.Series,
                               benchmark_returns: pd.Series) -> float:
    """
    Calculate information ratio (active return / tracking error).

    Args:
        asset_returns: Asset return time series
        benchmark_returns: Benchmark return time series

    Returns:
        Information ratio
    """
    # Align the series
    aligned_data = pd.DataFrame({
        'asset': asset_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned_data) == 0:
        return 0.0

    # Active returns
    active_returns = aligned_data['asset'] - aligned_data['benchmark']

    # Tracking error (standard deviation of active returns)
    tracking_error = active_returns.std()

    if tracking_error == 0:
        return 0.0

    return active_returns.mean() / tracking_error


def calculate_treynor_ratio(returns: pd.Series, market_returns: pd.Series,
                           risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """
    Calculate Treynor ratio (excess return / beta).

    Args:
        returns: Asset return time series
        market_returns: Market return time series
        risk_free_rate: Risk-free rate (annual)
        periods: Number of periods per year

    Returns:
        Treynor ratio
    """
    beta = calculate_beta(returns, market_returns)

    if beta == 0:
        return 0.0

    rf_period = risk_free_rate / periods
    excess_return = returns.mean() - rf_period

    return (excess_return * periods) / beta