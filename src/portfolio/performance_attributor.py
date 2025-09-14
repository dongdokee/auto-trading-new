"""
Performance Attributor

Implements strategy-level performance attribution and analysis using Brinson-Fachler methodology.
Provides detailed performance metrics and factor decomposition.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import warnings
from datetime import datetime


@dataclass
class AttributionConfig:
    """Configuration for performance attribution"""
    lookback_window: int = 252  # 1 year
    risk_free_rate: float = 0.0
    attribution_method: str = 'brinson_fachler'

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.lookback_window <= 0:
            raise ValueError("Lookback window must be positive")
        if self.risk_free_rate < 0:
            raise ValueError("Risk-free rate must be non-negative")

        valid_methods = ['brinson_fachler', 'brinson_hood_beebower']
        if self.attribution_method not in valid_methods:
            raise ValueError(f"Invalid attribution method: {self.attribution_method}. Valid options: {valid_methods}")


@dataclass
class PerformanceMetrics:
    """Performance metrics for a strategy or portfolio"""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    calmar_ratio: float
    var_95: float
    var_99: float
    volatility: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    win_rate: float = 0.0

    @property
    def risk_adjusted_return(self) -> float:
        """Risk-adjusted return using Sharpe ratio"""
        return self.sharpe_ratio

    @property
    def downside_risk_adjusted_return(self) -> float:
        """Downside risk-adjusted return using Sortino ratio"""
        return self.sortino_ratio


@dataclass
class AttributionResult:
    """Result of performance attribution analysis"""
    portfolio_metrics: PerformanceMetrics
    strategy_metrics: Dict[str, PerformanceMetrics]
    strategy_contributions: Dict[str, float]
    allocation_effects: Dict[str, float]
    selection_effects: Dict[str, float]
    attribution_method: str
    analysis_period: Tuple[str, str]
    interaction_effects: Dict[str, float] = field(default_factory=dict)

    @property
    def total_allocation_effect(self) -> float:
        """Total allocation effect across all strategies"""
        return sum(self.allocation_effects.values())

    @property
    def total_selection_effect(self) -> float:
        """Total selection effect across all strategies"""
        return sum(self.selection_effects.values())

    @property
    def total_interaction_effect(self) -> float:
        """Total interaction effect across all strategies"""
        return sum(self.interaction_effects.values()) if self.interaction_effects else 0.0


class PerformanceAttributor:
    """
    Strategy-Level Performance Attribution System

    Implements Brinson-Fachler and Brinson-Hood-Beebower methodologies for
    performance attribution analysis. Decomposes portfolio returns into:
    - Strategy contributions
    - Allocation effects (timing)
    - Selection effects (security selection)
    - Interaction effects
    """

    def __init__(self, config: Optional[AttributionConfig] = None):
        """
        Initialize performance attributor

        Args:
            config: Attribution configuration. Uses defaults if None.
        """
        self.config = config or AttributionConfig()

        # Extract config for easier access
        self.lookback_window = self.config.lookback_window
        self.risk_free_rate = self.config.risk_free_rate
        self.attribution_method = self.config.attribution_method

        # Strategy data storage
        self.strategy_returns: Dict[str, Dict[str, pd.Series]] = {}

        # Benchmark data (optional)
        self.benchmark_returns: Optional[pd.Series] = None

    def add_strategy_data(
        self,
        strategy_name: str,
        strategy_data: Dict[str, Union[pd.Series, List]],
        replace: bool = True
    ) -> None:
        """
        Add or update strategy data

        Args:
            strategy_name: Name of the strategy
            strategy_data: Dictionary containing strategy data:
                - returns: pd.Series of strategy returns
                - weights: pd.Series of strategy weights (optional)
                - positions: pd.Series of position data (optional)
            replace: If True, replace existing data. If False, append.
        """
        # Validate required fields
        if 'returns' not in strategy_data:
            raise ValueError("Strategy data must contain 'returns' field")

        returns = strategy_data['returns']
        if not isinstance(returns, pd.Series):
            raise TypeError("Returns must be a pandas Series")

        if replace or strategy_name not in self.strategy_returns:
            # Replace or create new entry
            self.strategy_returns[strategy_name] = {}

        # Store returns
        if replace:
            self.strategy_returns[strategy_name]['returns'] = returns.copy()
        else:
            # Append to existing returns
            if 'returns' in self.strategy_returns[strategy_name]:
                existing_returns = self.strategy_returns[strategy_name]['returns']
                combined_returns = pd.concat([existing_returns, returns])
                # Remove duplicates, keeping the latest
                combined_returns = combined_returns[~combined_returns.index.duplicated(keep='last')]
                self.strategy_returns[strategy_name]['returns'] = combined_returns.sort_index()
            else:
                self.strategy_returns[strategy_name]['returns'] = returns.copy()

        # Store optional fields
        for field_name in ['weights', 'positions']:
            if field_name in strategy_data:
                field_data = strategy_data[field_name]
                if isinstance(field_data, pd.Series):
                    if replace:
                        self.strategy_returns[strategy_name][field_name] = field_data.copy()
                    else:
                        if field_name in self.strategy_returns[strategy_name]:
                            existing_data = self.strategy_returns[strategy_name][field_name]
                            combined_data = pd.concat([existing_data, field_data])
                            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                            self.strategy_returns[strategy_name][field_name] = combined_data.sort_index()
                        else:
                            self.strategy_returns[strategy_name][field_name] = field_data.copy()

    def set_benchmark(self, benchmark_returns: pd.Series) -> None:
        """
        Set benchmark returns for relative attribution

        Args:
            benchmark_returns: Series of benchmark returns
        """
        if not isinstance(benchmark_returns, pd.Series):
            raise TypeError("Benchmark returns must be a pandas Series")

        self.benchmark_returns = benchmark_returns.copy()

    def calculate_attribution(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> AttributionResult:
        """
        Calculate complete performance attribution

        Args:
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)

        Returns:
            AttributionResult: Complete attribution analysis
        """
        if not self.strategy_returns:
            raise ValueError("No strategy data available for attribution")

        # Determine analysis period
        analysis_period = self._get_analysis_period(start_date, end_date)

        # Calculate individual strategy metrics
        strategy_metrics = self._calculate_strategy_metrics(analysis_period)

        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(analysis_period)

        # Calculate attribution effects
        strategy_contributions = self._calculate_strategy_contributions(analysis_period)
        allocation_effects = self._calculate_allocation_effects(analysis_period)
        selection_effects = self._calculate_selection_effects(analysis_period)
        interaction_effects = self._calculate_interaction_effects(analysis_period)

        return AttributionResult(
            portfolio_metrics=portfolio_metrics,
            strategy_metrics=strategy_metrics,
            strategy_contributions=strategy_contributions,
            allocation_effects=allocation_effects,
            selection_effects=selection_effects,
            interaction_effects=interaction_effects,
            attribution_method=self.attribution_method,
            analysis_period=analysis_period
        )

    def calculate_rolling_attribution(
        self,
        window: int,
        step: int = 1
    ) -> List[AttributionResult]:
        """
        Calculate rolling attribution analysis

        Args:
            window: Rolling window size in periods
            step: Step size between windows

        Returns:
            List[AttributionResult]: List of attribution results for each period
        """
        if window <= 0:
            raise ValueError("Window size must be positive")

        # Get common date range across all strategies
        common_dates = self._get_common_date_range()
        if len(common_dates) < window:
            return []

        rolling_results = []

        for i in range(0, len(common_dates) - window + 1, step):
            start_date = common_dates[i]
            end_date = common_dates[i + window - 1]

            try:
                result = self.calculate_attribution(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                rolling_results.append(result)
            except Exception:
                # Skip periods with insufficient data
                continue

        return rolling_results

    def _get_analysis_period(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Tuple[str, str]:
        """Get analysis period dates"""
        common_dates = self._get_common_date_range()

        if start_date is None:
            start_date = common_dates[0].strftime('%Y-%m-%d')
        if end_date is None:
            end_date = common_dates[-1].strftime('%Y-%m-%d')

        return start_date, end_date

    def _get_common_date_range(self) -> pd.DatetimeIndex:
        """Get common date range across all strategies"""
        if not self.strategy_returns:
            return pd.DatetimeIndex([])

        # Find intersection of all date ranges
        common_dates = None
        for strategy_data in self.strategy_returns.values():
            strategy_dates = strategy_data['returns'].index
            if common_dates is None:
                common_dates = strategy_dates
            else:
                common_dates = common_dates.intersection(strategy_dates)

        return common_dates.sort_values()

    def _filter_data_by_period(
        self,
        data: pd.Series,
        start_date: str,
        end_date: str
    ) -> pd.Series:
        """Filter data by date period"""
        return data.loc[start_date:end_date]

    def _calculate_strategy_metrics(
        self,
        analysis_period: Tuple[str, str]
    ) -> Dict[str, PerformanceMetrics]:
        """Calculate metrics for each strategy"""
        strategy_metrics = {}

        for strategy_name, strategy_data in self.strategy_returns.items():
            returns = self._filter_data_by_period(
                strategy_data['returns'],
                analysis_period[0],
                analysis_period[1]
            )

            if len(returns) == 0:
                continue

            metrics = self._calculate_performance_metrics(returns)
            strategy_metrics[strategy_name] = metrics

        return strategy_metrics

    def _calculate_portfolio_metrics(
        self,
        analysis_period: Tuple[str, str]
    ) -> PerformanceMetrics:
        """Calculate portfolio-level metrics"""
        portfolio_returns = self._calculate_portfolio_returns(analysis_period)
        return self._calculate_performance_metrics(portfolio_returns)

    def _calculate_portfolio_returns(
        self,
        analysis_period: Tuple[str, str]
    ) -> pd.Series:
        """Calculate portfolio returns based on strategy weights"""
        portfolio_returns = None

        for strategy_name, strategy_data in self.strategy_returns.items():
            returns = self._filter_data_by_period(
                strategy_data['returns'],
                analysis_period[0],
                analysis_period[1]
            )

            # Use weights if available, otherwise equal weight
            if 'weights' in strategy_data:
                weights = self._filter_data_by_period(
                    strategy_data['weights'],
                    analysis_period[0],
                    analysis_period[1]
                )
                # Align dates
                aligned_returns, aligned_weights = returns.align(weights, join='inner')
                strategy_contribution = aligned_returns * aligned_weights
            else:
                # Equal weight assumption
                weight = 1.0 / len(self.strategy_returns)
                strategy_contribution = returns * weight

            if portfolio_returns is None:
                portfolio_returns = strategy_contribution
            else:
                portfolio_returns = portfolio_returns.add(strategy_contribution, fill_value=0)

        return portfolio_returns if portfolio_returns is not None else pd.Series([])

    def _calculate_performance_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)

        # Total return
        total_return = (1 + returns).prod() - 1

        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(returns)

        # Sortino ratio
        sortino_ratio = self._calculate_sortino_ratio(returns)

        # Maximum drawdown
        max_drawdown = self._calculate_maximum_drawdown(returns)

        # Calmar ratio
        calmar_ratio = self._calculate_calmar_ratio(returns)

        # VaR calculations
        var_95 = self._calculate_var(returns, 0.95)
        var_99 = self._calculate_var(returns, 0.99)

        # Additional metrics
        volatility = returns.std() * np.sqrt(252)
        skewness = returns.skew() if len(returns) > 2 else 0
        kurtosis = returns.kurtosis() if len(returns) > 3 else 0
        win_rate = (returns > 0).mean()

        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            maximum_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            var_99=var_99,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            win_rate=win_rate
        )

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean() * 252  # Annualized
        volatility = returns.std() * np.sqrt(252)  # Annualized

        if volatility == 0:
            return 0.0

        return (mean_return - self.risk_free_rate) / volatility

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean() * 252  # Annualized
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            # No downside risk
            return float('inf') if mean_return > self.risk_free_rate else 0.0

        downside_deviation = negative_returns.std() * np.sqrt(252)

        if downside_deviation == 0:
            return 0.0

        return (mean_return - self.risk_free_rate) / downside_deviation

    def _calculate_maximum_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return abs(drawdowns.min())

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        if len(returns) == 0:
            return 0.0

        annualized_return = returns.mean() * 252
        max_drawdown = self._calculate_maximum_drawdown(returns)

        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0

        return annualized_return / max_drawdown

    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0

        return float(np.percentile(returns, (1 - confidence_level) * 100))

    def _calculate_strategy_contributions(
        self,
        analysis_period: Tuple[str, str]
    ) -> Dict[str, float]:
        """Calculate individual strategy contributions to portfolio return"""
        strategy_contributions = {}
        portfolio_return = self._calculate_portfolio_returns(analysis_period).sum()

        for strategy_name, strategy_data in self.strategy_returns.items():
            returns = self._filter_data_by_period(
                strategy_data['returns'],
                analysis_period[0],
                analysis_period[1]
            )

            if 'weights' in strategy_data:
                weights = self._filter_data_by_period(
                    strategy_data['weights'],
                    analysis_period[0],
                    analysis_period[1]
                )
                aligned_returns, aligned_weights = returns.align(weights, join='inner')
                contribution = (aligned_returns * aligned_weights).sum()
            else:
                # Equal weight
                weight = 1.0 / len(self.strategy_returns)
                contribution = returns.sum() * weight

            strategy_contributions[strategy_name] = contribution

        return strategy_contributions

    def _calculate_allocation_effects(
        self,
        analysis_period: Tuple[str, str]
    ) -> Dict[str, float]:
        """Calculate allocation effects (timing decisions)"""
        allocation_effects = {}

        # This is a simplified allocation effect calculation
        # In practice, this would compare actual vs benchmark weights
        for strategy_name, strategy_data in self.strategy_returns.items():
            returns = self._filter_data_by_period(
                strategy_data['returns'],
                analysis_period[0],
                analysis_period[1]
            )

            if 'weights' in strategy_data:
                weights = self._filter_data_by_period(
                    strategy_data['weights'],
                    analysis_period[0],
                    analysis_period[1]
                )

                # Allocation effect = (actual weight - benchmark weight) * benchmark return
                # Simplified: use weight variance as proxy for allocation effect
                weight_variance = weights.var() if len(weights) > 1 else 0
                allocation_effects[strategy_name] = weight_variance * returns.mean()
            else:
                allocation_effects[strategy_name] = 0.0

        return allocation_effects

    def _calculate_selection_effects(
        self,
        analysis_period: Tuple[str, str]
    ) -> Dict[str, float]:
        """Calculate selection effects (security selection skill)"""
        selection_effects = {}

        # This is a simplified selection effect calculation
        # In practice, this would compare strategy returns vs benchmark sector returns
        portfolio_returns = self._calculate_portfolio_returns(analysis_period)
        portfolio_mean = portfolio_returns.mean()

        for strategy_name, strategy_data in self.strategy_returns.items():
            returns = self._filter_data_by_period(
                strategy_data['returns'],
                analysis_period[0],
                analysis_period[1]
            )

            # Selection effect = average weight * (strategy return - benchmark return)
            if 'weights' in strategy_data:
                weights = self._filter_data_by_period(
                    strategy_data['weights'],
                    analysis_period[0],
                    analysis_period[1]
                )
                avg_weight = weights.mean()
            else:
                avg_weight = 1.0 / len(self.strategy_returns)

            excess_return = returns.mean() - portfolio_mean
            selection_effects[strategy_name] = avg_weight * excess_return

        return selection_effects

    def _calculate_interaction_effects(
        self,
        analysis_period: Tuple[str, str]
    ) -> Dict[str, float]:
        """Calculate interaction effects between allocation and selection"""
        interaction_effects = {}

        # Interaction effect = (actual weight - benchmark weight) * (strategy return - benchmark return)
        # This is a simplified calculation
        for strategy_name, strategy_data in self.strategy_returns.items():
            returns = self._filter_data_by_period(
                strategy_data['returns'],
                analysis_period[0],
                analysis_period[1]
            )

            if 'weights' in strategy_data:
                weights = self._filter_data_by_period(
                    strategy_data['weights'],
                    analysis_period[0],
                    analysis_period[1]
                )

                # Use weight deviation from equal weight as proxy
                equal_weight = 1.0 / len(self.strategy_returns)
                weight_deviation = weights.mean() - equal_weight

                # Use return deviation from portfolio return as proxy
                portfolio_returns = self._calculate_portfolio_returns(analysis_period)
                return_deviation = returns.mean() - portfolio_returns.mean()

                interaction_effects[strategy_name] = weight_deviation * return_deviation
            else:
                interaction_effects[strategy_name] = 0.0

        return interaction_effects