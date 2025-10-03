"""
Performance Analysis for Adaptive Allocation

Handles performance score calculation and risk metrics analysis.
Extracted from adaptive_allocator.py for single responsibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta

from .models import AdaptiveConfig


class PerformanceAnalyzer:
    """
    Performance analysis for strategy allocation decisions

    Calculates composite performance scores and risk metrics
    for adaptive allocation optimization.
    """

    def __init__(self, config: AdaptiveConfig):
        self.config = config

    def calculate_performance_scores(
        self,
        strategy_performance: Dict[str, Dict[str, pd.Series]],
        as_of_date: datetime,
        decay_factor: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate composite performance scores for each strategy"""
        decay_factor = decay_factor or self.config.decay_factor
        performance_scores = {}

        for strategy_name, perf_data in strategy_performance.items():
            if 'returns' not in perf_data:
                continue

            returns = perf_data['returns']

            # Filter by lookback period
            start_date = as_of_date - timedelta(days=self.config.performance_lookback)
            filtered_returns = returns.loc[start_date:as_of_date]

            if len(filtered_returns) < 20:  # Need minimum data
                performance_scores[strategy_name] = 0.0
                continue

            # Calculate base metrics
            total_return = (1 + filtered_returns).prod() - 1
            volatility = filtered_returns.std() * np.sqrt(252)
            sharpe_ratio = self._calculate_sharpe_ratio(filtered_returns)

            # Use provided metrics if available
            if 'sharpe_ratio' in perf_data:
                sharpe_series = perf_data['sharpe_ratio'].loc[start_date:as_of_date]
                if len(sharpe_series) > 0:
                    sharpe_ratio = sharpe_series.mean()

            max_drawdown = 0.0
            if 'max_drawdown' in perf_data:
                dd_series = perf_data['max_drawdown'].loc[start_date:as_of_date]
                if len(dd_series) > 0:
                    max_drawdown = dd_series.mean()
            else:
                max_drawdown = self._calculate_max_drawdown(filtered_returns)

            # Apply exponential weighting to recent performance
            if len(filtered_returns) > 1:
                weights = np.array([decay_factor ** (len(filtered_returns) - i - 1)
                                  for i in range(len(filtered_returns))])
                weights = weights / weights.sum()
                weighted_return = np.dot(filtered_returns.values, weights)
            else:
                weighted_return = filtered_returns.iloc[0] if len(filtered_returns) > 0 else 0

            # Composite score calculation with stronger differentiation
            return_score = total_return * 5  # Weight returns heavily
            risk_adjusted_score = sharpe_ratio * 3  # Sharpe ratio component
            drawdown_penalty = max_drawdown * 4  # Penalize drawdowns more
            recent_performance = weighted_return * 252 * 2  # Recent performance matters more

            composite_score = (
                return_score + risk_adjusted_score + recent_performance - drawdown_penalty
            )

            # Add volatility penalty for extremely volatile strategies
            if volatility > self.config.max_strategy_volatility:
                volatility_penalty = (volatility - self.config.max_strategy_volatility) * 5
                composite_score -= volatility_penalty

            performance_scores[strategy_name] = composite_score

        return performance_scores

    def calculate_risk_metrics(
        self,
        strategy_performance: Dict[str, Dict[str, pd.Series]],
        as_of_date: datetime
    ) -> Dict[str, float]:
        """Calculate risk metrics for each strategy"""
        risk_metrics = {}

        for strategy_name, perf_data in strategy_performance.items():
            if 'returns' not in perf_data:
                continue

            returns = perf_data['returns']
            start_date = as_of_date - timedelta(days=self.config.performance_lookback)
            filtered_returns = returns.loc[start_date:as_of_date]

            if len(filtered_returns) < 20:
                risk_metrics[strategy_name] = 0.5  # Neutral risk score
                continue

            volatility = filtered_returns.std() * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(filtered_returns)
            var_95 = np.percentile(filtered_returns, 5)  # 5% VaR

            # Risk score (lower is better)
            risk_score = volatility * 0.4 + max_drawdown * 0.4 + abs(var_95) * 252 * 0.2

            risk_metrics[strategy_name] = risk_score

        return risk_metrics

    def calculate_confidence_score(
        self,
        performance_scores: Dict[str, float],
        strategy_performance: Dict[str, Dict[str, pd.Series]],
        as_of_date: datetime
    ) -> float:
        """Calculate confidence in allocation decision"""
        if not performance_scores:
            return 0.0

        # Data sufficiency score
        min_data_points = 60  # Prefer 60+ days of data
        data_sufficiency_scores = []

        for strategy_name in performance_scores:
            if strategy_name in strategy_performance:
                returns = strategy_performance[strategy_name]['returns']
                start_date = as_of_date - timedelta(days=self.config.performance_lookback)
                data_points = len(returns.loc[start_date:as_of_date])
                sufficiency = min(1.0, data_points / min_data_points)
                data_sufficiency_scores.append(sufficiency)

        avg_data_sufficiency = np.mean(data_sufficiency_scores) if data_sufficiency_scores else 0.5

        # Performance dispersion (higher dispersion = more confident in differentiation)
        score_values = list(performance_scores.values())
        if len(score_values) > 1:
            score_std = np.std(score_values)
            score_mean = np.mean(score_values)
            if score_mean != 0:
                coefficient_variation = abs(score_std / score_mean)
                dispersion_confidence = min(1.0, coefficient_variation)
            else:
                dispersion_confidence = 0.5
        else:
            dispersion_confidence = 0.5

        # Combined confidence score
        confidence = (avg_data_sufficiency * 0.6 + dispersion_confidence * 0.4)

        return min(1.0, max(0.0, confidence))

    def estimate_performance_improvement(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        performance_scores: Dict[str, float]
    ) -> float:
        """Estimate expected performance improvement from reallocation"""
        current_score = sum(
            current_allocation.get(s, 0) * performance_scores.get(s, 0)
            for s in performance_scores
        )

        target_score = sum(
            target_allocation.get(s, 0) * performance_scores.get(s, 0)
            for s in performance_scores
        )

        # Convert to annualized improvement estimate
        improvement = (target_score - current_score) * 0.1  # Conservative scaling factor

        return max(0.0, improvement)

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)

        if volatility == 0:
            return 0.0

        return mean_return / volatility

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max

        return abs(drawdowns.min())