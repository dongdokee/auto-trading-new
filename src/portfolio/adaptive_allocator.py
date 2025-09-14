"""
Adaptive Allocator

Implements performance-based dynamic strategy allocation and rebalancing.
Provides intelligent rebalancing decisions with transaction cost awareness.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive allocation"""
    performance_lookback: int = 126  # 6 months lookback
    rebalance_threshold: float = 0.05  # 5% threshold for rebalancing
    min_rebalance_interval: int = 21  # Minimum days between rebalances
    max_strategy_weight: float = 0.6  # Maximum weight per strategy
    min_strategy_weight: float = 0.05  # Minimum weight per strategy
    decay_factor: float = 0.94  # Exponential decay for performance weighting
    transaction_cost_rate: float = 0.001  # Transaction cost rate
    risk_adjustment_factor: float = 0.3  # Risk adjustment in allocation
    max_strategy_volatility: float = 0.5  # Maximum allowed strategy volatility

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.performance_lookback <= 0:
            raise ValueError("Performance lookback must be positive")
        if self.rebalance_threshold <= 0:
            raise ValueError("Rebalance threshold must be positive")
        if self.min_rebalance_interval <= 0:
            raise ValueError("Min rebalance interval must be positive")
        if not 0 < self.max_strategy_weight <= 1:
            raise ValueError("Max strategy weight must be between 0 and 1")
        if self.min_strategy_weight <= 0:
            raise ValueError("Min strategy weight must be positive")
        if not 0 < self.decay_factor <= 1:
            raise ValueError("Decay factor must be between 0 and 1")


@dataclass
class PerformanceWindow:
    """Performance metrics over a specific window"""
    start_date: datetime
    end_date: datetime
    metrics: Dict[str, float]


@dataclass
class AllocationUpdate:
    """Result of allocation calculation"""
    new_weights: Dict[str, float]
    weight_changes: Dict[str, float]
    turnover: float
    confidence_score: float
    expected_improvement: float
    timestamp: datetime
    performance_scores: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RebalanceRecommendation:
    """Rebalancing recommendation with trade details"""
    should_rebalance: bool
    urgency: str  # 'LOW', 'MEDIUM', 'HIGH'
    expected_performance_improvement: float
    estimated_transaction_cost: float
    net_benefit: float
    trades: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""


class AdaptiveAllocator:
    """
    Performance-Based Adaptive Strategy Allocation

    Implements intelligent dynamic allocation based on:
    - Recent strategy performance
    - Risk-adjusted returns
    - Transaction cost awareness
    - Rebalancing frequency optimization
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """
        Initialize adaptive allocator

        Args:
            config: Adaptive allocation configuration. Uses defaults if None.
        """
        self.config = config or AdaptiveConfig()

        # Extract config for easier access
        self.performance_lookback = self.config.performance_lookback
        self.rebalance_threshold = self.config.rebalance_threshold
        self.min_rebalance_interval = self.config.min_rebalance_interval
        self.max_strategy_weight = self.config.max_strategy_weight
        self.min_strategy_weight = self.config.min_strategy_weight
        self.decay_factor = self.config.decay_factor
        self.transaction_cost_rate = self.config.transaction_cost_rate
        self.risk_adjustment_factor = self.config.risk_adjustment_factor
        self.max_strategy_volatility = self.config.max_strategy_volatility

        # Strategy performance data storage
        self.strategy_performance: Dict[str, Dict[str, pd.Series]] = {}

        # Rebalancing history
        self.last_rebalance_date: Optional[datetime] = None
        self.rebalance_history: List[AllocationUpdate] = []

    def add_strategy_performance(
        self,
        strategy_name: str,
        performance_data: Dict[str, Union[pd.Series, List]],
        replace: bool = True
    ) -> None:
        """
        Add or update strategy performance data

        Args:
            strategy_name: Name of the strategy
            performance_data: Dictionary containing performance metrics:
                - returns: pd.Series of strategy returns (required)
                - sharpe_ratio: pd.Series of Sharpe ratios (optional)
                - max_drawdown: pd.Series of maximum drawdowns (optional)
                - volatility: pd.Series of volatilities (optional)
            replace: If True, replace existing data. If False, append.
        """
        # Validate required fields
        if 'returns' not in performance_data:
            raise ValueError("Performance data must contain 'returns' field")

        returns = performance_data['returns']
        if not isinstance(returns, pd.Series):
            raise TypeError("Returns must be a pandas Series")

        if len(returns) == 0:
            raise ValueError("Returns series cannot be empty")

        if replace or strategy_name not in self.strategy_performance:
            # Replace or create new entry
            self.strategy_performance[strategy_name] = {}

        # Store all performance metrics
        for metric_name, metric_data in performance_data.items():
            if isinstance(metric_data, pd.Series):
                if replace:
                    self.strategy_performance[strategy_name][metric_name] = metric_data.copy()
                else:
                    # Append to existing data
                    if metric_name in self.strategy_performance[strategy_name]:
                        existing_data = self.strategy_performance[strategy_name][metric_name]
                        combined_data = pd.concat([existing_data, metric_data])
                        # Remove duplicates, keeping the latest
                        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                        self.strategy_performance[strategy_name][metric_name] = combined_data.sort_index()
                    else:
                        self.strategy_performance[strategy_name][metric_name] = metric_data.copy()

    def calculate_allocation_update(
        self,
        current_allocation: Dict[str, float],
        as_of_date: Optional[datetime] = None
    ) -> AllocationUpdate:
        """
        Calculate updated allocation based on recent performance

        Args:
            current_allocation: Current strategy weights
            as_of_date: Date for calculation. Uses latest if None.

        Returns:
            AllocationUpdate: New allocation with metadata
        """
        if not self.strategy_performance:
            raise ValueError("No strategy performance data available")

        as_of_date = as_of_date or datetime.now()

        # Calculate performance scores
        performance_scores = self._calculate_performance_scores(as_of_date)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(as_of_date)

        # Calculate new weights based on performance and risk
        new_weights = self._calculate_optimal_weights(
            performance_scores, risk_metrics, current_allocation
        )

        # Apply constraints
        new_weights = self._apply_weight_constraints(new_weights)

        # Calculate changes and turnover
        weight_changes = {
            strategy: new_weights[strategy] - current_allocation.get(strategy, 0)
            for strategy in new_weights
        }

        turnover = sum(abs(change) for change in weight_changes.values())

        # Calculate confidence and expected improvement
        confidence_score = self._calculate_confidence_score(performance_scores, as_of_date)
        expected_improvement = self._estimate_performance_improvement(
            current_allocation, new_weights, performance_scores
        )

        allocation_update = AllocationUpdate(
            new_weights=new_weights,
            weight_changes=weight_changes,
            turnover=turnover,
            confidence_score=confidence_score,
            expected_improvement=expected_improvement,
            timestamp=as_of_date,
            performance_scores=performance_scores,
            risk_metrics=risk_metrics
        )

        return allocation_update

    def get_rebalance_recommendation(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        as_of_date: Optional[datetime] = None
    ) -> RebalanceRecommendation:
        """
        Get rebalancing recommendation with cost-benefit analysis

        Args:
            current_allocation: Current strategy weights
            target_allocation: Target strategy weights
            as_of_date: Date for calculation

        Returns:
            RebalanceRecommendation: Detailed recommendation
        """
        as_of_date = as_of_date or datetime.now()

        # Calculate transaction costs
        transaction_cost = self._calculate_transaction_cost(current_allocation, target_allocation)

        # Estimate performance improvement
        performance_scores = self._calculate_performance_scores(as_of_date)
        performance_improvement = self._estimate_performance_improvement(
            current_allocation, target_allocation, performance_scores
        )

        # Net benefit
        net_benefit = performance_improvement - transaction_cost

        # Decision logic
        should_rebalance = self._should_rebalance(current_allocation, target_allocation, as_of_date)

        # Determine urgency
        urgency = self._determine_urgency(current_allocation, target_allocation, net_benefit)

        # Create trades
        trades = self._create_rebalance_trades(current_allocation, target_allocation)

        # Generate rationale
        rationale = self._generate_rebalance_rationale(
            current_allocation, target_allocation, net_benefit, should_rebalance
        )

        return RebalanceRecommendation(
            should_rebalance=should_rebalance,
            urgency=urgency,
            expected_performance_improvement=performance_improvement,
            estimated_transaction_cost=transaction_cost,
            net_benefit=net_benefit,
            trades=trades,
            rationale=rationale
        )

    def _calculate_performance_scores(
        self,
        as_of_date: datetime,
        decay_factor: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate composite performance scores for each strategy"""
        decay_factor = decay_factor or self.decay_factor
        performance_scores = {}

        for strategy_name, perf_data in self.strategy_performance.items():
            if 'returns' not in perf_data:
                continue

            returns = perf_data['returns']

            # Filter by lookback period
            start_date = as_of_date - timedelta(days=self.performance_lookback)
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
            if volatility > self.max_strategy_volatility:
                volatility_penalty = (volatility - self.max_strategy_volatility) * 5
                composite_score -= volatility_penalty

            performance_scores[strategy_name] = composite_score

        return performance_scores

    def _calculate_risk_metrics(self, as_of_date: datetime) -> Dict[str, float]:
        """Calculate risk metrics for each strategy"""
        risk_metrics = {}

        for strategy_name, perf_data in self.strategy_performance.items():
            if 'returns' not in perf_data:
                continue

            returns = perf_data['returns']
            start_date = as_of_date - timedelta(days=self.performance_lookback)
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

    def _calculate_optimal_weights(
        self,
        performance_scores: Dict[str, float],
        risk_metrics: Dict[str, float],
        current_allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate optimal weights based on performance and risk"""
        if not performance_scores:
            return current_allocation

        # Risk-adjusted performance scores
        risk_adjusted_scores = {}
        for strategy in performance_scores:
            perf_score = performance_scores[strategy]
            risk_score = risk_metrics.get(strategy, 0.5)

            # Adjust performance by risk
            risk_adjustment = 1 - (risk_score * self.risk_adjustment_factor)
            risk_adjustment = max(0.1, risk_adjustment)  # Minimum adjustment

            risk_adjusted_scores[strategy] = perf_score * risk_adjustment

        # Convert to weights using a more aggressive differentiation
        if not risk_adjusted_scores:
            n_strategies = len(performance_scores)
            return {strategy: 1.0 / n_strategies for strategy in performance_scores}

        # Normalize scores to make differences more pronounced
        score_values = list(risk_adjusted_scores.values())
        if len(score_values) <= 1:
            return {list(risk_adjusted_scores.keys())[0]: 1.0}

        min_score = min(score_values)
        max_score = max(score_values)
        score_range = max_score - min_score

        if score_range < 1e-6:
            # Scores are essentially equal - use equal weights
            n_strategies = len(performance_scores)
            return {strategy: 1.0 / n_strategies for strategy in performance_scores}

        # Normalize scores to [0, 1] and add base weight
        normalized_scores = {}
        for strategy, score in risk_adjusted_scores.items():
            normalized_score = (score - min_score) / score_range
            # Apply exponential amplification to create more differentiation
            amplified_score = np.exp(normalized_score * 3)  # Stronger amplification
            normalized_scores[strategy] = amplified_score

        # Convert to weights
        total_score = sum(normalized_scores.values())
        raw_weights = {s: score / total_score for s, score in normalized_scores.items()}

        return raw_weights

    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight constraints using iterative projection method"""
        import numpy as np

        strategies = list(weights.keys())
        n = len(strategies)

        if n == 0:
            return {}

        # Check if constraints are feasible
        min_total = n * self.min_strategy_weight
        max_total = n * self.max_strategy_weight

        if min_total > 1.0:
            # Infeasible - reduce min constraint
            adjusted_min = max(0.01, 1.0 / n * 0.9)  # Allow slight violation
        else:
            adjusted_min = self.min_strategy_weight

        if max_total < 1.0:
            # Infeasible - increase max constraint
            adjusted_max = min(1.0, 1.0 / n * 1.1)  # Allow slight violation
        else:
            adjusted_max = self.max_strategy_weight

        # Convert to numpy array for optimization
        w = np.array([weights[s] for s in strategies])

        # Iterative projection onto constraints
        max_iterations = 100
        tolerance = 1e-8

        for _ in range(max_iterations):
            # Project onto simplex (sum = 1)
            w_mean = (w.sum() - 1.0) / n
            w = w - w_mean

            # Project onto box constraints
            w = np.clip(w, adjusted_min, adjusted_max)

            # Check convergence
            current_sum = w.sum()
            if abs(current_sum - 1.0) < tolerance:
                break

        # Ensure exact sum to 1 (final normalization if needed)
        current_sum = w.sum()
        if current_sum > 0:
            w = w / current_sum

        # Final clipping to ensure constraints (with small tolerance)
        w = np.clip(w, adjusted_min - 1e-10, adjusted_max + 1e-10)

        # Convert back to dictionary
        final_weights = {strategies[i]: float(w[i]) for i in range(n)}

        return final_weights

    def _calculate_confidence_score(
        self,
        performance_scores: Dict[str, float],
        as_of_date: datetime
    ) -> float:
        """Calculate confidence in allocation decision"""
        if not performance_scores:
            return 0.0

        # Data sufficiency score
        min_data_points = 60  # Prefer 60+ days of data
        data_sufficiency_scores = []

        for strategy_name in performance_scores:
            if strategy_name in self.strategy_performance:
                returns = self.strategy_performance[strategy_name]['returns']
                start_date = as_of_date - timedelta(days=self.performance_lookback)
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

    def _estimate_performance_improvement(
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

    def _should_rebalance(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        as_of_date: Optional[datetime] = None
    ) -> bool:
        """Decide whether rebalancing is warranted"""
        as_of_date = as_of_date or datetime.now()

        # Time constraint check
        if self.last_rebalance_date is not None:
            days_since_rebalance = (as_of_date - self.last_rebalance_date).days
            if days_since_rebalance < self.min_rebalance_interval:
                return False

        # Deviation threshold check
        max_deviation = max(
            abs(target_allocation.get(s, 0) - current_allocation.get(s, 0))
            for s in set(list(current_allocation.keys()) + list(target_allocation.keys()))
        )

        return max_deviation >= self.rebalance_threshold

    def _calculate_transaction_cost(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float]
    ) -> float:
        """Calculate estimated transaction costs for rebalancing"""
        turnover = sum(
            abs(target_allocation.get(s, 0) - current_allocation.get(s, 0))
            for s in set(list(current_allocation.keys()) + list(target_allocation.keys()))
        )

        return turnover * self.transaction_cost_rate

    def _determine_urgency(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        net_benefit: float
    ) -> str:
        """Determine urgency level of rebalancing"""
        max_deviation = max(
            abs(target_allocation.get(s, 0) - current_allocation.get(s, 0))
            for s in set(list(current_allocation.keys()) + list(target_allocation.keys()))
        )

        if net_benefit > 0.02 or max_deviation > 0.15:  # 2% benefit or 15% deviation
            return 'HIGH'
        elif net_benefit > 0.005 or max_deviation > 0.08:  # 0.5% benefit or 8% deviation
            return 'MEDIUM'
        else:
            return 'LOW'

    def _create_rebalance_trades(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Create specific trades for rebalancing"""
        trades = []

        for strategy in set(list(current_allocation.keys()) + list(target_allocation.keys())):
            current_weight = current_allocation.get(strategy, 0)
            target_weight = target_allocation.get(strategy, 0)
            change = target_weight - current_weight

            if abs(change) > 0.001:  # Only include meaningful changes
                trades.append({
                    'strategy': strategy,
                    'action': 'BUY' if change > 0 else 'SELL',
                    'amount': abs(change),
                    'current_weight': current_weight,
                    'target_weight': target_weight
                })

        return trades

    def _generate_rebalance_rationale(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        net_benefit: float,
        should_rebalance: bool
    ) -> str:
        """Generate human-readable rationale for rebalancing decision"""
        if not should_rebalance:
            return "No rebalancing recommended: insufficient benefit or recent rebalancing"

        max_change_strategy = max(
            current_allocation.keys(),
            key=lambda s: abs(target_allocation.get(s, 0) - current_allocation.get(s, 0))
        )

        max_change = abs(target_allocation.get(max_change_strategy, 0) - current_allocation.get(max_change_strategy, 0))

        rationale_parts = []

        if net_benefit > 0.01:
            rationale_parts.append(f"Expected performance improvement: {net_benefit:.2%}")

        if max_change > 0.1:
            rationale_parts.append(f"Large allocation drift in {max_change_strategy}: {max_change:.1%}")

        if not rationale_parts:
            rationale_parts.append("Periodic rebalancing to maintain target allocation")

        return "; ".join(rationale_parts)

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

    def _create_rebalance_recommendation(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        urgency: str
    ) -> RebalanceRecommendation:
        """Create rebalance recommendation (internal helper)"""
        trades = self._create_rebalance_trades(current_allocation, target_allocation)
        transaction_cost = self._calculate_transaction_cost(current_allocation, target_allocation)

        return RebalanceRecommendation(
            should_rebalance=True,
            urgency=urgency,
            expected_performance_improvement=0.0,  # To be calculated
            estimated_transaction_cost=transaction_cost,
            net_benefit=0.0,  # To be calculated
            trades=trades
        )