"""
Adaptive Allocator - Main Coordinator

Performance-based dynamic strategy allocation with transaction cost awareness.
This is now a lightweight wrapper that coordinates specialized modules.
"""

import warnings
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime

from .models import AdaptiveConfig, AllocationUpdate, RebalanceRecommendation
from .performance_analyzer import PerformanceAnalyzer
from .weight_optimizer import WeightOptimizer
from .rebalance_engine import RebalanceEngine


class AdaptiveAllocator:
    """
    Performance-Based Adaptive Strategy Allocation

    Lightweight coordinator that delegates to specialized modules:
    - PerformanceAnalyzer: Performance scoring and risk metrics
    - WeightOptimizer: Weight calculation and constraints
    - RebalanceEngine: Rebalancing decisions and trade generation
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """
        Initialize adaptive allocator

        Args:
            config: Adaptive allocation configuration. Uses defaults if None.
        """
        self.config = config or AdaptiveConfig()

        # Initialize specialized components
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        self.weight_optimizer = WeightOptimizer(self.config)
        self.rebalance_engine = RebalanceEngine(self.config)

        # Strategy performance data storage
        self.strategy_performance: Dict[str, Dict[str, pd.Series]] = {}

        # Rebalancing history
        self.rebalance_history: List[AllocationUpdate] = []

        # Backward compatibility: expose config parameters as direct attributes
        self.performance_lookback = self.config.performance_lookback
        self.rebalance_threshold = self.config.rebalance_threshold
        self.min_rebalance_interval = self.config.min_rebalance_interval
        self.max_strategy_weight = self.config.max_strategy_weight
        self.min_strategy_weight = self.config.min_strategy_weight
        self.decay_factor = self.config.decay_factor
        self.transaction_cost_rate = self.config.transaction_cost_rate
        self.risk_adjustment_factor = self.config.risk_adjustment_factor
        self.max_strategy_volatility = self.config.max_strategy_volatility

    @property
    def last_rebalance_date(self) -> Optional[datetime]:
        """Get last rebalance date"""
        return self.rebalance_engine.last_rebalance_date

    @last_rebalance_date.setter
    def last_rebalance_date(self, value: Optional[datetime]) -> None:
        """Set last rebalance date"""
        self.rebalance_engine.last_rebalance_date = value

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

        # Calculate performance scores and risk metrics
        performance_scores = self.performance_analyzer.calculate_performance_scores(
            self.strategy_performance, as_of_date
        )
        risk_metrics = self.performance_analyzer.calculate_risk_metrics(
            self.strategy_performance, as_of_date
        )

        # Calculate new weights
        new_weights = self.weight_optimizer.calculate_optimal_weights(
            performance_scores, risk_metrics, current_allocation
        )

        # Calculate changes and turnover
        weight_changes = {
            strategy: new_weights[strategy] - current_allocation.get(strategy, 0)
            for strategy in new_weights
        }

        turnover = sum(abs(change) for change in weight_changes.values())

        # Calculate confidence and expected improvement
        confidence_score = self.performance_analyzer.calculate_confidence_score(
            performance_scores, self.strategy_performance, as_of_date
        )
        expected_improvement = self.performance_analyzer.estimate_performance_improvement(
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

        # Calculate performance improvement
        performance_scores = self.performance_analyzer.calculate_performance_scores(
            self.strategy_performance, as_of_date
        )
        performance_improvement = self.performance_analyzer.estimate_performance_improvement(
            current_allocation, target_allocation, performance_scores
        )

        return self.rebalance_engine.get_rebalance_recommendation(
            current_allocation, target_allocation, performance_improvement, as_of_date
        )

    # Backward compatibility methods - delegate to specialized components
    def _calculate_performance_scores(
        self,
        as_of_date: datetime,
        decay_factor: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate composite performance scores for each strategy (backward compatibility)"""
        return self.performance_analyzer.calculate_performance_scores(
            self.strategy_performance, as_of_date, decay_factor
        )

    def _calculate_risk_metrics(self, as_of_date: datetime) -> Dict[str, float]:
        """Calculate risk metrics for each strategy (backward compatibility)"""
        return self.performance_analyzer.calculate_risk_metrics(
            self.strategy_performance, as_of_date
        )

    def _calculate_optimal_weights(
        self,
        performance_scores: Dict[str, float],
        risk_metrics: Dict[str, float],
        current_allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate optimal weights (backward compatibility)"""
        return self.weight_optimizer.calculate_optimal_weights(
            performance_scores, risk_metrics, current_allocation
        )

    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight constraints (backward compatibility)"""
        return self.weight_optimizer.apply_weight_constraints(weights)

    def _calculate_confidence_score(
        self,
        performance_scores: Dict[str, float],
        as_of_date: datetime
    ) -> float:
        """Calculate confidence in allocation decision (backward compatibility)"""
        return self.performance_analyzer.calculate_confidence_score(
            performance_scores, self.strategy_performance, as_of_date
        )

    def _estimate_performance_improvement(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        performance_scores: Dict[str, float]
    ) -> float:
        """Estimate expected performance improvement (backward compatibility)"""
        return self.performance_analyzer.estimate_performance_improvement(
            current_allocation, target_allocation, performance_scores
        )

    def _should_rebalance(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        as_of_date: Optional[datetime] = None
    ) -> bool:
        """Decide whether rebalancing is warranted (backward compatibility)"""
        return self.rebalance_engine.should_rebalance(
            current_allocation, target_allocation, as_of_date
        )

    def _calculate_transaction_cost(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float]
    ) -> float:
        """Calculate estimated transaction costs (backward compatibility)"""
        return self.rebalance_engine.calculate_transaction_cost(
            current_allocation, target_allocation
        )

    def _determine_urgency(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        net_benefit: float
    ) -> str:
        """Determine urgency level (backward compatibility)"""
        return self.rebalance_engine.determine_urgency(
            current_allocation, target_allocation, net_benefit
        )

    def _create_rebalance_trades(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float]
    ) -> List[Dict[str, Union[str, float]]]:
        """Create rebalance trades (backward compatibility)"""
        return self.rebalance_engine.create_rebalance_trades(
            current_allocation, target_allocation
        )

    def _generate_rebalance_rationale(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        net_benefit: float,
        should_rebalance: bool
    ) -> str:
        """Generate rebalance rationale (backward compatibility)"""
        return self.rebalance_engine.generate_rebalance_rationale(
            current_allocation, target_allocation, net_benefit, should_rebalance
        )

    def _create_rebalance_recommendation(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        urgency: str
    ) -> RebalanceRecommendation:
        """Create rebalance recommendation (backward compatibility)"""
        transaction_cost = self.rebalance_engine.calculate_transaction_cost(
            current_allocation, target_allocation
        )
        trades = self.rebalance_engine.create_rebalance_trades(
            current_allocation, target_allocation
        )

        return RebalanceRecommendation(
            should_rebalance=True,
            urgency=urgency,
            expected_performance_improvement=0.0,  # To be calculated by caller
            estimated_transaction_cost=transaction_cost,
            net_benefit=0.0,  # To be calculated by caller
            trades=trades
        )


# Backward compatibility warning for direct imports
def _warn_deprecated_import():
    warnings.warn(
        "Direct import from adaptive_allocator.py is deprecated. "
        "Please use 'from src.portfolio.allocation import AdaptiveAllocator' instead. "
        "This wrapper will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

# Export the main class for backward compatibility
__all__ = ['AdaptiveAllocator', 'AdaptiveConfig', 'AllocationUpdate', 'RebalanceRecommendation']

# Show deprecation warning when this module is imported directly
import sys
if __name__ != '__main__' and 'src.portfolio.adaptive_allocator' in sys.modules:
    _warn_deprecated_import()