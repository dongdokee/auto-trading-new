"""
Rebalancing Engine for Adaptive Allocation

Handles rebalancing decisions and trade generation.
Extracted from adaptive_allocator.py for single responsibility.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from .models import AdaptiveConfig, RebalanceRecommendation


class RebalanceEngine:
    """
    Rebalancing decision engine

    Handles rebalancing logic, transaction cost analysis,
    and trade generation for portfolio rebalancing.
    """

    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.last_rebalance_date: Optional[datetime] = None

    def get_rebalance_recommendation(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        performance_improvement: float,
        as_of_date: Optional[datetime] = None
    ) -> RebalanceRecommendation:
        """
        Get rebalancing recommendation with cost-benefit analysis

        Args:
            current_allocation: Current strategy weights
            target_allocation: Target strategy weights
            performance_improvement: Expected performance improvement
            as_of_date: Date for calculation

        Returns:
            RebalanceRecommendation: Detailed recommendation
        """
        as_of_date = as_of_date or datetime.now()

        # Calculate transaction costs
        transaction_cost = self.calculate_transaction_cost(current_allocation, target_allocation)

        # Net benefit
        net_benefit = performance_improvement - transaction_cost

        # Decision logic
        should_rebalance = self.should_rebalance(current_allocation, target_allocation, as_of_date)

        # Determine urgency
        urgency = self.determine_urgency(current_allocation, target_allocation, net_benefit)

        # Create trades
        trades = self.create_rebalance_trades(current_allocation, target_allocation)

        # Generate rationale
        rationale = self.generate_rebalance_rationale(
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

    def should_rebalance(
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
            if days_since_rebalance < self.config.min_rebalance_interval:
                return False

        # Deviation threshold check
        max_deviation = max(
            abs(target_allocation.get(s, 0) - current_allocation.get(s, 0))
            for s in set(list(current_allocation.keys()) + list(target_allocation.keys()))
        )

        return max_deviation >= self.config.rebalance_threshold

    def calculate_transaction_cost(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float]
    ) -> float:
        """Calculate estimated transaction costs for rebalancing"""
        turnover = sum(
            abs(target_allocation.get(s, 0) - current_allocation.get(s, 0))
            for s in set(list(current_allocation.keys()) + list(target_allocation.keys()))
        )

        return turnover * self.config.transaction_cost_rate

    def determine_urgency(
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

    def create_rebalance_trades(
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

    def generate_rebalance_rationale(
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

    def update_last_rebalance_date(self, date: datetime) -> None:
        """Update the last rebalance date"""
        self.last_rebalance_date = date