"""
Data Models for Adaptive Allocation

Contains all data structures and configurations for portfolio allocation.
Extracted from adaptive_allocator.py for better organization.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


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