"""
Strategy Matrix for Regime-Based Strategy Allocation

Manages strategy weights and allocation based on current market regime and volatility.
Implements dynamic strategy selection to optimize performance across different market conditions.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class StrategyAllocation:
    """Strategy allocation with weight and configuration"""
    strategy_name: str
    weight: float
    confidence_multiplier: float = 1.0
    enabled: bool = True


class StrategyMatrix:
    """
    Dynamic strategy allocation based on market regime and volatility

    Manages the allocation of capital across different trading strategies based on:
    - Current market regime (BULL/BEAR/SIDEWAYS/NEUTRAL)
    - Market volatility level (HIGH/LOW)
    - Strategy performance history
    - Confidence levels in regime detection
    """

    def __init__(self):
        """Initialize strategy matrix with default allocations"""

        # Base allocation matrix: (regime, volatility) -> {strategy: weight}
        self.base_allocations = {
            ('BULL', 'LOW'): {
                'TrendFollowing': 0.7,
                'MeanReversion': 0.2,
                'RangeTrading': 0.1
            },
            ('BULL', 'HIGH'): {
                'TrendFollowing': 0.5,
                'MeanReversion': 0.3,
                'FundingArbitrage': 0.2
            },
            ('BEAR', 'LOW'): {
                'MeanReversion': 0.6,
                'TrendFollowing': 0.3,
                'FundingArbitrage': 0.1
            },
            ('BEAR', 'HIGH'): {
                'MeanReversion': 0.4,
                'TrendFollowing': 0.3,
                'FundingArbitrage': 0.3
            },
            ('SIDEWAYS', 'LOW'): {
                'RangeTrading': 0.5,
                'MeanReversion': 0.4,
                'FundingArbitrage': 0.1
            },
            ('SIDEWAYS', 'HIGH'): {
                'MeanReversion': 0.5,
                'RangeTrading': 0.3,
                'FundingArbitrage': 0.2
            },
            ('NEUTRAL', 'LOW'): {
                'MeanReversion': 0.4,
                'TrendFollowing': 0.3,
                'RangeTrading': 0.3
            },
            ('NEUTRAL', 'HIGH'): {
                'FundingArbitrage': 0.5,
                'MeanReversion': 0.3,
                'TrendFollowing': 0.2
            }
        }

        # Strategy performance tracking for dynamic adjustment
        self.strategy_performance: Dict[str, Dict[str, float]] = {}

        # Current allocation cache
        self._current_allocation: Dict[str, StrategyAllocation] = {}
        self._last_regime_info: Dict[str, Any] = {}

    def get_strategy_allocation(self, regime_info: Dict[str, Any]) -> Dict[str, StrategyAllocation]:
        """
        Get current strategy allocation based on regime information

        Args:
            regime_info: Dictionary containing:
                - regime: Current market regime
                - volatility_forecast: Predicted volatility
                - confidence: Confidence in regime detection
                - duration: How long current regime has lasted

        Returns:
            dict: Strategy name -> StrategyAllocation mapping
        """
        regime = regime_info.get('regime', 'NEUTRAL')
        volatility = regime_info.get('volatility_forecast', 0.02)
        confidence = regime_info.get('confidence', 0.5)

        # Determine volatility level
        vol_level = 'HIGH' if volatility > 0.03 else 'LOW'

        # Get base allocation
        allocation_key = (regime, vol_level)
        base_weights = self.base_allocations.get(allocation_key, self.base_allocations[('NEUTRAL', 'LOW')])

        # Apply confidence adjustments
        adjusted_allocation = self._adjust_for_confidence(base_weights, confidence)

        # Apply performance-based adjustments
        performance_adjusted = self._adjust_for_performance(adjusted_allocation)

        # Create StrategyAllocation objects
        strategy_allocations = {}
        for strategy_name, weight in performance_adjusted.items():
            strategy_allocations[strategy_name] = StrategyAllocation(
                strategy_name=strategy_name,
                weight=weight,
                confidence_multiplier=self._get_confidence_multiplier(strategy_name, regime_info),
                enabled=weight > 0.01  # Disable strategies with very low weights
            )

        # Cache for analysis
        self._current_allocation = strategy_allocations
        self._last_regime_info = regime_info.copy()

        return strategy_allocations

    def _adjust_for_confidence(self, base_weights: Dict[str, float], confidence: float) -> Dict[str, float]:
        """
        Adjust strategy weights based on regime detection confidence

        Args:
            base_weights: Base strategy weights
            confidence: Confidence in regime detection [0, 1]

        Returns:
            dict: Confidence-adjusted weights
        """
        if confidence < 0.6:
            # Low confidence: move towards neutral/diversified allocation
            neutral_weight = 1.0 / len(base_weights)
            blend_factor = (0.6 - confidence) / 0.6  # Higher blend for lower confidence

            adjusted_weights = {}
            for strategy, weight in base_weights.items():
                # Blend between base weight and neutral weight
                adjusted_weights[strategy] = weight * (1 - blend_factor) + neutral_weight * blend_factor

        else:
            # High confidence: use base weights with slight amplification
            amplification = min(1.2, 1.0 + (confidence - 0.6) * 0.5)
            total_weight = 0
            adjusted_weights = {}

            for strategy, weight in base_weights.items():
                adjusted_weights[strategy] = weight * amplification
                total_weight += adjusted_weights[strategy]

            # Renormalize
            if total_weight > 0:
                for strategy in adjusted_weights:
                    adjusted_weights[strategy] /= total_weight

        return adjusted_weights

    def _adjust_for_performance(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust strategy weights based on recent performance

        Args:
            base_weights: Base strategy weights

        Returns:
            dict: Performance-adjusted weights
        """
        if not self.strategy_performance:
            return base_weights

        # Calculate performance scores
        performance_scores = {}
        for strategy in base_weights:
            perf = self.strategy_performance.get(strategy, {})

            # Weighted performance score combining multiple metrics
            sharpe = perf.get('sharpe_ratio', 0)
            win_rate = perf.get('win_rate', 0.5)
            recent_pnl = perf.get('recent_pnl_normalized', 0)  # Normalized by volatility

            # Combine metrics (emphasize risk-adjusted returns)
            performance_scores[strategy] = (sharpe * 0.5 + (win_rate - 0.5) * 0.3 + recent_pnl * 0.2)

        # Apply performance adjustment (conservative approach)
        max_adjustment = 0.3  # Maximum 30% weight adjustment
        adjusted_weights = {}
        total_weight = 0

        for strategy, base_weight in base_weights.items():
            perf_score = performance_scores.get(strategy, 0)

            # Convert performance score to adjustment factor
            adjustment = np.tanh(perf_score) * max_adjustment  # Bounded adjustment
            adjusted_weight = base_weight * (1 + adjustment)

            adjusted_weights[strategy] = max(0.01, adjusted_weight)  # Minimum 1% allocation
            total_weight += adjusted_weights[strategy]

        # Renormalize to sum to 1
        if total_weight > 0:
            for strategy in adjusted_weights:
                adjusted_weights[strategy] /= total_weight

        return adjusted_weights

    def _get_confidence_multiplier(self, strategy_name: str, regime_info: Dict[str, Any]) -> float:
        """
        Get confidence multiplier for strategy signals

        Args:
            strategy_name: Name of the strategy
            regime_info: Current regime information

        Returns:
            float: Confidence multiplier for strategy signals
        """
        base_confidence = regime_info.get('confidence', 0.5)
        regime = regime_info.get('regime', 'NEUTRAL')

        # Strategy-specific confidence adjustments
        strategy_regime_fit = {
            'TrendFollowing': {'BULL': 1.2, 'BEAR': 1.1, 'SIDEWAYS': 0.7, 'NEUTRAL': 0.8},
            'MeanReversion': {'BULL': 0.9, 'BEAR': 0.9, 'SIDEWAYS': 1.2, 'NEUTRAL': 1.0},
            'RangeTrading': {'BULL': 0.6, 'BEAR': 0.6, 'SIDEWAYS': 1.3, 'NEUTRAL': 1.0},
            'FundingArbitrage': {'BULL': 1.0, 'BEAR': 1.0, 'SIDEWAYS': 1.1, 'NEUTRAL': 1.0}
        }

        regime_fit = strategy_regime_fit.get(strategy_name, {}).get(regime, 1.0)

        # Combine base confidence with regime fit
        confidence_multiplier = base_confidence * regime_fit

        return min(1.2, max(0.5, confidence_multiplier))

    def update_strategy_performance(self, strategy_name: str, performance_metrics: Dict[str, float]) -> None:
        """
        Update strategy performance metrics for dynamic allocation

        Args:
            strategy_name: Name of the strategy
            performance_metrics: Dictionary containing:
                - sharpe_ratio: Risk-adjusted returns
                - win_rate: Percentage of winning trades
                - recent_pnl_normalized: Recent PnL normalized by volatility
                - total_trades: Total number of trades
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {}

        # Update metrics with exponential decay for recent performance emphasis
        decay_factor = 0.9
        current_metrics = self.strategy_performance[strategy_name]

        for metric, new_value in performance_metrics.items():
            if metric in current_metrics:
                # Exponentially weighted average
                current_metrics[metric] = (current_metrics[metric] * decay_factor +
                                        new_value * (1 - decay_factor))
            else:
                current_metrics[metric] = new_value

        self.strategy_performance[strategy_name] = current_metrics

    def get_current_allocation_summary(self) -> Dict[str, Any]:
        """
        Get summary of current strategy allocation

        Returns:
            dict: Allocation summary with weights and metadata
        """
        summary = {
            'total_strategies': len(self._current_allocation),
            'regime_info': self._last_regime_info,
            'allocations': {},
            'total_weight': 0
        }

        for strategy_name, allocation in self._current_allocation.items():
            summary['allocations'][strategy_name] = {
                'weight': allocation.weight,
                'confidence_multiplier': allocation.confidence_multiplier,
                'enabled': allocation.enabled
            }
            summary['total_weight'] += allocation.weight

        return summary

    def reset_performance_tracking(self) -> None:
        """Reset all strategy performance tracking"""
        self.strategy_performance.clear()

    def disable_strategy(self, strategy_name: str) -> None:
        """
        Disable a specific strategy

        Args:
            strategy_name: Name of strategy to disable
        """
        if strategy_name in self._current_allocation:
            self._current_allocation[strategy_name].enabled = False
            self._current_allocation[strategy_name].weight = 0.0

    def enable_strategy(self, strategy_name: str) -> None:
        """
        Re-enable a disabled strategy

        Args:
            strategy_name: Name of strategy to enable
        """
        if strategy_name in self._current_allocation:
            self._current_allocation[strategy_name].enabled = True
            # Weight will be recalculated on next allocation call