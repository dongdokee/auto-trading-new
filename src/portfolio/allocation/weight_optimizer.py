"""
Weight Optimization for Adaptive Allocation

Handles weight calculation and constraint application.
Extracted from adaptive_allocator.py for single responsibility.
"""

import numpy as np
from typing import Dict

from .models import AdaptiveConfig


class WeightOptimizer:
    """
    Weight optimization for strategy allocation

    Calculates optimal weights based on performance and risk metrics,
    with constraint enforcement using iterative projection.
    """

    def __init__(self, config: AdaptiveConfig):
        self.config = config

    def calculate_optimal_weights(
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
            risk_adjustment = 1 - (risk_score * self.config.risk_adjustment_factor)
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

        # Apply constraints
        return self.apply_weight_constraints(raw_weights)

    def apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight constraints using iterative projection method"""
        strategies = list(weights.keys())
        n = len(strategies)

        if n == 0:
            return {}

        # Check if constraints are feasible
        min_total = n * self.config.min_strategy_weight
        max_total = n * self.config.max_strategy_weight

        if min_total > 1.0:
            # Infeasible - reduce min constraint
            adjusted_min = max(0.01, 1.0 / n * 0.9)  # Allow slight violation
        else:
            adjusted_min = self.config.min_strategy_weight

        if max_total < 1.0:
            # Infeasible - increase max constraint
            adjusted_max = min(1.0, 1.0 / n * 1.1)  # Allow slight violation
        else:
            adjusted_max = self.config.max_strategy_weight

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