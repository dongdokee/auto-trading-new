# src/optimization/tuner/analytics.py
"""
Analytics for Hyperparameter Tuning

Provides analysis methods for parameter importance and convergence analysis.
"""

import logging
from typing import Dict, List, Any
import numpy as np
from .models import OptimizationResult

logger = logging.getLogger(__name__)


class TuningAnalytics:
    """Analytics utilities for hyperparameter tuning results"""

    def __init__(self, optimization_history: List[OptimizationResult]):
        self.optimization_history = optimization_history

    def calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate parameter importance based on optimization history."""
        if len(self.optimization_history) < 2:
            return {}

        importance = {}

        # Get all parameter names that appear in the history
        all_param_names = set()
        for result in self.optimization_history:
            all_param_names.update(result.parameters.keys())

        for param_name in all_param_names:
            # Calculate correlation between parameter values and objective values
            param_values = []
            objective_values = []

            for r in self.optimization_history:
                if param_name in r.parameters:
                    param_values.append(r.parameters[param_name])
                    objective_values.append(r.objective_value)

            if len(param_values) > 1 and len(set(param_values)) > 1:  # Avoid division by zero
                correlation = np.corrcoef(param_values, objective_values)[0, 1]
                importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                importance[param_name] = 0.0

        return importance

    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        if len(self.optimization_history) < 5:
            return {'status': 'insufficient_data'}

        # Calculate running best
        running_best = []
        current_best = float('-inf')

        for result in self.optimization_history:
            if result.objective_value > current_best:
                current_best = result.objective_value
            running_best.append(current_best)

        # Check for convergence
        recent_improvements = sum(
            1 for i in range(len(running_best) - 5, len(running_best))
            if i > 0 and running_best[i] > running_best[i - 1]
        )

        return {
            'status': 'converged' if recent_improvements == 0 else 'improving',
            'improvement_rate': recent_improvements / 5,
            'final_best': running_best[-1],
            'total_trials': len(running_best),
            'running_best': running_best
        }

    def get_best_parameters_by_metric(self, metric_name: str) -> OptimizationResult:
        """Get best parameters for a specific metric."""
        if not self.optimization_history:
            return None

        best_result = None
        best_metric_value = float('-inf')

        for result in self.optimization_history:
            if metric_name in result.metrics:
                metric_value = result.metrics[metric_name]
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_result = result

        return best_result

    def get_parameter_distribution(self, param_name: str) -> Dict[str, Any]:
        """Get distribution statistics for a parameter."""
        param_values = []

        for result in self.optimization_history:
            if param_name in result.parameters:
                param_values.append(result.parameters[param_name])

        if not param_values:
            return {}

        return {
            'mean': np.mean(param_values),
            'std': np.std(param_values),
            'min': np.min(param_values),
            'max': np.max(param_values),
            'median': np.median(param_values),
            'count': len(param_values),
            'values': param_values
        }

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        if not self.optimization_history:
            return {'status': 'no_data'}

        objective_values = [r.objective_value for r in self.optimization_history]
        best_result = max(self.optimization_history, key=lambda r: r.objective_value)

        # Parameter importance
        param_importance = self.calculate_parameter_importance()

        # Convergence analysis
        convergence = self.analyze_convergence()

        return {
            'total_trials': len(self.optimization_history),
            'best_objective': max(objective_values),
            'best_parameters': best_result.parameters,
            'objective_mean': np.mean(objective_values),
            'objective_std': np.std(objective_values),
            'parameter_importance': param_importance,
            'convergence_analysis': convergence,
            'improvement_over_trials': self._calculate_improvement_over_trials()
        }

    def _calculate_improvement_over_trials(self) -> List[float]:
        """Calculate cumulative improvement over trials."""
        if not self.optimization_history:
            return []

        improvements = []
        best_so_far = float('-inf')

        for result in self.optimization_history:
            if result.objective_value > best_so_far:
                improvement = result.objective_value - best_so_far
                best_so_far = result.objective_value
            else:
                improvement = 0.0
            improvements.append(improvement)

        return improvements