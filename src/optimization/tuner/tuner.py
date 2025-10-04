# src/optimization/tuner/tuner.py
"""
Hyperparameter Tuner Main Class

Comprehensive hyperparameter optimization framework that combines all tuning methods.
"""

import logging
from typing import Dict, Any, List, Optional, Callable

from .models import ValidationError, OptimizationResult, ParameterSpace
from .search_algorithms import SearchAlgorithms
from .bayesian_optimizer import BayesianOptimizer
from .cross_validation import CrossValidator
from .analytics import TuningAnalytics

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Comprehensive hyperparameter optimization framework.

    Supports multiple optimization algorithms including random search, grid search,
    and Bayesian optimization with cross-validation and walk-forward analysis.
    """

    def __init__(self, parameter_space: ParameterSpace, objective_function: Callable):
        """
        Initialize hyperparameter tuner.

        Args:
            parameter_space: Parameter space definition
            objective_function: Async function to optimize (higher is better)
        """
        self.parameter_space = parameter_space
        self.objective_function = objective_function

        # Initialize component classes
        self.search_algorithms = SearchAlgorithms(parameter_space, objective_function)
        self.bayesian_optimizer = BayesianOptimizer(parameter_space, objective_function)
        self.cross_validator = CrossValidator(objective_function)

        # Shared optimization history
        self.optimization_history: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None

    async def random_search(self, n_trials: int) -> OptimizationResult:
        """
        Perform random search optimization.

        Args:
            n_trials: Number of random trials to perform

        Returns:
            Best optimization result
        """
        result = await self.search_algorithms.random_search(n_trials)
        self._update_shared_history(self.search_algorithms.optimization_history)
        return result

    async def grid_search(self, grid_size: int) -> OptimizationResult:
        """
        Perform grid search optimization.

        Args:
            grid_size: Number of grid points per parameter

        Returns:
            Best optimization result
        """
        result = await self.search_algorithms.grid_search(grid_size)
        self._update_shared_history(self.search_algorithms.optimization_history)
        return result

    async def bayesian_optimization(
        self,
        n_trials: int,
        n_initial: int = 5,
        early_stopping_rounds: Optional[int] = None,
        tolerance: float = 1e-6
    ) -> OptimizationResult:
        """
        Perform Bayesian optimization using Gaussian Process.

        Args:
            n_trials: Total number of trials
            n_initial: Number of initial random trials
            early_stopping_rounds: Stop if no improvement for this many rounds
            tolerance: Minimum improvement threshold

        Returns:
            Best optimization result
        """
        result = await self.bayesian_optimizer.bayesian_optimization(
            n_trials, n_initial, early_stopping_rounds, tolerance
        )
        self._update_shared_history(self.bayesian_optimizer.optimization_history)
        return result

    async def cross_validate_parameters(
        self,
        parameters: Dict[str, float],
        fold_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Perform cross-validation for given parameters.

        Args:
            parameters: Parameters to validate
            fold_data: List of fold data for cross-validation

        Returns:
            Cross-validation metrics
        """
        return await self.cross_validator.cross_validate_parameters(parameters, fold_data)

    async def walk_forward_optimization(
        self,
        parameters: Dict[str, float],
        time_series_data: List[Dict[str, Any]],
        window_size: int,
        step_size: int = 1
    ) -> Dict[str, Any]:
        """
        Perform walk-forward optimization for time series data.

        Args:
            parameters: Parameters to optimize
            time_series_data: Time series data for walk-forward analysis
            window_size: Size of the training window
            step_size: Step size for moving the window

        Returns:
            Walk-forward optimization results
        """
        return await self.cross_validator.walk_forward_optimization(
            parameters, time_series_data, window_size, step_size
        )

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        analytics = TuningAnalytics(self.optimization_history)
        return analytics.get_optimization_summary()

    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance analysis."""
        analytics = TuningAnalytics(self.optimization_history)
        return analytics.calculate_parameter_importance()

    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        analytics = TuningAnalytics(self.optimization_history)
        return analytics.analyze_convergence()

    def get_best_result(self) -> Optional[OptimizationResult]:
        """Get the best result found so far."""
        return self.best_result

    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get the complete optimization history."""
        return self.optimization_history.copy()

    def reset(self):
        """Reset all optimization history and results."""
        self.optimization_history.clear()
        self.best_result = None
        self.search_algorithms.reset_history()
        self.bayesian_optimizer.reset_history()

    def _update_shared_history(self, new_history: List[OptimizationResult]):
        """Update shared optimization history with new results."""
        for result in new_history:
            if result not in self.optimization_history:
                self.optimization_history.append(result)
                self._update_best_result(result)

    def _update_best_result(self, result: OptimizationResult):
        """Update the best result found so far."""
        if self.best_result is None or result.objective_value > self.best_result.objective_value:
            self.best_result = result