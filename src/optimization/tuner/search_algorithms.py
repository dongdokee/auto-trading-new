# src/optimization/tuner/search_algorithms.py
"""
Search Algorithms for Hyperparameter Optimization

Implements random search and grid search algorithms.
"""

import logging
from typing import List, Callable
from .models import ValidationError, OptimizationResult, ParameterSpace

logger = logging.getLogger(__name__)


class SearchAlgorithms:
    """Base search algorithms for hyperparameter optimization"""

    def __init__(self, parameter_space: ParameterSpace, objective_function: Callable):
        self.parameter_space = parameter_space
        self.objective_function = objective_function
        self.optimization_history: List[OptimizationResult] = []
        self.best_result = None

    async def random_search(self, n_trials: int) -> OptimizationResult:
        """
        Perform random search optimization.

        Args:
            n_trials: Number of random trials to perform

        Returns:
            Best optimization result
        """
        if n_trials <= 0:
            raise ValidationError("n_trials must be positive")

        logger.info(f"Starting random search with {n_trials} trials")

        for trial in range(n_trials):
            try:
                # Generate random parameters
                parameters = self.parameter_space.generate_random_parameters()

                # Evaluate objective function
                objective_value = await self.objective_function(parameters)

                # Create result
                result = OptimizationResult(
                    parameters=parameters,
                    objective_value=objective_value
                )

                # Update history and best result
                self.optimization_history.append(result)
                self._update_best_result(result)

                logger.debug(f"Trial {trial + 1}/{n_trials}: objective = {objective_value:.4f}")

            except Exception as e:
                logger.error(f"Trial {trial + 1} failed: {e}")
                raise ValidationError(f"Objective function evaluation failed: {e}")

        logger.info(f"Random search completed. Best objective: {self.best_result.objective_value:.4f}")
        return self.best_result

    async def grid_search(self, grid_size: int) -> OptimizationResult:
        """
        Perform grid search optimization.

        Args:
            grid_size: Number of grid points per parameter

        Returns:
            Best optimization result
        """
        if grid_size <= 0:
            raise ValidationError("grid_size must be positive")

        logger.info(f"Starting grid search with grid size {grid_size}")

        # Generate grid parameters
        grid_parameters = self.parameter_space.generate_grid_parameters(grid_size)
        total_trials = len(grid_parameters)

        for trial, parameters in enumerate(grid_parameters):
            try:
                # Evaluate objective function
                objective_value = await self.objective_function(parameters)

                # Create result
                result = OptimizationResult(
                    parameters=parameters,
                    objective_value=objective_value
                )

                # Update history and best result
                self.optimization_history.append(result)
                self._update_best_result(result)

                logger.debug(f"Trial {trial + 1}/{total_trials}: objective = {objective_value:.4f}")

            except Exception as e:
                logger.error(f"Grid trial {trial + 1} failed: {e}")
                continue

        logger.info(f"Grid search completed. Best objective: {self.best_result.objective_value:.4f}")
        return self.best_result

    def _update_best_result(self, result: OptimizationResult):
        """Update the best result found so far."""
        if self.best_result is None or result.objective_value > self.best_result.objective_value:
            self.best_result = result

    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get the complete optimization history."""
        return self.optimization_history.copy()

    def get_best_result(self) -> OptimizationResult:
        """Get the best result found so far."""
        return self.best_result

    def reset_history(self):
        """Reset optimization history and best result."""
        self.optimization_history.clear()
        self.best_result = None