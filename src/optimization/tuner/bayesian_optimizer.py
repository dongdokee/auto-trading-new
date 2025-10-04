# src/optimization/tuner/bayesian_optimizer.py
"""
Bayesian Optimization for Hyperparameter Tuning

Implements Bayesian optimization using Gaussian Process.
"""

import logging
from typing import Optional, Dict, List, Tuple
import numpy as np
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .models import ValidationError, OptimizationResult, ParameterSpace
from .search_algorithms import SearchAlgorithms

logger = logging.getLogger(__name__)


class BayesianOptimizer(SearchAlgorithms):
    """Bayesian optimization using Gaussian Process"""

    def __init__(self, parameter_space: ParameterSpace, objective_function):
        super().__init__(parameter_space, objective_function)

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
        if n_trials <= 0:
            raise ValidationError("n_trials must be positive")
        if n_initial <= 0:
            raise ValidationError("n_initial must be positive")
        if n_initial >= n_trials:
            raise ValidationError("n_initial must be less than n_trials")

        logger.info(f"Starting Bayesian optimization with {n_trials} trials")

        # Perform initial random trials
        await self.random_search(n_initial)

        # Setup Gaussian Process
        gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )

        no_improvement_rounds = 0
        last_best_value = self.best_result.objective_value if self.best_result else float('-inf')

        for trial in range(n_initial, n_trials):
            try:
                # Prepare training data for GP
                X_train, y_train = self._prepare_gp_data()

                # Fit Gaussian Process
                gp.fit(X_train, y_train)

                # Find next point using acquisition function
                next_params = self._acquisition_function_optimization(gp)

                # Evaluate objective function
                objective_value = await self.objective_function(next_params)

                # Create result
                result = OptimizationResult(
                    parameters=next_params,
                    objective_value=objective_value
                )

                # Update history and best result
                self.optimization_history.append(result)
                self._update_best_result(result)

                # Check for early stopping
                if early_stopping_rounds is not None:
                    if objective_value > last_best_value + tolerance:
                        no_improvement_rounds = 0
                        last_best_value = objective_value
                    else:
                        no_improvement_rounds += 1

                    if no_improvement_rounds >= early_stopping_rounds:
                        logger.info(f"Early stopping at trial {trial + 1}")
                        break

                logger.debug(f"Bayesian trial {trial + 1}/{n_trials}: objective = {objective_value:.4f}")

            except Exception as e:
                logger.error(f"Bayesian trial {trial + 1} failed: {e}")
                continue

        logger.info(f"Bayesian optimization completed. Best objective: {self.best_result.objective_value:.4f}")
        return self.best_result

    def _prepare_gp_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for Gaussian Process."""
        if not self.optimization_history:
            raise ValidationError("No optimization history available")

        param_names = list(self.parameter_space.parameters.keys())
        X_train = []
        y_train = []

        for result in self.optimization_history:
            # Normalize parameters to [0, 1] range
            normalized_params = self.parameter_space.normalize_parameters(result.parameters)
            x_row = [normalized_params.get(name, 0.0) for name in param_names]
            X_train.append(x_row)
            y_train.append(result.objective_value)

        return np.array(X_train), np.array(y_train)

    def _acquisition_function_optimization(self, gp: GaussianProcessRegressor) -> Dict[str, float]:
        """Optimize acquisition function to find next evaluation point."""

        def expected_improvement(x: np.ndarray) -> float:
            """Expected Improvement acquisition function."""
            if len(x.shape) == 1:
                x = x.reshape(1, -1)

            mean, std = gp.predict(x, return_std=True)

            # Current best observed value
            best_f = max(result.objective_value for result in self.optimization_history)

            with np.errstate(divide='warn'):
                improvement = mean - best_f
                Z = improvement / std
                ei = improvement * self._normal_cdf(Z) + std * self._normal_pdf(Z)

            return -ei[0]  # Negative because we minimize

        def acquisition_function(x: np.ndarray) -> float:
            """Wrapper for acquisition function."""
            return expected_improvement(x)

        # Optimize acquisition function
        best_acquisition = float('inf')
        best_x = None

        for _ in range(10):  # Multiple restarts
            x0 = np.random.random(len(self.parameter_space.parameters))

            result = optimize.minimize(
                acquisition_function,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * len(self.parameter_space.parameters)
            )

            if result.fun < best_acquisition:
                best_acquisition = result.fun
                best_x = result.x

        # Convert normalized parameters back to original scale
        param_names = list(self.parameter_space.parameters.keys())
        normalized_params = {name: best_x[i] for i, name in enumerate(param_names)}
        return self.parameter_space.denormalize_parameters(normalized_params)

    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + np.tanh(x / np.sqrt(2)))

    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        """Standard normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)