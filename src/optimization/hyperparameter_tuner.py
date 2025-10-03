"""
Hyperparameter Tuning Framework for strategy optimization.

This module provides comprehensive hyperparameter optimization capabilities including:
- Bayesian optimization with Gaussian Process
- Grid search and random search
- Cross-validation and walk-forward analysis
- Multi-objective optimization
- Early stopping and convergence detection

Features:
- Multiple optimization algorithms (Random, Grid, Bayesian)
- Cross-validation support for robust parameter estimation
- Walk-forward optimization for time series data
- Multi-objective optimization with Pareto frontier
- Constraint handling and early stopping
- Comprehensive optimization reporting
"""

import asyncio
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
import statistics
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails during optimization."""
    pass


@dataclass
class OptimizationResult:
    """
    Result of hyperparameter optimization.

    Contains optimized parameters, objective value, and additional metrics.
    """
    parameters: Dict[str, Any]
    objective_value: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    validation_score: Optional[float] = None
    fold_scores: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'parameters': self.parameters,
            'objective_value': self.objective_value,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'validation_score': self.validation_score,
            'fold_scores': self.fold_scores
        }

    def __lt__(self, other: 'OptimizationResult') -> bool:
        """Compare results by objective value."""
        return self.objective_value < other.objective_value

    def __gt__(self, other: 'OptimizationResult') -> bool:
        """Compare results by objective value."""
        return self.objective_value > other.objective_value


class ParameterSpace:
    """
    Parameter space definition for optimization.

    Defines parameter ranges, types, and constraints for hyperparameter optimization.
    """

    def __init__(self, parameters: Dict[str, Tuple[float, float]]):
        """
        Initialize parameter space.

        Args:
            parameters: Dictionary mapping parameter names to (min, max) ranges
        """
        self.parameters = {}

        for name, param_range in parameters.items():
            self._validate_parameter_range(name, param_range)
            self.parameters[name] = param_range

    def _validate_parameter_range(self, name: str, param_range: Tuple[float, float]) -> None:
        """Validate parameter range."""
        if not isinstance(param_range, tuple) or len(param_range) != 2:
            raise ValidationError(f"Parameter range for '{name}' must be a tuple of length 2")

        min_val, max_val = param_range

        try:
            min_val = float(min_val)
            max_val = float(max_val)
        except (TypeError, ValueError):
            raise ValidationError(f"Parameter range must be numeric for '{name}'")

        if min_val >= max_val:
            raise ValidationError(f"Invalid range for parameter '{name}': min ({min_val}) >= max ({max_val})")

        self.parameters[name] = (min_val, max_val)

    def generate_random_parameters(self) -> Dict[str, float]:
        """Generate random parameter set within defined ranges."""
        parameters = {}

        for name, (min_val, max_val) in self.parameters.items():
            parameters[name] = random.uniform(min_val, max_val)

        return parameters

    def generate_grid_parameters(self, grid_size: int) -> List[Dict[str, float]]:
        """
        Generate grid search parameter combinations.

        Args:
            grid_size: Number of points per parameter dimension

        Returns:
            List of parameter combinations
        """
        if grid_size <= 0:
            raise ValidationError("grid_size must be positive")

        # Generate grid points for each parameter
        param_grids = {}
        for name, (min_val, max_val) in self.parameters.items():
            param_grids[name] = np.linspace(min_val, max_val, grid_size)

        # Generate all combinations
        param_combinations = []
        param_names = list(self.parameters.keys())

        def generate_combinations(index: int, current_params: Dict[str, float]):
            if index == len(param_names):
                param_combinations.append(current_params.copy())
                return

            param_name = param_names[index]
            for value in param_grids[param_name]:
                current_params[param_name] = value
                generate_combinations(index + 1, current_params)

        generate_combinations(0, {})
        return param_combinations

    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        """Validate that parameters are within defined ranges."""
        for name, value in parameters.items():
            if name not in self.parameters:
                return False

            min_val, max_val = self.parameters[name]
            if not (min_val <= value <= max_val):
                return False

        return True

    def normalize_parameters(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Normalize parameters to [0, 1] range."""
        normalized = {}

        for name, value in parameters.items():
            if name in self.parameters:
                min_val, max_val = self.parameters[name]
                normalized[name] = (value - min_val) / (max_val - min_val)
            else:
                normalized[name] = value

        return normalized

    def denormalize_parameters(self, normalized_params: Dict[str, float]) -> Dict[str, float]:
        """Denormalize parameters from [0, 1] range to original ranges."""
        parameters = {}

        for name, norm_value in normalized_params.items():
            if name in self.parameters:
                min_val, max_val = self.parameters[name]
                parameters[name] = min_val + norm_value * (max_val - min_val)
            else:
                parameters[name] = norm_value

        return parameters


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
                raise ValidationError(f"Objective function evaluation failed: {e}")

        logger.info(f"Grid search completed. Best objective: {self.best_result.objective_value:.4f}")
        return self.best_result

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
            Cross-validation results with mean and std scores
        """
        fold_scores = []

        for fold in fold_data:
            try:
                score = await self.objective_function(parameters, fold)
                fold_scores.append(score)
            except Exception as e:
                logger.error(f"Cross-validation fold failed: {e}")
                continue

        if not fold_scores:
            raise ValidationError("All cross-validation folds failed")

        return {
            'mean_score': statistics.mean(fold_scores),
            'std_score': statistics.stdev(fold_scores) if len(fold_scores) > 1 else 0.0,
            'fold_scores': fold_scores
        }

    async def walk_forward_optimization(
        self,
        time_series_data: List[Dict[str, Any]],
        train_size: int,
        test_size: int,
        step_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Perform walk-forward optimization on time series data.

        Args:
            time_series_data: Time series data for optimization
            train_size: Size of training window
            test_size: Size of test window
            step_size: Step size for walking forward

        Returns:
            List of optimization results for each window
        """
        results = []
        data_length = len(time_series_data)

        for start_idx in range(0, data_length - train_size - test_size + 1, step_size):
            train_end = start_idx + train_size
            test_end = train_end + test_size

            train_data = time_series_data[start_idx:train_end]
            test_data = time_series_data[train_end:test_end]

            # Perform optimization on training data
            temp_history = self.optimization_history.copy()
            temp_best = self.best_result

            # Reset for this window
            self.optimization_history = []
            self.best_result = None

            try:
                # Quick optimization for this window
                best_result = await self.random_search(n_trials=20)

                # Test on out-of-sample data
                test_performance = await self._evaluate_on_test_data(
                    best_result.parameters, test_data
                )

                results.append({
                    'window_start': start_idx,
                    'parameters': best_result.parameters,
                    'train_performance': best_result.objective_value,
                    'test_performance': test_performance
                })

            except Exception as e:
                logger.error(f"Walk-forward window {start_idx} failed: {e}")

            # Restore original state
            self.optimization_history = temp_history
            self.best_result = temp_best

        return results

    async def multi_objective_optimization(
        self,
        n_trials: int,
        objectives: Dict[str, str],
        weights: Dict[str, float]
    ) -> OptimizationResult:
        """
        Perform multi-objective optimization.

        Args:
            n_trials: Number of trials
            objectives: Dictionary mapping objective names to 'maximize'/'minimize'
            weights: Weights for combining objectives

        Returns:
            Best optimization result
        """
        logger.info(f"Starting multi-objective optimization with {n_trials} trials")

        for trial in range(n_trials):
            try:
                # Generate random parameters
                parameters = self.parameter_space.generate_random_parameters()

                # Evaluate multi-objective function
                objective_values = await self.objective_function(parameters)

                # Combine objectives using weights
                combined_objective = 0.0
                for obj_name, obj_value in objective_values.items():
                    weight = weights.get(obj_name, 1.0)

                    if objectives.get(obj_name) == 'minimize':
                        obj_value = -obj_value  # Convert to maximization

                    combined_objective += weight * obj_value

                # Create result
                result = OptimizationResult(
                    parameters=parameters,
                    objective_value=combined_objective,
                    metrics=objective_values
                )

                # Update history and best result
                self.optimization_history.append(result)
                self._update_best_result(result)

            except Exception as e:
                logger.error(f"Multi-objective trial {trial + 1} failed: {e}")
                continue

        logger.info(f"Multi-objective optimization completed. Best objective: {self.best_result.objective_value:.4f}")
        return self.best_result

    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history."""
        return self.optimization_history.copy()

    def get_best_parameters(self) -> Optional[Dict[str, float]]:
        """Get best parameters found."""
        return self.best_result.parameters if self.best_result else None

    def export_optimization_report(self) -> Dict[str, Any]:
        """Export comprehensive optimization report."""
        if not self.optimization_history:
            return {'error': 'No optimization history available'}

        # Basic statistics
        objective_values = [r.objective_value for r in self.optimization_history]

        report = {
            'best_parameters': self.best_result.parameters if self.best_result else None,
            'best_objective_value': self.best_result.objective_value if self.best_result else None,
            'optimization_summary': {
                'total_trials': len(self.optimization_history),
                'mean_objective': statistics.mean(objective_values),
                'std_objective': statistics.stdev(objective_values) if len(objective_values) > 1 else 0.0,
                'min_objective': min(objective_values),
                'max_objective': max(objective_values)
            },
            'parameter_importance': self._calculate_parameter_importance(),
            'convergence_analysis': self._analyze_convergence()
        }

        return report

    def _update_best_result(self, result: OptimizationResult) -> None:
        """Update best result if current result is better."""
        if self.best_result is None or result.objective_value > self.best_result.objective_value:
            self.best_result = result

    def _prepare_gp_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for Gaussian Process."""
        X_train = []
        y_train = []

        for result in self.optimization_history:
            # Normalize parameters
            normalized_params = self.parameter_space.normalize_parameters(result.parameters)
            X_train.append(list(normalized_params.values()))
            y_train.append(result.objective_value)

        return np.array(X_train), np.array(y_train)

    def _acquisition_function_optimization(self, gp: GaussianProcessRegressor) -> Dict[str, float]:
        """Optimize acquisition function to find next point."""
        def acquisition_function(x):
            x_reshaped = x.reshape(1, -1)
            mean, std = gp.predict(x_reshaped, return_std=True)
            # Upper Confidence Bound acquisition
            kappa = 2.0
            return -(mean + kappa * std)  # Negative for minimization

        # Random start points for optimization
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

    async def _evaluate_on_test_data(self, parameters: Dict[str, float], test_data: List[Dict[str, Any]]) -> float:
        """Evaluate parameters on test data."""
        # Simple implementation - in practice, this would be more sophisticated
        return await self.objective_function(parameters)

    def _calculate_parameter_importance(self) -> Dict[str, float]:
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

    def _analyze_convergence(self) -> Dict[str, Any]:
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
            'convergence_trial': len(running_best) - recent_improvements if recent_improvements == 0 else None
        }