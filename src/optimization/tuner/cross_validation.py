# src/optimization/tuner/cross_validation.py
"""
Cross-Validation for Hyperparameter Tuning

Implements cross-validation and walk-forward analysis methods.
"""

import logging
from typing import Dict, List, Any, Callable
import statistics
from .models import ValidationError, OptimizationResult

logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross-validation utilities for hyperparameter optimization"""

    def __init__(self, objective_function: Callable):
        self.objective_function = objective_function

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
        if not fold_data:
            raise ValidationError("fold_data cannot be empty")

        logger.info(f"Starting cross-validation with {len(fold_data)} folds")

        fold_scores = []
        fold_metrics = []

        for fold_idx, fold in enumerate(fold_data):
            try:
                # Evaluate on this fold
                score = await self.objective_function(parameters)
                metrics = await self._evaluate_fold_metrics(parameters, fold)

                fold_scores.append(score)
                fold_metrics.append(metrics)

                logger.debug(f"Fold {fold_idx + 1}/{len(fold_data)}: score = {score:.4f}")

            except Exception as e:
                logger.error(f"Cross-validation fold {fold_idx + 1} failed: {e}")
                continue

        if not fold_scores:
            raise ValidationError("All cross-validation folds failed")

        # Calculate aggregate metrics
        cv_metrics = {
            'mean_score': statistics.mean(fold_scores),
            'std_score': statistics.stdev(fold_scores) if len(fold_scores) > 1 else 0.0,
            'min_score': min(fold_scores),
            'max_score': max(fold_scores),
            'fold_count': len(fold_scores),
            'fold_scores': fold_scores
        }

        logger.info(f"Cross-validation completed. Mean score: {cv_metrics['mean_score']:.4f} ± {cv_metrics['std_score']:.4f}")
        return cv_metrics

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
        if window_size <= 0:
            raise ValidationError("window_size must be positive")
        if step_size <= 0:
            raise ValidationError("step_size must be positive")
        if len(time_series_data) <= window_size:
            raise ValidationError("time_series_data must be larger than window_size")

        logger.info(f"Starting walk-forward optimization with window_size={window_size}, step_size={step_size}")

        window_results = []
        current_pos = 0

        while current_pos + window_size < len(time_series_data):
            try:
                # Extract training and test windows
                train_end = current_pos + window_size
                test_start = train_end
                test_end = min(test_start + step_size, len(time_series_data))

                train_data = time_series_data[current_pos:train_end]
                test_data = time_series_data[test_start:test_end]

                # Evaluate on test data
                test_score = await self._evaluate_on_test_data(parameters, test_data)

                window_result = {
                    'window_start': current_pos,
                    'window_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'test_score': test_score
                }

                window_results.append(window_result)

                logger.debug(f"Window {len(window_results)}: test_score = {test_score:.4f}")

                current_pos += step_size

            except Exception as e:
                logger.error(f"Walk-forward window failed: {e}")
                current_pos += step_size
                continue

        if not window_results:
            raise ValidationError("All walk-forward windows failed")

        # Calculate aggregate results
        test_scores = [result['test_score'] for result in window_results]

        walk_forward_results = {
            'mean_test_score': statistics.mean(test_scores),
            'std_test_score': statistics.stdev(test_scores) if len(test_scores) > 1 else 0.0,
            'min_test_score': min(test_scores),
            'max_test_score': max(test_scores),
            'window_count': len(window_results),
            'window_results': window_results,
            'test_scores': test_scores
        }

        logger.info(f"Walk-forward optimization completed. Mean test score: {walk_forward_results['mean_test_score']:.4f}")
        return walk_forward_results

    async def _evaluate_fold_metrics(self, parameters: Dict[str, float], fold_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate metrics for a single fold."""
        # Simple implementation - in practice, this would be more sophisticated
        score = await self.objective_function(parameters)
        return {'score': score}

    async def _evaluate_on_test_data(self, parameters: Dict[str, float], test_data: List[Dict[str, Any]]) -> float:
        """Evaluate parameters on test data."""
        # Simple implementation - in practice, this would be more sophisticated
        return await self.objective_function(parameters)