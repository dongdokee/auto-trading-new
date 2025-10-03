"""
Tests for Hyperparameter Tuning Framework.

Following TDD methodology: Red -> Green -> Refactor
Tests for Bayesian optimization, strategy parameter tuning, and walk-forward validation.
"""

import pytest
import asyncio
import numpy as np
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.optimization.hyperparameter_tuner import (
    HyperparameterTuner,
    OptimizationResult,
    ParameterSpace,
    ValidationError
)


class TestParameterSpace:
    """Test suite for ParameterSpace class."""

    def test_should_initialize_with_parameter_ranges(self):
        """Test that ParameterSpace initializes with parameter ranges."""
        param_space = ParameterSpace({
            'lookback_period': (10, 50),
            'entry_threshold': (1.5, 3.0),
            'exit_threshold': (0.5, 2.0)
        })

        assert 'lookback_period' in param_space.parameters
        assert 'entry_threshold' in param_space.parameters
        assert 'exit_threshold' in param_space.parameters
        assert param_space.parameters['lookback_period'] == (10, 50)

    def test_should_validate_parameter_ranges(self):
        """Test that ParameterSpace validates parameter ranges."""
        # Test invalid range (min > max)
        with pytest.raises(ValidationError, match="Invalid range for parameter"):
            ParameterSpace({
                'lookback_period': (50, 10)  # Invalid: min > max
            })

        # Test non-numeric range
        with pytest.raises(ValidationError, match="Parameter range must be numeric"):
            ParameterSpace({
                'lookback_period': ('a', 'b')  # Invalid: non-numeric
            })

    def test_should_generate_random_parameters(self):
        """Test that ParameterSpace can generate random parameter sets."""
        param_space = ParameterSpace({
            'lookback_period': (10, 50),
            'entry_threshold': (1.5, 3.0)
        })

        params = param_space.generate_random_parameters()

        assert isinstance(params, dict)
        assert 'lookback_period' in params
        assert 'entry_threshold' in params
        assert 10 <= params['lookback_period'] <= 50
        assert 1.5 <= params['entry_threshold'] <= 3.0

    def test_should_generate_grid_parameters(self):
        """Test that ParameterSpace can generate grid search parameters."""
        param_space = ParameterSpace({
            'lookback_period': (10, 30),
            'entry_threshold': (1.5, 2.5)
        })

        grid_params = param_space.generate_grid_parameters(grid_size=3)

        assert isinstance(grid_params, list)
        assert len(grid_params) == 9  # 3x3 grid
        assert all(isinstance(params, dict) for params in grid_params)

    def test_should_validate_parameter_values(self):
        """Test that ParameterSpace validates parameter values."""
        param_space = ParameterSpace({
            'lookback_period': (10, 50),
            'entry_threshold': (1.5, 3.0)
        })

        # Valid parameters
        valid_params = {'lookback_period': 20, 'entry_threshold': 2.0}
        assert param_space.validate_parameters(valid_params) is True

        # Invalid parameters (out of range)
        invalid_params = {'lookback_period': 60, 'entry_threshold': 2.0}
        assert param_space.validate_parameters(invalid_params) is False

    def test_should_normalize_parameters(self):
        """Test that ParameterSpace can normalize parameters to [0,1] range."""
        param_space = ParameterSpace({
            'lookback_period': (10, 50),
            'entry_threshold': (1.5, 3.0)
        })

        params = {'lookback_period': 30, 'entry_threshold': 2.25}
        normalized = param_space.normalize_parameters(params)

        assert 0 <= normalized['lookback_period'] <= 1
        assert 0 <= normalized['entry_threshold'] <= 1
        assert normalized['lookback_period'] == 0.5  # (30-10)/(50-10)
        assert normalized['entry_threshold'] == 0.5  # (2.25-1.5)/(3.0-1.5)


class TestOptimizationResult:
    """Test suite for OptimizationResult class."""

    def test_should_initialize_with_required_fields(self):
        """Test that OptimizationResult initializes with required fields."""
        result = OptimizationResult(
            parameters={'lookback_period': 20},
            objective_value=0.15,
            metrics={'sharpe_ratio': 1.5, 'max_drawdown': 0.05}
        )

        assert result.parameters == {'lookback_period': 20}
        assert result.objective_value == 0.15
        assert result.metrics['sharpe_ratio'] == 1.5
        assert result.timestamp is not None

    def test_should_compare_results_by_objective_value(self):
        """Test that OptimizationResult can be compared by objective value."""
        result1 = OptimizationResult(
            parameters={'a': 1},
            objective_value=0.15,
            metrics={}
        )

        result2 = OptimizationResult(
            parameters={'a': 2},
            objective_value=0.20,
            metrics={}
        )

        assert result2 > result1  # Higher objective value is better
        assert result1 < result2

    def test_should_convert_to_dictionary(self):
        """Test that OptimizationResult can be converted to dictionary."""
        result = OptimizationResult(
            parameters={'lookback_period': 20},
            objective_value=0.15,
            metrics={'sharpe_ratio': 1.5}
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict['parameters'] == {'lookback_period': 20}
        assert result_dict['objective_value'] == 0.15
        assert 'timestamp' in result_dict


class TestHyperparameterTuner:
    """Test suite for HyperparameterTuner class."""

    @pytest.fixture
    def parameter_space(self):
        """Create parameter space for testing."""
        return ParameterSpace({
            'lookback_period': (10, 50),
            'entry_threshold': (1.5, 3.0),
            'exit_threshold': (0.5, 2.0)
        })

    @pytest.fixture
    def objective_function(self):
        """Create mock objective function for testing."""
        async def mock_objective(params):
            # Simple objective: maximize negative distance from center
            center = {'lookback_period': 30, 'entry_threshold': 2.25, 'exit_threshold': 1.25}
            distance = sum((params[k] - center[k]) ** 2 for k in params.keys())
            return 1.0 / (1.0 + distance)  # Higher is better

        return mock_objective

    @pytest.fixture
    def hyperparameter_tuner(self, parameter_space, objective_function):
        """Create HyperparameterTuner instance for testing."""
        return HyperparameterTuner(parameter_space, objective_function)

    def test_should_initialize_with_parameter_space_and_objective(self, hyperparameter_tuner):
        """Test that HyperparameterTuner initializes correctly."""
        assert hyperparameter_tuner.parameter_space is not None
        assert hyperparameter_tuner.objective_function is not None
        assert hyperparameter_tuner.optimization_history == []
        assert hyperparameter_tuner.best_result is None

    @pytest.mark.asyncio
    async def test_should_perform_random_search(self, hyperparameter_tuner):
        """Test that HyperparameterTuner can perform random search optimization."""
        result = await hyperparameter_tuner.random_search(n_trials=10)

        assert isinstance(result, OptimizationResult)
        assert len(hyperparameter_tuner.optimization_history) == 10
        assert hyperparameter_tuner.best_result is not None
        assert result == hyperparameter_tuner.best_result

    @pytest.mark.asyncio
    async def test_should_perform_grid_search(self, hyperparameter_tuner):
        """Test that HyperparameterTuner can perform grid search optimization."""
        result = await hyperparameter_tuner.grid_search(grid_size=3)

        assert isinstance(result, OptimizationResult)
        assert len(hyperparameter_tuner.optimization_history) == 27  # 3^3 grid
        assert hyperparameter_tuner.best_result is not None

    @pytest.mark.asyncio
    async def test_should_perform_bayesian_optimization(self, hyperparameter_tuner):
        """Test that HyperparameterTuner can perform Bayesian optimization."""
        result = await hyperparameter_tuner.bayesian_optimization(n_trials=15, n_initial=5)

        assert isinstance(result, OptimizationResult)
        assert len(hyperparameter_tuner.optimization_history) == 15
        assert hyperparameter_tuner.best_result is not None

        # Check that optimization improved over random initial points
        initial_results = hyperparameter_tuner.optimization_history[:5]
        later_results = hyperparameter_tuner.optimization_history[10:]
        avg_initial = sum(r.objective_value for r in initial_results) / len(initial_results)
        avg_later = sum(r.objective_value for r in later_results) / len(later_results)

        # Bayesian optimization should improve over time
        assert avg_later >= avg_initial - 0.1  # Allow some variance

    @pytest.mark.asyncio
    async def test_should_handle_objective_function_errors(self, parameter_space):
        """Test that HyperparameterTuner handles objective function errors gracefully."""
        async def failing_objective(params):
            raise ValueError("Simulation failed")

        tuner = HyperparameterTuner(parameter_space, failing_objective)

        with pytest.raises(ValidationError, match="Objective function evaluation failed"):
            await tuner.random_search(n_trials=5)

    @pytest.mark.asyncio
    async def test_should_validate_optimization_parameters(self, hyperparameter_tuner):
        """Test that HyperparameterTuner validates optimization parameters."""
        # Test invalid n_trials
        with pytest.raises(ValidationError, match="n_trials must be positive"):
            await hyperparameter_tuner.random_search(n_trials=0)

        # Test invalid grid_size
        with pytest.raises(ValidationError, match="grid_size must be positive"):
            await hyperparameter_tuner.grid_search(grid_size=0)

        # Test invalid Bayesian parameters
        with pytest.raises(ValidationError, match="n_initial must be positive"):
            await hyperparameter_tuner.bayesian_optimization(n_trials=10, n_initial=0)

    @pytest.mark.asyncio
    async def test_should_support_early_stopping(self, hyperparameter_tuner):
        """Test that HyperparameterTuner supports early stopping."""
        # Create objective that reaches optimal quickly
        async def converging_objective(params):
            # Returns high value when lookback_period is close to 30
            return 1.0 - abs(params['lookback_period'] - 30) / 40

        hyperparameter_tuner.objective_function = converging_objective

        result = await hyperparameter_tuner.bayesian_optimization(
            n_trials=50,
            n_initial=5,
            early_stopping_rounds=5,
            tolerance=0.01
        )

        # Should stop early when no improvement
        assert len(hyperparameter_tuner.optimization_history) < 50

    @pytest.mark.asyncio
    async def test_should_perform_cross_validation(self, parameter_space):
        """Test that HyperparameterTuner can perform cross-validation."""
        async def cv_objective(params, fold_data):
            # Mock cross-validation objective
            base_score = 1.0 / (1.0 + sum((params[k] - 25) ** 2 for k in params.keys()))
            fold_variance = np.random.normal(0, 0.1)  # Add some variance per fold
            return base_score + fold_variance

        tuner = HyperparameterTuner(parameter_space, cv_objective)

        # Mock fold data
        fold_data = [{'train': [1, 2, 3], 'test': [4, 5]} for _ in range(5)]

        result = await tuner.cross_validate_parameters(
            parameters={'lookback_period': 25, 'entry_threshold': 2.0, 'exit_threshold': 1.0},
            fold_data=fold_data
        )

        assert isinstance(result, dict)
        assert 'mean_score' in result
        assert 'std_score' in result
        assert 'fold_scores' in result
        assert len(result['fold_scores']) == 5

    @pytest.mark.asyncio
    async def test_should_perform_walk_forward_optimization(self, hyperparameter_tuner):
        """Test that HyperparameterTuner can perform walk-forward optimization."""
        # Mock time series data
        time_series_data = [
            {'date': datetime.now() - timedelta(days=i), 'price': 100 + i}
            for i in range(100, 0, -1)
        ]

        result = await hyperparameter_tuner.walk_forward_optimization(
            time_series_data=time_series_data,
            train_size=60,
            test_size=20,
            step_size=10
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert all('parameters' in r and 'train_performance' in r and 'test_performance' in r for r in result)

    def test_should_get_optimization_history(self, hyperparameter_tuner):
        """Test that HyperparameterTuner provides optimization history."""
        # Add mock results to history
        for i in range(5):
            result = OptimizationResult(
                parameters={'lookback_period': 20 + i},
                objective_value=0.1 + i * 0.02,
                metrics={}
            )
            hyperparameter_tuner.optimization_history.append(result)

        history = hyperparameter_tuner.get_optimization_history()

        assert len(history) == 5
        assert all(isinstance(r, OptimizationResult) for r in history)

    def test_should_get_best_parameters(self, hyperparameter_tuner):
        """Test that HyperparameterTuner returns best parameters."""
        # Add mock results
        results = [
            OptimizationResult({'a': 1}, 0.1, {}),
            OptimizationResult({'a': 2}, 0.3, {}),  # Best
            OptimizationResult({'a': 3}, 0.2, {})
        ]

        for result in results:
            hyperparameter_tuner.optimization_history.append(result)

        hyperparameter_tuner.best_result = max(results, key=lambda r: r.objective_value)

        best_params = hyperparameter_tuner.get_best_parameters()
        assert best_params == {'a': 2}

    def test_should_export_optimization_report(self, hyperparameter_tuner):
        """Test that HyperparameterTuner can export optimization report."""
        # Add mock results
        for i in range(10):
            result = OptimizationResult(
                parameters={'lookback_period': 20 + i},
                objective_value=0.1 + i * 0.01,
                metrics={'sharpe_ratio': 1.0 + i * 0.1}
            )
            hyperparameter_tuner.optimization_history.append(result)

        hyperparameter_tuner.best_result = hyperparameter_tuner.optimization_history[-1]

        report = hyperparameter_tuner.export_optimization_report()

        assert isinstance(report, dict)
        assert 'best_parameters' in report
        assert 'best_objective_value' in report
        assert 'optimization_summary' in report
        assert 'parameter_importance' in report
        assert 'convergence_analysis' in report

    @pytest.mark.asyncio
    async def test_should_optimize_with_constraints(self, parameter_space):
        """Test that HyperparameterTuner supports constrained optimization."""
        async def constrained_objective(params):
            # Constraint: entry_threshold must be > exit_threshold
            if params['entry_threshold'] <= params['exit_threshold']:
                return float('-inf')  # Invalid solution
            return 1.0 / (1.0 + sum((params[k] - 25) ** 2 for k in params.keys()))

        tuner = HyperparameterTuner(parameter_space, constrained_objective)

        result = await tuner.random_search(n_trials=20)

        # All valid results should satisfy constraint
        valid_results = [r for r in tuner.optimization_history if r.objective_value > float('-inf')]
        assert len(valid_results) > 0
        assert all(r.parameters['entry_threshold'] > r.parameters['exit_threshold']
                  for r in valid_results)

    @pytest.mark.asyncio
    async def test_should_handle_multi_objective_optimization(self, parameter_space):
        """Test that HyperparameterTuner can handle multi-objective optimization."""
        async def multi_objective(params):
            # Return multiple objectives
            obj1 = 1.0 / (1.0 + abs(params['lookback_period'] - 30))  # Maximize
            obj2 = abs(params['entry_threshold'] - 2.0)  # Minimize
            return {'return': obj1, 'risk': obj2}

        tuner = HyperparameterTuner(parameter_space, multi_objective)

        result = await tuner.multi_objective_optimization(
            n_trials=20,
            objectives={'return': 'maximize', 'risk': 'minimize'},
            weights={'return': 0.7, 'risk': 0.3}
        )

        assert isinstance(result, OptimizationResult)
        assert 'return' in result.metrics
        assert 'risk' in result.metrics