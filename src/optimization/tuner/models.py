# src/optimization/tuner/models.py
"""
Hyperparameter Tuning Models

Core data models and parameter space definitions for hyperparameter optimization.
"""

import random
import statistics
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np


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
        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            raise ValidationError(f"Parameter range for '{name}' must contain numeric values")

        if min_val >= max_val:
            raise ValidationError(f"Parameter '{name}' min value must be less than max value")

    def generate_random_parameters(self) -> Dict[str, float]:
        """Generate random parameters within the defined space."""
        parameters = {}
        for name, (min_val, max_val) in self.parameters.items():
            parameters[name] = random.uniform(min_val, max_val)
        return parameters

    def generate_grid_parameters(self, grid_size: int) -> List[Dict[str, float]]:
        """Generate grid of parameters for grid search."""
        if grid_size <= 0:
            raise ValidationError("grid_size must be positive")

        # Generate grid points for each parameter
        param_grids = {}
        for name, (min_val, max_val) in self.parameters.items():
            param_grids[name] = np.linspace(min_val, max_val, grid_size)

        # Generate all combinations
        parameters_list = []
        param_names = list(self.parameters.keys())

        def generate_combinations(index: int, current_params: Dict[str, float]):
            if index == len(param_names):
                parameters_list.append(current_params.copy())
                return

            param_name = param_names[index]
            for value in param_grids[param_name]:
                current_params[param_name] = float(value)
                generate_combinations(index + 1, current_params)

        generate_combinations(0, {})
        return parameters_list

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
        """Denormalize parameters from [0, 1] range to original range."""
        denormalized = {}
        for name, norm_value in normalized_params.items():
            if name in self.parameters:
                min_val, max_val = self.parameters[name]
                denormalized[name] = min_val + norm_value * (max_val - min_val)
            else:
                denormalized[name] = norm_value
        return denormalized

    def is_valid_parameters(self, parameters: Dict[str, float]) -> bool:
        """Check if parameters are within valid ranges."""
        for name, value in parameters.items():
            if name in self.parameters:
                min_val, max_val = self.parameters[name]
                if not (min_val <= value <= max_val):
                    return False
        return True

    def clip_parameters(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Clip parameters to valid ranges."""
        clipped = {}
        for name, value in parameters.items():
            if name in self.parameters:
                min_val, max_val = self.parameters[name]
                clipped[name] = max(min_val, min(max_val, value))
            else:
                clipped[name] = value
        return clipped