# src/optimization/hyperparameter_tuner_new.py
"""
Hyperparameter Tuner - Backward Compatibility Wrapper

This is a backward compatibility wrapper for the refactored hyperparameter tuner.
All classes and functionality are imported from the new modular structure.

DEPRECATION WARNING: This wrapper will be removed in a future version.
Please update imports to use:
  from src.optimization.tuner import HyperparameterTuner, ParameterSpace, OptimizationResult
"""

import warnings
from .tuner import (
    HyperparameterTuner,
    ValidationError,
    OptimizationResult,
    ParameterSpace
)

# Issue deprecation warning
warnings.warn(
    "src.optimization.hyperparameter_tuner is deprecated. "
    "Use 'from src.optimization.tuner import HyperparameterTuner' instead. "
    "This compatibility layer will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all classes for backward compatibility
__all__ = [
    "ValidationError",
    "OptimizationResult",
    "ParameterSpace",
    "HyperparameterTuner"
]