"""
Optimization module for production performance tuning.

This module provides comprehensive optimization capabilities including:
- Dynamic configuration management with hot-reload
- Hyperparameter tuning with Bayesian optimization
- Performance enhancement through parallelization
- Caching layer implementation
- Database query optimization

Phase 6.1: Production Optimization
Target: 15-35% monthly ROI through system optimization
"""

from .config_optimizer import ConfigOptimizer, DynamicConfig, ConfigValidationError

__all__ = [
    'ConfigOptimizer',
    'DynamicConfig',
    'ConfigValidationError'
]

# Additional modules will be imported as they are implemented
try:
    from .hyperparameter_tuner import HyperparameterTuner, OptimizationResult
    __all__.extend(['HyperparameterTuner', 'OptimizationResult'])
except ImportError:
    pass

try:
    from .performance_enhancer import PerformanceEnhancer, ParallelProcessor
    __all__.extend(['PerformanceEnhancer', 'ParallelProcessor'])
except ImportError:
    pass

try:
    from .cache_manager import CacheManager, CacheConfig
    __all__.extend(['CacheManager', 'CacheConfig'])
except ImportError:
    pass

try:
    from .db_optimizer import DatabaseOptimizer, QueryOptimizer
    __all__.extend(['DatabaseOptimizer', 'QueryOptimizer'])
except ImportError:
    pass

__version__ = "1.0.0"