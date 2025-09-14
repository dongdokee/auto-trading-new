"""
Portfolio Management Module

This module implements portfolio optimization and management functionality including:

- PortfolioOptimizer: Markowitz optimization with transaction costs and constraints
- PerformanceAttributor: Strategy-level performance attribution and analysis
- CorrelationAnalyzer: Multi-strategy correlation analysis and risk decomposition
- AdaptiveAllocator: Performance-based dynamic strategy allocation

Key Components:
- Markowitz mean-variance optimization with modern portfolio theory
- Transaction cost modeling for realistic optimization
- Strategy performance attribution with risk-adjusted metrics
- Correlation-aware position sizing across strategies
- Adaptive allocation based on rolling performance windows

Integration Points:
- Uses StrategyManager signals for portfolio construction
- Integrates with RiskController for portfolio-level risk management
- Provides optimized allocations to PositionSizer for execution
"""

from typing import Dict, List, Optional, Any

# Core portfolio optimization
from .portfolio_optimizer import PortfolioOptimizer, OptimizationResult, OptimizationConfig

# Performance attribution
from .performance_attributor import PerformanceAttributor, AttributionResult, AttributionConfig

# Correlation analysis
from .correlation_analyzer import CorrelationAnalyzer, CorrelationMatrix, CorrelationConfig

# Adaptive allocation
from .adaptive_allocator import AdaptiveAllocator, AllocationUpdate, AdaptiveConfig

__all__ = [
    # Core optimization
    'PortfolioOptimizer',
    'OptimizationResult',
    'OptimizationConfig',

    # Performance attribution
    'PerformanceAttributor',
    'AttributionResult',
    'AttributionConfig',

    # Correlation analysis
    'CorrelationAnalyzer',
    'CorrelationMatrix',
    'CorrelationConfig',

    # Adaptive allocation
    'AdaptiveAllocator',
    'AllocationUpdate',
    'AdaptiveConfig',
]

__version__ = "1.0.0"