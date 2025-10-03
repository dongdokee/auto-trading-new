"""
Execution Algorithms - Refactored Module

This module now serves as a backward-compatible wrapper around the refactored algorithms package.
All original functionality is preserved through imports from specialized submodules.

DEPRECATION NOTICE:
This file is maintained for backward compatibility only.
New code should import directly from the algorithms package:
    from .algorithms import TWAPAlgorithm, VWAPAlgorithm, etc.

Original file has been split into:
- algorithms/base.py: Common base classes and utilities
- algorithms/twap_algorithms.py: TWAP variants
- algorithms/vwap_algorithms.py: VWAP variants
- algorithms/adaptive_algorithms.py: Multi-signal adaptive algorithms
- algorithms/participation_algorithms.py: Participation rate control
- algorithms/analytics.py: Performance analysis and metrics
"""

# Backward compatibility imports
from .algorithms import (
    BaseExecutionAlgorithm,
    ExecutionMetrics,
    TWAPAlgorithm,
    DynamicTWAPAlgorithm,
    TWAPWithFallback,
    VWAPAlgorithm,
    AdaptiveVWAPAlgorithm,
    AdaptiveAlgorithm,
    SignalCalculator,
    ParticipationRateAlgorithm,
    ExecutionAnalytics,
    ImplementationShortfall,
    PerformanceMetrics
)

# For compatibility with existing tests and usage
from .models import Order, OrderSide, OrderUrgency
from typing import Dict, List
from decimal import Decimal
import warnings


class ExecutionAlgorithms:
    """Legacy wrapper for execution algorithms - DEPRECATED

    This class maintains backward compatibility with the original ExecutionAlgorithms
    interface while delegating to the new specialized algorithm classes.
    """

    def __init__(self):
        warnings.warn(
            "ExecutionAlgorithms class is deprecated. Use specific algorithm classes "
            "from src.execution.algorithms package instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Initialize algorithm instances
        self.twap_algo = TWAPAlgorithm()
        self.dynamic_twap_algo = DynamicTWAPAlgorithm()
        self.twap_fallback_algo = TWAPWithFallback()
        self.vwap_algo = VWAPAlgorithm()
        self.adaptive_vwap_algo = AdaptiveVWAPAlgorithm()
        self.adaptive_algo = AdaptiveAlgorithm()
        self.participation_algo = ParticipationRateAlgorithm()

        # Legacy attributes
        self.execution_history = []
        self.performance_cache = {}

    async def execute_twap(self, order: Order, market_analysis: Dict) -> Dict:
        """Enhanced TWAP algorithm - DEPRECATED

        Use TWAPAlgorithm.execute() instead.
        """
        return await self.twap_algo.execute(order, market_analysis)

    async def execute_dynamic_twap(self, order: Order, market_analysis: Dict) -> Dict:
        """Dynamic TWAP algorithm - DEPRECATED

        Use DynamicTWAPAlgorithm.execute() instead.
        """
        return await self.dynamic_twap_algo.execute(order, market_analysis)

    async def execute_vwap(self, order: Order, market_analysis: Dict, volume_profile: Dict) -> Dict:
        """VWAP algorithm - DEPRECATED

        Use VWAPAlgorithm.execute() instead.
        """
        return await self.vwap_algo.execute(order, market_analysis, volume_profile)

    async def execute_adaptive_vwap(self, order: Order, market_analysis: Dict, volume_profile: Dict) -> Dict:
        """Adaptive VWAP algorithm - DEPRECATED

        Use AdaptiveVWAPAlgorithm.execute() instead.
        """
        return await self.adaptive_vwap_algo.execute(order, market_analysis, volume_profile)

    async def execute_adaptive(self, order: Order, market_analysis: Dict) -> Dict:
        """Multi-signal adaptive algorithm - DEPRECATED

        Use AdaptiveAlgorithm.execute() instead.
        """
        return await self.adaptive_algo.execute(order, market_analysis)

    async def execute_participation_rate(self, order: Order, market_analysis: Dict, target_rate: float) -> Dict:
        """Participation rate control algorithm - DEPRECATED

        Use ParticipationRateAlgorithm.execute() instead.
        """
        return await self.participation_algo.execute(order, market_analysis, target_rate)

    async def execute_twap_with_fallback(self, order: Order, market_analysis: Dict) -> Dict:
        """TWAP with fallback recovery - DEPRECATED

        Use TWAPWithFallback.execute() instead.
        """
        return await self.twap_fallback_algo.execute(order, market_analysis)

    # Legacy method delegates to analytics
    def calculate_momentum_signal(self, market_analysis: Dict) -> float:
        """DEPRECATED - Use SignalCalculator.calculate_momentum_signal() instead"""
        return SignalCalculator.calculate_momentum_signal(market_analysis)

    def calculate_liquidity_signal(self, market_analysis: Dict) -> float:
        """DEPRECATED - Use SignalCalculator.calculate_liquidity_signal() instead"""
        return SignalCalculator.calculate_liquidity_signal(market_analysis)

    def calculate_volatility_signal(self, market_analysis: Dict) -> float:
        """DEPRECATED - Use SignalCalculator.calculate_volatility_signal() instead"""
        return SignalCalculator.calculate_volatility_signal(market_analysis)

    def optimize_slice_timing(self, slice_count: int, total_duration: float, microstructure_data: Dict) -> List[float]:
        """DEPRECATED - Use ExecutionAnalytics.optimize_slice_timing() instead"""
        return ExecutionAnalytics.optimize_slice_timing(slice_count, total_duration, microstructure_data)

    def calculate_implementation_shortfall(self, execution_result: Dict, decision_price: Decimal, order: Order) -> Dict:
        """DEPRECATED - Use ExecutionAnalytics.calculate_implementation_shortfall() instead"""
        shortfall = ExecutionAnalytics.calculate_implementation_shortfall(execution_result, decision_price, order)
        return {
            'market_impact': shortfall.market_impact,
            'timing_cost': shortfall.timing_cost,
            'commission_cost': shortfall.commission_cost,
            'total_shortfall': shortfall.total_shortfall
        }

    def calculate_performance_metrics(self, execution_result: Dict, benchmarks: List[str]) -> Dict:
        """DEPRECATED - Use ExecutionAnalytics.calculate_performance_metrics() instead"""
        metrics = ExecutionAnalytics.calculate_performance_metrics(execution_result, benchmarks)
        return {
            'vs_twap': metrics.vs_twap,
            'vs_vwap': metrics.vs_vwap,
            'vs_arrival_price': metrics.vs_arrival_price,
            'sharpe_ratio': metrics.sharpe_ratio,
            'information_ratio': metrics.information_ratio
        }

    def generate_execution_analytics(self, execution_history: List[Dict]) -> Dict:
        """DEPRECATED - Use ExecutionAnalytics.generate_execution_analytics() instead"""
        return ExecutionAnalytics.generate_execution_analytics(execution_history)

    # Validation methods (delegate to ParticipationRateAlgorithm)
    def validate_participation_rate_params(self, params: Dict):
        """DEPRECATED - Use ParticipationRateAlgorithm.validate_participation_rate_params() instead"""
        self.participation_algo.validate_participation_rate_params(params)

    def validate_slice_size_params(self, params: Dict):
        """DEPRECATED - Use ParticipationRateAlgorithm.validate_slice_size_params() instead"""
        self.participation_algo.validate_slice_size_params(params)

    def validate_timing_params(self, params: Dict):
        """DEPRECATED - Use ParticipationRateAlgorithm.validate_timing_params() instead"""
        self.participation_algo.validate_timing_params(params)

    # Utility methods (delegate to base algorithm)
    def _aggregate_results(self, result: Dict):
        """DEPRECATED - Use BaseExecutionAlgorithm._aggregate_results() instead"""
        self.twap_algo._aggregate_results(result)

    async def get_updated_market_conditions(self, symbol: str) -> Dict:
        """DEPRECATED - Use BaseExecutionAlgorithm.get_updated_market_conditions() instead"""
        return await self.twap_algo.get_updated_market_conditions(symbol)

    async def place_order(self, **kwargs) -> Dict:
        """DEPRECATED - Use BaseExecutionAlgorithm.place_order() instead"""
        return await self.twap_algo.place_order(**kwargs)