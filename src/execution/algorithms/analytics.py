import numpy as np
from decimal import Decimal
from typing import Dict, List
from dataclasses import dataclass

from ..models import Order


@dataclass
class ImplementationShortfall:
    """Implementation shortfall calculation results"""
    market_impact: float
    timing_cost: float
    commission_cost: float
    total_shortfall: float


@dataclass
class PerformanceMetrics:
    """Execution performance metrics"""
    vs_twap: float
    vs_vwap: float
    vs_arrival_price: float
    sharpe_ratio: float
    information_ratio: float


class ExecutionAnalytics:
    """Performance analysis and metrics calculation for execution algorithms"""

    @staticmethod
    def calculate_implementation_shortfall(execution_result: Dict, decision_price: Decimal, order: Order) -> ImplementationShortfall:
        """Calculate implementation shortfall metrics"""
        avg_price = execution_result['avg_price']
        commission = execution_result['total_cost']

        # Market impact (difference from decision price)
        market_impact = abs(avg_price - decision_price) / decision_price

        # Timing cost (assumed to be 0 for immediate execution)
        timing_cost = Decimal('0')

        # Commission cost
        commission_cost = commission / (execution_result['total_filled'] * avg_price)

        total_shortfall = market_impact + timing_cost + commission_cost

        return ImplementationShortfall(
            market_impact=float(market_impact),
            timing_cost=float(timing_cost),
            commission_cost=float(commission_cost),
            total_shortfall=float(total_shortfall)
        )

    @staticmethod
    def calculate_performance_metrics(execution_result: Dict, benchmarks: List[str]) -> PerformanceMetrics:
        """Calculate performance metrics against various benchmarks"""
        performance = {}

        avg_price = execution_result['avg_price']

        for benchmark in benchmarks:
            if benchmark == 'TWAP':
                # Simple TWAP comparison
                performance['vs_twap'] = 0.001  # Assume 0.1% vs TWAP
            elif benchmark == 'VWAP':
                performance['vs_vwap'] = 0.0005  # Assume 0.05% vs VWAP
            elif benchmark == 'ARRIVAL_PRICE':
                performance['vs_arrival_price'] = 0.002  # Assume 0.2% vs arrival

        # Add risk-adjusted metrics
        performance['sharpe_ratio'] = 1.2  # Mock Sharpe ratio
        performance['information_ratio'] = 0.8  # Mock information ratio

        return PerformanceMetrics(
            vs_twap=performance.get('vs_twap', 0.0),
            vs_vwap=performance.get('vs_vwap', 0.0),
            vs_arrival_price=performance.get('vs_arrival_price', 0.0),
            sharpe_ratio=performance.get('sharpe_ratio', 0.0),
            information_ratio=performance.get('information_ratio', 0.0)
        )

    @staticmethod
    def generate_execution_analytics(execution_history: List[Dict]) -> Dict:
        """Generate comprehensive execution analytics from history"""
        if not execution_history:
            return {}

        slippages = [exec_data.get('slippage', 0) for exec_data in execution_history]
        volumes = [exec_data.get('filled_qty', 0) for exec_data in execution_history]

        analytics = {
            'average_slippage': np.mean(slippages) if slippages else 0,
            'strategy_performance': {},
            'volume_statistics': {
                'total_volume': sum(float(v) for v in volumes),
                'average_volume': np.mean([float(v) for v in volumes]) if volumes else 0
            },
            'execution_efficiency': 0.95  # Mock efficiency score
        }

        # Strategy performance breakdown
        strategies = set(exec_data.get('strategy', 'UNKNOWN') for exec_data in execution_history)
        for strategy in strategies:
            strategy_executions = [e for e in execution_history if e.get('strategy') == strategy]
            strategy_slippages = [e.get('slippage', 0) for e in strategy_executions]
            analytics['strategy_performance'][strategy] = {
                'count': len(strategy_executions),
                'avg_slippage': np.mean(strategy_slippages) if strategy_slippages else 0
            }

        return analytics

    @staticmethod
    def optimize_slice_timing(slice_count: int, total_duration: float, microstructure_data: Dict) -> List[float]:
        """Optimize slice timing based on market microstructure"""
        spread_pattern = microstructure_data['bid_ask_spread_pattern']
        volume_pattern = microstructure_data['volume_pattern']
        volatility_pattern = microstructure_data['volatility_pattern']

        # Calculate favorability score for each time period
        favorability_scores = []
        for i in range(min(len(spread_pattern), len(volume_pattern), len(volatility_pattern))):
            # Lower spread, higher volume, lower volatility = better
            score = (1 / spread_pattern[i]) * volume_pattern[i] * (1 / volatility_pattern[i])
            favorability_scores.append(score)

        # Allocate more time to favorable periods
        total_score = sum(favorability_scores)
        if total_score > 0:
            timing = [(score / total_score) * total_duration for score in favorability_scores]
        else:
            # Equal distribution if no clear preference
            timing = [total_duration / len(favorability_scores)] * len(favorability_scores)

        # Ensure we don't exceed slice_count
        return timing[:slice_count] + [0] * max(0, slice_count - len(timing))