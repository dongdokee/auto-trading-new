import asyncio
import numpy as np
from decimal import Decimal
from typing import Dict

from .base import BaseExecutionAlgorithm
from ..models import Order, OrderSide


class SignalCalculator:
    """Calculate various market signals for adaptive execution"""

    @staticmethod
    def calculate_momentum_signal(market_analysis: Dict) -> float:
        """Calculate momentum signal based on market imbalance"""
        imbalance = market_analysis.get('imbalance', 0)
        return np.tanh(imbalance * 5)  # Normalize to [-1, 1]

    @staticmethod
    def calculate_liquidity_signal(market_analysis: Dict) -> float:
        """Calculate liquidity signal"""
        liquidity_score = market_analysis.get('liquidity_score', 0.5)
        return (liquidity_score - 0.5) * 2  # Convert [0, 1] to [-1, 1]

    @staticmethod
    def calculate_volatility_signal(market_analysis: Dict) -> float:
        """Calculate volatility signal (negative for high volatility)"""
        volatility = market_analysis.get('volatility', 0.02)
        # Negative signal for high volatility (indicates caution)
        return -(volatility - 0.02) * 20  # Normalize around 2% baseline


class AdaptiveAlgorithm(BaseExecutionAlgorithm):
    """Multi-signal adaptive execution algorithm"""

    def __init__(self):
        super().__init__()
        self.signal_calculator = SignalCalculator()

    async def execute(self, order: Order, market_analysis: Dict) -> Dict:
        """Execute order using adaptive algorithm with multiple signals"""

        # Calculate initial signals
        momentum_signal = self.signal_calculator.calculate_momentum_signal(market_analysis)
        liquidity_signal = self.signal_calculator.calculate_liquidity_signal(market_analysis)
        volatility_signal = self.signal_calculator.calculate_volatility_signal(market_analysis)

        result = {
            'strategy': 'ADAPTIVE',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0'),
            'signals': {
                'momentum': momentum_signal,
                'liquidity': liquidity_signal,
                'volatility': volatility_signal
            },
            'slice_adjustments': []
        }

        remaining_size = order.size
        base_slice_size = order.size / 10  # Start with 10% slices

        while remaining_size > 0:
            # Calculate combined signal score
            signal_score = (momentum_signal + liquidity_signal - abs(volatility_signal)) / 3

            # Adjust slice size based on signal
            if signal_score > 0.3:  # Favorable conditions
                current_slice_size = min(base_slice_size * Decimal('1.5'), remaining_size)
                adjustment_type = 'increase'
            elif signal_score < -0.3:  # Unfavorable conditions
                current_slice_size = base_slice_size * Decimal('0.5')
                adjustment_type = 'decrease'
            else:  # Neutral conditions
                current_slice_size = base_slice_size
                adjustment_type = 'maintain'

            result['slice_adjustments'].append({
                'signal_score': signal_score,
                'adjustment': adjustment_type,
                'slice_size': float(current_slice_size)
            })

            slice_result = await self.place_order(
                symbol=order.symbol,
                side=order.side.value,
                size=current_slice_size,
                order_type='LIMIT',
                price=market_analysis['best_bid'] if order.side == OrderSide.BUY else market_analysis['best_ask']
            )

            result['slices'].append(slice_result)
            filled = slice_result['filled_qty']
            remaining_size -= filled

            # Update signals for next iteration
            market_analysis = await self.get_updated_market_conditions(order.symbol)
            momentum_signal = self.signal_calculator.calculate_momentum_signal(market_analysis)
            liquidity_signal = self.signal_calculator.calculate_liquidity_signal(market_analysis)
            volatility_signal = self.signal_calculator.calculate_volatility_signal(market_analysis)

            # Random wait between slices (optimized for testing)
            await asyncio.sleep(np.random.uniform(0.01, 0.05))

        self._aggregate_results(result)
        return result