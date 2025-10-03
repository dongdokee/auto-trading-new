import asyncio
from decimal import Decimal
from typing import Dict

from .base import BaseExecutionAlgorithm
from ..models import Order, OrderSide


class ParticipationRateAlgorithm(BaseExecutionAlgorithm):
    """Market participation rate control algorithm"""

    async def execute(self, order: Order, market_analysis: Dict, target_rate: float = 0.2) -> Dict:
        """Execute order with controlled market participation rate"""

        if not 0 < target_rate <= 1.0:
            raise ValueError("Participation rate must be between 0 and 1")

        market_volume_rate = market_analysis['avg_volume_1min']  # Volume per minute
        target_execution_rate = market_volume_rate * target_rate

        result = {
            'strategy': 'PARTICIPATION_RATE',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0'),
            'target_rate': target_rate,
            'actual_rate': 0.0
        }

        remaining_size = order.size
        total_market_volume = Decimal('0')

        while remaining_size > 0:
            # Execute at target participation rate
            slice_size = min(Decimal(str(target_execution_rate)), remaining_size)

            slice_result = await self.place_order(
                symbol=order.symbol,
                side=order.side.value,
                size=slice_size,
                order_type='LIMIT',
                price=market_analysis['best_bid'] if order.side == OrderSide.BUY else market_analysis['best_ask']
            )

            result['slices'].append(slice_result)
            filled = slice_result['filled_qty']
            remaining_size -= filled
            total_market_volume += Decimal(str(market_volume_rate))

            await asyncio.sleep(0.01)  # Wait 1 minute (simulated as 0.01 second for testing)
            market_analysis = await self.get_updated_market_conditions(order.symbol)
            market_volume_rate = market_analysis['avg_volume_1min']
            target_execution_rate = market_volume_rate * target_rate

        self._aggregate_results(result)

        # Calculate actual participation rate
        if total_market_volume > 0:
            result['actual_rate'] = float(result['total_filled'] / total_market_volume)

        return result

    def validate_participation_rate_params(self, params: Dict):
        """Validate participation rate parameters"""
        rate = params.get('participation_rate', 0.2)
        if not 0 < rate <= 1.0:
            raise ValueError("Participation rate must be between 0 and 1")

    def validate_slice_size_params(self, params: Dict):
        """Validate slice size parameters"""
        max_size = params.get('max_slice_size', 1.0)
        if max_size <= 0:
            raise ValueError("Slice size must be positive")

    def validate_timing_params(self, params: Dict):
        """Validate timing parameters"""
        interval = params.get('min_slice_interval', 1.0)
        if interval <= 0:
            raise ValueError("Slice interval must be positive")