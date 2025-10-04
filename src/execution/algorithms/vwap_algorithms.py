import asyncio
from decimal import Decimal
from typing import Dict

from .base import BaseExecutionAlgorithm
from src.execution.models import Order, OrderSide


class VWAPAlgorithm(BaseExecutionAlgorithm):
    """Volume-Weighted Average Price execution algorithm"""

    async def execute(self, order: Order, market_analysis: Dict, volume_profile: Dict) -> Dict:
        """Execute VWAP algorithm following volume profile"""

        # Create volume schedule
        volume_schedule = self._create_volume_schedule(order.size, volume_profile)
        vwap_benchmark = self._calculate_vwap_benchmark(market_analysis, volume_profile)

        result = {
            'strategy': 'VWAP',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0'),
            'volume_schedule': volume_schedule,
            'vwap_benchmark': vwap_benchmark,
            'vwap_slippage': Decimal('0')
        }

        for hour, scheduled_volume in volume_schedule.items():
            if scheduled_volume > 0:
                # Execute volume for this hour
                slice_result = await self.place_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    size=scheduled_volume,
                    order_type='LIMIT',
                    price=market_analysis['vwap_bid'] if order.side == OrderSide.BUY else market_analysis['vwap_ask']
                )

                result['slices'].append(slice_result)

                # Simulate hour progression (shortened for testing)
                await asyncio.sleep(0.1)  # Simulate 1 hour with 0.1 second

        self._aggregate_results(result)

        # Calculate VWAP slippage
        if result['total_filled'] > 0:
            result['vwap_slippage'] = (result['avg_price'] - Decimal(str(vwap_benchmark))) / Decimal(str(vwap_benchmark))

        return result

    def _create_volume_schedule(self, total_size: Decimal, volume_profile: Dict) -> Dict[int, Decimal]:
        """Create volume execution schedule based on historical profile"""
        hourly_volumes = volume_profile['hourly_volumes']
        total_volume = sum(hourly_volumes)

        schedule = {}
        for hour, volume in enumerate(hourly_volumes):
            if total_volume > 0:
                proportion = volume / total_volume
                schedule[hour] = total_size * Decimal(str(proportion))
            else:
                schedule[hour] = Decimal('0')

        return schedule

    def _calculate_vwap_benchmark(self, market_analysis: Dict, volume_profile: Dict) -> float:
        """Calculate VWAP benchmark price"""
        # Simplified VWAP calculation
        return (market_analysis['vwap_bid'] + market_analysis['vwap_ask']) / 2


class AdaptiveVWAPAlgorithm(BaseExecutionAlgorithm):
    """Adaptive VWAP algorithm with pace adjustment"""

    async def execute(self, order: Order, market_analysis: Dict, volume_profile: Dict) -> Dict:
        """Execute adaptive VWAP with execution pace adjustment"""

        volume_schedule = self._create_volume_schedule(order.size, volume_profile)
        target_vwap = self._calculate_vwap_benchmark(market_analysis, volume_profile)

        result = {
            'strategy': 'ADAPTIVE_VWAP',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0'),
            'pace_adjustments': [],
            'target_vwap': target_vwap
        }

        executed_so_far = Decimal('0')
        scheduled_so_far = Decimal('0')

        for hour, scheduled_volume in volume_schedule.items():
            scheduled_so_far += scheduled_volume

            # Calculate pace deviation
            pace_deviation = (executed_so_far - scheduled_so_far) / order.size if order.size > 0 else 0

            # Adjust volume based on pace
            adjusted_volume = scheduled_volume
            if pace_deviation < -0.1:  # Behind schedule
                adjusted_volume *= Decimal('1.2')  # Increase pace
                result['pace_adjustments'].append({
                    'hour': hour,
                    'type': 'accelerate',
                    'deviation': float(pace_deviation)
                })
            elif pace_deviation > 0.1:  # Ahead of schedule
                adjusted_volume *= Decimal('0.8')  # Slow down
                result['pace_adjustments'].append({
                    'hour': hour,
                    'type': 'decelerate',
                    'deviation': float(pace_deviation)
                })

            slice_result = await self.place_order(
                symbol=order.symbol,
                side=order.side.value,
                size=adjusted_volume,
                order_type='LIMIT',
                price=market_analysis['vwap_bid'] if order.side == OrderSide.BUY else market_analysis['vwap_ask']
            )

            result['slices'].append(slice_result)
            executed_so_far += slice_result['filled_qty']

            await asyncio.sleep(0.1)  # Simulate time progression

        self._aggregate_results(result)
        return result

    def _create_volume_schedule(self, total_size: Decimal, volume_profile: Dict) -> Dict[int, Decimal]:
        """Create volume execution schedule"""
        hourly_volumes = volume_profile['hourly_volumes']
        total_volume = sum(hourly_volumes)

        schedule = {}
        for hour, volume in enumerate(hourly_volumes):
            if total_volume > 0:
                proportion = volume / total_volume
                schedule[hour] = total_size * Decimal(str(proportion))
            else:
                schedule[hour] = Decimal('0')

        return schedule

    def _calculate_vwap_benchmark(self, market_analysis: Dict, volume_profile: Dict) -> float:
        """Calculate VWAP benchmark"""
        return (market_analysis['vwap_bid'] + market_analysis['vwap_ask']) / 2