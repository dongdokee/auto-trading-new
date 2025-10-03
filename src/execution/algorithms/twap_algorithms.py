import asyncio
import numpy as np
from decimal import Decimal
from typing import Dict
from datetime import datetime

from .base import BaseExecutionAlgorithm
from ..models import Order, OrderSide, OrderUrgency


class TWAPAlgorithm(BaseExecutionAlgorithm):
    """Time-Weighted Average Price execution algorithm"""

    async def execute(self, order: Order, market_analysis: Dict) -> Dict:
        """Execute enhanced TWAP algorithm"""

        # Calculate optimal duration using Almgren-Chriss model
        optimal_duration = self._calculate_optimal_duration(order.size, market_analysis)

        # Adjust slice count based on volatility
        volatility = market_analysis.get('volatility', 0.02)
        base_slices = max(5, int(optimal_duration / 60))

        # Higher volatility = more slices
        volatility_multiplier = 1 + (volatility - 0.02) * 10
        n_slices = max(5, int(base_slices * volatility_multiplier))

        slice_size = order.size / n_slices
        slice_interval = optimal_duration / n_slices

        result = {
            'strategy': 'TWAP',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0'),
            'duration': optimal_duration,
            'slice_count': n_slices,
            'volatility_adjustment': volatility_multiplier
        }

        for i in range(n_slices):
            slice_order = Order(
                symbol=order.symbol,
                side=order.side,
                size=slice_size,
                urgency=OrderUrgency.MEDIUM
            )

            # Execute slice
            slice_result = await self.place_order(
                symbol=slice_order.symbol,
                side=slice_order.side.value,
                size=slice_order.size,
                order_type='LIMIT',
                price=market_analysis['best_bid'] if order.side == OrderSide.BUY else market_analysis['best_ask']
            )

            result['slices'].append(slice_result)

            if i < n_slices - 1:
                await asyncio.sleep(slice_interval)
                # Update market conditions for next slice
                market_analysis = await self.get_updated_market_conditions(order.symbol)

        self._aggregate_results(result)
        return result

    def _calculate_optimal_duration(self, size: Decimal, market_analysis: Dict) -> float:
        """Calculate optimal execution duration using Almgren-Chriss model"""
        daily_volume = market_analysis.get('daily_volume', 10000000)
        volatility = market_analysis.get('volatility', 0.02)

        # Simplified Almgren-Chriss model
        risk_aversion = 1.0
        temporary_impact = 0.1

        optimal_time = np.sqrt(
            float(size) * volatility / (risk_aversion * daily_volume * temporary_impact)
        )

        # Constrain to reasonable range (30 seconds to 30 minutes)
        return np.clip(optimal_time * 3600, 30, 1800)


class DynamicTWAPAlgorithm(BaseExecutionAlgorithm):
    """Dynamic TWAP algorithm with market condition adaptation"""

    async def execute(self, order: Order, market_analysis: Dict) -> Dict:
        """Execute dynamic TWAP with market feedback"""

        initial_duration = self._calculate_optimal_duration(order.size, market_analysis)
        n_slices = max(5, int(initial_duration / 60))
        slice_size = order.size / n_slices

        result = {
            'strategy': 'DYNAMIC_TWAP',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0'),
            'adaptations': [],
            'initial_duration': initial_duration
        }

        remaining_size = order.size
        slice_count = 0

        while remaining_size > 0 and slice_count < n_slices * 2:  # Safety limit
            current_slice_size = min(slice_size, remaining_size)

            # Evaluate market conditions
            current_analysis = await self.get_updated_market_conditions(order.symbol)

            # Reduce slice size if conditions deteriorated
            if current_analysis['spread_bps'] > market_analysis['spread_bps'] * 1.5:
                current_slice_size *= Decimal('0.7')
                result['adaptations'].append({
                    'type': 'reduce_size',
                    'reason': 'spread_widening',
                    'slice': slice_count
                })

            # Increase slice size if conditions improved
            elif current_analysis['spread_bps'] < market_analysis['spread_bps'] * 0.8:
                current_slice_size = min(current_slice_size * Decimal('1.3'), remaining_size)
                result['adaptations'].append({
                    'type': 'increase_size',
                    'reason': 'spread_tightening',
                    'slice': slice_count
                })

            slice_result = await self.place_order(
                symbol=order.symbol,
                side=order.side.value,
                size=current_slice_size,
                order_type='LIMIT',
                price=current_analysis['best_bid'] if order.side == OrderSide.BUY else current_analysis['best_ask']
            )

            result['slices'].append(slice_result)
            filled = slice_result['filled_qty']
            remaining_size -= filled
            slice_count += 1

            # Adaptive wait time
            wait_time = self._calculate_adaptive_wait_time(current_analysis, market_analysis)
            if remaining_size > 0:
                await asyncio.sleep(wait_time)

            market_analysis = current_analysis  # Update for next iteration

        self._aggregate_results(result)
        return result

    def _calculate_optimal_duration(self, size: Decimal, market_analysis: Dict) -> float:
        """Calculate optimal execution duration"""
        daily_volume = market_analysis.get('daily_volume', 10000000)
        volatility = market_analysis.get('volatility', 0.02)

        risk_aversion = 1.0
        temporary_impact = 0.1

        optimal_time = np.sqrt(
            float(size) * volatility / (risk_aversion * daily_volume * temporary_impact)
        )

        return np.clip(optimal_time * 3600, 30, 1800)

    def _calculate_adaptive_wait_time(self, current_analysis: Dict, initial_analysis: Dict) -> float:
        """Calculate adaptive wait time based on market conditions"""
        base_wait = 10.0  # 10 seconds base

        # Adjust based on spread change
        spread_ratio = current_analysis['spread_bps'] / initial_analysis['spread_bps']
        if spread_ratio > 1.2:  # Spread widened
            return base_wait * 1.5  # Wait longer
        elif spread_ratio < 0.8:  # Spread tightened
            return base_wait * 0.7  # Wait less

        return base_wait


class TWAPWithFallback(BaseExecutionAlgorithm):
    """TWAP algorithm with error recovery and fallback mechanisms"""

    async def execute(self, order: Order, market_analysis: Dict) -> Dict:
        """Execute TWAP with fallback recovery"""

        result = {
            'strategy': 'TWAP_WITH_FALLBACK',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0'),
            'errors': [],
            'recovery_actions': []
        }

        n_slices = 5
        slice_size = order.size / n_slices

        for i in range(n_slices):
            try:
                slice_result = await self.place_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    size=slice_size,
                    order_type='LIMIT',
                    price=market_analysis['best_bid'] if order.side == OrderSide.BUY else market_analysis['best_ask']
                )
                result['slices'].append(slice_result)

            except Exception as e:
                # Record error and implement recovery
                result['errors'].append({
                    'slice': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

                # Recovery action: retry with smaller slice
                try:
                    recovery_slice_size = slice_size * Decimal('0.5')
                    recovery_result = await self.place_order(
                        symbol=order.symbol,
                        side=order.side.value,
                        size=recovery_slice_size,
                        order_type='MARKET',  # Use market order for recovery
                        price=None
                    )
                    result['slices'].append(recovery_result)
                    result['recovery_actions'].append({
                        'slice': i,
                        'action': 'retry_with_smaller_size',
                        'recovery_size': float(recovery_slice_size)
                    })
                except Exception as recovery_error:
                    result['errors'].append({
                        'slice': i,
                        'error': f"Recovery failed: {recovery_error}",
                        'timestamp': datetime.now().isoformat()
                    })

            await asyncio.sleep(1)  # Wait between slices

        self._aggregate_results(result)
        return result