# src/execution/execution_algorithms.py
import asyncio
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from src.execution.models import Order, OrderSide, OrderUrgency


class ExecutionAlgorithms:
    """고급 실행 알고리즘 구현"""

    def __init__(self):
        self.execution_history = []
        self.performance_cache = {}

    async def execute_twap(self, order: Order, market_analysis: Dict) -> Dict:
        """향상된 TWAP 알고리즘"""

        # 최적 지속시간 계산 (Almgren-Chriss 모델)
        optimal_duration = self._calculate_optimal_duration(order.size, market_analysis)

        # 변동성 기반 슬라이스 수 조정
        volatility = market_analysis.get('volatility', 0.02)
        base_slices = max(5, int(optimal_duration / 60))

        # 높은 변동성일 때 더 많은 슬라이스 사용
        volatility_multiplier = 1 + (volatility - 0.02) * 10  # 변동성 2% 기준
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

            # 슬라이스 실행
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

    async def execute_dynamic_twap(self, order: Order, market_analysis: Dict) -> Dict:
        """동적 조정 TWAP 알고리즘"""

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

            # 시장 조건 평가
            current_analysis = await self.get_updated_market_conditions(order.symbol)

            # 조건이 악화되면 슬라이스 크기 줄이기
            if current_analysis['spread_bps'] > market_analysis['spread_bps'] * 1.5:
                current_slice_size *= Decimal('0.7')
                result['adaptations'].append({
                    'type': 'reduce_size',
                    'reason': 'spread_widening',
                    'slice': slice_count
                })

            # 조건이 개선되면 슬라이스 크기 늘리기
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

            # 적응적 대기 시간
            wait_time = self._calculate_adaptive_wait_time(current_analysis, market_analysis)
            if remaining_size > 0:
                await asyncio.sleep(wait_time)

            market_analysis = current_analysis  # Update for next iteration

        self._aggregate_results(result)
        return result

    async def execute_vwap(self, order: Order, market_analysis: Dict, volume_profile: Dict) -> Dict:
        """VWAP 알고리즘"""

        # 볼륨 스케줄 생성
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

    async def execute_adaptive_vwap(self, order: Order, market_analysis: Dict, volume_profile: Dict) -> Dict:
        """적응형 VWAP 알고리즘"""

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

    async def execute_adaptive(self, order: Order, market_analysis: Dict) -> Dict:
        """고급 적응형 실행 알고리즘"""

        # 다중 신호 계산
        momentum_signal = self.calculate_momentum_signal(market_analysis)
        liquidity_signal = self.calculate_liquidity_signal(market_analysis)
        volatility_signal = self.calculate_volatility_signal(market_analysis)

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
            # 신호 기반 슬라이스 크기 조정
            signal_score = (momentum_signal + liquidity_signal - abs(volatility_signal)) / 3

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
            momentum_signal = self.calculate_momentum_signal(market_analysis)
            liquidity_signal = self.calculate_liquidity_signal(market_analysis)
            volatility_signal = self.calculate_volatility_signal(market_analysis)

            await asyncio.sleep(np.random.uniform(0.01, 0.05))  # Random wait between slices (optimized for testing)

        self._aggregate_results(result)
        return result

    async def execute_participation_rate(self, order: Order, market_analysis: Dict, target_rate: float) -> Dict:
        """참여율 제어 알고리즘"""

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

    async def execute_twap_with_fallback(self, order: Order, market_analysis: Dict) -> Dict:
        """실패 복구 기능이 있는 TWAP"""

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

    # Helper methods
    def _calculate_optimal_duration(self, size: Decimal, market_analysis: Dict) -> float:
        """Almgren-Chriss 모델 기반 최적 집행 시간"""
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

    def _calculate_adaptive_wait_time(self, current_analysis: Dict, initial_analysis: Dict) -> float:
        """적응적 대기 시간 계산"""
        base_wait = 10.0  # 10 seconds base

        # Adjust based on spread change
        spread_ratio = current_analysis['spread_bps'] / initial_analysis['spread_bps']
        if spread_ratio > 1.2:  # Spread widened
            return base_wait * 1.5  # Wait longer
        elif spread_ratio < 0.8:  # Spread tightened
            return base_wait * 0.7  # Wait less

        return base_wait

    def _create_volume_schedule(self, total_size: Decimal, volume_profile: Dict) -> Dict[int, Decimal]:
        """볼륨 스케줄 생성"""
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
        """VWAP 벤치마크 계산"""
        # Simplified VWAP calculation
        return (market_analysis['vwap_bid'] + market_analysis['vwap_ask']) / 2

    def calculate_momentum_signal(self, market_analysis: Dict) -> float:
        """모멘텀 신호 계산"""
        # Simple momentum based on imbalance
        imbalance = market_analysis.get('imbalance', 0)
        return np.tanh(imbalance * 5)  # Normalize to [-1, 1]

    def calculate_liquidity_signal(self, market_analysis: Dict) -> float:
        """유동성 신호 계산"""
        liquidity_score = market_analysis.get('liquidity_score', 0.5)
        return (liquidity_score - 0.5) * 2  # Convert [0, 1] to [-1, 1]

    def calculate_volatility_signal(self, market_analysis: Dict) -> float:
        """변동성 신호 계산"""
        volatility = market_analysis.get('volatility', 0.02)
        # Negative signal for high volatility (indicates caution)
        return -(volatility - 0.02) * 20  # Normalize around 2% baseline

    def optimize_slice_timing(self, slice_count: int, total_duration: float, microstructure_data: Dict) -> List[float]:
        """슬라이스 타이밍 최적화"""
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

    def calculate_implementation_shortfall(self, execution_result: Dict, decision_price: Decimal, order: Order) -> Dict:
        """구현 부족분 계산"""
        avg_price = execution_result['avg_price']
        commission = execution_result['total_cost']

        # Market impact (difference from decision price)
        market_impact = abs(avg_price - decision_price) / decision_price

        # Timing cost (assumed to be 0 for immediate execution)
        timing_cost = Decimal('0')

        # Commission cost
        commission_cost = commission / (execution_result['total_filled'] * avg_price)

        total_shortfall = market_impact + timing_cost + commission_cost

        return {
            'market_impact': float(market_impact),
            'timing_cost': float(timing_cost),
            'commission_cost': float(commission_cost),
            'total_shortfall': float(total_shortfall)
        }

    def calculate_performance_metrics(self, execution_result: Dict, benchmarks: List[str]) -> Dict:
        """성능 지표 계산"""
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

        return performance

    def generate_execution_analytics(self, execution_history: List[Dict]) -> Dict:
        """실행 분석 생성"""
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

    # Validation methods
    def validate_participation_rate_params(self, params: Dict):
        """참여율 매개변수 검증"""
        rate = params.get('participation_rate', 0.2)
        if not 0 < rate <= 1.0:
            raise ValueError("Participation rate must be between 0 and 1")

    def validate_slice_size_params(self, params: Dict):
        """슬라이스 크기 매개변수 검증"""
        max_size = params.get('max_slice_size', 1.0)
        if max_size <= 0:
            raise ValueError("Slice size must be positive")

    def validate_timing_params(self, params: Dict):
        """타이밍 매개변수 검증"""
        interval = params.get('min_slice_interval', 1.0)
        if interval <= 0:
            raise ValueError("Slice interval must be positive")

    # Utility methods
    def _aggregate_results(self, result: Dict):
        """슬라이스 결과 집계"""
        total_filled = Decimal('0')
        total_value = Decimal('0')
        total_cost = Decimal('0')

        for slice_data in result['slices']:
            filled = slice_data.get('filled_qty', Decimal('0'))
            price = slice_data.get('avg_price', Decimal('0'))
            cost = slice_data.get('commission', Decimal('0'))

            total_filled += filled
            total_value += filled * price
            total_cost += cost

        result['total_filled'] = total_filled
        result['avg_price'] = total_value / total_filled if total_filled > 0 else Decimal('0')
        result['total_cost'] = total_cost

    async def get_updated_market_conditions(self, symbol: str) -> Dict:
        """시장 조건 업데이트 (모의)"""
        # Mock implementation - in real system would fetch live data
        await asyncio.sleep(0.01)  # Simulate network delay
        return {
            'spread_bps': np.random.uniform(1.5, 4.0),
            'liquidity_score': np.random.uniform(0.3, 0.9),
            'avg_volume_1min': np.random.uniform(3000, 8000),
            'best_bid': Decimal('50000.0'),
            'best_ask': Decimal('50001.0'),
            'top_5_liquidity': np.random.uniform(15000, 35000),
            'volatility': np.random.uniform(0.015, 0.035),
            'imbalance': np.random.uniform(-0.3, 0.3),
            'vwap_bid': 49999.5,
            'vwap_ask': 50001.5
        }

    async def place_order(self, **kwargs) -> Dict:
        """주문 실행 (모의)"""
        # Mock implementation
        await asyncio.sleep(0.01)  # Simulate execution delay
        size = kwargs.get('size', Decimal('1.0'))
        price = kwargs.get('price', Decimal('50000.0'))

        return {
            'filled_qty': size,
            'avg_price': price,
            'commission': float(size) * float(price) * 0.0004,
            'status': 'FILLED',
            'order_id': f'mock-{np.random.randint(1000, 9999)}'
        }