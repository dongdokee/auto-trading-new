# src/execution/order_router.py
import asyncio
import numpy as np
from decimal import Decimal
from typing import Dict, Callable
from src.execution.models import Order, OrderSide, OrderUrgency


class SmartOrderRouter:
    """최적 집행 전략 선택 및 실행"""

    def __init__(self):
        self.execution_strategies = {
            'AGGRESSIVE': self.execute_aggressive,
            'PASSIVE': self.execute_passive,
            'TWAP': self.execute_twap,
            'ADAPTIVE': self.execute_adaptive
        }

    async def route_order(self, order: Order) -> Dict:
        """
        주문을 최적 집행 전략으로 라우팅

        Args:
            order: Order object

        Returns:
            ExecutionResult
        """

        # 시장 상태 분석
        market_analysis = await self.analyze_market_conditions(order.symbol)

        # 집행 전략 선택
        strategy = self._select_execution_strategy(order, market_analysis)

        # 전략 실행
        execution_func = self.execution_strategies[strategy]
        return await execution_func(order, market_analysis)

    def _select_execution_strategy(self, order: Order,
                                  market_analysis: Dict) -> str:
        """시장 상태와 주문 특성에 따른 전략 선택"""

        spread_bps = market_analysis['spread_bps']
        liquidity_score = market_analysis['liquidity_score']
        order_size_pct = float(order.size) / market_analysis['avg_volume_1min']

        # 긴급도별 전략
        if order.urgency.value == 'IMMEDIATE':
            return 'AGGRESSIVE'

        # 소액 주문
        if order_size_pct < 0.1:
            if spread_bps > 5 and order.urgency.value == 'LOW':
                return 'PASSIVE'
            else:
                return 'AGGRESSIVE'

        # 대액 주문
        if order_size_pct > 0.5:
            if liquidity_score > 0.7:
                return 'TWAP'
            else:
                return 'ADAPTIVE'

        # 중간 크기
        return 'ADAPTIVE'

    async def execute_aggressive(self, order: Order,
                                market_analysis: Dict) -> Dict:
        """즉시 체결 전략 (Market/IOC)"""

        result = {
            'strategy': 'AGGRESSIVE',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0')
        }

        # IOC 주문으로 즉시 체결 시도
        response = await self.place_order(
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            order_type='IOC',
            price=self._get_aggressive_price(order, market_analysis)
        )

        result['slices'].append(response)
        result['total_filled'] = response['filled_qty']
        result['avg_price'] = response['avg_price']
        result['total_cost'] = response['commission']

        return result

    async def execute_passive(self, order: Order,
                             market_analysis: Dict) -> Dict:
        """수수료 절감 전략 (Post-Only)"""

        result = {
            'strategy': 'PASSIVE',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0')
        }

        # Post-Only 주문
        best_bid = market_analysis['best_bid']
        best_ask = market_analysis['best_ask']

        if order.side == OrderSide.BUY:
            limit_price = best_bid
        else:
            limit_price = best_ask

        response = await self.place_order(
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            order_type='POST_ONLY',
            price=limit_price
        )

        result['slices'].append(response)

        # 일정 시간 대기 후 미체결분 처리
        if response['status'] == 'PARTIALLY_FILLED':
            await asyncio.sleep(5)

            remaining = order.size - response['filled_qty']
            if remaining > 0:
                remaining_order = Order(
                    symbol=order.symbol,
                    side=order.side,
                    size=remaining,
                    urgency=OrderUrgency.HIGH
                )
                ioc_response = await self.execute_aggressive(
                    remaining_order, market_analysis
                )
                result['slices'].extend(ioc_response['slices'])

        # 결과 집계
        self._aggregate_results(result)
        return result

    async def execute_twap(self, order: Order,
                          market_analysis: Dict) -> Dict:
        """시간 가중 평균 가격 전략"""

        # 최적 집행 시간 계산
        optimal_duration = self._calculate_optimal_duration(
            order.size, market_analysis
        )

        # 슬라이스 수와 간격
        n_slices = max(5, int(optimal_duration / 60))
        slice_size = order.size / n_slices
        slice_interval = optimal_duration / n_slices

        result = {
            'strategy': 'TWAP',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0'),
            'duration': optimal_duration
        }

        for i in range(n_slices):
            slice_order = Order(
                symbol=order.symbol,
                side=order.side,
                size=slice_size,
                urgency=OrderUrgency.MEDIUM
            )
            slice_result = await self.execute_aggressive(
                slice_order, market_analysis
            )

            result['slices'].append(slice_result)

            if i < n_slices - 1:
                await asyncio.sleep(slice_interval)
                market_analysis = await self.analyze_market_conditions(order.symbol)

        self._aggregate_results(result)
        return result

    async def execute_adaptive(self, order: Order,
                              market_analysis: Dict) -> Dict:
        """시장 상태에 적응하는 동적 집행"""

        result = {
            'strategy': 'ADAPTIVE',
            'slices': [],
            'total_filled': Decimal('0'),
            'avg_price': Decimal('0'),
            'total_cost': Decimal('0')
        }

        remaining_size = order.size

        while remaining_size > 0:
            # 현재 시장 상태 기반 슬라이스 크기
            current_liquidity = market_analysis['top_5_liquidity']
            slice_size = min(
                remaining_size,
                Decimal(str(current_liquidity)) * Decimal('0.2')
            )

            # 스프레드 기반 전략 선택
            if market_analysis['spread_bps'] < 3:
                execution_method = self.execute_aggressive
            else:
                execution_method = self.execute_passive

            # 슬라이스 실행
            slice_order = Order(
                symbol=order.symbol,
                side=order.side,
                size=slice_size,
                urgency=OrderUrgency.MEDIUM
            )
            slice_result = await execution_method(slice_order, market_analysis)

            result['slices'].append(slice_result)
            filled = slice_result.get('total_filled', Decimal('0'))
            remaining_size -= filled

            # 피드백 기반 조정
            if filled < slice_size * Decimal('0.5'):
                await asyncio.sleep(np.random.uniform(1, 3))

            market_analysis = await self.analyze_market_conditions(order.symbol)

        self._aggregate_results(result)
        return result

    def _calculate_optimal_duration(self, size: Decimal,
                                   market_analysis: Dict) -> float:
        """Almgren-Chriss 모델 기반 최적 집행 시간"""

        daily_volume = market_analysis.get('daily_volume', 10000000)
        volatility = market_analysis.get('volatility', 0.02)

        # 간소화된 Almgren-Chriss
        risk_aversion = 1.0
        temp_impact = 0.1

        optimal_time = np.sqrt(
            float(size) * volatility / (risk_aversion * daily_volume * temp_impact)
        )

        # 실용적 범위로 제한 (30초 ~ 30분)
        return np.clip(optimal_time * 3600, 30, 1800)

    def _aggregate_results(self, result: Dict):
        """슬라이스 결과 집계"""
        total_filled = Decimal('0')
        total_value = Decimal('0')
        total_cost = Decimal('0')

        for slice_data in result['slices']:
            if isinstance(slice_data, dict):
                # Single slice
                filled = slice_data.get('total_filled', slice_data.get('filled_qty', Decimal('0')))
                price = slice_data.get('avg_price', Decimal('0'))
                cost = slice_data.get('total_cost', slice_data.get('commission', Decimal('0')))
            else:
                # Nested result structure
                filled = slice_data.get('total_filled', Decimal('0'))
                price = slice_data.get('avg_price', Decimal('0'))
                cost = slice_data.get('total_cost', Decimal('0'))

            total_filled += filled
            total_value += filled * price
            total_cost += cost

        result['total_filled'] = total_filled

        if total_filled > 0:
            result['avg_price'] = total_value / total_filled
        else:
            result['avg_price'] = Decimal('0')

        result['total_cost'] = total_cost

    async def analyze_market_conditions(self, symbol: str) -> Dict:
        """시장 상태 분석 (TODO: 실제 구현 필요)"""
        return {
            'spread_bps': 2.5,
            'liquidity_score': 0.8,
            'avg_volume_1min': 10000,
            'best_bid': Decimal('50000'),
            'best_ask': Decimal('50001'),
            'top_5_liquidity': 50000,
            'daily_volume': 10000000,
            'volatility': 0.02,
            'imbalance': 0.1
        }

    async def place_order(self, **kwargs) -> Dict:
        """실제 주문 실행 (TODO: 실제 구현 필요)"""
        return {
            'filled_qty': kwargs['size'],
            'avg_price': kwargs.get('price', Decimal('50000')),
            'commission': float(kwargs['size']) * float(kwargs.get('price', 50000)) * 0.0004,
            'status': 'FILLED',
            'order_id': 'mock-order-123'
        }

    def _get_aggressive_price(self, order: Order, market_analysis: Dict) -> Decimal:
        """공격적 주문의 가격 결정"""
        if order.side == OrderSide.BUY:
            return market_analysis['best_ask'] * Decimal('1.001')
        else:
            return market_analysis['best_bid'] * Decimal('0.999')