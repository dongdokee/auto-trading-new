# src/execution/market_analyzer.py
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any
import re


class MarketConditionAnalyzer:
    """실시간 주문북 분석 및 유동성 평가"""

    def analyze_orderbook(self, orderbook_snapshot: Dict) -> Dict:
        """
        주문북 마이크로구조 분석

        Args:
            orderbook_snapshot: 주문북 스냅샷 데이터

        Returns:
            dict: 분석 결과
        """
        self._validate_orderbook(orderbook_snapshot)

        bids = orderbook_snapshot.get('bids', [])
        asks = orderbook_snapshot.get('asks', [])

        analysis = {}

        # 1. 스프레드 분석
        self._analyze_spread(bids, asks, analysis)

        # 2. 주문북 불균형
        self._analyze_imbalance(bids, asks, analysis)

        # 3. 유동성 깊이
        analysis['liquidity_score'] = self._calculate_liquidity_score(bids, asks)

        # 4. 가격 충격 함수
        analysis['price_impact'] = self._estimate_price_impact(bids, asks)

        # 5. 실효 스프레드
        analysis['effective_spread'] = self._calculate_effective_spread(bids, asks)

        # 6. 주문북 형태 분석
        analysis['book_shape'] = self._analyze_book_shape(bids, asks)

        # 7. 큰 주문 감지
        analysis['large_orders'] = self._detect_large_orders(bids, asks)

        # 8. VWAP 계산
        analysis['vwap_bid'] = self._calculate_vwap(bids)
        analysis['vwap_ask'] = self._calculate_vwap(asks)

        # 9. 변동성 추정
        analysis['volatility_estimate'] = self._estimate_volatility(bids, asks)

        return analysis

    def _validate_orderbook(self, orderbook: Dict):
        """주문북 데이터 검증"""
        if 'symbol' not in orderbook:
            raise ValueError("Symbol is required")

        symbol = orderbook['symbol']
        if not re.match(r'^[A-Z]+$', symbol):
            raise ValueError("Invalid symbol format")

        if 'timestamp' not in orderbook:
            raise ValueError("Timestamp is required")

        # 주문 데이터 유효성 검증
        for side in ['bids', 'asks']:
            if side in orderbook:
                for order in orderbook[side]:
                    try:
                        Decimal(str(order['price']))
                        Decimal(str(order['size']))
                    except (ValueError, TypeError, KeyError, Exception):
                        raise ValueError("Invalid orderbook data")

    def _analyze_spread(self, bids: List, asks: List, analysis: Dict):
        """스프레드 분석"""
        # 개별적으로 최고 가격 설정
        analysis['best_bid'] = Decimal(str(bids[0]['price'])) if bids else None
        analysis['best_ask'] = Decimal(str(asks[0]['price'])) if asks else None

        if bids and asks:
            best_bid = analysis['best_bid']
            best_ask = analysis['best_ask']
            mid_price = (best_bid + best_ask) / 2

            spread = best_ask - best_bid
            # 정밀도 향상을 위해 Decimal 연산 사용
            spread_bps = float((spread / mid_price) * Decimal('10000'))

            analysis['mid_price'] = mid_price
            analysis['spread'] = spread
            analysis['spread_bps'] = spread_bps
        else:
            analysis['mid_price'] = None
            analysis['spread'] = None
            analysis['spread_bps'] = None

    def _analyze_imbalance(self, bids: List, asks: List, analysis: Dict):
        """주문북 불균형 분석"""
        bid_volume_5 = sum(Decimal(str(level['size'])) for level in bids[:5])
        ask_volume_5 = sum(Decimal(str(level['size'])) for level in asks[:5])

        total_volume = bid_volume_5 + ask_volume_5
        if total_volume > 0:
            imbalance = (bid_volume_5 - ask_volume_5) / total_volume
        else:
            imbalance = Decimal('0')

        analysis['imbalance'] = imbalance
        analysis['bid_volume_5'] = bid_volume_5
        analysis['ask_volume_5'] = ask_volume_5
        analysis['top_5_liquidity'] = total_volume

    def _calculate_liquidity_score(self, bids: List, asks: List) -> float:
        """유동성 점수 (0~1)"""
        if not bids and not asks:
            return 0.0

        # 상위 10호가 총 수량
        bid_liquidity = sum(Decimal(str(level['size'])) for level in bids[:10])
        ask_liquidity = sum(Decimal(str(level['size'])) for level in asks[:10])
        total_liquidity = float(bid_liquidity + ask_liquidity)

        # 호가 간 가격 차이 균일성
        price_uniformity = self._calculate_price_uniformity(bids)

        # 종합 점수
        liquidity_score = min(1.0, total_liquidity / 10000) * price_uniformity

        return float(liquidity_score)

    def _calculate_price_uniformity(self, orders: List) -> float:
        """가격 차이 균일성 계산"""
        if len(orders) < 2:
            return 0.0

        price_diffs = []
        for i in range(min(9, len(orders)-1)):
            diff = abs(float(orders[i]['price']) - float(orders[i+1]['price']))
            price_diffs.append(diff)

        if price_diffs:
            std_dev = np.std(price_diffs) if len(price_diffs) > 1 else 0
            return 1.0 / (1.0 + std_dev)
        else:
            return 0.0

    def _estimate_price_impact(self, bids: List, asks: List) -> Callable:
        """Square-root 시장 충격 모델"""

        total_liquidity = sum(
            Decimal(str(level['size'])) for level in (bids[:10] + asks[:10])
        )

        def impact_function(size: Decimal, side: str = 'BUY') -> float:
            """주문 크기에 따른 예상 가격 충격"""
            if total_liquidity == 0:
                return float(size) * 0.001  # 기본 충격

            # Square-root impact model
            impact = float(size) / float(total_liquidity)
            return np.sqrt(impact) * 0.001

        return impact_function

    def _calculate_effective_spread(self, bids: List, asks: List) -> float:
        """실효 스프레드 계산"""
        if not bids or not asks:
            return 0.0

        # 간단한 실효 스프레드 모델
        best_bid = float(bids[0]['price'])
        best_ask = float(asks[0]['price'])
        mid_price = (best_bid + best_ask) / 2

        return abs(best_ask - best_bid) / mid_price

    def _analyze_book_shape(self, bids: List, asks: List) -> Dict:
        """주문북 형태 분석"""
        if not bids and not asks:
            return {'depth_ratio': 0.0, 'concentration': 0.0}

        # 깊이 비율 (상위 5 vs 전체)
        total_volume = sum(
            Decimal(str(level['size'])) for level in (bids + asks)
        )
        top_5_volume = sum(
            Decimal(str(level['size'])) for level in (bids[:5] + asks[:5])
        )

        depth_ratio = float(top_5_volume / total_volume) if total_volume > 0 else 0.0

        # 집중도 (최고호가 vs 상위 5호가)
        if bids or asks:
            best_volume = max(
                float(bids[0]['size']) if bids else 0,
                float(asks[0]['size']) if asks else 0
            )
            concentration = best_volume / float(top_5_volume) if top_5_volume > 0 else 0.0
        else:
            concentration = 0.0

        return {
            'depth_ratio': depth_ratio,
            'concentration': concentration
        }

    def _detect_large_orders(self, bids: List, asks: List) -> Dict:
        """대량 주문 감지"""
        def detect_large_in_side(orders: List) -> List:
            if not orders:
                return []

            volumes = [float(order['size']) for order in orders]
            if len(volumes) < 3:
                return []

            avg_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            threshold = avg_volume + 2 * std_volume

            large_orders = []
            for i, order in enumerate(orders):
                if float(order['size']) > threshold:
                    large_orders.append({
                        'index': i,
                        'price': float(order['price']),
                        'size': float(order['size'])
                    })

            return large_orders

        return {
            'bid_side': detect_large_in_side(bids),
            'ask_side': detect_large_in_side(asks)
        }

    def _calculate_vwap(self, orders: List) -> float:
        """볼륨 가중 평균 가격 계산"""
        if not orders:
            return 0.0

        total_value = sum(
            float(order['price']) * float(order['size'])
            for order in orders[:5]
        )
        total_volume = sum(float(order['size']) for order in orders[:5])

        return total_value / total_volume if total_volume > 0 else 0.0

    def _estimate_volatility(self, bids: List, asks: List) -> float:
        """시장 변동성 추정"""
        if not bids or not asks:
            return 0.0

        # 간단한 변동성 모델: 스프레드 대비 중간가격
        best_bid = float(bids[0]['price'])
        best_ask = float(asks[0]['price'])
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        # 정규화된 변동성 추정
        volatility = (spread / mid_price) * 100 if mid_price > 0 else 0.0

        return volatility