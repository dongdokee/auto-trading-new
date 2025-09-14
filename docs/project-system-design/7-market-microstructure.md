# 코인 선물 자동매매 시스템 - 마켓 마이크로구조

## 7.1 주문북 분석

```python
class OrderBookAnalyzer:
    """실시간 주문북 분석 및 유동성 평가"""
    
    def analyze_orderbook(self, orderbook_snapshot: Dict) -> Dict:
        """
        주문북 마이크로구조 분석
        
        Returns:
            dict: 분석 결과
        """
        
        bids = orderbook_snapshot['bids']
        asks = orderbook_snapshot['asks']
        
        analysis = {}
        
        # 1. 스프레드 분석
        if bids and asks:
            best_bid = bids[0]['price']
            best_ask = asks[0]['price']
            mid_price = (best_bid + best_ask) / 2
            
            spread = best_ask - best_bid
            spread_bps = (spread / mid_price) * 10000
            
            analysis['best_bid'] = best_bid
            analysis['best_ask'] = best_ask
            analysis['mid_price'] = mid_price
            analysis['spread'] = spread
            analysis['spread_bps'] = spread_bps
        
        # 2. 주문북 불균형
        bid_volume_5 = sum(level['size'] for level in bids[:5])
        ask_volume_5 = sum(level['size'] for level in asks[:5])
        
        total_volume = bid_volume_5 + ask_volume_5
        if total_volume > 0:
            imbalance = (bid_volume_5 - ask_volume_5) / total_volume
        else:
            imbalance = 0
        
        analysis['imbalance'] = imbalance
        analysis['bid_volume_5'] = bid_volume_5
        analysis['ask_volume_5'] = ask_volume_5
        analysis['top_5_liquidity'] = total_volume
        
        # 3. 유동성 깊이
        analysis['liquidity_score'] = self._calculate_liquidity_score(bids, asks)
        
        # 4. 가격 충격 함수
        analysis['price_impact'] = self._estimate_price_impact(bids, asks)
        
        # 5. 실효 스프레드
        analysis['effective_spread'] = self._calculate_effective_spread()
        
        # 6. 주문북 형태 분석
        analysis['book_shape'] = self._analyze_book_shape(bids, asks)
        
        # 7. 큰 주문 감지
        analysis['large_orders'] = self._detect_large_orders(bids, asks)
        
        return analysis
    
    def _calculate_liquidity_score(self, bids: List, asks: List) -> float:
        """유동성 점수 (0~1)"""
        
        # 상위 10호가 총 수량
        bid_liquidity = sum(level['size'] for level in bids[:10])
        ask_liquidity = sum(level['size'] for level in asks[:10])
        total_liquidity = bid_liquidity + ask_liquidity
        
        # 호가 간 가격 차이 균일성
        bid_price_diffs = []
        for i in range(min(9, len(bids)-1)):
            diff = bids[i]['price'] - bids[i+1]['price']
            bid_price_diffs.append(diff)
        
        if bid_price_diffs:
            price_uniformity = 1 / (1 + np.std(bid_price_diffs))
        else:
            price_uniformity = 0
        
        # 종합 점수
        liquidity_score = min(1.0, total_liquidity / 10000) * price_uniformity
        
        return float(liquidity_score)
    
    def _estimate_price_impact(self, bids: List, asks: List) -> Callable:
        """Square-root 시장 충격 모델"""
        
        def impact_function(size: float, side: str = 'BUY') -> float:
            """
            주문 크기에 따른 예상 가격 충격
            
            Args:
                size: 주문 크기
                side: BUY or SELL
                
            Returns:
                float: 예상 충격 (%)
            """
            
            levels = asks if side == 'BUY' else bids
            
            cumulative_size = 0
            weighted_price = 0
            
            for level in levels:
                level_size = level['size']
                level_price = level['price']
                
                if cumulative_size + level_size >= size:
                    remaining = size - cumulative_size
                    weighted_price += remaining * level_price
                    cumulative_size = size
                    break
                else:
                    weighted_price += level_size * level_price
                    cumulative_size += level_size
            
            if cumulative_size >= size and cumulative_size > 0:
                avg_price = weighted_price / size
                mid_price = (bids[0]['price'] + asks[0]['price']) / 2
                impact = abs(avg_price - mid_price) / mid_price
                return float(impact)
            else:
                return 0.05  # 5% 페널티
        
        return impact_function
    
    def _calculate_effective_spread(self) -> Optional[float]:
        """실제 체결 데이터 기반 실효 스프레드"""
        
        trades = self.get_recent_trades(minutes=5)
        if not trades:
            return None
        
        effective_spreads = []
        
        for trade in trades:
            mid_at_trade = trade['mid_price_at_execution']
            side_sign = 1 if trade.get('side') == 'BUY' else -1
            
            eff_spread = 2.0 * side_sign * (trade['price'] - mid_at_trade) / mid_at_trade
            effective_spreads.append(eff_spread)
        
        return float(np.mean(effective_spreads) * 10000)
    
    def _analyze_book_shape(self, bids: List, asks: List) -> Dict:
        """주문북 형태 분석"""
        
        shape = {}
        
        # 기울기 계산
        if len(bids) >= 5:
            bid_prices = [b['price'] for b in bids[:5]]
            bid_volumes = [b['size'] for b in bids[:5]]
            bid_slope = np.polyfit(bid_prices, bid_volumes, 1)[0]
            shape['bid_slope'] = bid_slope
        
        if len(asks) >= 5:
            ask_prices = [a['price'] for a in asks[:5]]
            ask_volumes = [a['size'] for a in asks[:5]]
            ask_slope = np.polyfit(ask_prices, ask_volumes, 1)[0]
            shape['ask_slope'] = ask_slope
        
        # 형태 분류
        if 'bid_slope' in shape and 'ask_slope' in shape:
            if abs(bid_slope) < 0.1 and abs(ask_slope) < 0.1:
                shape['type'] = 'FLAT'
            elif bid_slope > ask_slope:
                shape['type'] = 'BID_HEAVY'
            else:
                shape['type'] = 'ASK_HEAVY'
        
        return shape
    
    def _detect_large_orders(self, bids: List, asks: List) -> List[Dict]:
        """큰 주문 감지"""
        
        large_orders = []
        
        # 평균 크기 계산
        all_sizes = [b['size'] for b in bids[:20]] + [a['size'] for a in asks[:20]]
        if all_sizes:
            avg_size = np.mean(all_sizes)
            threshold = avg_size * 3  # 평균의 3배
            
            # 큰 매수 주문
            for i, bid in enumerate(bids[:10]):
                if bid['size'] > threshold:
                    large_orders.append({
                        'side': 'BID',
                        'level': i,
                        'price': bid['price'],
                        'size': bid['size'],
                        'size_ratio': bid['size'] / avg_size
                    })
            
            # 큰 매도 주문
            for i, ask in enumerate(asks[:10]):
                if ask['size'] > threshold:
                    large_orders.append({
                        'side': 'ASK',
                        'level': i,
                        'price': ask['price'],
                        'size': ask['size'],
                        'size_ratio': ask['size'] / avg_size
                    })
        
        return large_orders
    
    def get_recent_trades(self, minutes: int = 5) -> List[Dict]:
        """최근 체결 데이터 조회"""
        # TODO: 실제 구현 필요
        return []

## 7.2 시장 충격 모델

class MarketImpactModel:
    """동적 캘리브레이션 기반 시장 충격 모델"""
    
    def __init__(self):
        self.temp_impact_coef = 0.1
        self.perm_impact_coef = 0.05
        self.last_calibration = None
        self.calibration_interval = 86400
        self.temp_impact_model = None
        self.perm_impact_model = None
        
    def calibrate_from_trades(self, execution_history: List[Dict]):
        """실제 체결 데이터로 충격 계수 캘리브레이션"""
        
        if not execution_history:
            return
        
        X, y_temp, y_perm = [], [], []
        
        for trade in execution_history:
            # 특징 추출
            features = [
                trade['size'] / max(1e-10, trade['avg_daily_volume']),
                trade['volatility'],
                trade['spread_bps'] / 100.0,
                trade.get('execution_speed', 1.0)
            ]
            X.append(features)
            
            # 충격 계산
            temp_impact = abs(trade['exec_price'] - trade['mid_before']) / trade['mid_before']
            perm_impact = abs(trade['mid_after_5min'] - trade['mid_before']) / trade['mid_before']
            
            y_temp.append(temp_impact)
            y_perm.append(perm_impact)
        
        # 배열 변환
        X = np.array(X)
        y_temp = np.array(y_temp)
        y_perm = np.array(y_perm)
        
        # 비선형 회귀
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Ridge 회귀 (과적합 방지)
        self.temp_impact_model = (poly, Ridge(alpha=0.1).fit(X_poly, y_temp))
        self.perm_impact_model = (poly, Ridge(alpha=0.1).fit(X_poly, y_perm))
        self.last_calibration = time.time()
    
    def estimate_impact(self, order_size: float, market_state: Dict) -> Dict:
        """주문의 예상 시장 충격"""
        
        # 캘리브레이션 체크
        if (self.last_calibration is None or 
            time.time() - self.last_calibration > self.calibration_interval):
            self.calibrate_from_trades(self.get_recent_executions())
        
        # 특징 구성
        features = np.array([[
            order_size / max(1e-10, market_state['daily_volume']),
            market_state['volatility'],
            market_state['spread_bps'] / 100.0,
            1.0
        ]])
        
        # 예측
        if self.temp_impact_model is not None:
            poly, model = self.temp_impact_model
            temp = float(model.predict(poly.transform(features))[0])
        else:
            # 폴백: Square-root 모델
            temp = self.temp_impact_coef * np.sqrt(
                order_size / max(1e-10, market_state['daily_volume'])
            )
        
        if self.perm_impact_model is not None:
            poly, model = self.perm_impact_model
            perm = float(model.predict(poly.transform(features))[0])
        else:
            perm = self.perm_impact_coef * (
                order_size / max(1e-10, market_state['daily_volume'])
            )
        
        return {
            'temporary': float(temp),
            'permanent': float(perm),
            'total': float(temp + perm),
            'breakdown': self._impact_breakdown(temp, perm, market_state)
        }
    
    def _impact_breakdown(self, temp: float, perm: float, 
                         market_state: Dict) -> Dict:
        """충격 요소별 분해"""
        
        return {
            'spread_component': market_state['spread_bps'] / 20000,
            'size_component': temp * 0.6,
            'volatility_component': temp * 0.3,
            'timing_component': temp * 0.1,
            'permanent_drift': perm
        }
    
    def get_recent_executions(self) -> List[Dict]:
        """최근 체결 이력 조회"""
        # TODO: 실제 DB 조회 구현
        return []

## 7.3 유동성 프로파일링

class LiquidityProfiler:
    """시간대별 유동성 프로파일링"""
    
    def __init__(self):
        self.liquidity_history = {}
        self.profile_window = 30  # days
        
    def update_profile(self, symbol: str, timestamp: pd.Timestamp, 
                      liquidity_metrics: Dict):
        """유동성 프로파일 업데이트"""
        
        if symbol not in self.liquidity_history:
            self.liquidity_history[symbol] = []
        
        record = {
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'day_of_week': timestamp.dayofweek,
            'spread': liquidity_metrics['spread_bps'],
            'depth': liquidity_metrics['top_5_liquidity'],
            'imbalance': liquidity_metrics['imbalance']
        }
        
        self.liquidity_history[symbol].append(record)
        
        # 오래된 데이터 제거
        cutoff = timestamp - pd.Timedelta(days=self.profile_window)
        self.liquidity_history[symbol] = [
            r for r in self.liquidity_history[symbol]
            if r['timestamp'] > cutoff
        ]
    
    def get_expected_liquidity(self, symbol: str, 
                              target_time: pd.Timestamp) -> Dict:
        """특정 시간의 예상 유동성"""
        
        if symbol not in self.liquidity_history:
            return self._default_liquidity()
        
        history = self.liquidity_history[symbol]
        target_hour = target_time.hour
        target_dow = target_time.dayofweek
        
        # 유사 시간대 데이터 필터링
        similar_times = [
            r for r in history
            if abs(r['hour'] - target_hour) <= 1 and r['day_of_week'] == target_dow
        ]
        
        if not similar_times:
            similar_times = [
                r for r in history
                if abs(r['hour'] - target_hour) <= 2
            ]
        
        if not similar_times:
            return self._default_liquidity()
        
        # 평균 계산
        avg_spread = np.mean([r['spread'] for r in similar_times])
        avg_depth = np.mean([r['depth'] for r in similar_times])
        std_depth = np.std([r['depth'] for r in similar_times])
        
        return {
            'expected_spread': avg_spread,
            'expected_depth': avg_depth,
            'depth_std': std_depth,
            'confidence': min(1.0, len(similar_times) / 10),
            'sample_size': len(similar_times)
        }
    
    def find_optimal_execution_window(self, symbol: str, 
                                     order_size: float) -> List[Dict]:
        """최적 실행 시간대 찾기"""
        
        if symbol not in self.liquidity_history:
            return []
        
        # 시간대별 평균 계산
        hourly_stats = {}
        
        for record in self.liquidity_history[symbol]:
            hour = record['hour']
            if hour not in hourly_stats:
                hourly_stats[hour] = {
                    'spreads': [],
                    'depths': []
                }
            hourly_stats[hour]['spreads'].append(record['spread'])
            hourly_stats[hour]['depths'].append(record['depth'])
        
        # 각 시간대 평가
        windows = []
        
        for hour, stats in hourly_stats.items():
            if len(stats['spreads']) < 5:
                continue
            
            avg_spread = np.mean(stats['spreads'])
            avg_depth = np.mean(stats['depths'])
            
            # 실행 비용 추정
            exec_cost = avg_spread / 10000 + order_size / (avg_depth + 1e-10) * 0.001
            
            windows.append({
                'hour': hour,
                'avg_spread': avg_spread,
                'avg_depth': avg_depth,
                'estimated_cost': exec_cost,
                'samples': len(stats['spreads'])
            })
        
        # 비용 기준 정렬
        windows.sort(key=lambda x: x['estimated_cost'])
        
        return windows[:5]  # Top 5 windows
    
    def _default_liquidity(self) -> Dict:
        """기본 유동성 값"""
        return {
            'expected_spread': 5.0,  # 5 bps
            'expected_depth': 10000,
            'depth_std': 5000,
            'confidence': 0,
            'sample_size': 0
        }

## 7.4 틱 데이터 분석

class TickDataAnalyzer:
    """틱 수준 시장 데이터 분석"""
    
    def __init__(self):
        self.tick_buffer = []
        self.buffer_size = 1000
        self.trade_flow_imbalance = 0
        
    def process_tick(self, tick: Dict):
        """틱 데이터 처리"""
        
        # 버퍼에 추가
        self.tick_buffer.append(tick)
        
        # 크기 제한
        if len(self.tick_buffer) > self.buffer_size:
            self.tick_buffer.pop(0)
        
        # Trade flow 업데이트
        if tick['type'] == 'TRADE':
            self._update_trade_flow(tick)
    
    def _update_trade_flow(self, trade_tick: Dict):
        """거래 플로우 업데이트"""
        
        # Tick rule 적용
        if len(self.tick_buffer) >= 2:
            prev_price = self.tick_buffer[-2]['price']
            curr_price = trade_tick['price']
            
            if curr_price > prev_price:
                direction = 1  # Buy pressure
            elif curr_price < prev_price:
                direction = -1  # Sell pressure
            else:
                direction = 0  # Neutral
            
            # Exponential decay
            self.trade_flow_imbalance = (
                0.95 * self.trade_flow_imbalance + 
                0.05 * direction * trade_tick['size']
            )
    
    def calculate_vpin(self, window: int = 50) -> float:
        """Volume-synchronized Probability of Informed Trading"""
        
        if len(self.tick_buffer) < window:
            return 0.5
        
        recent_ticks = self.tick_buffer[-window:]
        
        buy_volume = sum(
            t['size'] for t in recent_ticks 
            if t.get('side') == 'BUY'
        )
        sell_volume = sum(
            t['size'] for t in recent_ticks 
            if t.get('side') == 'SELL'
        )
        
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            vpin = abs(buy_volume - sell_volume) / total_volume
        else:
            vpin = 0
        
        return float(vpin)
    
    def detect_microstructure_patterns(self) -> Dict:
        """마이크로구조 패턴 감지"""
        
        patterns = {
            'quote_stuffing': False,
            'layering': False,
            'momentum_ignition': False,
            'ping_pong': False
        }
        
        if len(self.tick_buffer) < 100:
            return patterns
        
        # Quote stuffing 감지
        quote_rate = self._calculate_quote_rate()
        if quote_rate > 100:  # 100 quotes/second
            patterns['quote_stuffing'] = True
        
        # Layering 감지
        if self._detect_layering():
            patterns['layering'] = True
        
        # Momentum ignition 감지
        if self._detect_momentum_ignition():
            patterns['momentum_ignition'] = True
        
        return patterns
    
    def _calculate_quote_rate(self) -> float:
        """초당 호가 변경률"""
        
        if len(self.tick_buffer) < 2:
            return 0
        
        time_span = (
            self.tick_buffer[-1]['timestamp'] - 
            self.tick_buffer[0]['timestamp']
        ).total_seconds()
        
        if time_span > 0:
            quote_count = sum(1 for t in self.tick_buffer if t['type'] == 'QUOTE')
            return quote_count / time_span
        
        return 0
    
    def _detect_layering(self) -> bool:
        """레이어링 패턴 감지"""
        
        # 한쪽에 집중된 주문 후 빠른 취소
        order_events = [t for t in self.tick_buffer if t['type'] in ['ORDER', 'CANCEL']]
        
        if len(order_events) < 10:
            return False
        
        # 짧은 시간 내 대량 주문 후 취소
        recent_orders = order_events[-10:]
        order_count = sum(1 for e in recent_orders if e['type'] == 'ORDER')
        cancel_count = sum(1 for e in recent_orders if e['type'] == 'CANCEL')
        
        return cancel_count > order_count * 0.7
    
    def _detect_momentum_ignition(self) -> bool:
        """모멘텀 점화 패턴 감지"""
        
        trades = [t for t in self.tick_buffer if t['type'] == 'TRADE']
        
        if len(trades) < 5:
            return False
        
        # 연속된 같은 방향 거래
        recent_trades = trades[-5:]
        sides = [t.get('side') for t in recent_trades]
        
        # 모두 같은 방향
        if len(set(sides)) == 1:
            # 가격 움직임 체크
            price_change = (
                recent_trades[-1]['price'] - recent_trades[0]['price']
            ) / recent_trades[0]['price']
            
            return abs(price_change) > 0.002  # 0.2% move
        
        return False
```