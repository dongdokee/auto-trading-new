# 코인 선물 자동매매 시스템 - 주문 집행 엔진

## 6.1 스마트 주문 라우팅

```python
@dataclass
class Order:
    """주문 데이터 클래스"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    size: float
    urgency: str = 'MEDIUM'  # 'LOW', 'MEDIUM', 'HIGH', 'IMMEDIATE'
    price: Optional[float] = None

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
        order_size_pct = order.size / market_analysis['avg_volume_1min']
        
        # 긴급도별 전략
        if order.urgency == 'IMMEDIATE':
            return 'AGGRESSIVE'
        
        # 소액 주문
        if order_size_pct < 0.1:
            if spread_bps > 5 and order.urgency == 'LOW':
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
            'total_filled': 0,
            'avg_price': 0,
            'total_cost': 0
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
            'total_filled': 0,
            'avg_price': 0,
            'total_cost': 0
        }
        
        # Post-Only 주문
        best_bid = market_analysis['best_bid']
        best_ask = market_analysis['best_ask']
        
        if order.side == 'BUY':
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
                    order.symbol, order.side, remaining, 'HIGH'
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
            'total_filled': 0,
            'avg_price': 0,
            'total_cost': 0,
            'duration': optimal_duration
        }
        
        for i in range(n_slices):
            slice_order = Order(
                order.symbol, order.side, slice_size, 'MEDIUM'
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
            'total_filled': 0,
            'avg_price': 0,
            'total_cost': 0
        }
        
        remaining_size = order.size
        
        while remaining_size > 0:
            # 현재 시장 상태 기반 슬라이스 크기
            current_liquidity = market_analysis['top_5_liquidity']
            slice_size = min(
                remaining_size,
                current_liquidity * 0.2
            )
            
            # 스프레드 기반 전략 선택
            if market_analysis['spread_bps'] < 3:
                execution_method = self.execute_aggressive
            else:
                execution_method = self.execute_passive
            
            # 슬라이스 실행
            slice_order = Order(
                order.symbol, order.side, slice_size, 'MEDIUM'
            )
            slice_result = await execution_method(slice_order, market_analysis)
            
            result['slices'].append(slice_result)
            filled = slice_result.get('total_filled', 0)
            remaining_size -= filled
            
            # 피드백 기반 조정
            if filled < slice_size * 0.5:
                await asyncio.sleep(np.random.uniform(1, 3))
            
            market_analysis = await self.analyze_market_conditions(order.symbol)
        
        self._aggregate_results(result)
        return result
    
    def _calculate_optimal_duration(self, size: float, 
                                   market_analysis: Dict) -> float:
        """Almgren-Chriss 모델 기반 최적 집행 시간"""
        
        daily_volume = market_analysis['daily_volume']
        volatility = market_analysis['volatility']
        
        # 간소화된 Almgren-Chriss
        risk_aversion = 1.0
        temp_impact = 0.1
        
        optimal_time = np.sqrt(
            size * volatility / (risk_aversion * daily_volume * temp_impact)
        )
        
        # 실용적 범위로 제한 (30초 ~ 30분)
        return np.clip(optimal_time * 3600, 30, 1800)
    
    def _aggregate_results(self, result: Dict):
        """슬라이스 결과 집계"""
        result['total_filled'] = sum(
            s.get('total_filled', s.get('filled_qty', 0)) 
            for s in result['slices']
        )
        
        if result['total_filled'] > 0:
            total_value = sum(
                s.get('total_filled', s.get('filled_qty', 0)) * 
                s.get('avg_price', 0) 
                for s in result['slices']
            )
            result['avg_price'] = total_value / result['total_filled']
            result['total_cost'] = sum(
                s.get('total_cost', s.get('commission', 0)) 
                for s in result['slices']
            )
    
    async def analyze_market_conditions(self, symbol: str) -> Dict:
        """시장 상태 분석"""
        # TODO: 실제 구현 필요
        return {
            'spread_bps': 2.5,
            'liquidity_score': 0.8,
            'avg_volume_1min': 10000,
            'best_bid': 50000,
            'best_ask': 50001,
            'top_5_liquidity': 50000,
            'daily_volume': 10000000,
            'volatility': 0.02
        }
    
    async def place_order(self, **kwargs) -> Dict:
        """실제 주문 실행"""
        # TODO: 실제 구현 필요
        return {
            'filled_qty': kwargs['size'],
            'avg_price': kwargs.get('price', 50000),
            'commission': kwargs['size'] * kwargs.get('price', 50000) * 0.0004,
            'status': 'FILLED'
        }
    
    def _get_aggressive_price(self, order: Order, market_analysis: Dict) -> float:
        """공격적 주문의 가격 결정"""
        if order.side == 'BUY':
            return market_analysis['best_ask'] * 1.001
        else:
            return market_analysis['best_bid'] * 0.999

## 6.2 주문 관리

class OrderManager:
    """주문 생명주기 관리"""
    
    def __init__(self):
        self.active_orders = {}
        self.order_history = []
        self.max_order_age = 300  # 5 minutes
        
    async def submit_order(self, order: Order) -> str:
        """주문 제출"""
        
        order_id = self._generate_order_id()
        
        order_info = {
            'id': order_id,
            'order': order,
            'status': 'PENDING',
            'submitted_at': pd.Timestamp.now(),
            'filled_qty': 0,
            'avg_price': 0,
            'attempts': 0
        }
        
        self.active_orders[order_id] = order_info
        
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        
        if order_id not in self.active_orders:
            return False
        
        order_info = self.active_orders[order_id]
        
        if order_info['status'] in ['FILLED', 'CANCELLED']:
            return False
        
        # API 호출로 실제 취소
        # TODO: 실제 구현
        
        order_info['status'] = 'CANCELLED'
        order_info['cancelled_at'] = pd.Timestamp.now()
        
        # 기록 이동
        self.order_history.append(order_info)
        del self.active_orders[order_id]
        
        return True
    
    async def update_order_status(self, order_id: str, 
                                 filled_qty: float, 
                                 avg_price: float):
        """주문 상태 업데이트"""
        
        if order_id not in self.active_orders:
            return
        
        order_info = self.active_orders[order_id]
        order_info['filled_qty'] = filled_qty
        order_info['avg_price'] = avg_price
        
        # 완전 체결 체크
        if filled_qty >= order_info['order'].size * 0.999:  # 99.9%
            order_info['status'] = 'FILLED'
            order_info['filled_at'] = pd.Timestamp.now()
            
            # 기록 이동
            self.order_history.append(order_info)
            del self.active_orders[order_id]
        else:
            order_info['status'] = 'PARTIALLY_FILLED'
    
    async def check_stale_orders(self):
        """오래된 주문 처리"""
        
        current_time = pd.Timestamp.now()
        stale_orders = []
        
        for order_id, order_info in self.active_orders.items():
            age = (current_time - order_info['submitted_at']).total_seconds()
            
            if age > self.max_order_age:
                stale_orders.append(order_id)
        
        for order_id in stale_orders:
            await self.cancel_order(order_id)
    
    def _generate_order_id(self) -> str:
        """주문 ID 생성"""
        import uuid
        return str(uuid.uuid4())

## 6.3 슬리피지 컨트롤

class SlippageController:
    """슬리피지 모니터링 및 제어"""
    
    def __init__(self):
        self.slippage_history = []
        self.max_acceptable_slippage = 0.001  # 0.1%
        self.adaptive_threshold = True
        
    def calculate_slippage(self, expected_price: float, 
                          actual_price: float, 
                          side: str) -> float:
        """슬리피지 계산"""
        
        if side == 'BUY':
            slippage = (actual_price - expected_price) / expected_price
        else:  # SELL
            slippage = (expected_price - actual_price) / expected_price
        
        return slippage
    
    def record_execution(self, order: Order, execution: Dict):
        """실행 기록"""
        
        slippage = self.calculate_slippage(
            order.price or execution['expected_price'],
            execution['avg_price'],
            order.side
        )
        
        record = {
            'timestamp': pd.Timestamp.now(),
            'symbol': order.symbol,
            'size': order.size,
            'side': order.side,
            'slippage': slippage,
            'market_conditions': execution.get('market_conditions', {})
        }
        
        self.slippage_history.append(record)
        
        # 적응형 임계값 업데이트
        if self.adaptive_threshold:
            self._update_threshold()
    
    def _update_threshold(self):
        """동적 임계값 업데이트"""
        
        if len(self.slippage_history) < 100:
            return
        
        recent_slippages = [r['slippage'] for r in self.slippage_history[-100:]]
        
        # 95th percentile을 새 임계값으로
        new_threshold = np.percentile(np.abs(recent_slippages), 95)
        
        # 완만한 조정
        self.max_acceptable_slippage = (
            0.7 * self.max_acceptable_slippage + 
            0.3 * new_threshold
        )
    
    def predict_slippage(self, order: Order, 
                        market_conditions: Dict) -> float:
        """슬리피지 예측"""
        
        # 간단한 모델: 크기와 변동성 기반
        size_impact = order.size / market_conditions['liquidity']
        volatility_impact = market_conditions['volatility']
        spread_impact = market_conditions['spread_bps'] / 10000
        
        predicted_slippage = (
            size_impact * 0.5 + 
            volatility_impact * 0.3 + 
            spread_impact * 0.2
        )
        
        return predicted_slippage
    
    def should_execute(self, order: Order, 
                      market_conditions: Dict) -> bool:
        """실행 여부 결정"""
        
        predicted_slippage = self.predict_slippage(order, market_conditions)
        
        return predicted_slippage <= self.max_acceptable_slippage
```