# 코인 선물 자동매매 시스템 - 데이터 품질 관리

## 데이터 품질 관리자

```python
class DataQualityManager:
    """데이터 이상치 탐지 및 정제"""
    
    def __init__(self):
        self.anomaly_threshold_z = 5
        self.spike_threshold_pct = 0.1
        self.staleness_threshold = 1000  # ms
        
    def validate_and_clean(self, data_point: Dict, 
                          historical_data: pd.DataFrame) -> Tuple:
        """
        데이터 검증 및 정제
        
        Returns:
            tuple: (cleaned_data, is_anomaly, anomaly_type)
        """
        
        # 1. 타임스탬프 검증
        current_time = time.time() * 1000
        data_age = current_time - data_point['timestamp']
        
        if data_age > self.staleness_threshold:
            return None, True, 'STALE_DATA'
        
        # 2. 가격 유효성
        price = data_point['price']
        
        if price <= 0:
            return None, True, 'INVALID_PRICE'
        
        # 3. 통계적 이상치 (Modified Z-score)
        if len(historical_data) >= 100:
            recent_prices = historical_data['price'].tail(100)
            median = recent_prices.median()
            mad = np.median(np.abs(recent_prices - median))
            
            if mad > 0:
                modified_z = 0.6745 * (price - median) / mad
                
                if abs(modified_z) > self.anomaly_threshold_z:
                    return data_point, True, 'STATISTICAL_OUTLIER'
        
        # 4. 급격한 변화 감지
        if len(historical_data) > 0:
            last_price = historical_data['price'].iloc[-1]
            price_change = abs(price - last_price) / last_price
            
            if price_change > self.spike_threshold_pct:
                return data_point, True, 'PRICE_SPIKE'
        
        # 5. 주문북 일관성
        if 'orderbook' in data_point:
            ob_valid, ob_issues = self._validate_orderbook(data_point['orderbook'])
            
            if not ob_valid:
                return data_point, True, f'ORDERBOOK_INVALID: {ob_issues}'
        
        # 6. 볼륨 이상치
        if 'volume' in data_point:
            vol_valid, vol_issue = self._validate_volume(
                data_point['volume'], 
                historical_data
            )
            if not vol_valid:
                return data_point, True, vol_issue
        
        return data_point, False, None
    
    def _validate_orderbook(self, orderbook: Dict) -> Tuple[bool, List[str]]:
        """주문북 유효성 검증"""
        
        issues = []
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        # 매수 호가 정렬 검증
        if bids:
            bid_prices = [b['price'] for b in bids]
            if not all(bid_prices[i] >= bid_prices[i+1] for i in range(len(bid_prices)-1)):
                issues.append('BID_ORDER')
        
        # 매도 호가 정렬 검증
        if asks:
            ask_prices = [a['price'] for a in asks]
            if not all(ask_prices[i] <= ask_prices[i+1] for i in range(len(ask_prices)-1)):
                issues.append('ASK_ORDER')
        
        # 스프레드 검증
        if bids and asks:
            if bids[0]['price'] >= asks[0]['price']:
                issues.append('NEGATIVE_SPREAD')
            
            spread_pct = (asks[0]['price'] - bids[0]['price']) / bids[0]['price']
            if spread_pct > 0.05:  # 5% 이상
                issues.append('WIDE_SPREAD')
        
        # 유동성 검증
        if bids and asks:
            bid_depth = sum(b['size'] for b in bids[:5])
            ask_depth = sum(a['size'] for a in asks[:5])
            
            if bid_depth == 0 or ask_depth == 0:
                issues.append('NO_LIQUIDITY')
            elif abs(bid_depth - ask_depth) / (bid_depth + ask_depth) > 0.9:
                issues.append('LIQUIDITY_IMBALANCE')
        
        return len(issues) == 0, issues
    
    def _validate_volume(self, volume: float, 
                        historical_data: pd.DataFrame) -> Tuple[bool, str]:
        """거래량 유효성 검증"""
        
        if volume < 0:
            return False, 'NEGATIVE_VOLUME'
        
        if len(historical_data) >= 20:
            recent_volumes = historical_data['volume'].tail(20)
            avg_volume = recent_volumes.mean()
            
            if avg_volume > 0:
                volume_ratio = volume / avg_volume
                
                if volume_ratio > 100:  # 평균의 100배
                    return False, 'VOLUME_SPIKE'
                elif volume_ratio < 0.01:  # 평균의 1%
                    return False, 'VOLUME_DROP'
        
        return True, None
    
    def apply_data_policy(self, data_point: Dict, 
                         anomaly_info: Tuple) -> Optional[Dict]:
        """이상치 처리 정책 적용"""
        
        cleaned_data, is_anomaly, anomaly_type = anomaly_info
        
        if not is_anomaly:
            return cleaned_data
        
        # 정책별 처리
        if anomaly_type == 'STALE_DATA':
            # 오래된 데이터 폐기
            return None
            
        elif anomaly_type == 'INVALID_PRICE':
            # 잘못된 가격 폐기
            return None
            
        elif anomaly_type == 'STATISTICAL_OUTLIER':
            # 통계적 이상치: 보수적 모드
            cleaned_data['conservative_mode'] = True
            return cleaned_data
            
        elif anomaly_type == 'PRICE_SPIKE':
            # 급격한 가격 변화: 거래 중단
            cleaned_data['halt_trading'] = True
            return cleaned_data
            
        elif 'ORDERBOOK_INVALID' in anomaly_type:
            # 주문북 문제: 유동성 체크
            cleaned_data['liquidity_check'] = True
            return cleaned_data
            
        elif anomaly_type in ['VOLUME_SPIKE', 'VOLUME_DROP']:
            # 거래량 이상: 경고만
            cleaned_data['volume_warning'] = True
            return cleaned_data
        
        # 기본: 보수적 처리
        return cleaned_data

## 데이터 집계 및 리샘플링

class DataAggregator:
    """시계열 데이터 집계"""
    
    def __init__(self):
        self.timeframes = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
    def aggregate_ohlcv(self, tick_data: pd.DataFrame, 
                       timeframe: str) -> pd.DataFrame:
        """틱 데이터를 OHLCV로 집계"""
        
        if timeframe not in self.timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        resample_rule = self.timeframes[timeframe]
        
        ohlcv = tick_data.resample(resample_rule).agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # 빈 구간 처리
        ohlcv = self._handle_missing_data(ohlcv)
        
        return ohlcv
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        
        # Forward fill for price data
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].fillna(method='ffill')
        
        # Zero fill for volume
        df['volume'] = df['volume'].fillna(0)
        
        return df
    
    def calculate_vwap(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Volume Weighted Average Price"""
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(period).sum() / \
               df['volume'].rolling(period).sum()
        
        return vwap

## 실시간 데이터 버퍼

class RealTimeDataBuffer:
    """실시간 데이터 버퍼링 및 처리"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffers = {}
        self.locks = {}
        
    async def add_data(self, symbol: str, data_type: str, data: Dict):
        """데이터 추가"""
        
        key = f"{symbol}_{data_type}"
        
        if key not in self.buffers:
            self.buffers[key] = []
            self.locks[key] = asyncio.Lock()
        
        async with self.locks[key]:
            self.buffers[key].append(data)
            
            # 크기 제한
            if len(self.buffers[key]) > self.max_size:
                self.buffers[key].pop(0)
    
    async def get_recent_data(self, symbol: str, data_type: str, 
                             n: int = 100) -> List[Dict]:
        """최근 데이터 조회"""
        
        key = f"{symbol}_{data_type}"
        
        if key not in self.buffers:
            return []
        
        async with self.locks[key]:
            return self.buffers[key][-n:]
    
    async def clear_old_data(self, max_age_seconds: int = 3600):
        """오래된 데이터 정리"""
        
        current_time = time.time()
        
        for key in list(self.buffers.keys()):
            async with self.locks[key]:
                self.buffers[key] = [
                    d for d in self.buffers[key]
                    if current_time - d.get('timestamp', 0) < max_age_seconds
                ]

## 데이터 무결성 체크

class DataIntegrityChecker:
    """데이터 무결성 검증"""
    
    def __init__(self):
        self.check_history = []
        
    def check_continuity(self, df: pd.DataFrame) -> Dict:
        """시계열 연속성 검사"""
        
        issues = []
        
        # 시간 간격 체크
        time_diffs = df.index.to_series().diff()
        expected_freq = pd.infer_freq(df.index)
        
        if expected_freq:
            expected_delta = pd.Timedelta(expected_freq)
            gaps = time_diffs[time_diffs > expected_delta * 1.5]
            
            if len(gaps) > 0:
                issues.append({
                    'type': 'TIME_GAP',
                    'count': len(gaps),
                    'locations': gaps.index.tolist()
                })
        
        # 가격 연속성
        price_jumps = df['close'].pct_change().abs()
        large_jumps = price_jumps[price_jumps > 0.1]  # 10% 이상
        
        if len(large_jumps) > 0:
            issues.append({
                'type': 'PRICE_JUMP',
                'count': len(large_jumps),
                'locations': large_jumps.index.tolist()
            })
        
        return {
            'is_continuous': len(issues) == 0,
            'issues': issues
        }
    
    def check_consistency(self, df: pd.DataFrame) -> Dict:
        """OHLC 일관성 검사"""
        
        issues = []
        
        # High >= Low
        invalid_hl = df[df['high'] < df['low']]
        if len(invalid_hl) > 0:
            issues.append({
                'type': 'HIGH_LOW_INVALID',
                'count': len(invalid_hl),
                'locations': invalid_hl.index.tolist()
            })
        
        # High >= Close, Open
        invalid_high = df[(df['high'] < df['close']) | (df['high'] < df['open'])]
        if len(invalid_high) > 0:
            issues.append({
                'type': 'HIGH_INVALID',
                'count': len(invalid_high),
                'locations': invalid_high.index.tolist()
            })
        
        # Low <= Close, Open
        invalid_low = df[(df['low'] > df['close']) | (df['low'] > df['open'])]
        if len(invalid_low) > 0:
            issues.append({
                'type': 'LOW_INVALID',
                'count': len(invalid_low),
                'locations': invalid_low.index.tolist()
            })
        
        return {
            'is_consistent': len(issues) == 0,
            'issues': issues
        }
    
    def generate_report(self) -> Dict:
        """무결성 검사 리포트"""
        
        if not self.check_history:
            return {}
        
        total_checks = len(self.check_history)
        failed_checks = sum(1 for c in self.check_history if not c['passed'])
        
        issue_types = {}
        for check in self.check_history:
            for issue in check.get('issues', []):
                issue_type = issue['type']
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        return {
            'total_checks': total_checks,
            'failed_checks': failed_checks,
            'success_rate': (total_checks - failed_checks) / total_checks,
            'issue_breakdown': issue_types,
            'last_check': self.check_history[-1] if self.check_history else None
        }
```