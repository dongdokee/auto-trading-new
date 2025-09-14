# 코인 선물 자동매매 시스템 - 전략 엔진

## 3.1 레짐 감지 시스템

```python
class NoLookAheadRegimeDetector:
    """룩어헤드 바이어스가 제거된 레짐 감지"""
    
    def __init__(self):
        self.hmm_model = None
        self.garch_model = None
        self.last_train_index = -1
        self.retrain_interval = 180  # 6개월
        self.min_train_samples = 500
        
        # Sticky HMM parameters
        self.transition_penalty = 0.9
        self.min_regime_duration = 5
        self.current_regime = None
        self.regime_duration = 0
        self.state_label_map = {}
        
    def fit(self, historical_data: pd.DataFrame, end_index: int) -> bool:
        """
        특정 시점까지의 데이터로만 학습
        
        Args:
            historical_data: 전체 히스토리 데이터
            end_index: 학습에 사용할 마지막 인덱스 (exclusive)
        """
        if end_index < self.min_train_samples:
            return False
        
        # end_index 이전 데이터만 사용
        train_data = historical_data[:end_index]
        
        # Feature engineering
        returns = np.diff(np.log(train_data['close']))
        features = self._prepare_features(train_data)
        
        if features.size == 0:
            return False
        
        # HMM with sticky transitions
        from hmmlearn.hmm import GaussianHMM
        self.hmm_model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            init_params="mc",  # means, covars만 초기화
            params="mc"        # transition matrix는 학습 안 함
        )
        
        # Sticky transition matrix 설정
        trans_mat = np.full((3, 3), (1 - self.transition_penalty) / 2)
        np.fill_diagonal(trans_mat, self.transition_penalty)
        self.hmm_model.transmat_ = trans_mat
        self.hmm_model.startprob_ = np.array([1/3, 1/3, 1/3])
        
        # 학습
        self.hmm_model.fit(features)
        
        # 상태 레이블 재할당
        self.state_label_map = self._relabel_states_after_training(features)
        
        # GARCH model
        from arch import arch_model
        self.garch_model = arch_model(returns, vol='GARCH', p=1, q=1)
        self.garch_result = self.garch_model.fit(disp='off')
        
        self.last_train_index = end_index
        return True
    
    def _relabel_states_after_training(self, features: np.ndarray) -> Dict:
        """학습 후 상태 레이블 일관성 유지"""
        if self.hmm_model is None:
            return {}
            
        states = self.hmm_model.predict(features)
        
        # 원본 수익률 데이터 추출 (표준화 전)
        returns = features[:, 0]  # 첫 번째 특징이 수익률
        
        state_returns = {}
        for state in range(self.hmm_model.n_components):
            mask = (states == state)
            if mask.sum() > 0:
                state_returns[state] = returns[mask].mean()
        
        # 수익률 기준으로 정렬
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1], reverse=True)
        
        # 레이블 매핑
        label_map = {}
        if len(sorted_states) >= 3:
            label_map[sorted_states[0][0]] = 'BULL'
            label_map[sorted_states[2][0]] = 'BEAR'
            label_map[sorted_states[1][0]] = 'SIDEWAYS'
        else:
            label_map = {0: 'BULL', 1: 'BEAR', 2: 'SIDEWAYS'}
        
        return label_map
    
    def detect_regime(self, data: pd.DataFrame, current_index: int) -> Dict:
        """
        현재 시점의 레짐 감지 (미래 데이터 사용 안 함)
        
        Args:
            data: 전체 데이터
            current_index: 현재 시점 인덱스
            
        Returns:
            dict: 레짐 정보
        """
        # 재학습 필요 체크
        if current_index - self.last_train_index > self.retrain_interval:
            self.fit(data, current_index)
        
        # 현재까지의 데이터만 사용
        current_data = data[:current_index + 1]
        features = self._prepare_features(current_data)
        
        if features.size == 0:
            return {'regime': 'NEUTRAL', 'confidence': 0.5}
        
        # 현재 상태 예측
        if self.hmm_model is None:
            return {'regime': 'NEUTRAL', 'confidence': 0.5}
        
        # 상태 확률
        state_probs = self.hmm_model.predict_proba(features[-1:])
        predicted_state = np.argmax(state_probs[0])
        confidence = state_probs[0, predicted_state]
        
        # 레이블 매핑 적용
        candidate_regime = self.state_label_map.get(predicted_state, 'NEUTRAL')
        
        # Whipsaw 방지
        if self.current_regime is None:
            self.current_regime = candidate_regime
            self.regime_duration = 1
        elif candidate_regime == self.current_regime:
            self.regime_duration += 1
        else:
            # 충분한 지속 시간과 높은 확신도 필요
            if self.regime_duration >= self.min_regime_duration and confidence > 0.7:
                self.current_regime = candidate_regime
                self.regime_duration = 1
        
        # GARCH 변동성 예측
        if self.garch_result is not None:
            volatility_forecast = float(
                self.garch_result.forecast(horizon=1).variance.values[-1, 0]**0.5
            )
        else:
            volatility_forecast = 0.02
        
        return {
            'regime': self.current_regime,
            'confidence': float(confidence),
            'regime_probabilities': state_probs[0].tolist(),
            'volatility_forecast': volatility_forecast,
            'duration': self.regime_duration
        }
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """특징 추출 (표준화 포함)"""
        if len(data) < 21:
            return np.array([])
            
        returns = np.diff(np.log(data['close']))
        
        # 볼륨 프로파일
        volume_ma = data['volume'].rolling(20).mean()
        volume_ratio = data['volume'] / (volume_ma + 1e-10)
        
        # RSI
        rsi = self._calculate_rsi(data['close'])
        
        # 특징 결합
        min_length = min(len(returns), len(volume_ratio) - 1, len(rsi) - 1)
        if min_length < 20:
            return np.array([])
            
        features = np.column_stack([
            returns[-min_length:],
            volume_ratio.values[-min_length-1:-1],
            rsi.values[-min_length-1:-1]
        ])
        
        # Z-score 표준화
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
```

## 3.2 전략 매트릭스

```python
class StrategyMatrix:
    """레짐별 최적 전략 선택"""
    
    def __init__(self):
        self.base_weights = {
            ('BULL', 'LOW'): {'TREND_FOLLOWING': 0.7, 'MOMENTUM': 0.3},
            ('BULL', 'HIGH'): {'TREND_FOLLOWING': 0.4, 'FUNDING_ARB': 0.6},
            ('BEAR', 'LOW'): {'MEAN_REVERSION': 0.6, 'FUNDING_ARB': 0.4},
            ('BEAR', 'HIGH'): {'MEAN_REVERSION': 0.3, 'FUNDING_ARB': 0.7},
            ('SIDEWAYS', 'LOW'): {'RANGE_TRADING': 0.7, 'FUNDING_ARB': 0.3},
            ('SIDEWAYS', 'HIGH'): {'RANGE_TRADING': 0.4, 'FUNDING_ARB': 0.6},
            ('NEUTRAL', 'LOW'): {'NEUTRAL': 0.5, 'FUNDING_ARB': 0.5},
            ('NEUTRAL', 'HIGH'): {'FUNDING_ARB': 1.0}
        }
    
    def get_strategy_weights(self, regime_info: Dict) -> Dict[str, float]:
        """레짐과 변동성에 따른 전략 가중치"""
        
        regime = regime_info['regime']
        vol_level = 'HIGH' if regime_info['volatility_forecast'] > 0.03 else 'LOW'
        
        key = (regime, vol_level)
        weights = self.base_weights.get(key, {'NEUTRAL': 1.0})
        
        # 확신도에 따른 조정
        confidence = regime_info['confidence']
        if confidence < 0.6:
            # 낮은 확신도: 중립 전략 비중 증가
            adjusted = {}
            for strategy, weight in weights.items():
                adjusted[strategy] = weight * confidence
            adjusted['NEUTRAL'] = adjusted.get('NEUTRAL', 0) + (1 - confidence) * 0.5
            
            # 정규화
            total = sum(adjusted.values())
            if total > 0:
                return {k: v/total for k, v in adjusted.items()}
        
        return weights
```

## 3.3 펀딩 레이트 차익거래

```python
class FundingArbitrage:
    """펀딩 차익거래 전략 (USDT 기준)"""
    
    def __init__(self):
        self.funding_spike_threshold = 0.001  # 0.1%
        self.min_sharpe = 1.5
        
    def calculate_funding_opportunity(self, symbol: str, 
                                     market_data: Dict, 
                                     portfolio_state: Dict) -> Dict:
        """펀딩 차익 기회 평가"""
        
        # 시장 데이터
        current_funding = market_data['funding_rate']
        hours_to_funding = market_data['hours_to_next_funding']
        price = market_data['price']  # USDT per coin
        volatility_hourly = market_data['volatility_hourly']
        
        # 포지션 사이징 (VaR 제약)
        z = 1.65  # 95% 신뢰수준
        available_var = max(0, 
            portfolio_state['equity'] * 0.02 - 
            portfolio_state.get('current_var_usdt', 0)
        )
        
        # VaR 기반 최대 수량
        if volatility_hourly > 0 and price > 0:
            size_var = available_var / (
                z * volatility_hourly * np.sqrt(hours_to_funding) * price
            )
        else:
            size_var = 0
        
        # 자본 대비 상한
        size_cap = portfolio_state['equity'] * 0.2 / price
        
        # 최종 수량
        quantity = max(0.0, min(size_var, size_cap))
        
        if quantity <= 0:
            return {'action': 'SKIP'}
        
        # 예상 펀딩 수익 (USDT)
        notional = quantity * price
        expected_funding_pnl = notional * current_funding * (hours_to_funding/8)
        
        # 가격 변동 리스크 (USDT)
        price_risk = notional * volatility_hourly * np.sqrt(hours_to_funding)
        
        # 펀딩 전후 스파이크 리스크
        spike_multiplier = 1.0
        if hours_to_funding < 0.5 and abs(current_funding) > 0.002:
            spike_multiplier = 2.0
        elif hours_to_funding < 1.0:
            spike_multiplier = 1.2
        
        adjusted_risk = price_risk * spike_multiplier
        
        # 리스크 조정 수익률
        if adjusted_risk > 0:
            funding_sharpe = expected_funding_pnl / adjusted_risk
        else:
            funding_sharpe = 0
        
        # 진입 결정
        if funding_sharpe > self.min_sharpe:
            # 델타 중립 여부 결정
            if abs(current_funding) > 0.002:  # 극단적 펀딩
                strategy = 'DELTA_NEUTRAL'  # 현물-선물 페어
            else:
                strategy = 'DIRECTIONAL'  # 선물만
            
            return {
                'action': 'ENTER',
                'side': 'SHORT' if current_funding > 0 else 'LONG',
                'strategy': strategy,
                'quantity': quantity,
                'expected_sharpe': funding_sharpe,
                'size_multiplier': min(2.0, funding_sharpe / self.min_sharpe)
            }
        
        return {'action': 'SKIP'}
```

## 3.4 개별 전략 구현

```python
class TrendFollowingStrategy:
    """추세 추종 전략"""
    
    def __init__(self):
        self.fast_period = 20
        self.slow_period = 50
        self.atr_multiplier = 2.0
        self.min_trend_strength = 0.3
        
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """추세 추종 신호 생성"""
        
        if len(data) < self.slow_period:
            return {'action': 'SKIP'}
        
        # Moving averages
        fast_ma = data['close'].rolling(self.fast_period).mean()
        slow_ma = data['close'].rolling(self.slow_period).mean()
        
        # ATR for volatility-adjusted signals
        atr = self.calculate_atr(data)
        
        # Trend strength
        price = data['close'].iloc[-1]
        trend_strength = (fast_ma.iloc[-1] - slow_ma.iloc[-1]) / atr.iloc[-1]
        
        # Entry signals
        if trend_strength > self.min_trend_strength:
            return {
                'action': 'ENTER',
                'side': 'BUY',
                'strength': min(1.0, trend_strength / 2.0),
                'stop_loss': price - self.atr_multiplier * atr.iloc[-1],
                'take_profit': price + self.atr_multiplier * atr.iloc[-1] * 3
            }
        elif trend_strength < -self.min_trend_strength:
            return {
                'action': 'ENTER',
                'side': 'SELL',
                'strength': min(1.0, abs(trend_strength) / 2.0),
                'stop_loss': price + self.atr_multiplier * atr.iloc[-1],
                'take_profit': price - self.atr_multiplier * atr.iloc[-1] * 3
            }
        
        return {'action': 'SKIP'}
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR 계산"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr

class MeanReversionStrategy:
    """평균 회귀 전략"""
    
    def __init__(self):
        self.bb_period = 20
        self.bb_std = 2.0
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """평균 회귀 신호 생성"""
        
        if len(data) < self.bb_period:
            return {'action': 'SKIP'}
        
        # Bollinger Bands
        close = data['close']
        bb_middle = close.rolling(self.bb_period).mean()
        bb_std_val = close.rolling(self.bb_period).std()
        bb_upper = bb_middle + self.bb_std * bb_std_val
        bb_lower = bb_middle - self.bb_std * bb_std_val
        
        # RSI
        rsi = self.calculate_rsi(close, self.rsi_period)
        
        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Entry conditions
        if current_price < bb_lower.iloc[-1] and current_rsi < self.rsi_oversold:
            distance_to_mean = (bb_middle.iloc[-1] - current_price) / current_price
            return {
                'action': 'ENTER',
                'side': 'BUY',
                'strength': min(1.0, distance_to_mean * 10),
                'target': bb_middle.iloc[-1],
                'stop_loss': bb_lower.iloc[-1] * 0.98
            }
        elif current_price > bb_upper.iloc[-1] and current_rsi > self.rsi_overbought:
            distance_to_mean = (current_price - bb_middle.iloc[-1]) / current_price
            return {
                'action': 'ENTER',
                'side': 'SELL',
                'strength': min(1.0, distance_to_mean * 10),
                'target': bb_middle.iloc[-1],
                'stop_loss': bb_upper.iloc[-1] * 1.02
            }
        
        return {'action': 'SKIP'}
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

class RangeTradingStrategy:
    """레인지 트레이딩 전략"""
    
    def __init__(self):
        self.lookback = 100
        self.range_threshold = 0.02  # 2% range
        self.breakout_confirmation = 3
        
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """레인지 트레이딩 신호 생성"""
        
        if len(data) < self.lookback:
            return {'action': 'SKIP'}
        
        recent_data = data.tail(self.lookback)
        
        # Identify range
        resistance = recent_data['high'].max()
        support = recent_data['low'].min()
        range_size = (resistance - support) / support
        
        # Check if in range
        if range_size > self.range_threshold * 2:
            return {'action': 'SKIP'}  # Too volatile for range trading
        
        current_price = data['close'].iloc[-1]
        distance_to_support = (current_price - support) / support
        distance_to_resistance = (resistance - current_price) / resistance
        
        # Entry at range boundaries
        if distance_to_support < 0.01:  # Near support
            return {
                'action': 'ENTER',
                'side': 'BUY',
                'strength': 0.8,
                'target': resistance * 0.98,
                'stop_loss': support * 0.98
            }
        elif distance_to_resistance < 0.01:  # Near resistance
            return {
                'action': 'ENTER',
                'side': 'SELL',
                'strength': 0.8,
                'target': support * 1.02,
                'stop_loss': resistance * 1.02
            }
        
        return {'action': 'SKIP'}
```

## 3.5 알파 생명주기 관리

```python
class AlphaLifecycleManager:
    """전략 성과 추적 및 관리"""
    
    def __init__(self):
        self.strategies = {}
        self.performance_window = 30  # days
        self.decay_threshold = 0.5    # Sharpe threshold
        
    def track_performance(self, strategy_id: str, daily_pnl: float):
        """전략 성과 추적"""
        
        if strategy_id not in self.strategies:
            self.strategies[strategy_id] = {
                'pnl_history': [],
                'sharpe_history': [],
                'weight': 0.2,
                'stage': 'INCUBATION',
                'inception_date': pd.Timestamp.now()
            }
        
        strategy = self.strategies[strategy_id]
        strategy['pnl_history'].append(daily_pnl)
        
        # Rolling 메트릭 계산
        if len(strategy['pnl_history']) >= self.performance_window:
            recent_pnl = strategy['pnl_history'][-self.performance_window:]
            
            # Sharpe Ratio
            if np.std(recent_pnl) > 0:
                sharpe = np.mean(recent_pnl) / np.std(recent_pnl) * np.sqrt(252)
            else:
                sharpe = 0
            
            strategy['sharpe_history'].append(sharpe)
            
            # 생명주기 단계 업데이트
            self._update_lifecycle_stage(strategy_id, sharpe)
            
            # 가중치 동적 조정
            self._adjust_weight(strategy_id, sharpe)
    
    def _update_lifecycle_stage(self, strategy_id: str, current_sharpe: float):
        """전략 생명주기 단계 전환"""
        
        strategy = self.strategies[strategy_id]
        sharpe_history = strategy['sharpe_history']
        
        if len(sharpe_history) < 10:
            return
        
        # 추세 분석
        recent_sharpes = sharpe_history[-10:]
        trend = np.polyfit(range(10), recent_sharpes, 1)[0]
        
        current_stage = strategy['stage']
        
        # 단계 전환 로직
        transitions = {
            'INCUBATION': {
                'GROWTH': current_sharpe > 1.0 and trend > 0,
                'RETIRED': current_sharpe < 0 and len(sharpe_history) > 60
            },
            'GROWTH': {
                'MATURE': current_sharpe > 1.5 and abs(trend) < 0.01,
                'DECAY': current_sharpe < 0.8 or trend < -0.05
            },
            'MATURE': {
                'DECAY': trend < -0.02 or current_sharpe < 1.0
            },
            'DECAY': {
                'RETIRED': current_sharpe < self.decay_threshold
            }
        }
        
        if current_stage in transitions:
            for next_stage, condition in transitions[current_stage].items():
                if condition:
                    strategy['stage'] = next_stage
                    self._log_stage_transition(strategy_id, current_stage, next_stage)
                    break
    
    def _adjust_weight(self, strategy_id: str, current_sharpe: float):
        """성과 기반 가중치 조정"""
        
        strategy = self.strategies[strategy_id]
        stage = strategy['stage']
        
        # 단계별 가중치 조정
        weight_map = {
            'INCUBATION': min(0.1, max(0.01, current_sharpe * 0.05)),
            'GROWTH': min(0.3, max(0.1, current_sharpe * 0.15)),
            'MATURE': min(0.4, max(0.2, current_sharpe * 0.2)),
            'DECAY': max(0.05, strategy['weight'] * 0.8),
            'RETIRED': 0
        }
        
        strategy['weight'] = weight_map.get(stage, 0.1)
        
        # 전체 가중치 정규화
        self._normalize_weights()
    
    def get_active_strategies(self) -> Dict:
        """활성 전략 목록"""
        return {
            sid: s for sid, s in self.strategies.items()
            if s['stage'] not in ['RETIRED', 'DECAY']
        }
    
    def _normalize_weights(self):
        """전체 전략 가중치 정규화"""
        total_weight = sum(s['weight'] for s in self.strategies.values())
        if total_weight > 0:
            for strategy in self.strategies.values():
                strategy['weight'] /= total_weight
    
    def _log_stage_transition(self, strategy_id: str, from_stage: str, to_stage: str):
        """단계 전환 로깅"""
        print(f"Strategy {strategy_id}: {from_stage} → {to_stage}")
```