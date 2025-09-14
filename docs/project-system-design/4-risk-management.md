# 코인 선물 자동매매 시스템 - 리스크 관리

## 4.1 다층 리스크 제어

```python
class RiskController:
    """통합 리스크 관리자 (USDT 기준)"""
    
    def __init__(self, initial_capital_usdt: float):
        self.initial_capital = initial_capital_usdt
        
        # 리스크 한도 (USDT 기준)
        self.risk_limits = {
            'var_daily_return': 0.02,        # 일일 VaR 2% (수익률)
            'cvar_daily_return': 0.03,       # 일일 CVaR 3% (수익률)
            'var_daily_usdt': initial_capital_usdt * 0.02,  # VaR (USDT)
            'cvar_daily_usdt': initial_capital_usdt * 0.03,  # CVaR (USDT)
            'max_drawdown_pct': 0.12,        # 최대 드로다운 12%
            'correlation_threshold': 0.7,     # 상관관계 임계값
            'concentration_limit': 0.2,       # 단일 자산 집중도 20%
            'max_leverage': 10.0,             # 최대 레버리지 10x
            'liquidation_prob_24h': 0.005    # 24시간 청산 확률 0.5%
        }
        
        self.current_drawdown = 0
        self.high_water_mark = initial_capital_usdt
    
    def check_all_limits(self, portfolio_state: Dict) -> List[Tuple[str, float]]:
        """모든 리스크 한도 체크"""
        
        violations = []
        
        # VaR/CVaR 체크
        risk_metrics = RiskMetrics(
            portfolio_state['returns'], 
            portfolio_state['equity']
        )
        
        var_return = risk_metrics.calculate_var_return()
        if abs(var_return) > self.risk_limits['var_daily_return']:
            violations.append(('VAR_RETURN', var_return))
        
        cvar_return = risk_metrics.calculate_cvar_return()
        if abs(cvar_return) > self.risk_limits['cvar_daily_return']:
            violations.append(('CVAR_RETURN', cvar_return))
        
        # 드로다운 체크
        current_equity = portfolio_state['equity']
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
        
        self.current_drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
        if self.current_drawdown > self.risk_limits['max_drawdown_pct']:
            violations.append(('DRAWDOWN', self.current_drawdown))
        
        # 레버리지 체크
        total_leverage = portfolio_state['total_leverage']
        if total_leverage > self.risk_limits['max_leverage']:
            violations.append(('LEVERAGE', total_leverage))
        
        # 청산 확률 체크
        liquidation_prob = self._calculate_liquidation_probability(portfolio_state)
        if liquidation_prob > self.risk_limits['liquidation_prob_24h']:
            violations.append(('LIQUIDATION_RISK', liquidation_prob))
        
        return violations
    
    def _calculate_liquidation_probability(self, portfolio_state: Dict) -> float:
        """24시간 내 청산 확률 계산 (barrier model + fat-tail)"""
        
        min_log_distance = float('inf')
        
        for position in portfolio_state['positions']:
            if position['size'] == 0:
                continue
            
            current_price = position['current_price']
            liquidation_price = position['liquidation_price']
            
            # 로그 거리 계산
            if position['side'] == 'LONG':
                log_distance = np.log(current_price / liquidation_price)
            else:  # SHORT
                log_distance = np.log(liquidation_price / current_price)
            
            min_log_distance = min(min_log_distance, log_distance)
        
        if not np.isfinite(min_log_distance):
            return 0.0
        
        # 일일 로그 변동성
        daily_vol_log = portfolio_state.get('daily_volatility_log', 0.05)
        T = 1.0  # 1일
        
        # Barrier hitting probability (정규분포)
        prob_normal = 2.0 * (1.0 - stats.norm.cdf(
            min_log_distance / (daily_vol_log * np.sqrt(T))
        ))
        
        # Fat-tail 보정 (t-분포, df=4)
        prob_fat_tail = 2.0 * (1.0 - stats.t.cdf(
            min_log_distance / (daily_vol_log * np.sqrt(T)), 
            df=4
        ))
        
        # 더 보수적인 값 사용
        return float(max(prob_normal, prob_fat_tail))
    
    def calculate_position_limit(self, symbol: str, 
                                portfolio_state: Dict,
                                market_state: Dict) -> Dict:
        """심볼별 포지션 한도 계산"""
        
        equity = portfolio_state['equity']
        
        # 1. 집중도 한도
        concentration_limit = equity * self.risk_limits['concentration_limit']
        
        # 2. VaR 한도
        available_var = max(0,
            self.risk_limits['var_daily_usdt'] - 
            portfolio_state.get('current_var_usdt', 0)
        )
        
        # 3. 상관관계 조정
        correlation_adj = self._get_correlation_adjustment(
            symbol, portfolio_state
        )
        
        # 4. 레버리지 한도
        current_leverage = portfolio_state['total_leverage']
        leverage_room = max(0, 
            self.risk_limits['max_leverage'] - current_leverage
        )
        
        # 최종 한도
        price = market_state['price']
        
        max_notional = min(
            concentration_limit,
            available_var / (market_state['volatility'] * 1.65),
            equity * leverage_room
        ) * correlation_adj
        
        max_quantity = max_notional / price if price > 0 else 0
        
        return {
            'max_quantity': max_quantity,
            'max_notional': max_notional,
            'limiting_factor': self._get_limiting_factor(
                concentration_limit, available_var, leverage_room
            )
        }
    
    def _get_correlation_adjustment(self, symbol: str, 
                                   portfolio_state: Dict) -> float:
        """상관관계 기반 조정 계수"""
        
        if not portfolio_state.get('positions'):
            return 1.0
        
        correlations = []
        for position in portfolio_state['positions']:
            if position['symbol'] != symbol:
                corr = portfolio_state.get('correlation_matrix', {}).get(
                    (symbol, position['symbol']), 0
                )
                correlations.append(abs(corr))
        
        if not correlations:
            return 1.0
        
        avg_correlation = np.mean(correlations)
        
        if avg_correlation > 0.8:
            return 0.3
        elif avg_correlation > 0.6:
            return 0.5
        elif avg_correlation > 0.4:
            return 0.7
        else:
            return 1.0
    
    def _get_limiting_factor(self, concentration: float, 
                            var: float, leverage: float) -> str:
        """제한 요인 식별"""
        
        limits = {
            'CONCENTRATION': concentration,
            'VAR': var,
            'LEVERAGE': leverage
        }
        
        return min(limits, key=limits.get)
```

## 4.2 동적 포지션 사이징

```python
class PositionSizer:
    """청산 안전성을 고려한 포지션 사이징 (USDT 기준)"""
    
    def __init__(self, risk_controller: RiskController, 
                 kelly_optimizer: ContinuousKellyOptimizer):
        self.risk_controller = risk_controller
        self.kelly_optimizer = kelly_optimizer
        
    def calculate_position_size(self, signal: Dict, 
                               market_state: Dict, 
                               portfolio_state: Dict) -> float:
        """
        다중 제약 하의 최적 포지션 크기
        
        Returns:
            float: Position size in coin units
        """
        
        symbol = signal['symbol']
        account_equity = portfolio_state['equity']  # USDT
        price = market_state['price']  # USDT per coin
        
        # 1. Kelly 기반 크기
        kelly_fraction = self.kelly_optimizer.calculate_optimal_fraction(
            portfolio_state['recent_returns'],
            regime=market_state['regime']
        )
        kelly_notional = account_equity * kelly_fraction  # USDT
        kelly_size = kelly_notional / price  # coin units
        
        # 2. ATR 기반 리스크 크기
        atr = market_state['atr']  # USDT
        risk_per_trade = account_equity * 0.01  # 1% risk in USDT
        stop_distance = 2.0 * atr  # USDT
        if stop_distance > 0:
            atr_size = risk_per_trade / stop_distance  # coin units
        else:
            atr_size = 0
        
        # 3. 청산 안전 거리 기반 크기
        liquidation_safe_size = self._calculate_liquidation_safe_size(
            symbol, 
            signal['side'],
            market_state,
            portfolio_state
        )
        
        # 4. VaR 제약 크기
        var_constrained_size = self._calculate_var_constrained_size(
            symbol,
            market_state,
            portfolio_state
        )
        
        # 5. 상관관계 조정
        correlation_factor = self._calculate_correlation_adjustment(
            symbol,
            portfolio_state
        )
        
        # 최종 크기 (최소값 선택)
        base_size = min(
            kelly_size,
            atr_size,
            liquidation_safe_size,
            var_constrained_size
        )
        
        # 상관관계 조정 적용
        final_size = base_size * correlation_factor * signal['strength']
        
        # 최소/최대 제약
        min_notional = market_state.get('min_notional', 10)  # 10 USDT
        min_size = min_notional / price
        max_size = account_equity * 0.2 / price  # 단일 포지션 20% 상한
        
        return self._round_to_lot_size(
            np.clip(final_size, min_size, max_size),
            market_state['lot_size']
        )
    
    def _calculate_liquidation_safe_size(self, symbol: str, side: str, 
                                        market_state: Dict, 
                                        portfolio_state: Dict) -> float:
        """청산 안전 거리를 고려한 최대 크기"""
        
        price = market_state['price']
        atr = market_state['atr']
        
        # Binance USDT-M 증거금 규칙
        mmr = self._get_maintenance_margin_rate(symbol, portfolio_state['equity'])
        fee_buffer = 0.001  # Fee buffer
        k = 3.0  # 3 ATR safety distance
        
        # 안전 거리 (가격 대비 비율)
        safe_distance_pct = (k * atr) / price
        
        # 최대 허용 레버리지
        max_leverage_for_safety = 1.0 / max(1e-6, 
            mmr + fee_buffer + safe_distance_pct
        )
        
        # 실제 사용 레버리지
        symbol_leverage = market_state.get('symbol_leverage', 10)
        max_global_leverage = self.risk_controller.risk_limits['max_leverage']
        
        leverage_to_use = min(
            max_leverage_for_safety,
            symbol_leverage,
            max_global_leverage
        )
        
        # 포지션 크기 (coin units)
        notional = portfolio_state['equity'] * leverage_to_use
        size = notional / price
        
        return float(size)
    
    def _calculate_var_constrained_size(self, symbol: str, 
                                       market_state: Dict,
                                       portfolio_state: Dict) -> float:
        """VaR 제약 하의 최대 포지션 크기"""
        
        z = 1.65  # 95% confidence
        price = market_state['price']
        sigma = portfolio_state['symbol_volatilities'].get(symbol, 0.05)
        
        # VaR 한도 (USDT)
        var_limit = portfolio_state['equity'] * 0.02
        current_var = portfolio_state.get('current_var_usdt', 0)
        available_var = max(0, var_limit - current_var)
        
        if sigma <= 0 or price <= 0:
            return 0.0
        
        # VaR = z * σ * Price * Quantity
        # → Quantity = VaR / (z * σ * Price)
        max_quantity = available_var / (z * sigma * price)
        
        return float(max_quantity)
    
    def _calculate_correlation_adjustment(self, symbol: str, 
                                         portfolio_state: Dict) -> float:
        """포트폴리오 상관관계 기반 조정 계수"""
        
        if not portfolio_state.get('positions'):
            return 1.0
        
        # 기존 포지션들과의 평균 상관관계
        correlations = []
        for position in portfolio_state['positions']:
            corr = portfolio_state.get('correlation_matrix', {}).get(
                (symbol, position['symbol']), 0
            )
            correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations) if correlations else 0
        
        # 높은 상관관계일수록 크기 감소
        if avg_correlation > 0.8:
            return 0.3
        elif avg_correlation > 0.6:
            return 0.5
        elif avg_correlation > 0.4:
            return 0.7
        else:
            return 1.0
    
    def _get_maintenance_margin_rate(self, symbol: str, notional: float) -> float:
        """Binance 유지증거금률 (실제로는 티어별)"""
        # 간소화된 버전 - 실제로는 API에서 조회
        if notional < 10000:
            return 0.004  # 0.4%
        elif notional < 50000:
            return 0.005  # 0.5%
        else:
            return 0.01   # 1%
    
    def _round_to_lot_size(self, size: float, lot_size: float) -> float:
        """거래소 lot size로 반올림"""
        return float(np.floor(size / lot_size) * lot_size)


## 4.3 포지션 관리

class PositionManager:
    """포지션 생명주기 관리"""
    
    def __init__(self, risk_controller: RiskController):
        self.risk_controller = risk_controller
        self.positions = {}
        
    def open_position(self, symbol: str, side: str, size: float, 
                     price: float, leverage: float) -> Dict:
        """포지션 오픈"""
        
        position = {
            'symbol': symbol,
            'side': side,
            'size': size,
            'entry_price': price,
            'current_price': price,
            'leverage': leverage,
            'margin': (size * price) / leverage,
            'liquidation_price': self._calculate_liquidation_price(
                side, price, leverage
            ),
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'open_time': pd.Timestamp.now(),
            'trailing_stop': None,
            'take_profit': None
        }
        
        self.positions[symbol] = position
        return position
    
    def update_position(self, symbol: str, current_price: float) -> Dict:
        """포지션 업데이트"""
        
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position['current_price'] = current_price
        
        # PnL 계산
        if position['side'] == 'LONG':
            position['unrealized_pnl'] = (
                (current_price - position['entry_price']) * position['size']
            )
        else:  # SHORT
            position['unrealized_pnl'] = (
                (position['entry_price'] - current_price) * position['size']
            )
        
        # Trailing stop 업데이트
        if position['trailing_stop']:
            self._update_trailing_stop(position, current_price)
        
        return position
    
    def close_position(self, symbol: str, price: float, 
                      reason: str = 'MANUAL') -> Dict:
        """포지션 종료"""
        
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # 최종 PnL 계산
        if position['side'] == 'LONG':
            final_pnl = (price - position['entry_price']) * position['size']
        else:
            final_pnl = (position['entry_price'] - price) * position['size']
        
        position['realized_pnl'] = final_pnl
        position['close_price'] = price
        position['close_time'] = pd.Timestamp.now()
        position['close_reason'] = reason
        
        # 포지션 제거
        del self.positions[symbol]
        
        return position
    
    def check_stop_conditions(self, symbol: str, 
                            current_price: float) -> Optional[str]:
        """스톱 조건 체크"""
        
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Stop loss
        if position.get('stop_loss'):
            if position['side'] == 'LONG' and current_price <= position['stop_loss']:
                return 'STOP_LOSS'
            elif position['side'] == 'SHORT' and current_price >= position['stop_loss']:
                return 'STOP_LOSS'
        
        # Take profit
        if position.get('take_profit'):
            if position['side'] == 'LONG' and current_price >= position['take_profit']:
                return 'TAKE_PROFIT'
            elif position['side'] == 'SHORT' and current_price <= position['take_profit']:
                return 'TAKE_PROFIT'
        
        # Trailing stop
        if position.get('trailing_stop'):
            if position['side'] == 'LONG' and current_price <= position['trailing_stop']:
                return 'TRAILING_STOP'
            elif position['side'] == 'SHORT' and current_price >= position['trailing_stop']:
                return 'TRAILING_STOP'
        
        return None
    
    def _calculate_liquidation_price(self, side: str, 
                                    entry_price: float, 
                                    leverage: float) -> float:
        """청산가 계산"""
        
        # 간소화된 버전
        mmr = 0.005  # 0.5% maintenance margin
        
        if side == 'LONG':
            liquidation_price = entry_price * (1 - 1/leverage + mmr)
        else:  # SHORT
            liquidation_price = entry_price * (1 + 1/leverage - mmr)
        
        return liquidation_price
    
    def _update_trailing_stop(self, position: Dict, current_price: float):
        """Trailing stop 업데이트"""
        
        trailing_distance = position.get('trailing_distance', 0.02)  # 2%
        
        if position['side'] == 'LONG':
            new_stop = current_price * (1 - trailing_distance)
            if position['trailing_stop'] is None or new_stop > position['trailing_stop']:
                position['trailing_stop'] = new_stop
        else:  # SHORT
            new_stop = current_price * (1 + trailing_distance)
            if position['trailing_stop'] is None or new_stop < position['trailing_stop']:
                position['trailing_stop'] = new_stop
```