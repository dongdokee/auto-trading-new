# 코인 선물 자동매매 시스템 - 모니터링 시스템

## 9.1 실시간 대시보드

```python
class TradingDashboard:
    """실시간 모니터링 대시보드 (USDT 기준)"""
    
    def __init__(self, risk_controller: RiskController):
        self.risk_controller = risk_controller
        
        self.metrics = {
            # 수익성 지표 (USDT 기준)
            'pnl_realtime_usdt': 0,          # USDT
            'pnl_realtime_pct': 0,           # %
            'equity_usdt': 0,                # USDT
            'sharpe_rolling_30d': 0,         # ratio
            'sortino_rolling_30d': 0,        # ratio
            
            # 리스크 지표
            'var_daily_usdt': 0,             # USDT
            'var_limit_usdt': 0,             # USDT
            'cvar_daily_usdt': 0,            # USDT
            'current_drawdown_pct': 0,       # %
            'max_drawdown_pct': 0,           # %
            
            # 포지션 정보
            'total_leverage': 0,             # x
            'total_notional_usdt': 0,        # USDT
            'margin_used_usdt': 0,           # USDT
            'free_margin_usdt': 0,           # USDT
            'liquidation_distance_min': 0,   # %
            'correlation_max': 0,            # correlation
            
            # 실행 품질
            'avg_slippage_bps': 0,          # bps
            'fill_rate_pct': 0,             # %
            'post_only_success_pct': 0,     # %
            
            # 시스템 상태
            'ws_uptime_pct': 0,             # %
            'order_latency_p99_ms': 0,      # ms
            'api_error_rate_pct': 0,        # %
            
            # 전략 성과
            'strategy_sharpes': {},          # strategy_id -> sharpe
            'alpha_decay_scores': {},        # strategy_id -> decay score
            
            # 데이터 품질
            'data_anomaly_rate_pct': 0,     # %
            'data_staleness_seconds': 0,    # seconds
            
            # 자금 흐름
            'total_commission_usdt': 0,      # USDT
            'funding_pnl_usdt': 0,           # USDT
            'trading_pnl_usdt': 0,           # USDT
        }
        
        self.update_interval = 1  # 초
        self.history_window = 86400  # 24시간
        
    def update_metrics(self, portfolio_state: Dict, market_state: Dict):
        """메트릭 업데이트"""
        
        # PnL 계산
        current_equity = portfolio_state['equity']
        initial_equity = portfolio_state['initial_equity']
        
        self.metrics['equity_usdt'] = current_equity
        self.metrics['pnl_realtime_usdt'] = current_equity - initial_equity
        self.metrics['pnl_realtime_pct'] = (
            (current_equity - initial_equity) / initial_equity * 100
        )
        
        # 리스크 메트릭
        risk_calc = RiskMetrics(
            portfolio_state['returns_30d'],
            current_equity
        )
        
        self.metrics['var_daily_usdt'] = risk_calc.calculate_var_usdt()
        self.metrics['var_limit_usdt'] = self.risk_controller.risk_limits['var_daily_usdt']
        self.metrics['cvar_daily_usdt'] = risk_calc.calculate_cvar_usdt()
        
        # 드로다운
        high_water_mark = portfolio_state['high_water_mark']
        self.metrics['current_drawdown_pct'] = (
            (high_water_mark - current_equity) / high_water_mark * 100
        )
        
        # 나머지 메트릭 업데이트
        self._update_position_metrics(portfolio_state)
        self._update_execution_metrics(portfolio_state)
        self._update_system_metrics()
        self._update_strategy_metrics(portfolio_state)
        self._update_data_quality_metrics(market_state)
    
    def get_alerts(self) -> List[Dict]:
        """경고 조건 체크"""
        
        alerts = []
        
        # Critical 알림
        if self.metrics['current_drawdown_pct'] > 10:
            alerts.append({
                'level': 'CRITICAL',
                'message': f"Drawdown {self.metrics['current_drawdown_pct']:.1f}% exceeds limit",
                'metric': 'drawdown'
            })
        
        if self.metrics['var_daily_usdt'] > self.metrics['var_limit_usdt']:
            alerts.append({
                'level': 'CRITICAL',
                'message': f"VaR {self.metrics['var_daily_usdt']:.2f} USDT exceeds limit",
                'metric': 'var'
            })
        
        if self.metrics['liquidation_distance_min'] < 5:
            alerts.append({
                'level': 'CRITICAL',
                'message': f"Liquidation distance {self.metrics['liquidation_distance_min']:.1f}% too close",
                'metric': 'liquidation'
            })
        
        # Warning 알림
        if self.metrics['correlation_max'] > 0.8:
            alerts.append({
                'level': 'WARNING',
                'message': f"High correlation {self.metrics['correlation_max']:.2f}",
                'metric': 'correlation'
            })
        
        if self.metrics['avg_slippage_bps'] > 50:
            alerts.append({
                'level': 'WARNING',
                'message': f"High slippage {self.metrics['avg_slippage_bps']:.0f} bps",
                'metric': 'slippage'
            })
        
        # Info 알림
        if self.metrics['data_anomaly_rate_pct'] > 1:
            alerts.append({
                'level': 'INFO',
                'message': f"Data anomaly rate {self.metrics['data_anomaly_rate_pct']:.1f}%",
                'metric': 'data_quality'
            })
        
        return alerts
    
    def _update_position_metrics(self, portfolio_state: Dict):
        """포지션 관련 메트릭 업데이트"""
        
        total_notional = 0
        margin_used = 0
        min_liq_distance = float('inf')
        
        for position in portfolio_state.get('positions', {}).values():
            notional = position['size'] * position['current_price']
            total_notional += notional
            margin_used += position['margin']
            
            # 청산 거리
            if position['side'] == 'LONG':
                liq_distance = (
                    (position['current_price'] - position['liquidation_price']) / 
                    position['current_price'] * 100
                )
            else:
                liq_distance = (
                    (position['liquidation_price'] - position['current_price']) / 
                    position['current_price'] * 100
                )
            
            min_liq_distance = min(min_liq_distance, liq_distance)
        
        self.metrics['total_notional_usdt'] = total_notional
        self.metrics['margin_used_usdt'] = margin_used
        self.metrics['free_margin_usdt'] = portfolio_state['equity'] - margin_used
        self.metrics['liquidation_distance_min'] = min_liq_distance if min_liq_distance != float('inf') else 100
        
        # 레버리지
        if portfolio_state['equity'] > 0:
            self.metrics['total_leverage'] = total_notional / portfolio_state['equity']
        
        # 상관관계
        correlations = []
        for corr in portfolio_state.get('correlation_matrix', {}).values():
            correlations.append(abs(corr))
        
        if correlations:
            self.metrics['correlation_max'] = max(correlations)
    
    def _update_execution_metrics(self, portfolio_state: Dict):
        """실행 품질 메트릭 업데이트"""
        
        recent_trades = portfolio_state.get('recent_trades', [])
        
        if recent_trades:
            # 슬리피지
            slippages = [t.get('slippage_bps', 0) for t in recent_trades]
            self.metrics['avg_slippage_bps'] = np.mean(slippages)
            
            # Fill rate
            filled = sum(1 for t in recent_trades if t['status'] == 'FILLED')
            self.metrics['fill_rate_pct'] = filled / len(recent_trades) * 100
            
            # Post-only 성공률
            post_only = [t for t in recent_trades if t.get('order_type') == 'POST_ONLY']
            if post_only:
                successful = sum(1 for t in post_only if t['status'] == 'FILLED')
                self.metrics['post_only_success_pct'] = successful / len(post_only) * 100
    
    def _update_system_metrics(self):
        """시스템 상태 메트릭 업데이트"""
        # TODO: 실제 시스템 메트릭 수집 구현
        pass
    
    def _update_strategy_metrics(self, portfolio_state: Dict):
        """전략 성과 메트릭 업데이트"""
        
        for strategy_id, performance in portfolio_state.get('strategy_performance', {}).items():
            if len(performance['returns']) >= 30:
                returns = np.array(performance['returns'][-30:])
                if np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                    self.metrics['strategy_sharpes'][strategy_id] = sharpe
    
    def _update_data_quality_metrics(self, market_state: Dict):
        """데이터 품질 메트릭 업데이트"""
        
        self.metrics['data_anomaly_rate_pct'] = market_state.get('anomaly_rate', 0) * 100
        self.metrics['data_staleness_seconds'] = market_state.get('data_age', 0)

## 9.2 성과 추적

class PerformanceTracker:
    """성과 추적 및 분석"""
    
    def __init__(self):
        self.daily_performance = []
        self.monthly_performance = []
        self.benchmark_returns = []
        
    def record_daily_performance(self, date: pd.Timestamp, 
                                performance: Dict):
        """일일 성과 기록"""
        
        record = {
            'date': date,
            'return': performance['daily_return'],
            'equity': performance['ending_equity'],
            'trades': performance['trade_count'],
            'commission': performance['total_commission'],
            'funding': performance['funding_pnl'],
            'sharpe': performance.get('sharpe_ratio'),
            'drawdown': performance.get('drawdown')
        }
        
        self.daily_performance.append(record)
        
        # 월간 집계
        if self._is_month_end(date):
            self._aggregate_monthly(date)
    
    def calculate_performance_metrics(self, period: str = 'YTD') -> Dict:
        """성과 메트릭 계산"""
        
        if not self.daily_performance:
            return {}
        
        # 기간 필터링
        perf_data = self._filter_by_period(period)
        
        if not perf_data:
            return {}
        
        returns = [p['return'] for p in perf_data]
        
        # 기본 메트릭
        total_return = np.prod([1 + r for r in returns]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # 리스크 메트릭
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # 드로다운
        equity_curve = [p['equity'] for p in perf_data]
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # 비용 분석
        total_commission = sum(p['commission'] for p in perf_data)
        total_funding = sum(p['funding'] for p in perf_data)
        
        return {
            'period': period,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': sum(p['trades'] for p in perf_data),
            'total_commission': total_commission,
            'total_funding': total_funding,
            'avg_daily_return': np.mean(returns),
            'return_volatility': np.std(returns) * np.sqrt(252)
        }
    
    def compare_to_benchmark(self, benchmark_symbol: str = 'BTC') -> Dict:
        """벤치마크 대비 성과"""
        
        if not self.daily_performance or not self.benchmark_returns:
            return {}
        
        strategy_returns = [p['return'] for p in self.daily_performance]
        
        # 초과 수익률
        excess_returns = [
            s - b for s, b in zip(strategy_returns, self.benchmark_returns)
        ]
        
        # Information Ratio
        if np.std(excess_returns) > 0:
            info_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        else:
            info_ratio = 0
        
        # Beta
        if len(strategy_returns) > 30:
            beta = np.cov(strategy_returns, self.benchmark_returns)[0, 1] / np.var(self.benchmark_returns)
        else:
            beta = 1
        
        # Alpha
        alpha = np.mean(strategy_returns) - beta * np.mean(self.benchmark_returns)
        
        return {
            'information_ratio': info_ratio,
            'beta': beta,
            'alpha': alpha * 252,  # 연간화
            'correlation': np.corrcoef(strategy_returns, self.benchmark_returns)[0, 1],
            'tracking_error': np.std(excess_returns) * np.sqrt(252)
        }
    
    def _filter_by_period(self, period: str) -> List[Dict]:
        """기간별 필터링"""
        
        if not self.daily_performance:
            return []
        
        today = pd.Timestamp.now()
        
        if period == 'YTD':
            start_date = pd.Timestamp(today.year, 1, 1)
        elif period == '1M':
            start_date = today - pd.Timedelta(days=30)
        elif period == '3M':
            start_date = today - pd.Timedelta(days=90)
        elif period == '1Y':
            start_date = today - pd.Timedelta(days=365)
        else:
            return self.daily_performance
        
        return [p for p in self.daily_performance if p['date'] >= start_date]
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """최대 낙폭 계산"""
        
        if not equity_curve:
            return 0
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / (running_max + 1e-10)
        
        return float(drawdowns.min())
    
    def _is_month_end(self, date: pd.Timestamp) -> bool:
        """월말 체크"""
        next_day = date + pd.Timedelta(days=1)
        return next_day.month != date.month
    
    def _aggregate_monthly(self, date: pd.Timestamp):
        """월간 집계"""
        # TODO: 구현 필요
        pass

## 9.3 알림 시스템

class AlertSystem:
    """실시간 알림 시스템"""
    
    def __init__(self):
        self.alert_channels = []
        self.alert_history = []
        self.alert_thresholds = {
            'drawdown': 0.10,
            'var_breach': 0.02,
            'liquidation_distance': 0.05,
            'api_error_rate': 0.01
        }
        
    def add_channel(self, channel):
        """알림 채널 추가"""
        self.alert_channels.append(channel)
    
    async def send_alert(self, alert: Dict):
        """알림 발송"""
        
        # 알림 기록
        alert['timestamp'] = pd.Timestamp.now()
        self.alert_history.append(alert)
        
        # 모든 채널로 발송
        for channel in self.alert_channels:
            try:
                await channel.send(alert)
            except Exception as e:
                print(f"Alert sending failed: {e}")
    
    def check_conditions(self, metrics: Dict):
        """알림 조건 체크"""
        
        alerts = []
        
        # 드로다운 체크
        if metrics.get('current_drawdown_pct', 0) > self.alert_thresholds['drawdown'] * 100:
            alerts.append({
                'level': 'CRITICAL',
                'type': 'DRAWDOWN',
                'message': f"Drawdown {metrics['current_drawdown_pct']:.1f}% exceeded threshold",
                'value': metrics['current_drawdown_pct']
            })
        
        # VaR 위반
        if metrics.get('var_daily_return', 0) > self.alert_thresholds['var_breach']:
            alerts.append({
                'level': 'WARNING',
                'type': 'VAR_BREACH',
                'message': f"VaR breach detected: {metrics['var_daily_return']:.2%}",
                'value': metrics['var_daily_return']
            })
        
        return alerts

class SlackChannel:
    """Slack 알림 채널"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    async def send(self, alert: Dict):
        """Slack으로 알림 발송"""
        import aiohttp
        
        color = {
            'CRITICAL': 'danger',
            'WARNING': 'warning',
            'INFO': 'good'
        }.get(alert['level'], 'good')
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f"{alert['level']}: {alert['type']}",
                'text': alert['message'],
                'ts': alert['timestamp'].timestamp()
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook_url, json=payload) as response:
                return response.status == 200
```