# 코인 선물 자동매매 시스템 - 백테스트 및 시뮬레이션

## 8.1 Walk-Forward 최적화

```python
class WalkForwardOptimizer:
    """과적합 방지를 위한 Walk-Forward 분석"""
    
    def __init__(self):
        self.in_sample_period = 60  # days
        self.out_sample_period = 20  # days
        self.min_walks = 10
        
    def run_walk_forward(self, data: pd.DataFrame, 
                         strategy_class: type, 
                         param_grid: Dict) -> Dict:
        """
        Walk-Forward 최적화 및 검증
        
        Args:
            data: 전체 히스토리 데이터
            strategy_class: 전략 클래스
            param_grid: 파라미터 검색 공간
            
        Returns:
            dict: Walk-forward 결과
        """
        
        total_days = len(data)
        walk_size = self.out_sample_period
        n_walks = (total_days - self.in_sample_period) // walk_size
        
        if n_walks < self.min_walks:
            raise ValueError(f"데이터 부족: 최소 {self.min_walks} walks 필요")
        
        results = []
        
        for walk_idx in range(n_walks):
            # 데이터 분할
            in_start = walk_idx * walk_size
            in_end = in_start + self.in_sample_period
            out_start = in_end
            out_end = out_start + self.out_sample_period
            
            if out_end > total_days:
                break
            
            in_sample_data = data.iloc[in_start:in_end]
            out_sample_data = data.iloc[out_start:out_end]
            
            # In-sample 최적화
            best_params, in_sample_perf = self._optimize_parameters(
                in_sample_data,
                strategy_class,
                param_grid
            )
            
            # Out-of-sample 검증
            out_sample_perf = self._evaluate_strategy(
                out_sample_data,
                strategy_class,
                best_params
            )
            
            # Walk 결과 저장
            walk_result = {
                'walk': walk_idx,
                'in_sample_period': (in_start, in_end),
                'out_sample_period': (out_start, out_end),
                'best_params': best_params,
                'in_sample_sharpe': in_sample_perf['sharpe'],
                'out_sample_sharpe': out_sample_perf['sharpe'],
                'in_sample_return': in_sample_perf['total_return'],
                'out_sample_return': out_sample_perf['total_return'],
                'degradation': in_sample_perf['sharpe'] - out_sample_perf['sharpe']
            }
            results.append(walk_result)
        
        # 종합 분석
        analysis = self._analyze_walk_forward_results(results)
        
        return {
            'walks': results,
            'analysis': analysis,
            'is_robust': analysis['is_robust']
        }
    
    def _optimize_parameters(self, data: pd.DataFrame, 
                            strategy_class: type, 
                            param_grid: Dict) -> Tuple:
        """In-sample 파라미터 최적화"""
        
        from itertools import product
        
        best_sharpe = -float('inf')
        best_params = None
        best_performance = None
        
        # Grid search
        param_combinations = list(product(*param_grid.values()))
        param_keys = list(param_grid.keys())
        
        for param_values in param_combinations:
            params = dict(zip(param_keys, param_values))
            
            # 전략 실행
            strategy = strategy_class(**params)
            performance = self._backtest_strategy(strategy, data)
            
            if performance['sharpe'] > best_sharpe:
                best_sharpe = performance['sharpe']
                best_params = params
                best_performance = performance
        
        return best_params, best_performance
    
    def _evaluate_strategy(self, data: pd.DataFrame, 
                          strategy_class: type, 
                          params: Dict) -> Dict:
        """Out-of-sample 전략 평가"""
        
        strategy = strategy_class(**params)
        return self._backtest_strategy(strategy, data)
    
    def _backtest_strategy(self, strategy, data: pd.DataFrame) -> Dict:
        """전략 백테스트 (룩어헤드 방지)"""
        
        equity_curve = [100000]  # 초기 자본 (USDT)
        returns = []
        
        for i in range(1, len(data)):
            # i-1까지의 데이터만 사용
            historical_data = data.iloc[:i]
            
            # 신호 생성
            signal = strategy.generate_signal(historical_data)
            
            if signal is not None:
                # 포지션 크기
                position_size = equity_curve[-1] * 0.02
                
                # 수익률 계산
                entry_price = data.iloc[i]['open']
                exit_price = data.iloc[i]['close']
                
                if signal['side'] == 'BUY':
                    gross_return = (exit_price - entry_price) / entry_price
                else:
                    gross_return = (entry_price - exit_price) / entry_price
                
                # 비용 차감
                slippage = 0.0005
                commission = 0.0004
                net_return = gross_return - slippage - commission
                
                # 자본 업데이트
                pnl = position_size * net_return
                equity_curve.append(equity_curve[-1] + pnl)
                returns.append(net_return)
            else:
                equity_curve.append(equity_curve[-1])
                returns.append(0)
        
        # 성과 메트릭
        returns_array = np.array(returns)
        
        if len(returns_array) > 0 and np.std(returns_array) > 0:
            sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe = 0
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'returns': returns_array,
            'equity_curve': equity_curve
        }
    
    def _analyze_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Walk-forward 결과 종합 분석"""
        
        if not results:
            return {'is_robust': False}
        
        # 성과 저하 분석
        degradations = [r['degradation'] for r in results]
        avg_degradation = np.mean(degradations)
        std_degradation = np.std(degradations)
        
        # 파라미터 안정성 분석
        param_stability = self._analyze_parameter_stability(results)
        
        # Out-of-sample 일관성
        out_sharpes = [r['out_sample_sharpe'] for r in results]
        out_consistency = 1 - (np.std(out_sharpes) / (abs(np.mean(out_sharpes)) + 1e-6))
        
        # Robust 판정
        is_robust = (
            avg_degradation < 0.3 and
            out_consistency > 0.5 and
            param_stability > 0.6
        )
        
        return {
            'avg_degradation': avg_degradation,
            'std_degradation': std_degradation,
            'out_sample_consistency': out_consistency,
            'parameter_stability': param_stability,
            'avg_in_sample_sharpe': np.mean([r['in_sample_sharpe'] for r in results]),
            'avg_out_sample_sharpe': np.mean(out_sharpes),
            'is_robust': is_robust
        }
    
    def _analyze_parameter_stability(self, results: List[Dict]) -> float:
        """파라미터 안정성 분석"""
        
        if len(results) < 2:
            return 0
        
        param_series = {}
        
        for result in results:
            for param_name, param_value in result['best_params'].items():
                if param_name not in param_series:
                    param_series[param_name] = []
                param_series[param_name].append(param_value)
        
        stability_scores = []
        
        for param_name, values in param_series.items():
            if isinstance(values[0], (int, float)):
                mean_val = np.mean(values)
                std_val = np.std(values)
                if abs(mean_val) > 1e-10:
                    cv = std_val / abs(mean_val)
                    stability = 1 / (1 + cv)
                    stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0

## 8.2 Monte Carlo 시뮬레이션

class MonteCarloSimulator:
    """포트폴리오 시나리오 시뮬레이션"""
    
    def __init__(self, initial_capital_usdt: float):
        self.initial_capital = initial_capital_usdt
        
    def run_simulation(self, strategy_returns: np.ndarray, 
                      n_paths: int = 10000, 
                      horizon: int = 252) -> Dict:
        """
        Monte Carlo 시뮬레이션 실행
        
        Args:
            strategy_returns: 전략 수익률 히스토리
            n_paths: 시뮬레이션 경로 수
            horizon: 예측 기간 (일)
            
        Returns:
            dict: 시뮬레이션 결과
        """
        
        # 수익률 분포 파라미터 추정
        mu = np.mean(strategy_returns)
        sigma = np.std(strategy_returns)
        
        # 고차 모멘트
        skew = stats.skew(strategy_returns)
        kurt = stats.kurtosis(strategy_returns)
        
        # 시뮬레이션 결과 저장
        final_values = []
        max_drawdowns = []
        liquidation_count = 0
        var_breach_counts = []
        path_sharpes = []
        
        # 난수 생성기
        rng = np.random.default_rng()
        
        for path_idx in range(n_paths):
            # 경로 생성 (t-분포 사용)
            if kurt > 3:  # Leptokurtic
                df = 6  # 자유도
                path_returns = stats.t.rvs(df, loc=mu, scale=sigma, 
                                          size=horizon, random_state=rng)
            else:
                path_returns = rng.normal(mu, sigma, horizon)
            
            # GARCH 효과 추가
            path_returns = self._add_garch_effects(path_returns)
            
            # 점프 추가
            path_returns = self._add_jumps(path_returns, rng)
            
            # 자본 경로
            equity_path = self.initial_capital * np.cumprod(1 + path_returns)
            
            # 메트릭 계산
            final_values.append(equity_path[-1])
            
            # 최대 낙폭
            running_max = np.maximum.accumulate(equity_path)
            drawdowns = (equity_path - running_max) / (running_max + 1e-10)
            max_dd = drawdowns.min()
            max_drawdowns.append(max_dd)
            
            # 청산 체크
            if equity_path.min() < self.initial_capital * 0.5:
                liquidation_count += 1
            
            # VaR 위반 횟수
            daily_returns = np.diff(equity_path) / (equity_path[:-1] + 1e-10)
            var_threshold = -0.05
            var_breaches = sum(daily_returns < var_threshold)
            var_breach_counts.append(var_breaches)
            
            # 경로별 샤프
            if np.std(path_returns) > 0:
                path_sharpe = np.mean(path_returns) / np.std(path_returns) * np.sqrt(252)
                path_sharpes.append(path_sharpe)
        
        # 결과 집계
        final_values = np.array(final_values)
        max_drawdowns = np.array(max_drawdowns)
        
        return {
            'expected_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'percentiles': {
                '5%': np.percentile(final_values, 5),
                '25%': np.percentile(final_values, 25),
                '75%': np.percentile(final_values, 75),
                '95%': np.percentile(final_values, 95)
            },
            'probability_of_profit': sum(final_values > self.initial_capital) / n_paths,
            'probability_of_liquidation': liquidation_count / n_paths,
            'expected_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown': np.min(max_drawdowns),
            'avg_var_breaches': np.mean(var_breach_counts),
            'sharpe_distribution': self._calculate_sharpe_distribution(path_sharpes)
        }
    
    def _add_garch_effects(self, base_returns: np.ndarray) -> np.ndarray:
        """GARCH(1,1) 변동성 클러스터링 추가"""
        
        n = len(base_returns)
        omega = 1e-6
        alpha = 0.05
        beta = 0.90
        
        sigma2 = np.zeros(n)
        adjusted_returns = np.zeros(n)
        
        sigma2[0] = np.var(base_returns)
        adjusted_returns[0] = base_returns[0]
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * (adjusted_returns[t-1]**2) + beta * sigma2[t-1]
            
            base_vol = np.var(base_returns)
            if base_vol > 0:
                adjusted_returns[t] = base_returns[t] * np.sqrt(sigma2[t] / base_vol)
            else:
                adjusted_returns[t] = base_returns[t]
        
        return adjusted_returns
    
    def _add_jumps(self, returns: np.ndarray, rng) -> np.ndarray:
        """Merton jump diffusion 추가"""
        
        n = len(returns)
        jump_prob = 0.01
        jump_size_mean = 0
        jump_size_std = 0.05
        
        jumps = rng.binomial(1, jump_prob, n)
        jump_sizes = rng.normal(jump_size_mean, jump_size_std, n)
        
        return returns + jumps * jump_sizes
    
    def _calculate_sharpe_distribution(self, path_sharpes: List[float]) -> Dict:
        """Sharpe ratio 분포 계산"""
        
        if not path_sharpes:
            return None
        
        sharpes = np.array(path_sharpes)
        
        return {
            'mean': float(np.mean(sharpes)),
            'std': float(np.std(sharpes)),
            'percentiles': {
                '25%': float(np.percentile(sharpes, 25)),
                '50%': float(np.percentile(sharpes, 50)),
                '75%': float(np.percentile(sharpes, 75))
            }
        }

## 8.3 현실적 백테스트 엔진

class RealisticBacktester:
    """거래소 특성과 실제 비용을 반영한 백테스터"""
    
    def __init__(self, initial_capital_usdt: float, exchange: str = 'binance'):
        self.initial_capital = initial_capital_usdt
        self.exchange = exchange
        
        # 거래소별 설정
        self.fee_structure = {
            'maker': 0.0002,  # 0.02%
            'taker': 0.0004   # 0.04%
        }
        
        # 시뮬레이션 설정
        self.latency_mean = 50  # ms
        self.latency_std = 20   # ms
        self.api_failure_rate = 0.001
        
    def backtest(self, strategy, market_data: pd.DataFrame, 
                start_date, end_date) -> Dict:
        """
        현실적 백테스트 실행 (룩어헤드 방지)
        
        Returns:
            dict: 백테스트 결과
        """
        
        # 초기화
        portfolio = {
            'cash': self.initial_capital,  # USDT
            'positions': {},
            'equity_curve': [self.initial_capital],
            'trades': [],
            'returns': []
        }
        
        # 데이터 필터링
        test_data = market_data[
            (market_data.index >= start_date) & 
            (market_data.index <= end_date)
        ]
        
        for i in range(len(test_data)):
            current_time = test_data.index[i]
            
            # 현재 시점까지의 데이터만 사용
            historical_data = test_data.iloc[:i+1]
            
            if len(historical_data) < strategy.min_history:
                portfolio['equity_curve'].append(portfolio['equity_curve'][-1])
                continue
            
            # 시장 상태 스냅샷
            market_snapshot = self._create_market_snapshot(
                historical_data, current_time
            )
            
            # 포지션 업데이트 (마킹)
            self._update_positions(portfolio, market_snapshot)
            
            # 청산 체크
            liquidation = self._check_liquidation(portfolio, market_snapshot)
            if liquidation:
                self._handle_liquidation(portfolio, liquidation, current_time)
                continue
            
            # 펀딩 처리
            if self._is_funding_time(current_time):
                self._process_funding(portfolio, market_snapshot)
            
            # 전략 신호 생성
            signals = strategy.generate_signals(historical_data, portfolio)
            
            # 신호 실행
            for signal in signals:
                # API 실패 시뮬레이션
                if np.random.random() < self.api_failure_rate:
                    continue
                
                # 지연 시뮬레이션
                latency = max(0, np.random.normal(self.latency_mean, self.latency_std))
                
                # 가격 드리프트
                price_drift = self._simulate_price_drift(
                    market_snapshot['volatility'], latency / 1000
                )
                
                # 주문 실행
                execution = self._execute_order(
                    signal, market_snapshot, price_drift, portfolio
                )
                
                if execution['status'] == 'FILLED':
                    portfolio['trades'].append(execution)
                    self._update_portfolio(portfolio, execution)
            
            # 자본 기록
            total_equity = self._calculate_total_equity(portfolio, market_snapshot)
            portfolio['equity_curve'].append(total_equity)
            
            # 일일 수익률 기록
            if len(portfolio['equity_curve']) > 1:
                daily_return = (
                    (portfolio['equity_curve'][-1] - portfolio['equity_curve'][-2]) / 
                    portfolio['equity_curve'][-2]
                )
            else:
                daily_return = 0
            
            portfolio['returns'].append(daily_return)
        
        # 최종 성과 계산
        performance = self._calculate_performance(portfolio)
        
        return {
            'performance': performance,
            'portfolio': portfolio,
            'equity_curve': portfolio['equity_curve'],
            'trades': portfolio['trades']
        }
    
    def _create_market_snapshot(self, data: pd.DataFrame, 
                               current_time) -> Dict:
        """현재 시장 상태 스냅샷"""
        
        current = data.iloc[-1]
        
        # 변동성 (20일)
        if len(data) >= 20:
            returns = data['close'].pct_change()
            volatility = returns.tail(20).std() * np.sqrt(252)
        else:
            volatility = 0.05
        
        # ATR
        if len(data) >= 14:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
        else:
            atr = current['high'] - current['low']
        
        return {
            'time': current_time,
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'close': current['close'],
            'volume': current['volume'],
            'volatility': volatility,
            'atr': atr,
            'funding_rate': current.get('funding_rate', 0)
        }
    
    def _check_liquidation(self, portfolio: Dict, 
                          market_snapshot: Dict) -> Optional[Dict]:
        """청산 체크"""
        
        for symbol, position in portfolio['positions'].items():
            if position['size'] == 0:
                continue
            
            current_price = market_snapshot['close']
            
            # 증거금률 계산
            if position['side'] == 'LONG':
                pnl = (current_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - current_price) * position['size']
            
            margin_ratio = (position['margin'] + pnl) / (position['size'] * current_price)
            
            # 유지증거금률 체크
            if margin_ratio < 0.004:
                return {
                    'symbol': symbol,
                    'position': position,
                    'liquidation_price': current_price
                }
        
        return None
    
    def _execute_order(self, signal: Dict, market_snapshot: Dict, 
                      price_drift: float, portfolio: Dict) -> Dict:
        """주문 실행 시뮬레이션"""
        
        # 실행 가격
        if signal.get('order_type', 'MARKET') == 'MARKET':
            if signal['side'] == 'BUY':
                exec_price = market_snapshot['close'] * (1 + price_drift + 0.0001)
            else:
                exec_price = market_snapshot['close'] * (1 - price_drift - 0.0001)
            
            fee_rate = self.fee_structure['taker']
        else:
            exec_price = signal['price']
            fee_rate = self.fee_structure['maker']
        
        # 자금 체크
        required_margin = exec_price * signal['size'] / signal.get('leverage', 1)
        if required_margin > portfolio['cash']:
            return {'status': 'REJECTED', 'reason': 'Insufficient margin'}
        
        # 체결
        notional = exec_price * signal['size']
        commission = notional * fee_rate
        
        return {
            'status': 'FILLED',
            'symbol': signal['symbol'],
            'side': signal['side'],
            'size': signal['size'],
            'price': exec_price,
            'notional': notional,
            'commission': commission,
            'time': market_snapshot['time']
        }
    
    def _calculate_performance(self, portfolio: Dict) -> Dict:
        """성과 메트릭 계산"""
        
        returns = np.array(portfolio['returns'])
        equity_curve = np.array(portfolio['equity_curve'])
        
        # 기본 메트릭
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Sharpe Ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.sqrt(np.mean(downside_returns**2))
            if downside_std > 0:
                sortino = np.mean(returns) / downside_std * np.sqrt(252)
            else:
                sortino = float('inf') if np.mean(returns) > 0 else 0
        else:
            sortino = float('inf') if np.mean(returns) > 0 else 0
        
        # Maximum Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / (running_max + 1e-10)
        max_drawdown = drawdowns.min()
        
        # Calmar Ratio
        if max_drawdown != 0:
            calmar = total_return / abs(max_drawdown)
        else:
            calmar = float('inf') if total_return > 0 else 0
        
        # Win Rate
        trades = portfolio['trades']
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / len(trades)
        else:
            win_rate = 0
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / max(1, len(returns))) - 1,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_trade': np.mean([t.get('pnl', 0) for t in trades]) if trades else 0
        }
    
    def _calculate_total_equity(self, portfolio: Dict, 
                               market_snapshot: Dict) -> float:
        """총 자본 계산 (USDT)"""
        total = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            if position['size'] > 0:
                mark_price = market_snapshot['close']
                if position['side'] == 'LONG':
                    value = position['size'] * mark_price
                else:
                    value = position['margin'] + (
                        position['entry_price'] - mark_price
                    ) * position['size']
                total += value
        
        return total
    
    def _update_positions(self, portfolio: Dict, market_snapshot: Dict):
        """포지션 마크투마켓"""
        for symbol, position in portfolio['positions'].items():
            position['current_price'] = market_snapshot['close']
            
            if position['side'] == 'LONG':
                position['unrealized_pnl'] = (
                    (market_snapshot['close'] - position['entry_price']) * 
                    position['size']
                )
            else:
                position['unrealized_pnl'] = (
                    (position['entry_price'] - market_snapshot['close']) * 
                    position['size']
                )
    
    def _handle_liquidation(self, portfolio: Dict, liquidation: Dict, 
                          current_time):
        """청산 처리"""
        symbol = liquidation['symbol']
        position = liquidation['position']
        
        # 포지션 청산
        portfolio['cash'] = 0  # 증거금 손실
        del portfolio['positions'][symbol]
        
        # 청산 기록
        portfolio['trades'].append({
            'type': 'LIQUIDATION',
            'symbol': symbol,
            'time': current_time,
            'loss': position['margin']
        })
    
    def _is_funding_time(self, current_time) -> bool:
        """펀딩 시간 체크"""
        hour = current_time.hour
        return hour in [0, 8, 16]  # UTC 기준
    
    def _process_funding(self, portfolio: Dict, market_snapshot: Dict):
        """펀딩 처리"""
        funding_rate = market_snapshot.get('funding_rate', 0)
        
        for position in portfolio['positions'].values():
            notional = position['size'] * market_snapshot['close']
            
            if position['side'] == 'LONG':
                funding_payment = -notional * funding_rate
            else:
                funding_payment = notional * funding_rate
            
            portfolio['cash'] += funding_payment
    
    def _simulate_price_drift(self, volatility: float, 
                            time_seconds: float) -> float:
        """지연 동안 가격 드리프트"""
        return np.random.normal(0, volatility * np.sqrt(time_seconds / 86400))
    
    def _update_portfolio(self, portfolio: Dict, execution: Dict):
        """포트폴리오 업데이트"""
        symbol = execution['symbol']
        
        # 현금 차감
        portfolio['cash'] -= execution['commission']
        
        # 포지션 업데이트
        if symbol not in portfolio['positions']:
            portfolio['positions'][symbol] = {
                'side': execution['side'],
                'size': 0,
                'entry_price': 0,
                'margin': 0
            }
        
        position = portfolio['positions'][symbol]
        
        # 평균 진입가 계산
        if position['size'] == 0:
            position['entry_price'] = execution['price']
        else:
            total_value = position['size'] * position['entry_price'] + \
                         execution['size'] * execution['price']
            position['size'] += execution['size']
            position['entry_price'] = total_value / position['size']
        
        position['size'] += execution['size']
        position['margin'] += execution['notional'] / execution.get('leverage', 1)
```