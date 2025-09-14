# 코인 선물 자동매매 시스템 - 포트폴리오 최적화

## 5.1 Markowitz 최적화

```python
from scipy.optimize import minimize

class PortfolioOptimizer:
    """거래 비용과 제약을 고려한 포트폴리오 최적화"""
    
    def __init__(self, transaction_cost: float = 0.0004):
        self.transaction_cost = transaction_cost  # 0.04% taker fee
        self.use_shrinkage = True
        
    def optimize_weights(self, returns_df: pd.DataFrame, 
                        current_weights: Optional[np.ndarray] = None, 
                        constraints: Optional[Dict] = None) -> np.ndarray:
        """
        거래 비용을 고려한 포트폴리오 최적화
        
        Args:
            returns_df: DataFrame of asset returns
            current_weights: Current portfolio weights (for rebalancing cost)
            constraints: Additional constraints dict
            
        Returns:
            np.array: Optimal weights
        """
        
        n_assets = len(returns_df.columns)
        
        # 공분산 행렬 추정 (Ledoit-Wolf shrinkage)
        if self.use_shrinkage:
            cov_matrix = self._ledoit_wolf_covariance(returns_df)
        else:
            cov_matrix = returns_df.cov().values
        
        # 기대 수익률
        expected_returns = returns_df.mean().values
        
        # 현재 가중치 (리밸런싱 비용 계산용)
        if current_weights is None:
            current_weights = np.zeros(n_assets)
        
        # 목적 함수: Sharpe - 거래비용
        def objective(weights):
            portfolio_return = np.dot(expected_returns, weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # 거래 비용 (turnover)
            turnover = np.sum(np.abs(weights - current_weights))
            transaction_cost = turnover * self.transaction_cost
            
            # 비용 조정 Sharpe
            adjusted_return = portfolio_return - transaction_cost
            
            if portfolio_std > 1e-10:
                return -adjusted_return / portfolio_std  # 최대화를 위해 음수
            else:
                return 999999
        
        # 제약 조건
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # 가중치 합 = 1
        ]
        
        # 추가 제약 (레버리지, 섹터 등)
        if constraints:
            if 'max_leverage' in constraints:
                max_lev = constraints['max_leverage']
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w: max_lev - np.sum(np.abs(w))
                })
            
            if 'max_position' in constraints:
                max_pos = constraints['max_position']
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w: max_pos - np.max(np.abs(w))
                })
        
        # 경계 조건 (롱숏 허용, 개별 20% 제한)
        bounds = [(-0.2, 0.2) for _ in range(n_assets)]
        
        # 초기 추정치
        x0 = current_weights if np.sum(current_weights) > 0 else np.ones(n_assets) / n_assets
        
        # 최적화 실행
        result = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            # 최적화 실패 시 균등 가중
            return np.ones(n_assets) / n_assets
        
        return result.x
    
    def _ledoit_wolf_covariance(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf shrinkage covariance estimation"""
        
        returns = returns_df.values
        n, p = returns.shape
        
        # 표본 공분산
        sample_cov = np.cov(returns.T)
        
        # Shrinkage target: 대각 행렬
        mu = np.trace(sample_cov) / p
        target = np.eye(p) * mu
        
        # Ledoit-Wolf optimal shrinkage intensity
        y = returns - returns.mean(axis=0)
        y2 = y**2
        
        # 더 안정적인 계산
        phi_mat = (1/n) * np.dot(y2.T, y2) - sample_cov**2
        phi = np.sum(phi_mat)
        
        gamma = np.linalg.norm(sample_cov - target, 'fro')**2
        
        # Shrinkage intensity
        kappa = phi / gamma if gamma > 1e-10 else 0
        shrinkage = max(0, min(1, kappa/n))
        
        # Shrunk covariance
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target
        
        return shrunk_cov

## 5.2 Black-Litterman 통합

class BlackLittermanOptimizer:
    """전략 신호를 통합한 Black-Litterman 최적화"""
    
    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        self.risk_aversion = risk_aversion
        self.tau = tau  # 불확실성 스케일
        
    def optimize_with_views(self, market_data: Dict, 
                           strategy_signals: List[Dict]) -> np.ndarray:
        """
        Black-Litterman 모델로 전략 신호 통합
        
        Args:
            market_data: 시장 데이터 (returns, market_caps)
            strategy_signals: 전략별 신호
            
        Returns:
            np.array: BL 최적 가중치
        """
        
        returns = market_data['returns']
        market_caps = market_data['market_caps']
        
        # 시장 균형 가중치
        w_market = market_caps / market_caps.sum()
        
        # 공분산 행렬
        cov_matrix = returns.cov().values
        
        # 균형 기대수익률 (CAPM)
        pi = self.risk_aversion * cov_matrix @ w_market
        
        # 전략 신호를 Views로 변환
        P, Q, omega = self._construct_views(strategy_signals, returns, cov_matrix)
        
        if P is None:
            # Views가 없으면 시장 균형 사용
            return w_market
        
        # Black-Litterman 사후 분포
        tau_cov = self.tau * cov_matrix
        
        # 사후 기대수익률
        A = np.linalg.inv(tau_cov) @ pi
        B = P.T @ np.linalg.inv(omega) @ Q
        C = np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(omega) @ P
        
        mu_bl = np.linalg.inv(C) @ (A + B)
        
        # 사후 공분산
        cov_bl = cov_matrix + tau_cov - tau_cov @ P.T @ np.linalg.inv(
            P @ tau_cov @ P.T + omega
        ) @ P @ tau_cov
        
        # 최적 가중치 (사후 공분산 사용)
        w_optimal = (1 / self.risk_aversion) * np.linalg.inv(cov_bl) @ mu_bl
        
        # 제약 적용
        w_optimal = self._apply_constraints(w_optimal)
        
        return w_optimal
    
    def _construct_views(self, signals: List[Dict], 
                        returns: pd.DataFrame, 
                        cov_matrix: np.ndarray) -> Tuple:
        """전략 신호를 BL Views로 변환"""
        
        if not signals:
            return None, None, None
        
        n_assets = len(returns.columns)
        views = []
        
        for signal in signals:
            if signal['confidence'] > 0.6:  # 확신도 높은 신호만
                view = {
                    'assets': signal['assets'],
                    'weights': signal['weights'],
                    'return': signal['expected_return'],
                    'confidence': signal['confidence']
                }
                views.append(view)
        
        if not views:
            return None, None, None
        
        n_views = len(views)
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        confidences = []
        
        asset_names = list(returns.columns)
        
        for i, view in enumerate(views):
            # P 행렬 구성
            for asset, weight in zip(view['assets'], view['weights']):
                if asset in asset_names:
                    j = asset_names.index(asset)
                    P[i, j] = weight
            
            # 기대수익률
            Q[i] = view['return']
            confidences.append(view['confidence'])
        
        # Omega 구성 (시장 불확실성 스케일)
        tau_cov = self.tau * cov_matrix
        
        # View 불확실성 = 시장 불확실성 × 신뢰도 조정
        omega_diag = []
        for i in range(n_views):
            view_variance = P[i] @ tau_cov @ P[i].T
            confidence_adj = (1 - confidences[i]) / max(1e-3, confidences[i])
            omega_diag.append(view_variance * np.clip(confidence_adj, 0.01, 100))
        
        omega = np.diag(omega_diag)
        
        return P, Q, omega
    
    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """가중치 제약 적용"""
        
        # 개별 포지션 한도 (-20% ~ 20%)
        weights = np.clip(weights, -0.2, 0.2)
        
        # 총 레버리지 제약
        total_leverage = np.sum(np.abs(weights))
        if total_leverage > 2.0:  # 최대 2배 레버리지
            weights = weights * (2.0 / total_leverage)
        
        # 정규화 (합 = 1)
        weight_sum = np.sum(weights)
        if abs(weight_sum) > 1e-10:
            weights = weights / weight_sum
        
        return weights

## 5.3 리스크 패리티

class RiskParityOptimizer:
    """리스크 패리티 포트폴리오 최적화"""
    
    def __init__(self):
        self.max_iterations = 100
        self.tolerance = 1e-8
        
    def optimize(self, cov_matrix: np.ndarray, 
                target_risk_contributions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        리스크 패리티 가중치 계산
        
        Args:
            cov_matrix: 공분산 행렬
            target_risk_contributions: 목표 리스크 기여도 (None이면 균등)
            
        Returns:
            np.array: 최적 가중치
        """
        
        n_assets = len(cov_matrix)
        
        # 목표 리스크 기여도
        if target_risk_contributions is None:
            target_risk_contributions = np.ones(n_assets) / n_assets
        
        # 초기 가중치
        weights = np.ones(n_assets) / n_assets
        
        # Newton-Raphson 반복
        for iteration in range(self.max_iterations):
            # 포트폴리오 변동성
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            
            # Marginal risk contributions
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            
            # Risk contributions
            risk_contrib = weights * marginal_contrib
            risk_contrib = risk_contrib / risk_contrib.sum()
            
            # Check convergence
            if np.max(np.abs(risk_contrib - target_risk_contributions)) < self.tolerance:
                break
            
            # Newton step
            jacobian = self._calculate_jacobian(weights, cov_matrix)
            delta = np.linalg.solve(jacobian, target_risk_contributions - risk_contrib)
            
            # Update weights
            weights = weights + 0.5 * delta  # Step size = 0.5
            weights = np.maximum(weights, 1e-6)  # Ensure positive
            weights = weights / weights.sum()  # Normalize
        
        return weights
    
    def _calculate_jacobian(self, weights: np.ndarray, 
                           cov_matrix: np.ndarray) -> np.ndarray:
        """Jacobian 행렬 계산"""
        
        n = len(weights)
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        marginal_contrib = cov_matrix @ weights
        
        jacobian = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    jacobian[i, j] = marginal_contrib[i] / portfolio_vol - \
                                     weights[i] * marginal_contrib[i]**2 / portfolio_var**1.5
                else:
                    jacobian[i, j] = -weights[i] * marginal_contrib[i] * \
                                     marginal_contrib[j] / portfolio_var**1.5
        
        return jacobian

## 5.4 동적 리밸런싱

class DynamicRebalancer:
    """동적 포트폴리오 리밸런싱"""
    
    def __init__(self, optimizer: PortfolioOptimizer):
        self.optimizer = optimizer
        self.rebalance_threshold = 0.05  # 5% deviation
        self.min_rebalance_interval = 86400  # 24 hours
        self.last_rebalance_time = None
        
    def should_rebalance(self, current_weights: np.ndarray, 
                        target_weights: np.ndarray,
                        current_time: pd.Timestamp) -> bool:
        """리밸런싱 필요 여부 판단"""
        
        # 시간 체크
        if self.last_rebalance_time:
            time_since_last = (current_time - self.last_rebalance_time).total_seconds()
            if time_since_last < self.min_rebalance_interval:
                return False
        
        # 편차 체크
        weight_deviations = np.abs(current_weights - target_weights)
        max_deviation = np.max(weight_deviations)
        
        return max_deviation > self.rebalance_threshold
    
    def calculate_rebalance_trades(self, current_positions: Dict,
                                  target_weights: np.ndarray,
                                  portfolio_value: float) -> List[Dict]:
        """리밸런싱 거래 계산"""
        
        trades = []
        
        for symbol, target_weight in zip(current_positions.keys(), target_weights):
            current_value = current_positions[symbol]['value']
            current_weight = current_value / portfolio_value
            
            target_value = portfolio_value * target_weight
            trade_value = target_value - current_value
            
            if abs(trade_value) > 10:  # Minimum trade size
                trades.append({
                    'symbol': symbol,
                    'side': 'BUY' if trade_value > 0 else 'SELL',
                    'value': abs(trade_value),
                    'urgency': 'LOW'
                })
        
        return trades
```