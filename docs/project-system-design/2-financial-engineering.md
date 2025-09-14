# 코인 선물 자동매매 시스템 - 금융공학 프레임워크

## 2.1 Kelly Criterion 구현

```python
import numpy as np
import pandas as pd
import time
import asyncio
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler

class ContinuousKellyOptimizer:
    """연속 수익률용 Kelly 최적화"""
    
    def __init__(self, fractional: float = 0.25, allow_short: bool = False):
        self.fractional = fractional  # Fractional Kelly (1/4)
        self.ema_alpha = 0.2  # EMA decay parameter
        self.min_samples = 30  # 최소 샘플 수
        self.allow_short = allow_short  # 숏 포지션 허용 여부
        
    def _calculate_ema_weights(self, n: int) -> np.ndarray:
        """EMA 가중치 계산 (최근 데이터 높은 가중치)"""
        idx = np.arange(n)
        weights = np.exp(-self.ema_alpha * idx)
        return weights / weights.sum()
    
    def _calculate_weighted_variance(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """가중 분산 with proper correction"""
        weighted_mean = np.average(returns, weights=weights)
        
        # Effective sample size for weighted samples
        eff_n = (weights.sum()**2) / (weights**2).sum()
        
        # Weighted variance
        variance = np.average((returns - weighted_mean)**2, weights=weights)
        
        # Bessel-like correction for weighted samples
        if eff_n > 1:
            variance = variance * eff_n / (eff_n - 1)
        
        return float(variance)
    
    def calculate_optimal_fraction(self, returns_array: np.ndarray, 
                                  regime: str = 'NEUTRAL') -> float:
        """
        연속 수익률 Kelly 공식 적용
        
        Args:
            returns_array: numpy array of portfolio returns (decimal)
            regime: 현재 시장 레짐
            
        Returns:
            float: Optimal betting fraction
        """
        returns = np.asarray(returns_array)
        
        if len(returns) < self.min_samples:
            return 0.0  # 데이터 부족 시 0
        
        # EMA 가중치
        weights = self._calculate_ema_weights(len(returns))
        
        # 가중 평균
        mu = np.average(returns, weights=weights)
        
        # 가중 분산 (proper correction)
        variance = self._calculate_weighted_variance(returns, weights)
        
        if variance < 1e-10:
            return 0.0  # 분산이 너무 작으면 0
        
        # 연속 수익률 Kelly 공식: f* = μ/σ²
        kelly_fraction = mu / variance
        
        # 파라미터 추정 불확실성 보정 (Stein-type shrinkage)
        effective_samples = (weights.sum()**2) / (weights**2).sum()
        shrinkage = max(0.0, 1.0 - 2.0/effective_samples)
        kelly_fraction *= shrinkage
        
        # Fractional Kelly 적용
        kelly_fraction *= self.fractional
        
        # 레짐별 상한 적용
        if self.allow_short:
            regime_caps = {
                'BULL': (0.15, -0.05),    # 롱 선호
                'BEAR': (0.05, -0.15),    # 숏 선호
                'SIDEWAYS': (0.10, -0.10),
                'NEUTRAL': (0.08, -0.08)
            }
            cap_long, cap_short = regime_caps.get(regime, (0.08, -0.08))
            
            if kelly_fraction >= 0:
                return float(np.clip(kelly_fraction, 0.0, cap_long))
            else:
                return float(np.clip(kelly_fraction, cap_short, 0.0))
        else:
            # 롱 온리
            regime_caps = {'BULL': 0.15, 'BEAR': 0.05, 'SIDEWAYS': 0.10, 'NEUTRAL': 0.08}
            cap = regime_caps.get(regime, 0.08)
            return float(np.clip(max(0.0, kelly_fraction), 0.0, cap))
    
    def calculate_with_constraints(self, returns: np.ndarray, 
                                  current_leverage: float, 
                                  max_leverage: float = 10) -> float:
        """레버리지 제약을 고려한 Kelly 계산"""
        base_fraction = self.calculate_optimal_fraction(returns)
        
        # 현재 레버리지 고려
        remaining_leverage = max(0, max_leverage - current_leverage)
        leverage_constraint = remaining_leverage / max_leverage
        
        return base_fraction * min(1.0, leverage_constraint)
```

## 2.2 리스크 메트릭 체계

```python
class RiskMetrics:
    """단위 일관성이 보장된 리스크 메트릭 (USDT 기준)"""
    
    def __init__(self, returns_series: np.ndarray, portfolio_value_usdt: float):
        """
        Args:
            returns_series: Series/array of returns (decimal, e.g., 0.01 = 1%)
            portfolio_value_usdt: Current portfolio value (USDT)
        """
        self.returns = np.asarray(returns_series, dtype=float)
        self.portfolio_value = float(portfolio_value_usdt)
        
    def calculate_var_return(self, confidence: float = 0.95) -> float:
        """
        Value at Risk in return terms
        
        Returns:
            float: VaR as return (negative value for loss)
        """
        quantile = 1 - confidence
        var_return = np.percentile(self.returns, quantile * 100)
        return float(var_return)
    
    def calculate_cvar_return(self, confidence: float = 0.95) -> float:
        """
        Conditional VaR (Expected Shortfall) in return terms
        
        Returns:
            float: CVaR as return (negative value for loss)
        """
        var_threshold = self.calculate_var_return(confidence)
        tail_returns = self.returns[self.returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return var_threshold
        
        return float(tail_returns.mean())
    
    def calculate_var_usdt(self, confidence: float = 0.95) -> float:
        """
        VaR in USDT terms
        
        Returns:
            float: VaR as positive loss amount in USDT
        """
        var_return = self.calculate_var_return(confidence)
        return float(-var_return * self.portfolio_value)
    
    def calculate_cvar_usdt(self, confidence: float = 0.95) -> float:
        """
        CVaR in USDT terms
        
        Returns:
            float: CVaR as positive loss amount in USDT
        """
        cvar_return = self.calculate_cvar_return(confidence)
        return float(-cvar_return * self.portfolio_value)
    
    def calculate_sortino_ratio(self, target_return: float = 0, 
                                annualize: bool = True) -> float:
        """Sortino Ratio (하방 변동성만 고려)"""
        excess_returns = self.returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.sqrt(np.mean(downside_returns**2))
        if downside_std < 1e-10:
            return float('inf') if excess_returns.mean() > 0 else 0
            
        sortino = excess_returns.mean() / downside_std
        
        if annualize:
            sortino *= np.sqrt(252)
        
        return float(sortino)
    
    def calculate_max_drawdown(self) -> float:
        """최대 낙폭 계산"""
        r = np.asarray(self.returns, dtype=float)
        if len(r) == 0:
            return 0.0
            
        equity = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / (running_max + 1e-10)
        return float(drawdown.min())
    
    def calculate_calmar_ratio(self, annualize: bool = True) -> float:
        """Calmar Ratio (연간 수익률 / 최대 낙폭)"""
        max_dd = abs(self.calculate_max_drawdown())
        
        if max_dd < 1e-10:
            return float('inf') if self.returns.mean() > 0 else 0
        
        avg_return = self.returns.mean()
        if annualize:
            avg_return *= 252
        
        return float(avg_return / max_dd)
    
    def calculate_omega_ratio(self, threshold: float = 0) -> float:
        """Omega Ratio (상승 확률 가중 / 하락 확률 가중)"""
        gains = self.returns[self.returns > threshold] - threshold
        losses = threshold - self.returns[self.returns <= threshold]
        
        if len(losses) == 0 or losses.sum() == 0:
            return float('inf') if len(gains) > 0 else 1.0
        
        return float(gains.sum() / losses.sum())
```

## 2.3 수학적 모델 상세

### Kelly Criterion 유도

로그 효용 최대화 문제:
```
max E[log(W_t+1)]
where W_t+1 = W_t(1 + f*r_t)
```

1차 조건:
```
dE[log(W)]/df = E[r/(1+f*r)] = 0
```

Taylor 전개 및 근사:
```
f* ≈ μ/σ² (연속 수익률)
```

### VaR/CVaR 정의

**Value at Risk (VaR)**:
```
VaR_α = -inf{x : P(R ≤ x) ≥ 1-α}
```

**Conditional VaR (CVaR)**:
```
CVaR_α = -E[R | R ≤ -VaR_α]
```

### 청산 확률 (Barrier Hitting)

로그 거리 d, 변동성 σ, 기간 T에서:
```
P(hit within T) ≈ 2 * (1 - Φ(d / (σ√T)))
```

Fat-tail 보정 (t-분포):
```
P_fat(hit within T) ≈ 2 * (1 - T_df(d / (σ√T)))
```

## 2.4 고급 리스크 모델

```python
class AdvancedRiskModels:
    """고급 리스크 모델링"""
    
    @staticmethod
    def calculate_cornish_fisher_var(returns: np.ndarray, 
                                    confidence: float = 0.95) -> float:
        """Cornish-Fisher 전개를 이용한 VaR (비정규분포 고려)"""
        from scipy import stats
        
        mean = np.mean(returns)
        std = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        z = stats.norm.ppf(1 - confidence)
        
        # Cornish-Fisher expansion
        z_cf = z + (z**2 - 1) * skew / 6 + \
               (z**3 - 3*z) * kurt / 24 - \
               (2*z**3 - 5*z) * skew**2 / 36
        
        var_cf = mean + std * z_cf
        return float(-var_cf)
    
    @staticmethod
    def calculate_expected_shortfall_gaussian_mixture(returns: np.ndarray, 
                                                     confidence: float = 0.95) -> float:
        """Gaussian Mixture Model을 이용한 Expected Shortfall"""
        from sklearn.mixture import GaussianMixture
        
        # Fit GMM
        gmm = GaussianMixture(n_components=2, random_state=42)
        returns_reshaped = returns.reshape(-1, 1)
        gmm.fit(returns_reshaped)
        
        # Monte Carlo simulation for ES
        n_simulations = 10000
        samples = gmm.sample(n_simulations)[0].flatten()
        
        var_threshold = np.percentile(samples, (1 - confidence) * 100)
        tail_samples = samples[samples <= var_threshold]
        
        if len(tail_samples) > 0:
            es = tail_samples.mean()
        else:
            es = var_threshold
        
        return float(-es)
    
    @staticmethod
    def calculate_risk_parity_weights(cov_matrix: np.ndarray) -> np.ndarray:
        """Risk Parity 가중치 계산"""
        n_assets = len(cov_matrix)
        
        # Initial guess: equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Newton-Raphson iteration
        for _ in range(100):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            
            # Risk contribution
            contrib = weights * marginal_contrib
            
            # Target: equal risk contribution
            target_contrib = portfolio_vol / n_assets
            
            # Update weights
            weights = weights * target_contrib / contrib
            weights = weights / weights.sum()
        
        return weights
```

## 2.5 포트폴리오 리스크 분석

```python
class PortfolioRiskAnalyzer:
    """포트폴리오 수준 리스크 분석"""
    
    def __init__(self, positions: List[Dict], correlations: np.ndarray):
        self.positions = positions
        self.correlations = correlations
        
    def calculate_portfolio_var(self, confidence: float = 0.95) -> float:
        """포트폴리오 VaR (상관관계 고려)"""
        
        # Individual VaRs
        individual_vars = []
        weights = []
        
        for position in self.positions:
            var = position['var_usdt']
            weight = position['notional'] / sum(p['notional'] for p in self.positions)
            individual_vars.append(var)
            weights.append(weight)
        
        vars = np.array(individual_vars)
        w = np.array(weights)
        
        # Portfolio VaR with correlation
        portfolio_var = np.sqrt(w @ self.correlations @ w) * vars.mean()
        
        return float(portfolio_var)
    
    def calculate_marginal_var(self, position_index: int) -> float:
        """한계 VaR (포지션 추가 시 VaR 증가분)"""
        
        # Current portfolio VaR
        current_var = self.calculate_portfolio_var()
        
        # Remove position
        positions_without = self.positions[:position_index] + \
                          self.positions[position_index + 1:]
        
        if not positions_without:
            return current_var
        
        # Recalculate VaR without position
        analyzer_without = PortfolioRiskAnalyzer(
            positions_without, 
            self._reduce_correlation_matrix(position_index)
        )
        var_without = analyzer_without.calculate_portfolio_var()
        
        # Marginal VaR
        marginal_var = current_var - var_without
        
        return float(marginal_var)
    
    def calculate_component_var(self) -> List[float]:
        """Component VaR (각 포지션의 VaR 기여도)"""
        
        component_vars = []
        portfolio_var = self.calculate_portfolio_var()
        
        for i in range(len(self.positions)):
            marginal_var = self.calculate_marginal_var(i)
            weight = self.positions[i]['notional'] / \
                    sum(p['notional'] for p in self.positions)
            component_var = marginal_var * weight
            component_vars.append(component_var)
        
        # Normalize to sum to portfolio VaR
        total = sum(component_vars)
        if total > 0:
            component_vars = [cv * portfolio_var / total for cv in component_vars]
        
        return component_vars
    
    def _reduce_correlation_matrix(self, index: int) -> np.ndarray:
        """상관관계 행렬에서 특정 인덱스 제거"""
        mask = np.ones(len(self.correlations), dtype=bool)
        mask[index] = False
        return self.correlations[mask][:, mask]
```