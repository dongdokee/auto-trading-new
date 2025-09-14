# src/risk_management/risk_management.py
"""
리스크 관리 핵심 모듈
TDD 방식으로 구현됨 - 설정 가능한 파라미터 구조
"""
from typing import Dict, List, Tuple, Optional
import numpy as np


class RiskController:
    """통합 리스크 관리자 (USDT 기준)"""

    def __init__(self,
                 initial_capital_usdt: float,
                 var_daily_pct: float = 0.02,          # 일일 VaR 2% (기본값)
                 cvar_daily_pct: float = 0.03,         # 일일 CVaR 3% (기본값)
                 max_drawdown_pct: float = 0.12,       # 최대 드로다운 12% (기본값)
                 correlation_threshold: float = 0.7,   # 상관관계 임계값
                 concentration_limit: float = 0.2,     # 단일 자산 집중도 20%
                 max_leverage: float = 10.0,           # 최대 레버리지 10x (기본값)
                 liquidation_prob_24h: float = 0.005,  # 24시간 청산 확률 0.5%
                 allow_short: bool = False):           # 숏 포지션 허용 여부 (기본값: 롱 온리)

        self.initial_capital = initial_capital_usdt

        # 설정 가능한 리스크 한도 (USDT 기준)
        self.risk_limits = {
            'var_daily_return': var_daily_pct,
            'cvar_daily_return': cvar_daily_pct,
            'var_daily_usdt': initial_capital_usdt * var_daily_pct,
            'cvar_daily_usdt': initial_capital_usdt * cvar_daily_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'correlation_threshold': correlation_threshold,
            'concentration_limit': concentration_limit,
            'max_leverage': max_leverage,
            'liquidation_prob_24h': liquidation_prob_24h
        }

        self.current_drawdown = 0
        self.high_water_mark = initial_capital_usdt
        self.allow_short = allow_short

    def check_var_limit(self, portfolio_state: Dict) -> List[Tuple[str, float]]:
        """VaR 한도 체크"""
        violations = []

        current_var = portfolio_state.get('current_var_usdt', 0.0)
        var_limit = self.risk_limits['var_daily_usdt']

        if current_var > var_limit:
            violations.append(('VAR_USDT', current_var))

        return violations

    def calculate_optimal_position_fraction(self, returns: np.ndarray,
                                           regime: str = 'NEUTRAL',
                                           fractional: float = 0.25) -> float:
        """
        Kelly Criterion을 이용한 최적 포지션 fraction 계산

        Args:
            returns: numpy array of returns (decimal format, e.g. 0.01 = 1%)
            regime: 현재 시장 레짐 ('BULL', 'BEAR', 'SIDEWAYS', 'NEUTRAL')
            fractional: Fractional Kelly 계수 (기본값: 0.25 = 1/4 Kelly)

        Returns:
            float: 최적 betting fraction (양수: 롱, 음수: 숏, 0: 포지션 없음)
        """
        returns_array = np.asarray(returns, dtype=float)

        # 최소 샘플 수 체크
        if len(returns_array) < 30:
            return 0.0

        # EMA 가중치 계산 (최근 데이터에 높은 가중치)
        alpha = 0.2
        weights = np.exp(-alpha * np.arange(len(returns_array)))
        weights = weights / weights.sum()

        # 가중 평균 수익률
        mu = np.average(returns_array, weights=weights)

        # 가중 분산 계산
        variance = np.average((returns_array - mu) ** 2, weights=weights)

        # 분산이 너무 작으면 0 반환
        if variance < 1e-10:
            return 0.0

        # Kelly 공식: f* = μ/σ²
        kelly_fraction = mu / variance

        # 파라미터 추정 불확실성 보정 (shrinkage)
        effective_samples = (weights.sum() ** 2) / (weights ** 2).sum()
        shrinkage = max(0.0, 1.0 - 2.0 / effective_samples)
        kelly_fraction *= shrinkage

        # Fractional Kelly 적용
        kelly_fraction *= fractional

        # 레짐별 한도 적용
        if self.allow_short:
            # 양방향 거래 허용
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
            # 롱 온리 (기본값)
            regime_caps = {
                'BULL': 0.15,
                'BEAR': 0.05,
                'SIDEWAYS': 0.10,
                'NEUTRAL': 0.08
            }
            cap = regime_caps.get(regime, 0.08)
            return float(np.clip(max(0.0, kelly_fraction), 0.0, cap))