# src/risk_management/risk_management.py
"""
리스크 관리 핵심 모듈
TDD 방식으로 구현됨 - 설정 가능한 파라미터 구조
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from src.utils.logger import TradingLogger, get_trading_logger


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
                 max_consecutive_loss_days: int = 7,   # 🚀 NEW: 최대 연속 손실일 (기본값: 7일)
                 allow_short: bool = False,            # 숏 포지션 허용 여부 (기본값: 롱 온리)
                 logger: Optional[TradingLogger] = None):  # 로깅 시스템

        self.initial_capital = initial_capital_usdt

        # 로깅 시스템 초기화
        self.logger = logger or get_trading_logger("risk_controller", log_to_file=False)

        # 리스크 컨트롤러 초기화 로그
        self.logger.log_risk(
            "Risk controller initialized",
            level="INFO",
            initial_capital_usdt=initial_capital_usdt,
            var_daily_pct=var_daily_pct,
            max_drawdown_pct=max_drawdown_pct,
            max_leverage=max_leverage,
            allow_short=allow_short
        )

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
            'liquidation_prob_24h': liquidation_prob_24h,
            'max_consecutive_loss_days': max_consecutive_loss_days  # 🚀 NEW: 연속 손실일 한도
        }

        self.current_drawdown = 0
        self.high_water_mark = initial_capital_usdt
        self.allow_short = allow_short

        # 🚀 NEW: 드로다운 모니터링을 위한 추가 속성
        self.consecutive_loss_days = 0
        self.last_daily_pnl = 0.0
        self.drawdown_start_time = None  # 드로다운 시작 시간
        self.recovery_periods = []  # 드로다운 복구 기간 기록

    def check_var_limit(self, portfolio_state: Dict) -> List[Tuple[str, float]]:
        """VaR 한도 체크"""
        violations = []

        current_var = portfolio_state.get('current_var_usdt', 0.0)
        var_limit = self.risk_limits['var_daily_usdt']

        # VaR 상태 로깅 (INFO 레벨)
        self.logger.log_risk(
            "VaR limit check performed",
            level="INFO",
            current_var_usdt=current_var,
            var_limit_usdt=var_limit,
            utilization_pct=(current_var / var_limit * 100) if var_limit > 0 else 0
        )

        if current_var > var_limit:
            violations.append(('VAR_USDT', current_var))

            # VaR 한도 위반 경고 로깅 (WARNING 레벨)
            self.logger.log_risk(
                "VaR limit exceeded - Risk threshold violation detected",
                level="WARNING",
                event_type="VAR_LIMIT_VIOLATION",
                current_var_usdt=current_var,
                var_limit_usdt=var_limit,
                excess_usdt=current_var - var_limit,
                excess_pct=((current_var - var_limit) / var_limit * 100)
            )

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

    def check_leverage_limit(self, portfolio_state: Dict) -> List[Tuple[str, float]]:
        """레버리지 한도 체크"""
        violations = []

        # 총 레버리지 계산
        total_leverage = self._calculate_total_leverage(portfolio_state)
        leverage_limit = self.risk_limits['max_leverage']

        if total_leverage > leverage_limit:
            violations.append(('LEVERAGE', total_leverage))

        return violations

    def _calculate_total_leverage(self, portfolio_state: Dict) -> float:
        """포트폴리오의 총 레버리지 계산"""
        equity = portfolio_state.get('equity', self.initial_capital)
        positions = portfolio_state.get('positions', [])

        # 포지션이 없으면 레버리지 0
        if not positions:
            return 0.0

        # 총 notional 값 계산
        total_notional = sum(
            position.get('notional', 0.0) for position in positions
        )

        # 레버리지 = 총 notional / equity
        if equity > 0:
            return total_notional / equity
        else:
            return 0.0

    def calculate_safe_leverage_limit(self, portfolio_state: Dict) -> float:
        """청산 거리 기반 안전 레버리지 한도 계산"""
        positions = portfolio_state.get('positions', [])

        if not positions:
            return self.risk_limits['max_leverage']

        min_safe_leverage = self.risk_limits['max_leverage']

        for position in positions:
            current_price = position.get('current_price', 0)
            liquidation_price = position.get('liquidation_price', 0)
            daily_vol = position.get('daily_volatility', 0.05)  # 기본 5%
            side = position.get('side', 'LONG')

            if current_price <= 0 or liquidation_price <= 0:
                continue

            # 청산 거리 계산 (로그 기준)
            if side == 'LONG':
                log_distance = np.log(current_price / liquidation_price)
            else:  # SHORT
                log_distance = np.log(liquidation_price / current_price)

            # 안전 계수 (3-sigma 기준)
            safety_factor = 3.0
            required_distance = safety_factor * daily_vol

            # 안전 레버리지 계산
            if log_distance > required_distance:
                # 충분한 거리가 있으면 기본 한도 사용
                position_safe_leverage = self.risk_limits['max_leverage']
            else:
                # 거리가 부족하면 비례적으로 감소
                distance_ratio = max(0.1, log_distance / required_distance)
                position_safe_leverage = self.risk_limits['max_leverage'] * distance_ratio

            min_safe_leverage = min(min_safe_leverage, position_safe_leverage)

        return float(max(1.0, min_safe_leverage))  # 최소 1배 레버리지

    def calculate_volatility_adjusted_leverage(self, base_leverage: float,
                                             market_state: Dict) -> float:
        """변동성 기반 레버리지 조정"""
        daily_vol = market_state.get('daily_volatility', 0.05)
        regime = market_state.get('regime', 'NEUTRAL')

        # 기준 변동성 (5%)
        base_volatility = 0.05

        # 변동성 조정 계수
        vol_ratio = daily_vol / base_volatility

        if vol_ratio <= 1.0:
            # 낮은 변동성 - 레버리지 약간 증가 허용
            vol_adjustment = 1.0 + (1.0 - vol_ratio) * 0.2  # 최대 20% 증가
        else:
            # 높은 변동성 - 레버리지 감소
            vol_adjustment = 1.0 / (1.0 + (vol_ratio - 1.0) * 0.5)  # 변동성 2배면 레버리지 2/3

        # 레짐별 추가 조정
        regime_adjustments = {
            'BULL': 1.1,      # 강세장에서 약간 증가
            'BEAR': 0.8,      # 약세장에서 감소
            'SIDEWAYS': 0.9,  # 횡보에서 약간 감소
            'VOLATILE': 0.7,  # 고변동성에서 크게 감소
            'NEUTRAL': 1.0    # 기본
        }

        regime_adjustment = regime_adjustments.get(regime, 1.0)

        # 최종 조정된 레버리지
        adjusted_leverage = base_leverage * vol_adjustment * regime_adjustment

        # 최대 한도 적용
        max_allowed = self.risk_limits['max_leverage']

        return float(max(0.1, min(adjusted_leverage, max_allowed)))

    # ========== 🚀 NEW: 드로다운 모니터링 시스템 ==========

    def update_drawdown(self, current_equity: float) -> float:
        """
        현재 자본에 기반해 드로다운 업데이트

        Args:
            current_equity: 현재 총 자본 (USDT)

        Returns:
            float: 현재 드로다운 비율 (0.1 = 10% 드로다운)
        """
        previous_drawdown = self.current_drawdown

        if current_equity > self.high_water_mark:
            # 새로운 고점 달성
            old_hwm = self.high_water_mark
            self.high_water_mark = current_equity
            self.current_drawdown = 0.0

            # 새로운 고점 달성 로깅
            self.logger.log_portfolio(
                "New high water mark achieved",
                event_type="HIGH_WATER_MARK_UPDATE",
                previous_hwm=old_hwm,
                new_hwm=current_equity,
                equity_increase_usdt=current_equity - old_hwm,
                equity_increase_pct=((current_equity - old_hwm) / old_hwm * 100)
            )
        else:
            # 드로다운 계산
            self.current_drawdown = (self.high_water_mark - current_equity) / self.high_water_mark

            # 드로다운 상태 로깅
            if previous_drawdown != self.current_drawdown:
                severity = self.get_drawdown_severity_level()

                self.logger.log_risk(
                    "Drawdown updated",
                    level="INFO" if self.current_drawdown < 0.05 else "WARNING",
                    event_type="DRAWDOWN_UPDATE",
                    current_equity=current_equity,
                    high_water_mark=self.high_water_mark,
                    drawdown_pct=self.current_drawdown * 100,
                    drawdown_usdt=self.high_water_mark - current_equity,
                    severity_level=severity
                )

        return float(self.current_drawdown)

    def check_drawdown_limit(self, current_equity: float) -> List[Tuple[str, float]]:
        """
        드로다운 한도 위반 체크

        Args:
            current_equity: 현재 총 자본 (USDT)

        Returns:
            List[Tuple[str, float]]: 위반 목록 (위반 타입, 위반 값)
        """
        violations = []

        # 드로다운 업데이트
        current_drawdown = self.update_drawdown(current_equity)
        max_drawdown_limit = self.risk_limits['max_drawdown_pct']

        if current_drawdown > max_drawdown_limit:
            violations.append(('DRAWDOWN', current_drawdown))

            # 드로다운 한도 위반 경고 로깅 (CRITICAL 레벨)
            self.logger.log_risk(
                "CRITICAL: Maximum drawdown limit exceeded - Trading may be halted",
                level="CRITICAL",
                event_type="DRAWDOWN_LIMIT_VIOLATION",
                current_equity=current_equity,
                high_water_mark=self.high_water_mark,
                current_drawdown_pct=current_drawdown * 100,
                max_drawdown_limit_pct=max_drawdown_limit * 100,
                excess_drawdown_pct=(current_drawdown - max_drawdown_limit) * 100,
                drawdown_usdt=self.high_water_mark - current_equity
            )

        return violations

    def get_drawdown_severity_level(self) -> str:
        """
        현재 드로다운의 심각도 분류

        Returns:
            str: 'MILD' (0-5%), 'MODERATE' (5-10%), 'SEVERE' (10%+)
        """
        drawdown_pct = self.current_drawdown

        if drawdown_pct < 0.05:  # 5% 미만
            return 'MILD'
        elif drawdown_pct < 0.10:  # 5-10%
            return 'MODERATE'
        else:  # 10% 이상
            return 'SEVERE'

    def update_consecutive_loss_days(self, daily_pnl: float) -> int:
        """
        일일 손익을 기반으로 연속 손실일 업데이트

        Args:
            daily_pnl: 일일 손익 (USDT, 음수면 손실)

        Returns:
            int: 현재 연속 손실일 수
        """
        if daily_pnl < 0:
            # 손실 날짜
            self.consecutive_loss_days += 1
        else:
            # 수익 날짜 - 연속 손실 스트릭 리셋
            self.consecutive_loss_days = 0

        self.last_daily_pnl = daily_pnl
        return self.consecutive_loss_days

    def check_consecutive_loss_limit(self) -> List[Tuple[str, int]]:
        """
        연속 손실일 한도 위반 체크

        Returns:
            List[Tuple[str, int]]: 위반 목록 (위반 타입, 연속 손실일)
        """
        violations = []
        max_consecutive_loss_days = self.risk_limits.get('max_consecutive_loss_days', 7)  # 기본값 7일

        if self.consecutive_loss_days > max_consecutive_loss_days:
            violations.append(('CONSECUTIVE_LOSS_DAYS', self.consecutive_loss_days))

        return violations

    def track_drawdown_recovery(self, current_equity: float,
                               current_time=None) -> Optional[int]:
        """
        드로다운 복구 추적

        Args:
            current_equity: 현재 총 자본 (USDT)
            current_time: 현재 시간 (None이면 현재 시각 사용)

        Returns:
            Optional[int]: 복구 완료된 경우 복구 기간(일), 아직 복구 중이면 None
        """
        from datetime import datetime

        if current_time is None:
            current_time = datetime.now()

        # 드로다운 업데이트
        self.update_drawdown(current_equity)

        if self.current_drawdown > 0:
            # 드로다운 중
            if self.drawdown_start_time is None:
                # 드로다운 시작
                self.drawdown_start_time = current_time
            return None
        else:
            # 드로다운 복구됨
            if self.drawdown_start_time is not None:
                # 복구 기간 계산
                recovery_period_days = (current_time - self.drawdown_start_time).days

                # 복구 기록 저장
                recovery_record = {
                    'start_time': self.drawdown_start_time,
                    'end_time': current_time,
                    'recovery_days': recovery_period_days,
                    'recovered_equity': current_equity
                }
                self.recovery_periods.append(recovery_record)

                # 드로다운 상태 리셋
                self.drawdown_start_time = None

                return recovery_period_days
            return 0  # 드로다운이 없었던 경우

    def get_recovery_statistics(self) -> Dict:
        """
        드로다운 복구 통계 반환

        Returns:
            Dict: 복구 관련 통계
        """
        if not self.recovery_periods:
            return {
                'total_recoveries': 0,
                'average_recovery_days': 0.0,
                'max_recovery_days': 0,
                'min_recovery_days': 0
            }

        recovery_days = [period['recovery_days'] for period in self.recovery_periods]

        return {
            'total_recoveries': len(self.recovery_periods),
            'average_recovery_days': sum(recovery_days) / len(recovery_days),
            'max_recovery_days': max(recovery_days),
            'min_recovery_days': min(recovery_days),
            'recovery_history': self.recovery_periods.copy()
        }