# src/risk_management/position_sizing.py
"""
포지션 사이징 엔진
청산 안전성과 리스크 제약을 고려한 최적 포지션 크기 계산
"""
from typing import Dict, Optional
import numpy as np
from .risk_management import RiskController
from src.utils.logger import TradingLogger, get_trading_logger


class PositionSizer:
    """청산 안전성을 고려한 포지션 사이징 (USDT 기준)"""

    def __init__(self, risk_controller: RiskController, logger: Optional[TradingLogger] = None):
        self.risk_controller = risk_controller
        self.logger = logger or get_trading_logger("position_sizer", log_to_file=False)

        # 포지션 사이저 초기화 로그
        self.logger.info(
            "Position sizer initialized",
            component="PositionSizer",
            risk_controller_config={
                'initial_capital': risk_controller.initial_capital,
                'max_leverage': risk_controller.risk_limits['max_leverage']
            }
        )

    def calculate_position_size(self, signal: Dict,
                               market_state: Dict,
                               portfolio_state: Dict) -> float:
        """
        다중 제약 하의 최적 포지션 크기

        Args:
            signal: Trading signal with symbol, side, strength
            market_state: Market data (price, ATR, volatility, etc.)
            portfolio_state: Current portfolio state

        Returns:
            float: Position size in coin units
        """
        symbol = signal['symbol']
        signal_strength = signal.get('strength', 1.0)
        price = market_state['price']
        equity = portfolio_state['equity']

        # Early exit for zero signal strength
        if signal_strength == 0.0:
            return 0.0

        # 1. Kelly 기반 크기
        kelly_size = self._calculate_kelly_based_size(signal, market_state, portfolio_state)

        # 2. ATR 기반 리스크 크기
        atr_size = self._calculate_atr_based_size(signal, market_state, portfolio_state)

        # 3. 청산 안전 거리 기반 크기
        liquidation_safe_size = self._calculate_liquidation_safe_size(
            symbol, signal['side'], market_state, portfolio_state
        )

        # 4. VaR 제약 크기
        var_constrained_size = self._calculate_var_constrained_size(
            symbol, market_state, portfolio_state
        )

        # 5. 상관관계 조정
        correlation_factor = self._calculate_correlation_adjustment(
            symbol, portfolio_state
        )

        # 최종 크기 (최소값 선택 - 가장 제한적인 조건 적용)
        base_size = min(
            kelly_size,
            atr_size,
            liquidation_safe_size,
            var_constrained_size
        )

        # 상관관계 조정 및 신호 강도 적용
        final_size = base_size * correlation_factor * signal_strength

        # 최소/최대 제약 적용
        min_notional = market_state.get('min_notional', 10)  # 10 USDT 기본
        min_size = min_notional / price
        max_size = equity * 0.2 / price  # 단일 포지션 20% 상한

        # 범위 내로 클리핑
        constrained_size = max(min_size, min(final_size, max_size))

        # 거래소 lot size로 반올림
        final_position_size = self._round_to_lot_size(constrained_size, market_state.get('lot_size', 0.001))

        # 포지션 사이징 결과 로깅
        self.logger.log_trade(
            "Position size calculated",
            symbol=symbol,
            side=signal['side'],
            signal_strength=signal_strength,
            price=price,
            kelly_size=kelly_size,
            atr_size=atr_size,
            liquidation_safe_size=liquidation_safe_size,
            var_constrained_size=var_constrained_size,
            correlation_factor=correlation_factor,
            base_size=base_size,
            final_size=final_size,
            constrained_size=constrained_size,
            final_position_size=final_position_size,
            position_notional_usdt=final_position_size * price,
            limiting_factor=self._identify_limiting_factor(
                kelly_size, atr_size, liquidation_safe_size, var_constrained_size
            )
        )

        return final_position_size

    def _calculate_kelly_based_size(self, signal: Dict, market_state: Dict,
                                   portfolio_state: Dict) -> float:
        """Kelly Criterion 기반 포지션 크기"""
        recent_returns = portfolio_state.get('recent_returns', np.array([]))

        kelly_fraction = self.risk_controller.calculate_optimal_position_fraction(
            recent_returns,
            regime=market_state.get('regime', 'NEUTRAL'),
            fractional=0.25
        )

        # If Kelly returns 0 (insufficient data or other reasons), use conservative fallback
        if abs(kelly_fraction) < 1e-10:
            # Default to conservative 5% of equity if Kelly can't provide guidance
            kelly_fraction = 0.05

        kelly_notional = portfolio_state['equity'] * abs(kelly_fraction)  # Use absolute value for size
        return kelly_notional / market_state['price']

    def _calculate_atr_based_size(self, signal: Dict, market_state: Dict,
                                 portfolio_state: Dict) -> float:
        """ATR 기반 리스크 포지션 크기"""
        atr = market_state.get('atr', market_state['price'] * 0.02)  # Default 2% if no ATR
        price = market_state['price']
        equity = portfolio_state['equity']

        # 1% risk per trade
        risk_per_trade = equity * 0.01  # USDT
        stop_distance = 2.0 * atr  # 2 ATR stop distance

        if stop_distance > 0:
            return risk_per_trade / stop_distance  # Size in coin units
        else:
            return equity * 0.05 / price  # Fallback to 5% of equity

    def _calculate_liquidation_safe_size(self, symbol: str, side: str,
                                        market_state: Dict,
                                        portfolio_state: Dict) -> float:
        """청산 안전 거리를 고려한 최대 크기"""
        price = market_state['price']
        atr = market_state.get('atr', price * 0.02)
        equity = portfolio_state['equity']

        # Binance USDT-M 증거금 규칙 (간소화)
        mmr = self._get_maintenance_margin_rate(symbol, equity)
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
        notional = equity * leverage_to_use
        return notional / price

    def _calculate_var_constrained_size(self, symbol: str,
                                       market_state: Dict,
                                       portfolio_state: Dict) -> float:
        """VaR 제약 하의 최대 포지션 크기"""
        z = 1.65  # 95% confidence level
        price = market_state['price']
        sigma = portfolio_state.get('symbol_volatilities', {}).get(symbol, 0.05)

        # VaR 한도 (USDT)
        var_limit = portfolio_state['equity'] * 0.02  # 2% VaR
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
        positions = portfolio_state.get('positions', [])

        if not positions:
            return 1.0  # No correlation adjustment if no existing positions

        # 기존 포지션들과의 평균 상관관계
        correlations = []
        correlation_matrix = portfolio_state.get('correlation_matrix', {})

        for position in positions:
            existing_symbol = position.get('symbol')
            if existing_symbol and existing_symbol != symbol:
                corr = correlation_matrix.get((symbol, existing_symbol), 0)
                correlations.append(abs(corr))

        if not correlations:
            return 1.0

        avg_correlation = np.mean(correlations)

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
        if lot_size <= 0:
            return float(size)
        return float(np.floor(size / lot_size) * lot_size)

    def _identify_limiting_factor(self, kelly_size: float, atr_size: float,
                                liquidation_safe_size: float, var_constrained_size: float) -> str:
        """가장 제한적인 요인 식별"""
        min_size = min(kelly_size, atr_size, liquidation_safe_size, var_constrained_size)

        if min_size == kelly_size:
            return "KELLY_CRITERION"
        elif min_size == atr_size:
            return "ATR_RISK"
        elif min_size == liquidation_safe_size:
            return "LIQUIDATION_SAFETY"
        else:
            return "VAR_CONSTRAINT"