"""
Funding Arbitrage Strategy

Implements a funding rate arbitrage strategy for cryptocurrency perpetual futures.
Exploits funding rate differentials by taking positions to collect funding payments.
Supports both delta-neutral and directional positioning based on funding predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from ..base_strategy import BaseStrategy, StrategySignal, StrategyConfig


class FundingArbitrageStrategy(BaseStrategy):
    """
    Funding Rate Arbitrage Strategy

    Strategy Logic:
    - Predicts next funding rate based on historical patterns and current basis
    - Takes long positions when funding rates are expected to be negative (collect funding)
    - Takes short positions when funding rates are expected to be positive (collect funding)
    - Considers futures basis and volatility in positioning decisions
    - Supports delta-neutral positioning for market-neutral arbitrage
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize funding arbitrage strategy

        Args:
            config: Strategy configuration with parameters:
                - funding_threshold: Minimum annualized funding rate to trade (default: 0.01)
                - basis_threshold: Maximum unfavorable basis tolerance (default: 0.002)
                - delta_neutral: Whether to use delta neutral positioning (default: True)
                - funding_lookback: Periods to look back for funding prediction (default: 24)
                - position_hold_hours: Hours to hold positions (default: 8)
                - risk_factor: Risk adjustment factor (default: 0.5)
        """
        super().__init__(config)

        # Strategy parameters with defaults
        self.funding_threshold = self.parameters.get("funding_threshold", 0.01)  # 1% annualized
        self.basis_threshold = self.parameters.get("basis_threshold", 0.002)     # 0.2%
        self.delta_neutral = self.parameters.get("delta_neutral", True)
        self.funding_lookback = self.parameters.get("funding_lookback", 24)
        self.position_hold_hours = self.parameters.get("position_hold_hours", 8)
        self.risk_factor = self.parameters.get("risk_factor", 0.5)

        # Validate parameters
        if not (0.001 <= self.funding_threshold <= 0.1):
            raise ValueError("Funding threshold must be between 0.1% and 10%")

        if not (0.0005 <= self.basis_threshold <= 0.01):
            raise ValueError("Basis threshold must be between 0.05% and 1%")

        if self.funding_lookback < 8:
            raise ValueError("Funding lookback must be at least 8 periods")

        if self.position_hold_hours < 1:
            raise ValueError("Position hold hours must be at least 1")

        if not (0.1 <= self.risk_factor <= 1.0):
            raise ValueError("Risk factor must be between 0.1 and 1.0")

    def generate_signal(self, market_data: Dict[str, Any], current_index: int = -1) -> StrategySignal:
        """
        Generate funding arbitrage signal based on funding rates and basis

        Args:
            market_data: Dictionary containing:
                - symbol: Trading pair symbol
                - close: Current price
                - funding_rate: Current funding rate
                - basis: Current futures basis (futures - spot)
                - funding_data: Historical funding data DataFrame
            current_index: Current position in the data (for backtesting)

        Returns:
            StrategySignal: Generated trading signal
        """
        symbol = market_data.get('symbol', 'UNKNOWN')
        current_price = market_data.get('close', 0.0)
        current_funding = market_data.get('funding_rate', 0.0)
        current_basis = market_data.get('basis', 0.0)
        funding_data = market_data.get('funding_data')

        # Default HOLD signal
        default_signal = StrategySignal(
            symbol=symbol,
            action='HOLD',
            strength=0.0,
            confidence=0.0,
            metadata={'strategy': 'FundingArbitrage', 'reason': 'default'}
        )

        if funding_data is None or len(funding_data) < self.funding_lookback:
            return default_signal

        try:
            # Use current_index or last available data
            if current_index == -1:
                current_index = len(funding_data) - 1
            elif current_index >= len(funding_data):
                current_index = len(funding_data) - 1

            # Ensure we have enough historical data
            if current_index < self.funding_lookback:
                return default_signal

            # Predict next funding rate
            predicted_funding = self._predict_next_funding_rate(funding_data, current_index)

            if predicted_funding is None:
                return default_signal

            # Calculate funding statistics
            funding_stats = self._calculate_funding_statistics(funding_data, current_index)

            # Evaluate arbitrage opportunity
            signal_info = self._evaluate_arbitrage_opportunity(
                predicted_funding, current_funding, current_basis,
                funding_stats, current_price
            )

            if signal_info['action'] == 'HOLD':
                return default_signal

            # Apply risk adjustments
            signal_info = self._apply_risk_adjustments(
                signal_info, funding_stats, current_basis
            )

            # Create signal
            signal = StrategySignal(
                symbol=symbol,
                action=signal_info['action'],
                strength=signal_info['strength'],
                confidence=signal_info['confidence'],
                stop_loss=signal_info.get('stop_loss'),
                take_profit=signal_info.get('take_profit'),
                metadata={
                    'strategy': 'FundingArbitrage',
                    'predicted_funding': predicted_funding,
                    'current_funding': current_funding,
                    'annualized_funding': predicted_funding * 365 * 3,  # 8h periods
                    'basis': current_basis,
                    'basis_pct': current_basis / current_price if current_price > 0 else 0,
                    'delta_neutral': self.delta_neutral,
                    'hold_hours': self.position_hold_hours,
                    'funding_volatility': funding_stats.get('volatility', 0),
                    'funding_trend': funding_stats.get('trend', 0)
                }
            )

            # Track signal in history
            self.signal_history.append(signal)
            self.total_signals += 1

            return signal

        except Exception as e:
            # Return default signal on any error
            return StrategySignal(
                symbol=symbol,
                action='HOLD',
                strength=0.0,
                confidence=0.0,
                metadata={'strategy': 'FundingArbitrage', 'error': str(e)}
            )

    def _predict_next_funding_rate(self, data: pd.DataFrame, current_index: int) -> Optional[float]:
        """
        Predict next funding rate based on historical patterns and current basis

        Args:
            data: Funding data DataFrame
            current_index: Current position in data

        Returns:
            Predicted funding rate or None
        """
        try:
            # Get recent funding data
            start_idx = max(0, current_index - self.funding_lookback)
            recent_data = data.iloc[start_idx:current_index + 1].copy()

            if len(recent_data) < 8:
                return None

            funding_rates = recent_data['funding_rate'].values

            # Simple moving average prediction with trend adjustment
            recent_funding = funding_rates[-8:]  # Last 8 periods (2-3 days)

            # Calculate trend
            if len(recent_funding) >= 4:
                early_avg = np.mean(recent_funding[:4])
                late_avg = np.mean(recent_funding[-4:])
                trend = (late_avg - early_avg) / 4  # Trend per period
            else:
                trend = 0

            # Base prediction: weighted average of recent rates
            weights = np.exp(np.linspace(-1, 0, len(recent_funding)))  # More weight on recent
            weights /= weights.sum()
            base_prediction = np.dot(recent_funding, weights)

            # Add trend component (but dampen it significantly for small base rates)
            trend_dampening = min(1.0, abs(base_prediction) * 10000)  # Dampen for very small rates
            trend_component = trend * 0.5 * trend_dampening  # 50% trend continuation, dampened

            # Basis-adjusted prediction
            current_basis = recent_data['basis'].iloc[-1] if 'basis' in recent_data.columns else 0
            current_price = recent_data['futures_price'].iloc[-1] if 'futures_price' in recent_data.columns else 50000

            # High basis suggests funding might increase (more expensive futures)
            basis_pct = current_basis / current_price if current_price > 0 else 0
            # Dampen basis adjustment for very small funding rates
            basis_dampening = min(1.0, abs(base_prediction) * 5000)  # Stronger dampening
            basis_adjustment = basis_pct * 0.3 * basis_dampening  # 30% of basis, dampened

            # Combine components
            predicted_funding = base_prediction + trend_component + basis_adjustment

            # Bound the prediction to reasonable values
            predicted_funding = np.clip(predicted_funding, -0.01, 0.01)  # -1% to +1%

            return float(predicted_funding)

        except Exception:
            return None

    def _calculate_funding_statistics(self, data: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """
        Calculate funding rate statistics for risk assessment

        Args:
            data: Funding data DataFrame
            current_index: Current position

        Returns:
            Dictionary of funding statistics
        """
        try:
            # Get recent data
            start_idx = max(0, current_index - self.funding_lookback)
            recent_data = data.iloc[start_idx:current_index + 1].copy()

            if len(recent_data) < 5:
                return {'mean_funding': 0, 'std_funding': 0, 'trend': 0, 'volatility': 0}

            funding_rates = recent_data['funding_rate'].values

            # Basic statistics
            mean_funding = np.mean(funding_rates)
            std_funding = np.std(funding_rates)

            # Trend calculation
            if len(funding_rates) >= 6:
                x = np.arange(len(funding_rates))
                trend_slope = np.polyfit(x, funding_rates, 1)[0]  # Linear trend slope
            else:
                trend_slope = 0

            # Volatility (rolling standard deviation)
            if len(funding_rates) >= 8:
                rolling_std = pd.Series(funding_rates).rolling(window=8).std().iloc[-1]
                volatility = rolling_std if not np.isnan(rolling_std) else std_funding
            else:
                volatility = std_funding

            return {
                'mean_funding': float(mean_funding),
                'std_funding': float(std_funding),
                'trend': float(trend_slope),
                'volatility': float(volatility)
            }

        except Exception:
            return {'mean_funding': 0, 'std_funding': 0, 'trend': 0, 'volatility': 0}

    def _evaluate_arbitrage_opportunity(self, predicted_funding: float, current_funding: float,
                                      current_basis: float, funding_stats: Dict[str, Any],
                                      current_price: float) -> Dict[str, Any]:
        """
        Evaluate arbitrage opportunity based on predicted funding and basis

        Args:
            predicted_funding: Predicted funding rate
            current_funding: Current funding rate
            current_basis: Current basis (futures - spot)
            funding_stats: Funding statistics
            current_price: Current price

        Returns:
            Dict with action, strength, and confidence
        """
        # Annualize funding rate (assuming 8-hour periods, 3 times per day)
        annualized_funding = predicted_funding * 365 * 3

        # Check if funding opportunity exceeds threshold
        funding_abs = abs(annualized_funding)
        if funding_abs < self.funding_threshold:
            return {'action': 'HOLD', 'strength': 0.0, 'confidence': 0.0}

        # Convert threshold to per-period rate for comparison
        period_threshold = self.funding_threshold / (365 * 3)  # Convert annualized to per-8h-period

        # Determine direction - compare predicted_funding to period threshold
        if predicted_funding < -period_threshold:
            # Negative funding: Go long to collect funding
            action = 'BUY'
            strength_base = min(1.0, funding_abs / self.funding_threshold)
        elif predicted_funding > period_threshold:
            # Positive funding: Go short to collect funding
            action = 'SELL'
            strength_base = min(1.0, funding_abs / self.funding_threshold)
        else:
            return {'action': 'HOLD', 'strength': 0.0, 'confidence': 0.0}

        # Base confidence from funding predictability
        funding_volatility = funding_stats.get('volatility', 0)
        confidence_base = min(0.9, max(0.3, 1.0 - funding_volatility * 1000))  # Scale volatility

        # Adjust for basis
        basis_pct = abs(current_basis / current_price) if current_price > 0 else 0

        # Unfavorable basis reduces confidence
        if basis_pct > self.basis_threshold:
            basis_penalty = min(0.5, (basis_pct - self.basis_threshold) / self.basis_threshold)
            confidence_base *= (1 - basis_penalty)

        # Trend consistency bonus
        trend_direction = 1 if funding_stats.get('trend', 0) > 0 else -1
        prediction_direction = 1 if predicted_funding > 0 else -1

        if trend_direction == prediction_direction:
            confidence_base *= 1.1  # 10% bonus for trend consistency

        # Final adjustments
        strength = strength_base * self.risk_factor
        confidence = confidence_base * self.risk_factor

        # Ensure bounds
        strength = np.clip(strength, 0.0, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)

        return {
            'action': action,
            'strength': float(strength),
            'confidence': float(confidence)
        }

    def _apply_risk_adjustments(self, signal_info: Dict[str, Any], funding_stats: Dict[str, Any],
                              current_basis: float) -> Dict[str, Any]:
        """
        Apply additional risk adjustments to the signal

        Args:
            signal_info: Base signal information
            funding_stats: Funding statistics
            current_basis: Current basis

        Returns:
            Risk-adjusted signal information
        """
        # High volatility reduces position size
        volatility = funding_stats.get('volatility', 0)
        if volatility > 0.0005:  # High funding volatility
            volatility_penalty = min(0.3, volatility * 1000)
            signal_info['strength'] *= (1 - volatility_penalty)
            signal_info['confidence'] *= (1 - volatility_penalty)

        # Delta neutral mode adjustment
        if self.delta_neutral:
            # In delta neutral mode, we're less aggressive
            signal_info['strength'] *= 0.8
            signal_info['confidence'] *= 0.9

        # Very large basis is risky
        if abs(current_basis) > 100:  # Large basis
            signal_info['strength'] *= 0.7
            signal_info['confidence'] *= 0.8

        # Ensure final bounds
        signal_info['strength'] = np.clip(signal_info['strength'], 0.0, 1.0)
        signal_info['confidence'] = np.clip(signal_info['confidence'], 0.0, 1.0)

        return signal_info

    def update_parameters(self, **kwargs) -> None:
        """
        Update strategy parameters

        Args:
            **kwargs: Parameter updates
        """
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)

        # Re-validate critical parameters
        if not (0.001 <= self.funding_threshold <= 0.1):
            raise ValueError("Funding threshold must be between 0.1% and 10%")

        if not (0.0005 <= self.basis_threshold <= 0.01):
            raise ValueError("Basis threshold must be between 0.05% and 1%")