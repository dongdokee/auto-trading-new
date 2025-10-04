"""
Range Trading Strategy

Implements a support/resistance based trading strategy for sideways markets.
Identifies key support and resistance levels using pivot points and price action,
generates buy signals near support and sell signals near resistance.
Includes volume confirmation and breakout detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.strategy_engine.base_strategy import BaseStrategy, StrategySignal, StrategyConfig


class RangeTradingStrategy(BaseStrategy):
    """
    Support/Resistance Range Trading Strategy

    Strategy Logic:
    - Identifies support and resistance levels using pivot points and price action
    - Generates BUY signals when price approaches support with volume confirmation
    - Generates SELL signals when price approaches resistance with volume confirmation
    - Sets tight ATR-based stops and targets at opposite levels
    - Avoids trading during strong breakout conditions
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize range trading strategy

        Args:
            config: Strategy configuration with parameters:
                - lookback_period: Period for support/resistance detection (default: 20)
                - support_resistance_threshold: Distance from S/R levels (default: 0.02)
                - volume_confirmation: Require volume confirmation (default: True)
                - atr_period: ATR calculation period (default: 14)
                - stop_multiplier: ATR multiplier for stops (default: 1.5)
                - target_multiplier: ATR multiplier for targets (default: 2.0)
        """
        super().__init__(config)

        # Strategy parameters with defaults
        self.lookback_period = self.parameters.get("lookback_period", 20)
        self.support_resistance_threshold = self.parameters.get("support_resistance_threshold", 0.02)
        self.volume_confirmation = self.parameters.get("volume_confirmation", True)
        self.atr_period = self.parameters.get("atr_period", 14)
        self.stop_multiplier = self.parameters.get("stop_multiplier", 1.5)
        self.target_multiplier = self.parameters.get("target_multiplier", 2.0)

        # Validate parameters
        if self.lookback_period < 10:
            raise ValueError("Lookback period must be at least 10")

        if not (0.005 <= self.support_resistance_threshold <= 0.1):
            raise ValueError("Support/resistance threshold must be between 0.5% and 10%")

        if self.atr_period < 5:
            raise ValueError("ATR period must be at least 5")

        if self.stop_multiplier <= 0 or self.target_multiplier <= 0:
            raise ValueError("Stop and target multipliers must be positive")

    def generate_signal(self, market_data: Dict[str, Any], current_index: int = -1) -> StrategySignal:
        """
        Generate range trading signal based on support/resistance levels

        Args:
            market_data: Dictionary containing:
                - symbol: Trading pair symbol
                - close: Current price
                - ohlcv_data: Historical OHLC data
            current_index: Current position in the data (for backtesting)

        Returns:
            StrategySignal: Generated trading signal
        """
        symbol = market_data.get('symbol', 'UNKNOWN')
        current_price = market_data.get('close', 0.0)
        ohlcv_data = market_data.get('ohlcv_data')

        # Default HOLD signal
        default_signal = StrategySignal(
            symbol=symbol,
            action='HOLD',
            strength=0.0,
            confidence=0.0,
            metadata={'strategy': 'RangeTrading', 'reason': 'default'}
        )

        if ohlcv_data is None or len(ohlcv_data) < self.lookback_period:
            return default_signal

        try:
            # Use current_index or last available data
            if current_index == -1:
                current_index = len(ohlcv_data) - 1
            elif current_index >= len(ohlcv_data):
                current_index = len(ohlcv_data) - 1

            # Ensure we have enough data
            if current_index < self.lookback_period:
                return default_signal

            # Identify support and resistance levels
            support_level, resistance_level = self._identify_support_resistance(
                ohlcv_data, current_index
            )

            if support_level is None or resistance_level is None:
                return default_signal

            # Calculate ATR for stops and targets
            atr = self._calculate_atr(ohlcv_data, current_index)

            # Determine signal based on price relative to support/resistance
            signal_info = self._evaluate_range_position(
                current_price, support_level, resistance_level, atr
            )

            if signal_info['action'] == 'HOLD':
                return default_signal

            # Apply volume confirmation if enabled
            if self.volume_confirmation:
                volume_factor = self._check_volume_confirmation(ohlcv_data, current_index)
                signal_info['confidence'] *= volume_factor
                signal_info['strength'] *= volume_factor

            # Check for range breakout (reduces confidence)
            breakout_factor = self._check_breakout_conditions(
                current_price, support_level, resistance_level, ohlcv_data, current_index
            )
            signal_info['confidence'] *= breakout_factor

            # Set stop loss and take profit
            stop_loss, take_profit = self._calculate_stops_and_targets(
                signal_info['action'], current_price, atr, support_level, resistance_level
            )

            # Create signal
            signal = StrategySignal(
                symbol=symbol,
                action=signal_info['action'],
                strength=signal_info['strength'],
                confidence=signal_info['confidence'],
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'strategy': 'RangeTrading',
                    'support_level': support_level,
                    'resistance_level': resistance_level,
                    'atr': atr,
                    'range_width': (resistance_level - support_level) / support_level,
                    'volume_confirmed': self.volume_confirmation,
                    'breakout_factor': breakout_factor
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
                metadata={'strategy': 'RangeTrading', 'error': str(e)}
            )

    def _identify_support_resistance(self, data: pd.DataFrame, current_index: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Identify support and resistance levels using pivot points

        Args:
            data: OHLCV DataFrame
            current_index: Current position in data

        Returns:
            Tuple of (support_level, resistance_level) or (None, None)
        """
        try:
            # Get recent data for analysis
            start_idx = max(0, current_index - self.lookback_period)
            recent_data = data.iloc[start_idx:current_index + 1].copy()

            if len(recent_data) < 10:  # Need minimum data
                return None, None

            # Calculate pivot points (highs and lows)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            closes = recent_data['close'].values

            # Find pivot highs (local maxima)
            pivot_highs = []
            for i in range(1, len(highs) - 1):  # Simplified pivot detection
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    pivot_highs.append(highs[i])

            # Find pivot lows (local minima)
            pivot_lows = []
            for i in range(1, len(lows) - 1):  # Simplified pivot detection
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    pivot_lows.append(lows[i])

            # Use broader percentile approach for better range detection
            all_highs = highs[-15:]  # Recent highs
            all_lows = lows[-15:]   # Recent lows
            all_closes = closes[-15:]  # Recent closes

            # Calculate support and resistance with wider spread
            if len(pivot_lows) > 0:
                support_level = np.percentile(pivot_lows, 60)  # 60th percentile of pivot lows
            else:
                support_level = np.percentile(all_lows, 15)  # 15th percentile of lows

            if len(pivot_highs) > 0:
                resistance_level = np.percentile(pivot_highs, 40)  # 40th percentile of pivot highs
            else:
                resistance_level = np.percentile(all_highs, 85)  # 85th percentile of highs

            # Ensure minimum range width (at least 1.5%)
            current_price = closes[-1]
            min_range = current_price * 0.015  # 1.5% minimum range

            if (resistance_level - support_level) < min_range:
                # Expand the range
                mid_point = (support_level + resistance_level) / 2
                support_level = mid_point - min_range / 2
                resistance_level = mid_point + min_range / 2

            # Final validation
            if resistance_level > support_level:
                range_width = (resistance_level - support_level) / support_level
                if range_width >= 0.01:  # At least 1% range
                    return float(support_level), float(resistance_level)

            # Fallback: use percentile approach with guaranteed range
            support_level = np.percentile(all_closes, 10)  # 10th percentile
            resistance_level = np.percentile(all_closes, 90)  # 90th percentile

            return float(support_level), float(resistance_level)

        except Exception:
            return None, None

    def _evaluate_range_position(self, current_price: float, support: float,
                                resistance: float, atr: float) -> Dict[str, Any]:
        """
        Evaluate current price position within the range

        Args:
            current_price: Current market price
            support: Support level
            resistance: Resistance level
            atr: Average True Range

        Returns:
            Dict with action, strength, and confidence
        """
        range_width = resistance - support
        price_position = (current_price - support) / range_width  # 0 = support, 1 = resistance

        # Distance from support/resistance as percentage
        distance_from_support = abs(current_price - support) / support
        distance_from_resistance = abs(current_price - resistance) / resistance

        # More precise logic: check which level is closer
        distance_to_support = abs(current_price - support)
        distance_to_resistance = abs(current_price - resistance)

        # Check if near support (buy zone) AND closer to support than resistance
        if (distance_from_support <= self.support_resistance_threshold and
            distance_to_support < distance_to_resistance):
            # Strength based on how close to support
            strength = 1.0 - (distance_from_support / self.support_resistance_threshold)
            confidence = min(0.8, strength * 0.9)  # High confidence near support

            return {
                'action': 'BUY',
                'strength': strength,
                'confidence': confidence
            }

        # Check if near resistance (sell zone) AND closer to resistance than support
        elif (distance_from_resistance <= self.support_resistance_threshold and
              distance_to_resistance < distance_to_support):
            # Strength based on how close to resistance
            strength = 1.0 - (distance_from_resistance / self.support_resistance_threshold)
            confidence = min(0.8, strength * 0.9)  # High confidence near resistance

            return {
                'action': 'SELL',
                'strength': strength,
                'confidence': confidence
            }

        # In middle of range - no clear signal
        else:
            return {
                'action': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0
            }

    def _check_volume_confirmation(self, data: pd.DataFrame, current_index: int) -> float:
        """
        Check if current volume confirms the signal

        Args:
            data: OHLCV DataFrame
            current_index: Current position

        Returns:
            Volume confirmation factor (0.5 to 1.0)
        """
        try:
            if 'volume' not in data.columns:
                return 1.0  # No volume data available

            # Get recent volume data
            recent_volume = data['volume'].iloc[max(0, current_index-10):current_index+1]

            if len(recent_volume) < 5:
                return 1.0

            current_volume = recent_volume.iloc[-1]
            avg_volume = recent_volume.mean()

            # Volume confirmation factor
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Higher volume = better confirmation
            if volume_ratio >= 1.5:  # 50% above average
                return 1.0
            elif volume_ratio >= 1.2:  # 20% above average
                return 0.9
            elif volume_ratio >= 0.8:  # Normal volume
                return 0.8
            else:  # Low volume
                return 0.6

        except Exception:
            return 1.0

    def _check_breakout_conditions(self, current_price: float, support: float,
                                 resistance: float, data: pd.DataFrame,
                                 current_index: int) -> float:
        """
        Check for range breakout conditions that might invalidate range trading

        Args:
            current_price: Current price
            support: Support level
            resistance: Resistance level
            data: OHLCV DataFrame
            current_index: Current position

        Returns:
            Breakout factor (0.3 to 1.0) - lower values indicate breakout
        """
        try:
            # Check if price is significantly outside range
            range_width = resistance - support
            breakout_threshold = range_width * 0.1  # 10% of range width

            if current_price > resistance + breakout_threshold:
                # Strong upside breakout
                return 0.3
            elif current_price < support - breakout_threshold:
                # Strong downside breakout
                return 0.3

            # Check momentum (recent price movement)
            if current_index >= 5:
                recent_prices = data['close'].iloc[current_index-4:current_index+1]
                price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

                if abs(price_momentum) > 0.05:  # 5% momentum
                    return 0.6  # Moderate momentum reduces range trading confidence
                else:
                    return 1.0  # No strong momentum
            else:
                return 1.0

        except Exception:
            return 1.0

    def _calculate_atr(self, data: pd.DataFrame, current_index: int) -> float:
        """
        Calculate Average True Range for the given period

        Args:
            data: OHLCV DataFrame
            current_index: Current position in data

        Returns:
            ATR value
        """
        try:
            end_idx = current_index + 1
            start_idx = max(0, end_idx - self.atr_period)

            atr_data = data.iloc[start_idx:end_idx]

            if len(atr_data) < 2:
                return data['close'].iloc[current_index] * 0.02  # 2% fallback

            # Calculate True Range
            high_low = atr_data['high'] - atr_data['low']
            high_close_prev = abs(atr_data['high'] - atr_data['close'].shift(1))
            low_close_prev = abs(atr_data['low'] - atr_data['close'].shift(1))

            true_ranges = pd.DataFrame({
                'hl': high_low,
                'hcp': high_close_prev,
                'lcp': low_close_prev
            }).max(axis=1)

            # Remove NaN values
            true_ranges = true_ranges.dropna()

            if len(true_ranges) == 0:
                return data['close'].iloc[current_index] * 0.02

            return float(true_ranges.mean())

        except Exception:
            # Fallback ATR calculation
            return data['close'].iloc[current_index] * 0.02

    def _calculate_stops_and_targets(self, action: str, current_price: float, atr: float,
                                   support: float, resistance: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate stop loss and take profit levels

        Args:
            action: Signal action (BUY/SELL)
            current_price: Current market price
            atr: Average True Range
            support: Support level
            resistance: Resistance level

        Returns:
            Tuple of (stop_loss, take_profit)
        """
        try:
            if action == 'BUY':
                # For BUY: Stop below support, target near resistance
                stop_loss = current_price - (atr * self.stop_multiplier)
                # Target is conservative - don't go all the way to resistance
                target_distance = min(atr * self.target_multiplier, (resistance - current_price) * 0.8)
                take_profit = current_price + target_distance

            elif action == 'SELL':
                # For SELL: Stop above resistance, target near support
                stop_loss = current_price + (atr * self.stop_multiplier)
                # Target is conservative - don't go all the way to support
                target_distance = min(atr * self.target_multiplier, (current_price - support) * 0.8)
                take_profit = current_price - target_distance

            else:
                return None, None

            return float(stop_loss), float(take_profit)

        except Exception:
            return None, None

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
        if self.lookback_period < 10:
            raise ValueError("Lookback period must be at least 10")

        if not (0.005 <= self.support_resistance_threshold <= 0.1):
            raise ValueError("Support/resistance threshold must be between 0.5% and 10%")