"""
Trend Following Strategy

Implements a moving average crossover strategy with ATR-based position sizing and stops.
Uses fast and slow moving averages to identify trend direction and strength.
Includes volatility-adjusted signals and risk management through ATR-based stops.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..base_strategy import BaseStrategy, StrategySignal, StrategyConfig


class TrendFollowingStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy with ATR-based Risk Management

    Strategy Logic:
    - Uses fast and slow moving averages to identify trend direction
    - Requires minimum trend strength (MA spread / ATR) to generate signals
    - Sets stop losses and take profits based on ATR multiples
    - Scales signal strength based on trend magnitude relative to volatility
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize trend following strategy

        Args:
            config: Strategy configuration with parameters:
                - fast_period: Fast moving average period (default: 20)
                - slow_period: Slow moving average period (default: 50)
                - atr_period: ATR calculation period (default: 14)
                - atr_multiplier: ATR multiplier for stops (default: 2.0)
                - min_trend_strength: Minimum trend strength threshold (default: 0.3)
        """
        super().__init__(config)

        # Strategy parameters with defaults
        self.fast_period = self.parameters.get("fast_period", 20)
        self.slow_period = self.parameters.get("slow_period", 50)
        self.atr_period = self.parameters.get("atr_period", 14)
        self.atr_multiplier = self.parameters.get("atr_multiplier", 2.0)
        self.min_trend_strength = self.parameters.get("min_trend_strength", 0.3)

        # Validate parameters
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
        if self.atr_period <= 0 or self.atr_multiplier <= 0:
            raise ValueError("ATR period and multiplier must be positive")

    def generate_signal(self, market_data: Dict[str, Any]) -> StrategySignal:
        """
        Generate trend following signal based on moving average crossover

        Args:
            market_data: Dictionary containing:
                - symbol: Trading pair
                - close: Current price
                - ohlcv_data: Historical OHLCV DataFrame

        Returns:
            StrategySignal: Trading signal with trend-based action
        """
        # Validate market data
        if not self._validate_market_data(market_data):
            return self._hold_signal(market_data.get("symbol", "UNKNOWN"))

        # Extract data
        symbol = market_data["symbol"]
        current_price = float(market_data.get("close", 0))
        ohlcv_data = market_data.get("ohlcv_data")

        if not isinstance(ohlcv_data, pd.DataFrame) or len(ohlcv_data) < self.slow_period:
            return self._hold_signal(symbol)

        # Validate required columns
        required_columns = ['close', 'high', 'low']
        if not all(col in ohlcv_data.columns for col in required_columns):
            return self._hold_signal(symbol)

        try:
            # Calculate indicators
            fast_ma, slow_ma = self._calculate_moving_averages(ohlcv_data)
            atr = self.calculate_atr(ohlcv_data, period=self.atr_period)

            # Get latest values
            latest_fast = fast_ma.iloc[-1]
            latest_slow = slow_ma.iloc[-1]
            latest_atr = atr.iloc[-1]

            if pd.isna(latest_fast) or pd.isna(latest_slow) or pd.isna(latest_atr):
                return self._hold_signal(symbol)

            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(fast_ma, slow_ma, atr)

            # Generate signal based on trend direction and strength
            if trend_strength > self.min_trend_strength:
                # Bullish trend
                signal_strength = min(1.0, trend_strength / 2.0)
                confidence = min(0.9, 0.6 + signal_strength * 0.3)

                stop_loss = current_price - (self.atr_multiplier * latest_atr)
                take_profit = current_price + (self.atr_multiplier * latest_atr * 3)

                signal = StrategySignal(
                    symbol=symbol,
                    action="BUY",
                    strength=signal_strength,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "strategy_type": "trend_following",
                        "trend_strength": trend_strength,
                        "atr": latest_atr,
                        "fast_ma": latest_fast,
                        "slow_ma": latest_slow
                    }
                )

            elif trend_strength < -self.min_trend_strength:
                # Bearish trend
                signal_strength = min(1.0, abs(trend_strength) / 2.0)
                confidence = min(0.9, 0.6 + signal_strength * 0.3)

                stop_loss = current_price + (self.atr_multiplier * latest_atr)
                take_profit = current_price - (self.atr_multiplier * latest_atr * 3)

                signal = StrategySignal(
                    symbol=symbol,
                    action="SELL",
                    strength=signal_strength,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "strategy_type": "trend_following",
                        "trend_strength": trend_strength,
                        "atr": latest_atr,
                        "fast_ma": latest_fast,
                        "slow_ma": latest_slow
                    }
                )

            else:
                # No clear trend or insufficient strength
                signal = self._hold_signal(symbol)
                signal.metadata = {
                    "strategy_type": "trend_following",
                    "trend_strength": trend_strength,
                    "reason": "insufficient_trend_strength"
                }

            # Log signal for analysis
            self._log_signal(signal)

            return signal

        except Exception as e:
            # Handle any calculation errors gracefully
            return self._hold_signal(symbol, error=str(e))

    def _calculate_moving_averages(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate fast and slow moving averages

        Args:
            data: OHLCV DataFrame

        Returns:
            Tuple of (fast_ma, slow_ma) Series
        """
        close_prices = data['close']
        fast_ma = close_prices.rolling(window=self.fast_period).mean()
        slow_ma = close_prices.rolling(window=self.slow_period).mean()

        return fast_ma, slow_ma

    def _calculate_trend_strength(self, fast_ma: pd.Series, slow_ma: pd.Series, atr: pd.Series) -> float:
        """
        Calculate trend strength as MA spread normalized by ATR

        Args:
            fast_ma: Fast moving average series
            slow_ma: Slow moving average series
            atr: Average True Range series

        Returns:
            float: Trend strength (positive = bullish, negative = bearish)
        """
        latest_fast = fast_ma.iloc[-1]
        latest_slow = slow_ma.iloc[-1]
        latest_atr = atr.iloc[-1]

        if pd.isna(latest_fast) or pd.isna(latest_slow) or pd.isna(latest_atr) or latest_atr == 0:
            return 0.0

        # Normalize MA spread by ATR to account for volatility
        trend_strength = (latest_fast - latest_slow) / latest_atr

        return float(trend_strength)

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range

        Args:
            data: OHLCV DataFrame with 'high', 'low', 'close' columns
            period: ATR calculation period

        Returns:
            pd.Series: ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR as rolling mean of True Range
        atr = true_range.rolling(window=period).mean()

        return atr

    def _hold_signal(self, symbol: str, error: Optional[str] = None) -> StrategySignal:
        """
        Create a HOLD signal

        Args:
            symbol: Trading symbol
            error: Optional error message

        Returns:
            StrategySignal: HOLD signal
        """
        metadata = {"strategy_type": "trend_following"}
        if error:
            metadata["error"] = error

        return StrategySignal(
            symbol=symbol,
            action="HOLD",
            strength=0.0,
            confidence=0.5,
            metadata=metadata
        )

    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """
        Validate market data structure

        Args:
            market_data: Market data dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        if not super()._validate_market_data(market_data):
            return False

        # Check for required fields
        required_fields = ["ohlcv_data", "close"]
        for field in required_fields:
            if field not in market_data:
                return False

        # Validate ohlcv_data is DataFrame
        if not isinstance(market_data["ohlcv_data"], pd.DataFrame):
            return False

        # Validate close price is numeric
        try:
            float(market_data["close"])
        except (ValueError, TypeError):
            return False

        return True

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information including parameters

        Returns:
            dict: Strategy info with parameters
        """
        info = super().get_strategy_info()
        info.update({
            "strategy_type": "trend_following",
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "min_trend_strength": self.min_trend_strength
        })
        return info