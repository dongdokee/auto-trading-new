"""
Mean Reversion Strategy

Implements a Bollinger Bands + RSI mean reversion strategy.
Identifies overbought/oversold conditions and generates signals for mean reversion trades.
Uses Bollinger Bands to identify price extremes and RSI for momentum confirmation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..base_strategy import BaseStrategy, StrategySignal, StrategyConfig


class MeanReversionStrategy(BaseStrategy):
    """
    Bollinger Bands + RSI Mean Reversion Strategy

    Strategy Logic:
    - Uses Bollinger Bands to identify price extremes relative to moving average
    - Uses RSI to confirm overbought/oversold momentum conditions
    - Generates BUY signals when price is below lower BB and RSI is oversold
    - Generates SELL signals when price is above upper BB and RSI is overbought
    - Requires alignment between both indicators for signal generation
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize mean reversion strategy

        Args:
            config: Strategy configuration with parameters:
                - bb_period: Bollinger Bands period (default: 20)
                - bb_std: Bollinger Bands standard deviation (default: 2.0)
                - rsi_period: RSI calculation period (default: 14)
                - rsi_oversold: RSI oversold threshold (default: 30)
                - rsi_overbought: RSI overbought threshold (default: 70)
                - min_confidence: Minimum confidence for signal generation (default: 0.5)
        """
        super().__init__(config)

        # Strategy parameters with defaults
        self.bb_period = self.parameters.get("bb_period", 20)
        self.bb_std = self.parameters.get("bb_std", 2.0)
        self.rsi_period = self.parameters.get("rsi_period", 14)
        self.rsi_oversold = self.parameters.get("rsi_oversold", 30)
        self.rsi_overbought = self.parameters.get("rsi_overbought", 70)
        self.min_confidence = self.parameters.get("min_confidence", 0.5)

        # Validate parameters
        if self.bb_period <= 0 or self.bb_std <= 0:
            raise ValueError("Bollinger Bands parameters must be positive")
        if self.rsi_period <= 0:
            raise ValueError("RSI period must be positive")
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("RSI oversold threshold must be less than overbought threshold")
        if not (0 <= self.rsi_oversold <= 100) or not (0 <= self.rsi_overbought <= 100):
            raise ValueError("RSI thresholds must be between 0 and 100")

    def generate_signal(self, market_data: Dict[str, Any]) -> StrategySignal:
        """
        Generate mean reversion signal based on Bollinger Bands and RSI

        Args:
            market_data: Dictionary containing:
                - symbol: Trading pair
                - close: Current price
                - ohlcv_data: Historical OHLCV DataFrame

        Returns:
            StrategySignal: Trading signal with mean reversion action
        """
        # Validate market data
        if not self._validate_market_data(market_data):
            return self._hold_signal(market_data.get("symbol", "UNKNOWN"))

        # Extract data
        symbol = market_data["symbol"]
        current_price = float(market_data.get("close", 0))
        ohlcv_data = market_data.get("ohlcv_data")

        min_required_data = max(self.bb_period, self.rsi_period) + 1
        if not isinstance(ohlcv_data, pd.DataFrame) or len(ohlcv_data) < min_required_data:
            return self._hold_signal(symbol)

        # Validate required columns
        required_columns = ['close']
        if not all(col in ohlcv_data.columns for col in required_columns):
            return self._hold_signal(symbol)

        try:
            # Calculate indicators
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(ohlcv_data['close'])
            rsi = self.calculate_rsi(ohlcv_data['close'], period=self.rsi_period)

            # Get latest values
            latest_bb_upper = bb_upper.iloc[-1]
            latest_bb_middle = bb_middle.iloc[-1]
            latest_bb_lower = bb_lower.iloc[-1]
            latest_rsi = rsi.iloc[-1]

            if (pd.isna(latest_bb_upper) or pd.isna(latest_bb_middle) or
                pd.isna(latest_bb_lower) or pd.isna(latest_rsi)):
                return self._hold_signal(symbol)

            # Calculate position relative to Bollinger Bands
            bb_position = self._calculate_bb_position(
                current_price, latest_bb_upper, latest_bb_middle, latest_bb_lower
            )

            # Calculate mean reversion strength
            mr_strength = self._calculate_mean_reversion_strength(bb_position, latest_rsi)

            # Generate signals based on conditions
            signal = None

            # Buy signal: oversold conditions
            if (current_price < latest_bb_lower and
                latest_rsi < self.rsi_oversold and
                mr_strength > 0):

                confidence = self._calculate_confidence(bb_position, latest_rsi, "BUY")

                if confidence >= self.min_confidence:
                    signal = StrategySignal(
                        symbol=symbol,
                        action="BUY",
                        strength=min(1.0, mr_strength),
                        confidence=confidence,
                        take_profit=latest_bb_middle,  # Target middle band
                        stop_loss=latest_bb_lower * 0.98,  # Stop below lower band
                        metadata={
                            "strategy_type": "mean_reversion",
                            "bb_position": bb_position,
                            "rsi": latest_rsi,
                            "bb_upper": latest_bb_upper,
                            "bb_middle": latest_bb_middle,
                            "bb_lower": latest_bb_lower,
                            "mean_reversion_strength": mr_strength
                        }
                    )

            # Sell signal: overbought conditions
            elif (current_price > latest_bb_upper and
                  latest_rsi > self.rsi_overbought and
                  mr_strength < 0):

                confidence = self._calculate_confidence(bb_position, latest_rsi, "SELL")

                if confidence >= self.min_confidence:
                    signal = StrategySignal(
                        symbol=symbol,
                        action="SELL",
                        strength=min(1.0, abs(mr_strength)),
                        confidence=confidence,
                        take_profit=latest_bb_middle,  # Target middle band
                        stop_loss=latest_bb_upper * 1.02,  # Stop above upper band
                        metadata={
                            "strategy_type": "mean_reversion",
                            "bb_position": bb_position,
                            "rsi": latest_rsi,
                            "bb_upper": latest_bb_upper,
                            "bb_middle": latest_bb_middle,
                            "bb_lower": latest_bb_lower,
                            "mean_reversion_strength": mr_strength
                        }
                    )

            # Default to HOLD if no clear signal
            if signal is None:
                signal = self._hold_signal(symbol)
                signal.metadata = {
                    "strategy_type": "mean_reversion",
                    "bb_position": bb_position,
                    "rsi": latest_rsi,
                    "reason": "conditions_not_met"
                }

            # Log signal for analysis
            self._log_signal(signal)

            return signal

        except Exception as e:
            # Handle any calculation errors gracefully
            return self._hold_signal(symbol, error=str(e))

    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands

        Args:
            prices: Price series

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # Middle band (Simple Moving Average)
        middle_band = prices.rolling(window=self.bb_period).mean()

        # Standard deviation
        std = prices.rolling(window=self.bb_period).std()

        # Upper and lower bands
        upper_band = middle_band + (self.bb_std * std)
        lower_band = middle_band - (self.bb_std * std)

        return upper_band, middle_band, lower_band

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index

        Args:
            prices: Price series
            period: RSI calculation period

        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-10)  # Add small epsilon to avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_bb_position(self, price: float, bb_upper: float,
                              bb_middle: float, bb_lower: float) -> float:
        """
        Calculate position relative to Bollinger Bands

        Args:
            price: Current price
            bb_upper: Upper Bollinger Band
            bb_middle: Middle Bollinger Band (SMA)
            bb_lower: Lower Bollinger Band

        Returns:
            float: Position relative to bands (0 = middle, 1 = upper, -1 = lower)
        """
        band_width = bb_upper - bb_lower
        if band_width == 0:
            return 0.0

        # Position relative to middle band, normalized by band width
        bb_position = (price - bb_middle) / (band_width / 2)

        return float(bb_position)

    def _calculate_mean_reversion_strength(self, bb_position: float, rsi: float) -> float:
        """
        Calculate mean reversion strength based on BB position and RSI

        Args:
            bb_position: Position relative to Bollinger Bands
            rsi: Current RSI value

        Returns:
            float: Mean reversion strength (positive = bullish, negative = bearish)
        """
        # RSI contribution to strength
        if rsi < self.rsi_oversold:
            rsi_strength = (self.rsi_oversold - rsi) / self.rsi_oversold
        elif rsi > self.rsi_overbought:
            rsi_strength = -(rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
        else:
            rsi_strength = 0.0

        # BB position contribution
        if bb_position < -1:  # Below lower band
            bb_strength = abs(bb_position) - 1
        elif bb_position > 1:  # Above upper band
            bb_strength = -(bb_position - 1)
        else:
            bb_strength = 0.0

        # Combine strengths (both must align for strong signal)
        if (rsi_strength > 0 and bb_strength > 0) or (rsi_strength < 0 and bb_strength < 0):
            # Aligned signals - multiply strengths
            combined_strength = rsi_strength * 0.6 + bb_strength * 0.4
        else:
            # Conflicting signals - reduce strength
            combined_strength = (rsi_strength + bb_strength) * 0.3

        return float(combined_strength)

    def _calculate_confidence(self, bb_position: float, rsi: float, action: str) -> float:
        """
        Calculate confidence based on indicator alignment

        Args:
            bb_position: Position relative to Bollinger Bands
            rsi: Current RSI value
            action: Intended action ("BUY" or "SELL")

        Returns:
            float: Confidence level [0, 1]
        """
        if action == "BUY":
            # Buy confidence: how oversold and how far below lower band
            rsi_confidence = max(0, (self.rsi_oversold - rsi) / self.rsi_oversold)
            bb_confidence = max(0, min(1, abs(bb_position) - 1)) if bb_position < -1 else 0
        else:  # SELL
            # Sell confidence: how overbought and how far above upper band
            rsi_confidence = max(0, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought))
            bb_confidence = max(0, min(1, bb_position - 1)) if bb_position > 1 else 0

        # Base confidence from alignment
        base_confidence = 0.5 + (rsi_confidence * 0.3) + (bb_confidence * 0.2)

        return min(1.0, base_confidence)

    def _hold_signal(self, symbol: str, error: Optional[str] = None) -> StrategySignal:
        """
        Create a HOLD signal

        Args:
            symbol: Trading symbol
            error: Optional error message

        Returns:
            StrategySignal: HOLD signal
        """
        metadata = {"strategy_type": "mean_reversion"}
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
            "strategy_type": "mean_reversion",
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "min_confidence": self.min_confidence
        })
        return info