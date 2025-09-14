"""
Individual Trading Strategies

This module contains implementations of various trading strategies used by the system:

- TrendFollowingStrategy: Moving average crossover with ATR-based stops
- MeanReversionStrategy: Bollinger Bands + RSI combination for mean reversion
- RangeTradingStrategy: Support/resistance identification for sideways markets
- FundingArbitrageStrategy: USDT-based funding rate arbitrage (delta-neutral/directional)

Each strategy implements the BaseStrategy interface and provides:
- Signal generation based on market conditions
- Risk-adjusted position sizing recommendations
- Stop loss and take profit logic
- Strategy-specific performance metrics
"""

from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .range_trading import RangeTradingStrategy
from .funding_arbitrage import FundingArbitrageStrategy

__all__ = [
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'RangeTradingStrategy',
    'FundingArbitrageStrategy',
]