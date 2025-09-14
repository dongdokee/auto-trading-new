"""
Strategy Engine Module

This module implements the core strategy system for the automated trading platform,
including regime detection, individual trading strategies, and strategy lifecycle management.

Key Components:
- BaseStrategy: Abstract interface for all trading strategies
- NoLookAheadRegimeDetector: Market regime detection with HMM/GARCH
- Individual Strategies: Trend following, mean reversion, range trading, funding arbitrage
- StrategyMatrix: Regime-based strategy allocation
- AlphaLifecycleManager: Strategy performance tracking and lifecycle management
- StrategyManager: Signal aggregation and coordination

Integration Points:
- Uses RiskController and PositionSizer from risk_management module
- Provides signals for position management system
- Integrates with backtesting framework for strategy validation
"""

from typing import Dict, List, Optional, Any

# Core interfaces and base classes
from .base_strategy import BaseStrategy, StrategySignal, StrategyConfig
from .regime_detector import NoLookAheadRegimeDetector
from .strategy_matrix import StrategyMatrix, StrategyAllocation
from .strategy_manager import StrategyManager

# Individual strategy implementations
from .strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    RangeTradingStrategy,
    FundingArbitrageStrategy,
)

__all__ = [
    # Core interfaces
    'BaseStrategy',
    'StrategySignal',
    'StrategyConfig',

    # System components
    'NoLookAheadRegimeDetector',
    'StrategyMatrix',
    'StrategyAllocation',
    'StrategyManager',

    # Strategy implementations
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'RangeTradingStrategy',
    'FundingArbitrageStrategy',
]

__version__ = "1.0.0"