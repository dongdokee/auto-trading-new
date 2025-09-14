"""
Utilities package for the AutoTrading system.
Provides financial mathematics, time utilities, and logging capabilities.
"""

# Import financial math utilities
from .financial_math import (
    calculate_returns,
    calculate_log_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar,
    calculate_beta,
    calculate_correlation_matrix,
    calculate_compound_return,
    calculate_annualized_return,
    normalize_prices,
    calculate_rolling_correlation,
    calculate_information_ratio,
    calculate_treynor_ratio
)

# Import time utilities
from .time_utils import (
    get_market_timezone,
    is_market_open,
    get_next_market_open,
    get_next_market_close,
    get_trading_calendar,
    convert_to_market_time,
    is_weekend,
    is_business_day,
    get_business_days_between,
    format_duration,
    parse_duration,
    get_epoch_timestamp,
    get_utc_now,
    round_to_timeframe,
    get_timeframe_seconds,
    generate_trading_sessions,
    is_asian_trading_hours,
    is_european_trading_hours,
    is_us_trading_hours
)

# Import logging utilities
from .logger import TradingLogger, TradeContext, get_trading_logger

__all__ = [
    # Financial math
    'calculate_returns',
    'calculate_log_returns',
    'calculate_volatility',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_max_drawdown',
    'calculate_var',
    'calculate_cvar',
    'calculate_beta',
    'calculate_correlation_matrix',
    'calculate_compound_return',
    'calculate_annualized_return',
    'normalize_prices',
    'calculate_rolling_correlation',
    'calculate_information_ratio',
    'calculate_treynor_ratio',

    # Time utilities
    'get_market_timezone',
    'is_market_open',
    'get_next_market_open',
    'get_next_market_close',
    'get_trading_calendar',
    'convert_to_market_time',
    'is_weekend',
    'is_business_day',
    'get_business_days_between',
    'format_duration',
    'parse_duration',
    'get_epoch_timestamp',
    'get_utc_now',
    'round_to_timeframe',
    'get_timeframe_seconds',
    'generate_trading_sessions',
    'is_asian_trading_hours',
    'is_european_trading_hours',
    'is_us_trading_hours',

    # Logging
    'TradingLogger',
    'TradeContext',
    'get_trading_logger'
]