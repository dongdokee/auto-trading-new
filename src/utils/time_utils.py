"""
Time utility functions for the AutoTrading system.
Provides market hours, timezone handling, and trading calendar utilities.
"""

import re
from datetime import datetime, timedelta, timezone, time
from typing import Dict, List, Optional, Union, Tuple
import pytz
import pandas as pd
import numpy as np


# Market timezone mappings
MARKET_TIMEZONES = {
    'BINANCE': 'UTC',
    'BYBIT': 'UTC',
    'OKX': 'UTC',
    'COINBASE': 'UTC',
    'KRAKEN': 'UTC',
    'NYSE': 'America/New_York',
    'NASDAQ': 'America/New_York',
    'LSE': 'Europe/London',
    'TSE': 'Asia/Tokyo',
    'HKEX': 'Asia/Hong_Kong',
    'ASX': 'Australia/Sydney',
    'FOREX': 'UTC'
}

# Market trading hours (in local time)
MARKET_HOURS = {
    'NYSE': {'open': time(9, 30), 'close': time(16, 0)},
    'NASDAQ': {'open': time(9, 30), 'close': time(16, 0)},
    'LSE': {'open': time(8, 0), 'close': time(16, 30)},
    'TSE': {'open': time(9, 0), 'close': time(15, 0)},
    'HKEX': {'open': time(9, 30), 'close': time(16, 0)},
    'ASX': {'open': time(10, 0), 'close': time(16, 0)}
}

# Crypto exchanges are always open
CRYPTO_EXCHANGES = {'BINANCE', 'BYBIT', 'OKX', 'COINBASE', 'KRAKEN'}

# US market holidays (simplified - in practice would use a proper holiday calendar)
US_HOLIDAYS_2023 = [
    '2023-01-02',  # New Year's Day (observed)
    '2023-01-16',  # Martin Luther King Jr. Day
    '2023-02-20',  # Presidents' Day
    '2023-04-07',  # Good Friday
    '2023-05-29',  # Memorial Day
    '2023-06-19',  # Juneteenth
    '2023-07-04',  # Independence Day
    '2023-09-04',  # Labor Day
    '2023-11-23',  # Thanksgiving
    '2023-12-25',  # Christmas Day
]


def get_market_timezone(exchange: str) -> pytz.BaseTzInfo:
    """
    Get timezone for a specific exchange.

    Args:
        exchange: Exchange name

    Returns:
        Timezone object
    """
    timezone_str = MARKET_TIMEZONES.get(exchange.upper(), 'UTC')
    return pytz.timezone(timezone_str)


def convert_to_market_time(dt: datetime, exchange: str) -> datetime:
    """
    Convert datetime to market timezone.

    Args:
        dt: Input datetime
        exchange: Exchange name

    Returns:
        Datetime in market timezone
    """
    market_tz = get_market_timezone(exchange)

    # If datetime is naive, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(market_tz)


def is_market_open(exchange: str, dt: Optional[datetime] = None) -> bool:
    """
    Check if market is currently open.

    Args:
        exchange: Exchange name
        dt: Datetime to check (default: current time)

    Returns:
        True if market is open
    """
    if dt is None:
        dt = get_utc_now()

    exchange = exchange.upper()

    # Crypto markets are always open
    if exchange in CRYPTO_EXCHANGES:
        return True

    # Convert to market timezone
    market_time = convert_to_market_time(dt, exchange)

    # Check if it's a weekend
    if market_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Check if it's a holiday (simplified check)
    date_str = market_time.strftime('%Y-%m-%d')
    if date_str in US_HOLIDAYS_2023:
        return False

    # Check trading hours
    if exchange in MARKET_HOURS:
        hours = MARKET_HOURS[exchange]
        current_time = market_time.time()
        return hours['open'] <= current_time < hours['close']

    # Default: assume market is open during business hours
    return 9 <= market_time.hour < 17 and market_time.weekday() < 5


def get_next_market_open(exchange: str, dt: Optional[datetime] = None) -> datetime:
    """
    Get next market open time.

    Args:
        exchange: Exchange name
        dt: Reference datetime (default: current time)

    Returns:
        Next market open datetime
    """
    if dt is None:
        dt = get_utc_now()

    exchange = exchange.upper()

    # Crypto markets are always open
    if exchange in CRYPTO_EXCHANGES:
        return dt

    market_time = convert_to_market_time(dt, exchange)

    # If market is currently open, return next day's open
    if is_market_open(exchange, dt):
        next_day = market_time + timedelta(days=1)
    else:
        next_day = market_time

    # Find next business day
    while next_day.weekday() >= 5 or next_day.strftime('%Y-%m-%d') in US_HOLIDAYS_2023:
        next_day += timedelta(days=1)

    # Set to market open time
    if exchange in MARKET_HOURS:
        open_time = MARKET_HOURS[exchange]['open']
        next_open = next_day.replace(
            hour=open_time.hour,
            minute=open_time.minute,
            second=0,
            microsecond=0
        )
    else:
        next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)

    return next_open.astimezone(timezone.utc)


def get_next_market_close(exchange: str, dt: Optional[datetime] = None) -> datetime:
    """
    Get next market close time.

    Args:
        exchange: Exchange name
        dt: Reference datetime (default: current time)

    Returns:
        Next market close datetime
    """
    if dt is None:
        dt = get_utc_now()

    exchange = exchange.upper()

    # Crypto markets never close
    if exchange in CRYPTO_EXCHANGES:
        return dt + timedelta(days=365)  # Far future

    market_time = convert_to_market_time(dt, exchange)

    # If market is currently open, return today's close
    if is_market_open(exchange, dt):
        close_day = market_time
    else:
        # Find next business day
        close_day = market_time + timedelta(days=1)
        while close_day.weekday() >= 5 or close_day.strftime('%Y-%m-%d') in US_HOLIDAYS_2023:
            close_day += timedelta(days=1)

    # Set to market close time
    if exchange in MARKET_HOURS:
        close_time = MARKET_HOURS[exchange]['close']
        next_close = close_day.replace(
            hour=close_time.hour,
            minute=close_time.minute,
            second=0,
            microsecond=0
        )
    else:
        next_close = close_day.replace(hour=16, minute=0, second=0, microsecond=0)

    return next_close.astimezone(timezone.utc)


def get_trading_calendar(exchange: str, start_date: datetime,
                        end_date: datetime) -> List[datetime]:
    """
    Get trading calendar for date range.

    Args:
        exchange: Exchange name
        start_date: Start date
        end_date: End date

    Returns:
        List of trading days
    """
    exchange = exchange.upper()

    # Crypto exchanges trade every day
    if exchange in CRYPTO_EXCHANGES:
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    # Traditional markets: business days excluding holidays
    dates = []
    current = start_date

    while current <= end_date:
        # Skip weekends
        if current.weekday() < 5:
            # Skip holidays
            date_str = current.strftime('%Y-%m-%d')
            if date_str not in US_HOLIDAYS_2023:
                dates.append(current)
        current += timedelta(days=1)

    return dates


def is_weekend(dt: datetime) -> bool:
    """Check if datetime falls on weekend."""
    return dt.weekday() >= 5


def is_business_day(dt: datetime) -> bool:
    """Check if datetime falls on business day."""
    return dt.weekday() < 5


def get_business_days_between(start: datetime, end: datetime) -> int:
    """
    Get number of business days between two dates.

    Args:
        start: Start date
        end: End date

    Returns:
        Number of business days
    """
    current = start + timedelta(days=1)  # Exclude start date
    count = 0

    while current < end:  # Exclude end date
        if is_business_day(current):
            count += 1
        current += timedelta(days=1)

    return count


def round_to_timeframe(dt: datetime, timeframe: str) -> datetime:
    """
    Round datetime to specified timeframe.

    Args:
        dt: Input datetime
        timeframe: Timeframe string (e.g., '5m', '1h', '1d')

    Returns:
        Rounded datetime
    """
    if timeframe.endswith('m'):
        # Minutes
        minutes = int(timeframe[:-1])
        minute = (dt.minute // minutes) * minutes
        return dt.replace(minute=minute, second=0, microsecond=0)

    elif timeframe.endswith('h'):
        # Hours
        hours = int(timeframe[:-1])
        hour = (dt.hour // hours) * hours
        return dt.replace(hour=hour, minute=0, second=0, microsecond=0)

    elif timeframe.endswith('d'):
        # Days
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    elif timeframe.endswith('w'):
        # Weeks
        days_since_monday = dt.weekday()
        monday = dt - timedelta(days=days_since_monday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)

    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")


def get_timeframe_seconds(timeframe: str) -> int:
    """
    Convert timeframe string to seconds.

    Args:
        timeframe: Timeframe string

    Returns:
        Number of seconds
    """
    timeframe_map = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '4h': 14400,
        '6h': 21600,
        '8h': 28800,
        '12h': 43200,
        '1d': 86400,
        '1w': 604800
    }

    if timeframe in timeframe_map:
        return timeframe_map[timeframe]

    # Parse custom formats like "3h", "45m"
    match = re.match(r'^(\d+)([mhdw])$', timeframe)
    if match:
        value, unit = match.groups()
        value = int(value)

        unit_seconds = {
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800
        }

        return value * unit_seconds[unit]

    raise ValueError(f"Invalid timeframe format: {timeframe}")


def format_duration(duration: timedelta) -> str:
    """
    Format timedelta in human-readable form.

    Args:
        duration: Timedelta object

    Returns:
        Human-readable duration string
    """
    parts = []
    days = duration.days

    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")

    seconds = duration.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if remaining_seconds > 0 and not parts:  # Only show seconds if no larger units
        parts.append(f"{remaining_seconds} second{'s' if remaining_seconds != 1 else ''}")

    return " ".join(parts) if parts else "0 seconds"


def parse_duration(duration_str: str) -> timedelta:
    """
    Parse duration string to timedelta.

    Args:
        duration_str: Duration string (e.g., "2d 3h 45m")

    Returns:
        Timedelta object
    """
    total_seconds = 0

    # Find all number-unit pairs
    pattern = r'(\d+)([smhdw])'
    matches = re.findall(pattern, duration_str.lower())

    unit_seconds = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800
    }

    for value_str, unit in matches:
        value = int(value_str)
        total_seconds += value * unit_seconds.get(unit, 0)

    return timedelta(seconds=total_seconds)


def get_epoch_timestamp(dt: datetime) -> int:
    """
    Convert datetime to epoch timestamp in milliseconds.

    Args:
        dt: Datetime object

    Returns:
        Milliseconds since epoch
    """
    return int(dt.timestamp() * 1000)


def get_utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def is_asian_trading_hours(dt: datetime) -> bool:
    """
    Check if time falls within Asian trading session.

    Args:
        dt: Datetime in UTC

    Returns:
        True if within Asian trading hours
    """
    # Convert to Tokyo time
    tokyo_time = dt.astimezone(pytz.timezone('Asia/Tokyo'))
    hour = tokyo_time.hour

    # Asian session roughly 9:00 AM - 3:00 PM JST
    return 9 <= hour < 15 and tokyo_time.weekday() < 5


def is_european_trading_hours(dt: datetime) -> bool:
    """
    Check if time falls within European trading session.

    Args:
        dt: Datetime in UTC

    Returns:
        True if within European trading hours
    """
    # Convert to London time
    london_time = dt.astimezone(pytz.timezone('Europe/London'))
    hour = london_time.hour

    # European session roughly 8:00 AM - 4:30 PM GMT
    return 8 <= hour < 17 and london_time.weekday() < 5


def is_us_trading_hours(dt: datetime) -> bool:
    """
    Check if time falls within US trading session.

    Args:
        dt: Datetime in UTC

    Returns:
        True if within US trading hours
    """
    # Convert to New York time
    ny_time = dt.astimezone(pytz.timezone('America/New_York'))
    hour = ny_time.hour
    minute = ny_time.minute

    # US session: 9:30 AM - 4:00 PM ET
    start_minutes = 9 * 60 + 30  # 9:30 AM
    end_minutes = 16 * 60        # 4:00 PM
    current_minutes = hour * 60 + minute

    return (start_minutes <= current_minutes < end_minutes and
            ny_time.weekday() < 5)


def generate_trading_sessions(start_date: datetime,
                             end_date: datetime) -> List[Dict]:
    """
    Generate trading session information for date range.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of trading session dictionaries
    """
    sessions = []
    current = start_date

    while current <= end_date:
        # Skip weekends for traditional session analysis
        if current.weekday() < 5:
            session = {
                'date': current.date(),
                'asian_open': current.replace(hour=0, minute=0, second=0, microsecond=0),
                'european_open': current.replace(hour=7, minute=0, second=0, microsecond=0),
                'us_open': current.replace(hour=13, minute=30, second=0, microsecond=0),
                'session_type': 'full_trading_day'
            }

            # Add session overlap information
            if current.weekday() == 0:  # Monday
                session['session_type'] = 'week_start'
            elif current.weekday() == 4:  # Friday
                session['session_type'] = 'week_end'

            sessions.append(session)

        current += timedelta(days=1)

    return sessions