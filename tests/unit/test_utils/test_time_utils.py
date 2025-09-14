"""
Tests for time utility functions.
Following TDD methodology: Red -> Green -> Refactor
"""

import pytest
from datetime import datetime, timedelta, timezone, time
import pytz
import pandas as pd

# These imports will fail initially (Red phase)
# We'll implement them to make tests pass (Green phase)
try:
    from src.utils.time_utils import (
        get_market_timezone, is_market_open, get_next_market_open,
        get_next_market_close, get_trading_calendar, convert_to_market_time,
        is_weekend, is_business_day, get_business_days_between,
        format_duration, parse_duration, get_epoch_timestamp,
        get_utc_now, round_to_timeframe, get_timeframe_seconds,
        generate_trading_sessions, is_asian_trading_hours,
        is_european_trading_hours, is_us_trading_hours
    )
except ImportError:
    pytest.skip("Time utilities not yet implemented", allow_module_level=True)


class TestMarketTimezones:
    """Test market timezone utilities"""

    def test_should_get_market_timezone_for_exchanges(self):
        """Should return correct timezone for different exchanges"""
        # Test major cryptocurrency exchanges
        binance_tz = get_market_timezone('BINANCE')
        assert binance_tz.zone == 'UTC'  # Binance operates in UTC

        # Test traditional markets
        nyse_tz = get_market_timezone('NYSE')
        assert nyse_tz.zone == 'America/New_York'

        lse_tz = get_market_timezone('LSE')
        assert lse_tz.zone == 'Europe/London'

        # Test default fallback
        unknown_tz = get_market_timezone('UNKNOWN_EXCHANGE')
        assert unknown_tz.zone == 'UTC'

    def test_should_convert_to_market_time(self):
        """Should convert UTC time to market timezone"""
        utc_time = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)

        # Convert to NYSE time
        nyse_time = convert_to_market_time(utc_time, 'NYSE')
        assert nyse_time.tzinfo.zone == 'America/New_York'

        # Convert to Tokyo time
        tokyo_time = convert_to_market_time(utc_time, 'TSE')
        assert tokyo_time.tzinfo.zone == 'Asia/Tokyo'

    def test_should_handle_naive_datetime_conversion(self):
        """Should handle naive datetime by assuming UTC"""
        naive_time = datetime(2023, 6, 15, 14, 30, 0)

        market_time = convert_to_market_time(naive_time, 'NYSE')

        assert market_time.tzinfo is not None
        assert market_time.tzinfo.zone == 'America/New_York'


class TestMarketHours:
    """Test market hours and trading session utilities"""

    def test_should_detect_crypto_market_always_open(self):
        """Crypto markets should always be open"""
        # Test various times and days
        weekday_time = datetime(2023, 6, 15, 10, 30, 0, tzinfo=timezone.utc)  # Thursday
        weekend_time = datetime(2023, 6, 17, 15, 45, 0, tzinfo=timezone.utc)  # Saturday
        holiday_time = datetime(2023, 12, 25, 12, 0, 0, tzinfo=timezone.utc)  # Christmas

        assert is_market_open('BINANCE', weekday_time) is True
        assert is_market_open('BINANCE', weekend_time) is True
        assert is_market_open('BINANCE', holiday_time) is True

    def test_should_detect_traditional_market_hours(self):
        """Should correctly detect traditional market trading hours"""
        # NYSE trading hours: 9:30 AM - 4:00 PM ET
        market_open = datetime(2023, 6, 15, 13, 35, 0, tzinfo=timezone.utc)  # 9:35 AM ET
        market_closed = datetime(2023, 6, 15, 21, 30, 0, tzinfo=timezone.utc)  # 5:30 PM ET
        weekend = datetime(2023, 6, 17, 15, 0, 0, tzinfo=timezone.utc)  # Saturday

        assert is_market_open('NYSE', market_open) is True
        assert is_market_open('NYSE', market_closed) is False
        assert is_market_open('NYSE', weekend) is False

    def test_should_get_next_market_open_time(self):
        """Should calculate next market open time"""
        # Test during market hours - should return next day
        during_market = datetime(2023, 6, 15, 15, 0, 0, tzinfo=timezone.utc)  # Thursday during hours

        next_open = get_next_market_open('NYSE', during_market)

        assert next_open > during_market
        # Should be next trading day at 9:30 AM ET
        market_time = next_open.astimezone(pytz.timezone('America/New_York'))
        assert market_time.hour == 9
        assert market_time.minute == 30

    def test_should_get_next_market_close_time(self):
        """Should calculate next market close time"""
        during_market = datetime(2023, 6, 15, 15, 0, 0, tzinfo=timezone.utc)

        next_close = get_next_market_close('NYSE', during_market)

        assert next_close > during_market
        # Should be same day at 4:00 PM ET
        market_time = next_close.astimezone(pytz.timezone('America/New_York'))
        assert market_time.hour == 16
        assert market_time.minute == 0


class TestTradingCalendar:
    """Test trading calendar utilities"""

    def test_should_generate_trading_calendar(self):
        """Should generate trading calendar for date range"""
        start_date = datetime(2023, 6, 1)
        end_date = datetime(2023, 6, 30)

        calendar = get_trading_calendar('NYSE', start_date, end_date)

        assert isinstance(calendar, list)
        assert len(calendar) > 0
        # All dates should be business days
        for trading_day in calendar:
            assert trading_day.weekday() < 5  # Monday=0, Friday=4

    def test_should_exclude_market_holidays(self):
        """Should exclude known market holidays from trading calendar"""
        # Include Independence Day 2023 (July 4th)
        start_date = datetime(2023, 7, 1)
        end_date = datetime(2023, 7, 10)

        calendar = get_trading_calendar('NYSE', start_date, end_date)

        # July 4th, 2023 was on Tuesday - should be excluded
        july_4 = datetime(2023, 7, 4).date()
        calendar_dates = [day.date() for day in calendar]
        assert july_4 not in calendar_dates

    def test_should_handle_crypto_calendar(self):
        """Crypto markets should have all days in calendar"""
        start_date = datetime(2023, 6, 1)
        end_date = datetime(2023, 6, 7)  # Include weekend

        calendar = get_trading_calendar('BINANCE', start_date, end_date)

        # Should include all 7 days
        assert len(calendar) == 7


class TestBusinessDayUtilities:
    """Test business day calculation utilities"""

    def test_should_detect_weekend_days(self):
        """Should correctly identify weekend days"""
        saturday = datetime(2023, 6, 17)
        sunday = datetime(2023, 6, 18)
        monday = datetime(2023, 6, 19)

        assert is_weekend(saturday) is True
        assert is_weekend(sunday) is True
        assert is_weekend(monday) is False

    def test_should_detect_business_days(self):
        """Should correctly identify business days"""
        monday = datetime(2023, 6, 19)
        friday = datetime(2023, 6, 23)
        saturday = datetime(2023, 6, 24)

        assert is_business_day(monday) is True
        assert is_business_day(friday) is True
        assert is_business_day(saturday) is False

    def test_should_calculate_business_days_between_dates(self):
        """Should calculate business days between two dates"""
        # Monday to Friday (same week)
        start = datetime(2023, 6, 19)  # Monday
        end = datetime(2023, 6, 23)    # Friday

        business_days = get_business_days_between(start, end)

        assert business_days == 3  # Tue, Wed, Thu (excluding both Monday start and Friday end)

    def test_should_handle_business_days_across_weekends(self):
        """Should handle business day calculation across weekends"""
        # Friday to next Tuesday
        start = datetime(2023, 6, 23)  # Friday
        end = datetime(2023, 6, 27)    # Tuesday

        business_days = get_business_days_between(start, end)

        assert business_days == 1  # Only Monday (excluding both Friday start and Tuesday end)


class TestTimeFrameUtilities:
    """Test timeframe and duration utilities"""

    def test_should_round_to_timeframe(self):
        """Should round datetime to specified timeframe"""
        dt = datetime(2023, 6, 15, 14, 37, 23)

        # Round to 15-minute intervals
        rounded_15m = round_to_timeframe(dt, '15m')
        assert rounded_15m.minute in [0, 15, 30, 45]

        # Round to hour
        rounded_1h = round_to_timeframe(dt, '1h')
        assert rounded_1h.minute == 0
        assert rounded_1h.second == 0

        # Round to day
        rounded_1d = round_to_timeframe(dt, '1d')
        assert rounded_1d.hour == 0
        assert rounded_1d.minute == 0
        assert rounded_1d.second == 0

    def test_should_get_timeframe_seconds(self):
        """Should convert timeframe strings to seconds"""
        assert get_timeframe_seconds('1m') == 60
        assert get_timeframe_seconds('5m') == 300
        assert get_timeframe_seconds('15m') == 900
        assert get_timeframe_seconds('1h') == 3600
        assert get_timeframe_seconds('4h') == 14400
        assert get_timeframe_seconds('1d') == 86400
        assert get_timeframe_seconds('1w') == 604800

    def test_should_format_duration(self):
        """Should format duration in human-readable form"""
        duration = timedelta(days=2, hours=3, minutes=45)

        formatted = format_duration(duration)

        assert '2 days' in formatted
        assert '3 hours' in formatted
        assert '45 minutes' in formatted

    def test_should_parse_duration_string(self):
        """Should parse duration string back to timedelta"""
        duration_str = "2d 3h 45m"

        duration = parse_duration(duration_str)

        assert duration.days == 2
        assert duration.seconds == (3 * 3600) + (45 * 60)


class TestTimestampUtilities:
    """Test timestamp conversion utilities"""

    def test_should_get_epoch_timestamp(self):
        """Should convert datetime to epoch timestamp"""
        dt = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)

        timestamp = get_epoch_timestamp(dt)

        # Should be milliseconds since epoch
        assert isinstance(timestamp, int)
        assert timestamp > 1000000000000  # Should be in milliseconds

    def test_should_get_current_utc_time(self):
        """Should get current UTC time"""
        utc_now = get_utc_now()

        assert utc_now.tzinfo == timezone.utc
        # Should be recent (within last minute)
        assert (datetime.now(timezone.utc) - utc_now) < timedelta(minutes=1)


class TestTradingSessionUtilities:
    """Test trading session analysis utilities"""

    def test_should_identify_asian_trading_hours(self):
        """Should identify Asian trading session hours"""
        # Tokyo market hours (approximately 9:00 AM - 3:00 PM JST)
        asian_time = datetime(2023, 6, 15, 2, 0, 0, tzinfo=timezone.utc)  # 11 AM JST
        non_asian_time = datetime(2023, 6, 15, 15, 0, 0, tzinfo=timezone.utc)  # 12 AM JST

        assert is_asian_trading_hours(asian_time) is True
        assert is_asian_trading_hours(non_asian_time) is False

    def test_should_identify_european_trading_hours(self):
        """Should identify European trading session hours"""
        # London market hours (approximately 8:00 AM - 4:30 PM GMT)
        european_time = datetime(2023, 6, 15, 10, 0, 0, tzinfo=timezone.utc)  # 10 AM GMT
        non_european_time = datetime(2023, 6, 15, 20, 0, 0, tzinfo=timezone.utc)  # 8 PM GMT

        assert is_european_trading_hours(european_time) is True
        assert is_european_trading_hours(non_european_time) is False

    def test_should_identify_us_trading_hours(self):
        """Should identify US trading session hours"""
        # NYSE hours (9:30 AM - 4:00 PM ET)
        us_time = datetime(2023, 6, 15, 15, 0, 0, tzinfo=timezone.utc)  # 11 AM ET
        non_us_time = datetime(2023, 6, 15, 3, 0, 0, tzinfo=timezone.utc)  # 11 PM ET

        assert is_us_trading_hours(us_time) is True
        assert is_us_trading_hours(non_us_time) is False

    def test_should_generate_trading_sessions(self):
        """Should generate trading session information"""
        start_date = datetime(2023, 6, 15)
        end_date = datetime(2023, 6, 17)

        sessions = generate_trading_sessions(start_date, end_date)

        assert isinstance(sessions, list)
        assert len(sessions) > 0

        # Each session should have required fields
        for session in sessions:
            assert 'date' in session
            assert 'asian_open' in session
            assert 'european_open' in session
            assert 'us_open' in session
            assert 'session_type' in session


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_should_handle_invalid_exchange_names(self):
        """Should handle invalid exchange names gracefully"""
        invalid_time = datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Should not raise exception, should use sensible defaults
        result = is_market_open('INVALID_EXCHANGE', invalid_time)
        assert isinstance(result, bool)

    def test_should_handle_timezone_edge_cases(self):
        """Should handle timezone conversion edge cases"""
        # Test DST transitions, etc.
        dst_date = datetime(2023, 3, 12, 7, 0, 0, tzinfo=timezone.utc)  # US DST change

        market_time = convert_to_market_time(dst_date, 'NYSE')

        assert market_time.tzinfo is not None
        # Should handle DST correctly

    def test_should_validate_timeframe_formats(self):
        """Should validate and handle different timeframe formats"""
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']

        for tf in valid_timeframes:
            seconds = get_timeframe_seconds(tf)
            assert seconds > 0

        # Test invalid timeframe
        with pytest.raises(ValueError):
            get_timeframe_seconds('invalid_timeframe')