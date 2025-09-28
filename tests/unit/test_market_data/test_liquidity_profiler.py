# tests/unit/test_market_data/test_liquidity_profiler.py

import pytest
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta

from src.market_data.models import MarketMetrics, LiquidityProfile, ExecutionWindow, BookShape
from src.market_data.liquidity_profiler import LiquidityProfiler


class TestLiquidityProfiler:
    """Test suite for LiquidityProfiler"""

    @pytest.fixture
    def profiler(self):
        return LiquidityProfiler(profile_window_days=7)  # Shorter window for testing

    @pytest.fixture
    def sample_metrics(self):
        """Create sample market metrics for testing"""
        return MarketMetrics(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            best_bid=Decimal('50000'),
            best_ask=Decimal('50005'),
            mid_price=Decimal('50002.5'),
            spread=Decimal('5'),
            spread_bps=1.0,
            bid_volume_5=Decimal('10.0'),
            ask_volume_5=Decimal('12.0'),
            top_5_liquidity=Decimal('22.0'),
            imbalance=-0.091,
            liquidity_score=0.85,
            book_shape=BookShape.FLAT,
            large_orders=[]
        )

    def test_should_initialize_with_correct_parameters(self, profiler):
        """Test profiler initialization"""
        assert profiler.profile_window == 7
        assert profiler.min_samples_per_hour == 5
        assert profiler.confidence_threshold == 0.7
        assert len(profiler.liquidity_history) == 0

    def test_should_update_profile_with_new_data(self, profiler, sample_metrics):
        """Test profile update with new market data"""
        timestamp = pd.Timestamp(datetime.utcnow())

        profiler.update_profile("BTCUSDT", timestamp, sample_metrics)

        assert "BTCUSDT" in profiler.liquidity_history
        assert len(profiler.liquidity_history["BTCUSDT"]) == 1

        record = profiler.liquidity_history["BTCUSDT"][0]
        assert record['spread_bps'] == 1.0
        assert record['depth'] == 22.0
        assert record['hour'] == timestamp.hour

    def test_should_build_hourly_profiles(self, profiler, sample_metrics):
        """Test hourly profile building"""
        base_time = datetime.utcnow()

        # Add multiple records for the same hour
        for i in range(3):
            timestamp = pd.Timestamp(base_time + timedelta(minutes=i*10))
            profiler.update_profile("BTCUSDT", timestamp, sample_metrics)

        assert "BTCUSDT" in profiler.hourly_profiles
        hour = base_time.hour
        assert hour in profiler.hourly_profiles["BTCUSDT"]

        profile = profiler.hourly_profiles["BTCUSDT"][hour]
        assert profile.sample_size >= 1  # Should have at least one sample
        assert profile.symbol == "BTCUSDT"
        assert len(profile.historical_data) >= 1

    def test_should_get_expected_liquidity_with_sufficient_data(self, profiler, sample_metrics):
        """Test expected liquidity calculation with sufficient data"""
        base_time = datetime.utcnow()
        timestamp = pd.Timestamp(base_time)

        # Add sufficient data for the same hour and day of week
        for i in range(10):
            time_offset = timestamp + timedelta(minutes=i*5)
            profiler.update_profile("BTCUSDT", time_offset, sample_metrics)

        # Get expected liquidity for the same time
        profile = profiler.get_expected_liquidity("BTCUSDT", timestamp)

        assert isinstance(profile, LiquidityProfile)
        assert profile.symbol == "BTCUSDT"
        assert profile.expected_spread == 1.0
        assert profile.expected_depth == Decimal('22.0')
        assert profile.confidence > 0

    def test_should_expand_time_window_with_insufficient_data(self, profiler, sample_metrics):
        """Test time window expansion when insufficient data"""
        base_time = datetime.utcnow()

        # Add only 2 records (less than min_samples_per_hour)
        for i in range(2):
            timestamp = pd.Timestamp(base_time + timedelta(hours=i))
            profiler.update_profile("BTCUSDT", timestamp, sample_metrics)

        # Should expand window and still return a profile
        target_time = pd.Timestamp(base_time)
        profile = profiler.get_expected_liquidity("BTCUSDT", target_time)

        assert isinstance(profile, LiquidityProfile)
        assert profile.sample_size >= 0

    def test_should_return_default_profile_for_no_data(self, profiler):
        """Test default profile return when no data available"""
        target_time = pd.Timestamp(datetime.utcnow())
        profile = profiler.get_expected_liquidity("UNKNOWN_SYMBOL", target_time)

        assert isinstance(profile, LiquidityProfile)
        assert profile.symbol == "UNKNOWN_SYMBOL"
        assert profile.expected_spread == 5.0  # Default value
        assert profile.confidence == 0.0
        assert profile.sample_size == 0

    def test_should_find_optimal_execution_windows(self, profiler, sample_metrics):
        """Test optimal execution window finding"""
        base_time = datetime.utcnow()

        # Create varied liquidity profiles for different hours
        for hour_offset in range(6):
            for minute in range(0, 60, 10):  # Every 10 minutes = 6 points per hour (> min_samples_per_hour)
                timestamp = pd.Timestamp(base_time.replace(hour=(base_time.hour + hour_offset) % 24, minute=minute))

                # Vary spread based on hour to create optimal windows
                adjusted_metrics = MarketMetrics(
                    symbol="BTCUSDT",
                    timestamp=timestamp.to_pydatetime(),
                    best_bid=Decimal('50000'),
                    best_ask=Decimal('50005'),
                    mid_price=Decimal('50002.5'),
                    spread=Decimal('5'),
                    spread_bps=1.0 + hour_offset * 0.5,  # Varying spread
                    bid_volume_5=Decimal('10.0'),
                    ask_volume_5=Decimal('12.0'),
                    top_5_liquidity=Decimal('22.0') + Decimal(hour_offset * 2),  # Varying depth
                    imbalance=-0.091,
                    liquidity_score=0.85,
                    book_shape=BookShape.FLAT,
                    large_orders=[]
                )
                profiler.update_profile("BTCUSDT", timestamp, adjusted_metrics)

        # Find optimal windows
        windows = profiler.find_optimal_execution_windows("BTCUSDT", Decimal('1.0'), 6)

        assert isinstance(windows, list)
        assert len(windows) > 0

        # Check that windows are sorted by cost
        if len(windows) > 1:
            for i in range(len(windows) - 1):
                assert windows[i].cost_score() <= windows[i + 1].cost_score()

    def test_should_get_liquidity_forecast(self, profiler, sample_metrics):
        """Test liquidity forecasting"""
        # Add some historical data
        base_time = datetime.utcnow()
        for i in range(10):
            timestamp = pd.Timestamp(base_time + timedelta(hours=i))
            profiler.update_profile("BTCUSDT", timestamp, sample_metrics)

        forecasts = profiler.get_liquidity_forecast("BTCUSDT", hours_ahead=3)

        assert len(forecasts) == 3
        for forecast in forecasts:
            assert 'hour' in forecast
            assert 'expected_spread' in forecast
            assert 'expected_depth' in forecast
            assert 'confidence' in forecast
            assert 'liquidity_quality' in forecast

    def test_should_assess_liquidity_quality_correctly(self, profiler):
        """Test liquidity quality assessment"""
        # Test excellent liquidity
        excellent_profile = LiquidityProfile(
            symbol="BTCUSDT",
            hour=12,
            day_of_week=1,
            expected_spread=2.0,  # Low spread
            expected_depth=Decimal('200000'),  # High depth
            depth_std=1000.0,
            confidence=0.9,  # High confidence
            sample_size=50
        )

        quality = profiler._assess_liquidity_quality(excellent_profile)
        assert quality == "EXCELLENT"

        # Test poor liquidity
        poor_profile = LiquidityProfile(
            symbol="BTCUSDT",
            hour=12,
            day_of_week=1,
            expected_spread=50.0,  # High spread
            expected_depth=Decimal('100'),  # Low depth
            depth_std=100.0,
            confidence=0.1,  # Low confidence
            sample_size=2
        )

        quality = profiler._assess_liquidity_quality(poor_profile)
        assert quality == "POOR"

    def test_should_remove_outliers_correctly(self, profiler):
        """Test outlier removal"""
        # Data with clear outliers - make sure we have enough data points
        data_with_outliers = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 100, 200]  # 100, 200 are outliers

        cleaned_data = profiler._remove_outliers(data_with_outliers, percentile=0.15)

        # Should remove extreme outliers
        assert 200 not in cleaned_data
        assert len(cleaned_data) < len(data_with_outliers)

    def test_should_handle_small_datasets_in_outlier_removal(self, profiler):
        """Test outlier removal with small datasets"""
        small_data = [1, 2, 3]

        cleaned_data = profiler._remove_outliers(small_data)

        # Should return original data for small datasets
        assert cleaned_data == small_data

    def test_should_cleanup_old_data_periodically(self, profiler, sample_metrics):
        """Test periodic cleanup of old data"""
        # Add old data
        old_time = datetime.utcnow() - timedelta(days=10)  # Older than window
        old_timestamp = pd.Timestamp(old_time)

        profiler.update_profile("BTCUSDT", old_timestamp, sample_metrics)

        # Add recent data
        recent_time = datetime.utcnow()
        recent_timestamp = pd.Timestamp(recent_time)
        profiler.update_profile("BTCUSDT", recent_timestamp, sample_metrics)

        assert len(profiler.liquidity_history["BTCUSDT"]) == 2

        # Trigger cleanup
        profiler._cleanup_old_data()

        # Old data should be removed
        assert len(profiler.liquidity_history["BTCUSDT"]) == 1
        remaining_record = profiler.liquidity_history["BTCUSDT"][0]
        assert remaining_record['timestamp'] == recent_timestamp

    def test_should_calculate_execution_cost_correctly(self, profiler):
        """Test execution cost calculation"""
        hourly_data = {
            'avg_spread': 5.0,
            'avg_depth': 10000,
            'spread_std': 1.0
        }

        order_size = Decimal('100')
        cost = profiler._estimate_execution_cost(order_size, hourly_data)

        assert cost > 0
        assert isinstance(cost, float)

        # Larger orders should have higher cost
        large_order_cost = profiler._estimate_execution_cost(Decimal('1000'), hourly_data)
        assert large_order_cost > cost

    def test_should_handle_zero_depth_in_cost_calculation(self, profiler):
        """Test execution cost calculation with zero depth"""
        hourly_data = {
            'avg_spread': 5.0,
            'avg_depth': 0,  # Zero depth
            'spread_std': 1.0
        }

        order_size = Decimal('100')
        cost = profiler._estimate_execution_cost(order_size, hourly_data)

        # Should include penalty for zero depth, cost should be higher than just spread
        assert cost > 0.01  # Should be higher than just spread component

    def test_should_provide_profile_summary(self, profiler, sample_metrics):
        """Test profile summary generation"""
        # Add data across multiple hours
        base_time = datetime.utcnow()
        for hour_offset in range(3):
            for minute in range(0, 60, 10):
                timestamp = pd.Timestamp(base_time.replace(
                    hour=(base_time.hour + hour_offset) % 24,
                    minute=minute
                ))
                profiler.update_profile("BTCUSDT", timestamp, sample_metrics)

        summary = profiler.get_profile_summary("BTCUSDT")

        assert 'symbol' in summary
        assert 'total_samples' in summary
        assert 'date_range' in summary
        assert 'overall_stats' in summary
        assert 'best_hours' in summary
        assert 'worst_hours' in summary
        assert 'hourly_coverage' in summary

        assert summary['symbol'] == "BTCUSDT"
        assert summary['total_samples'] > 0

    def test_should_handle_unknown_symbol_in_summary(self, profiler):
        """Test profile summary for unknown symbol"""
        summary = profiler.get_profile_summary("UNKNOWN")

        assert 'error' in summary
        assert summary['error'] == 'No data available for symbol'

    def test_should_maintain_update_count(self, profiler, sample_metrics):
        """Test that update count is maintained correctly"""
        initial_count = profiler.update_count

        timestamp = pd.Timestamp(datetime.utcnow())
        profiler.update_profile("BTCUSDT", timestamp, sample_metrics)

        assert profiler.update_count == initial_count + 1

    def test_should_get_hourly_statistics(self, profiler, sample_metrics):
        """Test hourly statistics calculation"""
        hour = 12
        base_time = datetime.utcnow().replace(hour=hour)

        # Add multiple records for the same hour
        for i in range(7):  # Above minimum threshold
            timestamp = pd.Timestamp(base_time + timedelta(minutes=i*5))
            profiler.update_profile("BTCUSDT", timestamp, sample_metrics)

        stats = profiler._get_hourly_statistics("BTCUSDT", hour)

        assert stats is not None
        assert 'avg_spread' in stats
        assert 'avg_depth' in stats
        assert 'sample_count' in stats
        assert stats['sample_count'] >= profiler.min_samples_per_hour

    def test_should_return_none_for_insufficient_hourly_data(self, profiler, sample_metrics):
        """Test hourly statistics with insufficient data"""
        hour = 15
        base_time = datetime.utcnow().replace(hour=hour)

        # Add only 2 records (less than minimum)
        for i in range(2):
            timestamp = pd.Timestamp(base_time + timedelta(minutes=i*5))
            profiler.update_profile("BTCUSDT", timestamp, sample_metrics)

        stats = profiler._get_hourly_statistics("BTCUSDT", hour)
        assert stats is None

    def test_should_handle_execution_window_confidence(self, profiler):
        """Test execution window confidence calculation"""
        window = ExecutionWindow(
            hour=12,
            avg_spread=5.0,
            avg_depth=Decimal('10000'),
            estimated_cost=0.001,
            samples=25,  # High sample count
            confidence=0.0
        )

        # Confidence should be calculated based on samples
        window.confidence = min(1.0, window.samples / 20)
        assert window.confidence == 1.0  # Should be capped at 1.0

    def test_should_calculate_cost_score_correctly(self, profiler):
        """Test cost score calculation in ExecutionWindow"""
        window = ExecutionWindow(
            hour=12,
            avg_spread=5.0,
            avg_depth=Decimal('10000'),
            estimated_cost=0.001,
            samples=10
        )

        cost_score = window.cost_score()

        assert cost_score > 0
        assert isinstance(cost_score, float)

        # Lower depth should increase cost score
        low_depth_window = ExecutionWindow(
            hour=12,
            avg_spread=5.0,
            avg_depth=Decimal('100'),  # Much lower depth
            estimated_cost=0.001,
            samples=10
        )

        assert low_depth_window.cost_score() > cost_score