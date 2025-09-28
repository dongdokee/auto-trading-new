# src/market_data/liquidity_profiler.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta
from collections import defaultdict

from .models import LiquidityProfile, ExecutionWindow, MarketMetrics


class LiquidityProfiler:
    """Time-based liquidity profiling and optimal execution window analysis"""

    def __init__(self, profile_window_days: int = 30):
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.profile_window = profile_window_days
        self.min_samples_per_hour = 5
        self.confidence_threshold = 0.7

        # Data storage
        self.liquidity_history: Dict[str, List[Dict]] = {}
        self.hourly_profiles: Dict[str, Dict[int, LiquidityProfile]] = {}

        # Performance tracking
        self.update_count = 0
        self.last_cleanup = datetime.utcnow()

    def update_profile(self, symbol: str, timestamp: pd.Timestamp,
                      liquidity_metrics: MarketMetrics) -> None:
        """
        Update liquidity profile with new market data

        Args:
            symbol: Trading symbol
            timestamp: Timestamp of the data
            liquidity_metrics: Market metrics from order book analysis
        """
        if symbol not in self.liquidity_history:
            self.liquidity_history[symbol] = []

        # Create liquidity record
        record = {
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'day_of_week': timestamp.dayofweek,
            'minute': timestamp.minute,
            'spread_bps': liquidity_metrics.spread_bps,
            'depth': float(liquidity_metrics.top_5_liquidity),
            'imbalance': liquidity_metrics.imbalance,
            'liquidity_score': liquidity_metrics.liquidity_score,
            'bid_volume': float(liquidity_metrics.bid_volume_5),
            'ask_volume': float(liquidity_metrics.ask_volume_5),
            'large_order_count': len(liquidity_metrics.large_orders)
        }

        self.liquidity_history[symbol].append(record)
        self.update_count += 1

        # Cleanup old data periodically
        if self.update_count % 1000 == 0:
            self._cleanup_old_data()

        # Update hourly profile
        self._update_hourly_profile(symbol, record)

    def get_expected_liquidity(self, symbol: str,
                             target_time: pd.Timestamp) -> LiquidityProfile:
        """
        Get expected liquidity for a specific time

        Args:
            symbol: Trading symbol
            target_time: Target execution time

        Returns:
            LiquidityProfile: Expected liquidity conditions
        """
        if symbol not in self.liquidity_history:
            return self._default_liquidity_profile(symbol, target_time.hour,
                                                  target_time.dayofweek)

        history = self.liquidity_history[symbol]
        target_hour = target_time.hour
        target_dow = target_time.dayofweek

        # Find similar time periods (same hour, same day of week)
        similar_times = [
            r for r in history
            if (r['hour'] == target_hour and r['day_of_week'] == target_dow)
        ]

        # If not enough data, expand to +/- 1 hour
        if len(similar_times) < self.min_samples_per_hour:
            similar_times = [
                r for r in history
                if (abs(r['hour'] - target_hour) <= 1 and r['day_of_week'] == target_dow)
            ]

        # If still not enough, expand to +/- 2 hours, any day
        if len(similar_times) < self.min_samples_per_hour:
            similar_times = [
                r for r in history
                if abs(r['hour'] - target_hour) <= 2
            ]

        if not similar_times:
            return self._default_liquidity_profile(symbol, target_hour, target_dow)

        # Calculate statistics
        spreads = [r['spread_bps'] for r in similar_times]
        depths = [r['depth'] for r in similar_times]
        imbalances = [r['imbalance'] for r in similar_times]
        scores = [r['liquidity_score'] for r in similar_times]

        # Filter outliers (remove top and bottom 5% if enough samples)
        if len(similar_times) >= 20:
            spreads = self._remove_outliers(spreads)
            depths = self._remove_outliers(depths)

        return LiquidityProfile(
            symbol=symbol,
            hour=target_hour,
            day_of_week=target_dow,
            expected_spread=float(np.mean(spreads)),
            expected_depth=Decimal(str(np.mean(depths))),
            depth_std=float(np.std(depths)),
            confidence=min(1.0, len(similar_times) / 20),  # Full confidence at 20+ samples
            sample_size=len(similar_times),
            historical_data=[{
                'avg_spread': float(np.mean(spreads)),
                'avg_depth': float(np.mean(depths)),
                'avg_imbalance': float(np.mean(imbalances)),
                'avg_liquidity_score': float(np.mean(scores)),
                'spread_std': float(np.std(spreads)),
                'depth_std': float(np.std(depths))
            }]
        )

    def find_optimal_execution_windows(self, symbol: str, order_size: Decimal,
                                     lookahead_hours: int = 24) -> List[ExecutionWindow]:
        """
        Find optimal execution time windows for the next N hours

        Args:
            symbol: Trading symbol
            order_size: Size of order to execute
            lookahead_hours: Hours to look ahead

        Returns:
            List[ExecutionWindow]: Sorted list of optimal execution windows
        """
        if symbol not in self.liquidity_history:
            return []

        current_time = datetime.utcnow()
        windows = []

        # Analyze each hour in the lookahead period
        for hour_offset in range(lookahead_hours):
            target_time = current_time + timedelta(hours=hour_offset)
            target_hour = target_time.hour

            # Get historical data for this hour
            hourly_data = self._get_hourly_statistics(symbol, target_hour)

            if hourly_data and hourly_data['sample_count'] >= self.min_samples_per_hour:
                # Calculate execution cost estimate
                execution_cost = self._estimate_execution_cost(
                    order_size, hourly_data
                )

                window = ExecutionWindow(
                    hour=target_hour,
                    avg_spread=hourly_data['avg_spread'],
                    avg_depth=Decimal(str(hourly_data['avg_depth'])),
                    estimated_cost=execution_cost,
                    samples=hourly_data['sample_count'],
                    confidence=min(1.0, hourly_data['sample_count'] / 20)
                )

                windows.append(window)

        # Sort by execution cost (lower is better)
        windows.sort(key=lambda w: w.cost_score())

        return windows[:10]  # Return top 10 windows

    def get_liquidity_forecast(self, symbol: str, hours_ahead: int = 6) -> List[Dict]:
        """
        Get liquidity forecast for the next few hours

        Args:
            symbol: Trading symbol
            hours_ahead: Number of hours to forecast

        Returns:
            List[Dict]: Hourly liquidity forecasts
        """
        forecasts = []
        current_time = datetime.utcnow()

        for hour_offset in range(hours_ahead):
            target_time = current_time + timedelta(hours=hour_offset)
            profile = self.get_expected_liquidity(symbol, pd.Timestamp(target_time))

            forecasts.append({
                'hour': target_time.hour,
                'datetime': target_time,
                'expected_spread': profile.expected_spread,
                'expected_depth': float(profile.expected_depth),
                'confidence': profile.confidence,
                'liquidity_quality': self._assess_liquidity_quality(profile)
            })

        return forecasts

    def _update_hourly_profile(self, symbol: str, record: Dict) -> None:
        """Update hourly aggregated profile"""
        if symbol not in self.hourly_profiles:
            self.hourly_profiles[symbol] = {}

        hour = record['hour']
        if hour not in self.hourly_profiles[symbol]:
            # Create new hourly profile
            self.hourly_profiles[symbol][hour] = LiquidityProfile(
                symbol=symbol,
                hour=hour,
                day_of_week=-1,  # Aggregated across all days
                expected_spread=record['spread_bps'],
                expected_depth=Decimal(str(record['depth'])),
                depth_std=0.0,
                confidence=0.0,
                sample_size=1,
                historical_data=[record]
            )
        else:
            # Update existing profile
            profile = self.hourly_profiles[symbol][hour]
            profile.historical_data.append(record)
            profile.sample_size += 1

            # Recalculate statistics
            spreads = [r['spread_bps'] for r in profile.historical_data]
            depths = [r['depth'] for r in profile.historical_data]

            profile.expected_spread = float(np.mean(spreads))
            profile.expected_depth = Decimal(str(np.mean(depths)))
            profile.depth_std = float(np.std(depths))
            profile.confidence = min(1.0, profile.sample_size / 20)

    def _get_hourly_statistics(self, symbol: str, hour: int) -> Optional[Dict]:
        """Get aggregated statistics for a specific hour"""
        if symbol not in self.liquidity_history:
            return None

        hourly_records = [
            r for r in self.liquidity_history[symbol]
            if r['hour'] == hour
        ]

        if len(hourly_records) < self.min_samples_per_hour:
            return None

        spreads = [r['spread_bps'] for r in hourly_records]
        depths = [r['depth'] for r in hourly_records]
        imbalances = [r['imbalance'] for r in hourly_records]
        scores = [r['liquidity_score'] for r in hourly_records]

        return {
            'avg_spread': float(np.mean(spreads)),
            'avg_depth': float(np.mean(depths)),
            'avg_imbalance': float(np.mean(imbalances)),
            'avg_liquidity_score': float(np.mean(scores)),
            'spread_std': float(np.std(spreads)),
            'depth_std': float(np.std(depths)),
            'sample_count': len(hourly_records)
        }

    def _estimate_execution_cost(self, order_size: Decimal, hourly_data: Dict) -> float:
        """Estimate execution cost for given order size and market conditions"""
        # Base cost from spread
        spread_cost = hourly_data['avg_spread'] / 20000  # Half spread in decimal

        # Market impact cost (simplified model)
        avg_depth = hourly_data['avg_depth']
        if avg_depth > 0:
            impact_cost = float(order_size) / avg_depth * 0.001  # 0.1% per depth ratio
        else:
            impact_cost = 0.01  # 1% penalty for no liquidity

        # Volatility cost (higher spread std = higher cost)
        volatility_cost = hourly_data['spread_std'] / 100000  # Spread volatility component

        return spread_cost + impact_cost + volatility_cost

    def _assess_liquidity_quality(self, profile: LiquidityProfile) -> str:
        """Assess overall liquidity quality"""
        # Combine spread, depth, and confidence into quality score
        spread_score = max(0, 1 - profile.expected_spread / 20)  # Normalize to 20 bps
        depth_score = min(1, float(profile.expected_depth) / 100000)  # Normalize to 100k
        confidence_score = profile.confidence

        overall_score = (spread_score + depth_score + confidence_score) / 3

        if overall_score >= 0.8:
            return "EXCELLENT"
        elif overall_score >= 0.6:
            return "GOOD"
        elif overall_score >= 0.4:
            return "FAIR"
        else:
            return "POOR"

    def _remove_outliers(self, data: List[float], percentile: float = 0.05) -> List[float]:
        """Remove outliers from data"""
        if len(data) < 10:
            return data

        data_array = np.array(data)
        lower_bound = np.percentile(data_array, percentile * 100)
        upper_bound = np.percentile(data_array, (1 - percentile) * 100)

        return [x for x in data if lower_bound <= x <= upper_bound]

    def _cleanup_old_data(self) -> None:
        """Clean up old liquidity data"""
        cutoff_time = datetime.utcnow() - timedelta(days=self.profile_window)
        cutoff_timestamp = pd.Timestamp(cutoff_time)

        cleaned_count = 0
        for symbol in self.liquidity_history:
            old_count = len(self.liquidity_history[symbol])
            self.liquidity_history[symbol] = [
                record for record in self.liquidity_history[symbol]
                if record['timestamp'] > cutoff_timestamp
            ]
            cleaned_count += old_count - len(self.liquidity_history[symbol])

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old liquidity records")

        self.last_cleanup = datetime.utcnow()

    def _default_liquidity_profile(self, symbol: str, hour: int,
                                 day_of_week: int) -> LiquidityProfile:
        """Return default liquidity profile when no data available"""
        return LiquidityProfile(
            symbol=symbol,
            hour=hour,
            day_of_week=day_of_week,
            expected_spread=5.0,  # 5 bps default
            expected_depth=Decimal('10000'),  # 10k default depth
            depth_std=5000.0,
            confidence=0.0,
            sample_size=0,
            historical_data=[]
        )

    def get_profile_summary(self, symbol: str) -> Dict:
        """Get summary statistics for liquidity profile"""
        if symbol not in self.liquidity_history:
            return {'error': 'No data available for symbol'}

        history = self.liquidity_history[symbol]
        if not history:
            return {'error': 'No historical data'}

        # Overall statistics
        spreads = [r['spread_bps'] for r in history]
        depths = [r['depth'] for r in history]
        scores = [r['liquidity_score'] for r in history]

        # Best and worst hours
        hourly_stats = {}
        for record in history:
            hour = record['hour']
            if hour not in hourly_stats:
                hourly_stats[hour] = []
            hourly_stats[hour].append(record['spread_bps'])

        best_hours = []
        worst_hours = []
        for hour, hour_spreads in hourly_stats.items():
            if len(hour_spreads) >= self.min_samples_per_hour:
                avg_spread = np.mean(hour_spreads)
                best_hours.append((hour, avg_spread))
                worst_hours.append((hour, avg_spread))

        best_hours.sort(key=lambda x: x[1])  # Sort by spread (lower is better)
        worst_hours.sort(key=lambda x: x[1], reverse=True)

        return {
            'symbol': symbol,
            'total_samples': len(history),
            'date_range': {
                'start': min(r['timestamp'] for r in history),
                'end': max(r['timestamp'] for r in history)
            },
            'overall_stats': {
                'avg_spread': float(np.mean(spreads)),
                'avg_depth': float(np.mean(depths)),
                'avg_liquidity_score': float(np.mean(scores)),
                'spread_std': float(np.std(spreads))
            },
            'best_hours': best_hours[:3],  # Top 3 best hours
            'worst_hours': worst_hours[:3],  # Top 3 worst hours
            'hourly_coverage': {
                hour: len([r for r in history if r['hour'] == hour])
                for hour in range(24)
            }
        }