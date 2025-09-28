# src/market_data/data_aggregator.py

import logging
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Set
from decimal import Decimal
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pandas as pd

from .models import (
    OrderBookSnapshot, TickData, MarketMetrics, LiquidityProfile,
    MarketImpactEstimate, MicrostructurePatterns, AggregatedMarketData
)
from .orderbook_analyzer import OrderBookAnalyzer
from .market_impact import MarketImpactModel
from .liquidity_profiler import LiquidityProfiler
from .tick_processor import TickDataAnalyzer


class DataAggregator:
    """
    Central aggregator for multi-symbol market data processing and caching
    """

    def __init__(self, cache_ttl: int = 60, max_symbols: int = 50):
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.cache_ttl = cache_ttl  # seconds
        self.max_symbols = max_symbols
        self.performance_window = 3600  # 1 hour for performance tracking

        # Core analyzers
        self.orderbook_analyzer = OrderBookAnalyzer()
        self.market_impact_model = MarketImpactModel()
        self.liquidity_profiler = LiquidityProfiler()

        # Per-symbol tick analyzers
        self.tick_analyzers: Dict[str, TickDataAnalyzer] = {}

        # Data cache
        self.aggregated_data: Dict[str, AggregatedMarketData] = {}
        self.symbol_subscriptions: Set[str] = set()

        # Callback system
        self.update_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.pattern_callbacks: List[Callable] = []

        # Performance tracking
        self.performance_metrics = {
            'processed_orderbooks': 0,
            'processed_ticks': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': deque(maxlen=1000),
            'error_count': 0,
            'last_reset': datetime.utcnow()
        }

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the data aggregator"""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self.logger.info("DataAggregator started")

    async def stop(self) -> None:
        """Stop the data aggregator"""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()

        self.logger.info("DataAggregator stopped")

    def subscribe_symbol(self, symbol: str) -> None:
        """
        Subscribe to market data for a symbol

        Args:
            symbol: Trading symbol to subscribe to
        """
        if symbol in self.symbol_subscriptions:
            return

        if len(self.symbol_subscriptions) >= self.max_symbols:
            self.logger.warning(f"Maximum symbols ({self.max_symbols}) reached, cannot add {symbol}")
            return

        self.symbol_subscriptions.add(symbol)
        self.tick_analyzers[symbol] = TickDataAnalyzer()

        # Initialize empty aggregated data
        self.aggregated_data[symbol] = AggregatedMarketData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            current_metrics=None,  # Will be populated on first update
            liquidity_profile=self.liquidity_profiler._default_liquidity_profile(
                symbol, datetime.utcnow().hour, datetime.utcnow().weekday()
            )
        )

        self.logger.info(f"Subscribed to market data for {symbol}")

    def unsubscribe_symbol(self, symbol: str) -> None:
        """
        Unsubscribe from market data for a symbol

        Args:
            symbol: Trading symbol to unsubscribe from
        """
        if symbol not in self.symbol_subscriptions:
            return

        self.symbol_subscriptions.remove(symbol)

        # Cleanup
        if symbol in self.tick_analyzers:
            del self.tick_analyzers[symbol]
        if symbol in self.aggregated_data:
            del self.aggregated_data[symbol]
        if symbol in self.update_callbacks:
            del self.update_callbacks[symbol]

        self.logger.info(f"Unsubscribed from market data for {symbol}")

    async def process_orderbook_update(self, orderbook: OrderBookSnapshot) -> None:
        """
        Process order book update

        Args:
            orderbook: Order book snapshot
        """
        if orderbook.symbol not in self.symbol_subscriptions:
            return

        start_time = time.time()

        try:
            # Analyze order book
            metrics = self.orderbook_analyzer.analyze_orderbook(orderbook)

            # Update liquidity profile
            self.liquidity_profiler.update_profile(
                orderbook.symbol,
                pd.Timestamp(orderbook.timestamp),
                metrics
            )

            # Update aggregated data
            await self._update_aggregated_data(orderbook.symbol, orderbook=orderbook, metrics=metrics)

            # Track performance
            processing_time = time.time() - start_time
            self.performance_metrics['processing_times'].append(processing_time)
            self.performance_metrics['processed_orderbooks'] += 1

            # Trigger callbacks
            await self._trigger_update_callbacks(orderbook.symbol, 'orderbook', metrics)

        except Exception as e:
            self.logger.error(f"Error processing orderbook for {orderbook.symbol}: {e}")
            self.performance_metrics['error_count'] += 1

    async def process_tick_update(self, tick: TickData) -> None:
        """
        Process tick data update

        Args:
            tick: Tick data
        """
        if tick.symbol not in self.symbol_subscriptions:
            return

        start_time = time.time()

        try:
            # Process tick through analyzer
            tick_analyzer = self.tick_analyzers[tick.symbol]
            patterns = tick_analyzer.process_tick(tick)

            # Update aggregated data
            await self._update_aggregated_data(tick.symbol, tick=tick, patterns=patterns)

            # Track performance
            processing_time = time.time() - start_time
            self.performance_metrics['processing_times'].append(processing_time)
            self.performance_metrics['processed_ticks'] += 1

            # Trigger pattern callbacks if patterns detected
            if patterns and any([patterns.quote_stuffing, patterns.layering,
                               patterns.momentum_ignition, patterns.ping_pong]):
                await self._trigger_pattern_callbacks(patterns)

            # Trigger update callbacks
            await self._trigger_update_callbacks(tick.symbol, 'tick', tick)

        except Exception as e:
            self.logger.error(f"Error processing tick for {tick.symbol}: {e}")
            self.performance_metrics['error_count'] += 1

    async def get_market_data(self, symbol: str, force_refresh: bool = False) -> Optional[AggregatedMarketData]:
        """
        Get aggregated market data for a symbol

        Args:
            symbol: Trading symbol
            force_refresh: Force refresh cached data

        Returns:
            Optional[AggregatedMarketData]: Aggregated market data or None
        """
        if symbol not in self.symbol_subscriptions:
            self.logger.warning(f"Symbol {symbol} not subscribed")
            return None

        # Check cache
        if symbol in self.aggregated_data and not force_refresh:
            cached_data = self.aggregated_data[symbol]
            if cached_data.is_cache_valid():
                self.performance_metrics['cache_hits'] += 1
                return cached_data

        self.performance_metrics['cache_misses'] += 1

        # Return existing data even if stale (better than None)
        return self.aggregated_data.get(symbol)

    async def estimate_market_impact(self, symbol: str, order_size: Decimal,
                                   market_state: Optional[Dict] = None) -> Optional[MarketImpactEstimate]:
        """
        Estimate market impact for an order

        Args:
            symbol: Trading symbol
            order_size: Size of the order
            market_state: Optional market state override

        Returns:
            Optional[MarketImpactEstimate]: Market impact estimate
        """
        if symbol not in self.symbol_subscriptions:
            return None

        try:
            # Get current market state if not provided
            if market_state is None:
                aggregated_data = await self.get_market_data(symbol)
                if not aggregated_data or not aggregated_data.current_metrics:
                    return None

                market_state = {
                    'daily_volume': 1000000,  # Default, should come from market data
                    'volatility': 0.01,  # Default
                    'spread_bps': aggregated_data.current_metrics.spread_bps,
                    'execution_speed': 1.0,
                    'market_regime': 0.5
                }

            # Estimate impact
            impact_estimate = self.market_impact_model.estimate_impact(
                order_size, symbol, market_state
            )

            return impact_estimate

        except Exception as e:
            self.logger.error(f"Error estimating market impact for {symbol}: {e}")
            return None

    async def get_optimal_execution_windows(self, symbol: str, order_size: Decimal,
                                          hours_ahead: int = 24) -> List:
        """
        Get optimal execution windows for an order

        Args:
            symbol: Trading symbol
            order_size: Size of the order
            hours_ahead: Hours to look ahead

        Returns:
            List: Optimal execution windows
        """
        if symbol not in self.symbol_subscriptions:
            return []

        try:
            return self.liquidity_profiler.find_optimal_execution_windows(
                symbol, order_size, hours_ahead
            )
        except Exception as e:
            self.logger.error(f"Error finding execution windows for {symbol}: {e}")
            return []

    def add_update_callback(self, symbol: str, callback: Callable) -> None:
        """
        Add callback for market data updates

        Args:
            symbol: Symbol to monitor
            callback: Callback function
        """
        self.update_callbacks[symbol].append(callback)

    def add_pattern_callback(self, callback: Callable) -> None:
        """
        Add callback for pattern detection

        Args:
            callback: Callback function
        """
        self.pattern_callbacks.append(callback)

    async def _update_aggregated_data(self, symbol: str, orderbook: Optional[OrderBookSnapshot] = None,
                                    tick: Optional[TickData] = None, metrics: Optional[MarketMetrics] = None,
                                    patterns: Optional[MicrostructurePatterns] = None) -> None:
        """Update aggregated data for a symbol"""
        current_time = datetime.utcnow()

        if symbol not in self.aggregated_data:
            return

        aggregated = self.aggregated_data[symbol]

        # Update current metrics
        if metrics:
            aggregated.current_metrics = metrics

        # Update patterns
        if patterns:
            aggregated.patterns = patterns

        # Update liquidity profile
        if symbol in self.liquidity_profiler.hourly_profiles:
            hour = current_time.hour
            if hour in self.liquidity_profiler.hourly_profiles[symbol]:
                aggregated.liquidity_profile = self.liquidity_profiler.hourly_profiles[symbol][hour]

        # Update historical data (keep limited history)
        if orderbook:
            aggregated.orderbook_history.append(orderbook)
            if len(aggregated.orderbook_history) > 100:
                aggregated.orderbook_history = aggregated.orderbook_history[-100:]

        if tick:
            aggregated.tick_history.append(tick)
            if len(aggregated.tick_history) > 1000:
                aggregated.tick_history = aggregated.tick_history[-1000:]

        if metrics:
            aggregated.metrics_history.append(metrics)
            if len(aggregated.metrics_history) > 100:
                aggregated.metrics_history = aggregated.metrics_history[-100:]

        # Update cache timestamp
        aggregated.cache_timestamp = current_time
        aggregated.timestamp = current_time

    async def _trigger_update_callbacks(self, symbol: str, update_type: str, data: Any) -> None:
        """Trigger update callbacks for a symbol"""
        callbacks = self.update_callbacks.get(symbol, [])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, update_type, data)
                else:
                    callback(symbol, update_type, data)
            except Exception as e:
                self.logger.error(f"Error in update callback: {e}")

    async def _trigger_pattern_callbacks(self, patterns: MicrostructurePatterns) -> None:
        """Trigger pattern detection callbacks"""
        for callback in self.pattern_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(patterns)
                else:
                    callback(patterns)
            except Exception as e:
                self.logger.error(f"Error in pattern callback: {e}")

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old data and cache validation"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                current_time = datetime.utcnow()
                cleaned_count = 0

                # Clean up old aggregated data
                for symbol, data in self.aggregated_data.items():
                    # Clean old orderbook history
                    cutoff_time = current_time - timedelta(hours=1)
                    old_count = len(data.orderbook_history)
                    data.orderbook_history = [
                        ob for ob in data.orderbook_history
                        if ob.timestamp > cutoff_time
                    ]
                    cleaned_count += old_count - len(data.orderbook_history)

                    # Clean old tick history
                    cutoff_time = current_time - timedelta(minutes=30)
                    old_count = len(data.tick_history)
                    data.tick_history = [
                        tick for tick in data.tick_history
                        if tick.timestamp > cutoff_time
                    ]
                    cleaned_count += old_count - len(data.tick_history)

                # Clean up performance metrics
                if len(self.performance_metrics['processing_times']) > 1000:
                    self.performance_metrics['processing_times'] = deque(
                        list(self.performance_metrics['processing_times'])[-500:],
                        maxlen=1000
                    )

                if cleaned_count > 0:
                    self.logger.debug(f"Cleaned up {cleaned_count} old data points")

            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")

    def get_performance_metrics(self) -> Dict:
        """Get aggregator performance metrics"""
        processing_times = list(self.performance_metrics['processing_times'])

        if processing_times:
            avg_processing_time = sum(processing_times) / len(processing_times)
            max_processing_time = max(processing_times)
        else:
            avg_processing_time = 0
            max_processing_time = 0

        uptime = (datetime.utcnow() - self.performance_metrics['last_reset']).total_seconds()

        return {
            'uptime_seconds': uptime,
            'subscribed_symbols': len(self.symbol_subscriptions),
            'processed_orderbooks': self.performance_metrics['processed_orderbooks'],
            'processed_ticks': self.performance_metrics['processed_ticks'],
            'cache_hit_rate': (
                self.performance_metrics['cache_hits'] /
                max(1, self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
            ),
            'avg_processing_time_ms': avg_processing_time * 1000,
            'max_processing_time_ms': max_processing_time * 1000,
            'error_count': self.performance_metrics['error_count'],
            'memory_usage': {
                'aggregated_data_count': len(self.aggregated_data),
                'tick_analyzer_count': len(self.tick_analyzers),
                'callback_count': sum(len(callbacks) for callbacks in self.update_callbacks.values()),
                'pattern_callback_count': len(self.pattern_callbacks)
            }
        }

    def get_symbol_summary(self, symbol: str) -> Optional[Dict]:
        """Get summary for a specific symbol"""
        if symbol not in self.symbol_subscriptions:
            return None

        aggregated = self.aggregated_data.get(symbol)
        if not aggregated:
            return None

        tick_analyzer = self.tick_analyzers.get(symbol)

        summary = {
            'symbol': symbol,
            'last_update': aggregated.timestamp,
            'cache_valid': aggregated.is_cache_valid(),
            'orderbook_history_count': len(aggregated.orderbook_history),
            'tick_history_count': len(aggregated.tick_history),
            'metrics_history_count': len(aggregated.metrics_history)
        }

        if aggregated.current_metrics:
            summary['current_metrics'] = {
                'spread_bps': aggregated.current_metrics.spread_bps,
                'liquidity_score': aggregated.current_metrics.liquidity_score,
                'imbalance': aggregated.current_metrics.imbalance,
                'large_order_count': len(aggregated.current_metrics.large_orders)
            }

        if tick_analyzer:
            summary['tick_metrics'] = tick_analyzer.get_real_time_metrics()

        return summary