# src/market_data/data_aggregator.py
"""
Phase 8 Optimizations:
- Stream-based tick processing with generators
- Memory-efficient circular buffers for data caching
- Parallel processing for multi-symbol operations
- Batched callback notifications
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Set, AsyncIterator, Iterator
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

# Phase 8 imports
from src.core.patterns.async_utils import BatchProcessor, ConcurrentExecutor, process_concurrently
from src.core.patterns.memory_utils import (
    StreamProcessor, CircularBuffer, MemoryMonitor, create_stream_processor
)


class DataAggregator:
    """
    Central aggregator for multi-symbol market data processing and caching

    Phase 8 Features:
    - Stream-based processing for memory efficiency
    - Parallel multi-symbol operations
    - Circular buffers for fixed memory usage
    - Batched callback notifications
    """

    def __init__(self,
                 cache_ttl: int = 60,
                 max_symbols: int = 50,
                 stream_chunk_size: int = 100,
                 memory_threshold_mb: float = 500.0,
                 enable_parallel_processing: bool = True):
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.cache_ttl = cache_ttl  # seconds
        self.max_symbols = max_symbols
        self.performance_window = 3600  # 1 hour for performance tracking
        self.stream_chunk_size = stream_chunk_size
        self.memory_threshold_mb = memory_threshold_mb
        self.enable_parallel_processing = enable_parallel_processing

        # Core analyzers
        self.orderbook_analyzer = OrderBookAnalyzer()
        self.market_impact_model = MarketImpactModel()
        self.liquidity_profiler = LiquidityProfiler()

        # Per-symbol tick analyzers
        self.tick_analyzers: Dict[str, TickDataAnalyzer] = {}

        # Data cache with memory-efficient circular buffers
        self.aggregated_data: Dict[str, AggregatedMarketData] = {}
        self.symbol_subscriptions: Set[str] = set()

        # Phase 8: Memory-efficient buffers
        self.tick_buffer_size = 1000
        self.orderbook_buffer_size = 500

        # Per-symbol circular buffers for memory efficiency
        self.tick_buffers: Dict[str, CircularBuffer[TickData]] = {}
        self.orderbook_buffers: Dict[str, CircularBuffer[OrderBookSnapshot]] = {}

        # Callback system
        self.update_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.pattern_callbacks: List[Callable] = []

        # Phase 8: Stream processing utilities
        self.stream_processor = create_stream_processor(
            chunk_size=stream_chunk_size,
            memory_threshold_mb=memory_threshold_mb
        )

        # Phase 8: Async processing utilities
        if enable_parallel_processing:
            self.batch_processor = BatchProcessor(
                batch_size=20,
                max_concurrent_batches=3,
                timeout_seconds=10.0
            )
            self.concurrent_executor = ConcurrentExecutor(
                max_concurrent=30,
                timeout_seconds=15.0
            )
        else:
            self.batch_processor = None
            self.concurrent_executor = None

        # Phase 8: Memory monitoring
        self.memory_monitor = MemoryMonitor(alert_threshold_mb=memory_threshold_mb)

        # Performance tracking with circular buffer
        self.performance_metrics = {
            'processed_orderbooks': 0,
            'processed_ticks': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': CircularBuffer(maxsize=1000),
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

        # Phase 8: Initialize circular buffers for the symbol
        self.tick_buffers[symbol] = CircularBuffer(maxsize=self.tick_buffer_size)
        self.orderbook_buffers[symbol] = CircularBuffer(maxsize=self.orderbook_buffer_size)

        self.logger.info(f"Subscribed to symbol {symbol} with optimized buffers")

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

    # Phase 8 Optimization Methods

    async def process_tick_stream(self, tick_stream: AsyncIterator[TickData]) -> AsyncIterator[MicrostructurePatterns]:
        """
        Process stream of tick data with memory efficiency

        Args:
            tick_stream: Async iterator of tick data

        Yields:
            MicrostructurePatterns: Detected patterns
        """
        async def process_tick(tick: TickData) -> Optional[MicrostructurePatterns]:
            """Process single tick and return patterns if detected"""
            if tick.symbol not in self.symbol_subscriptions:
                return None

            try:
                # Add to circular buffer
                self.tick_buffers[tick.symbol].append(tick)

                # Process tick through analyzer
                tick_analyzer = self.tick_analyzers[tick.symbol]
                patterns = tick_analyzer.process_tick(tick)

                # Update metrics
                self.performance_metrics['processed_ticks'] += 1

                return patterns if patterns else None

            except Exception as e:
                self.logger.error(f"Error processing tick {tick.symbol}: {e}")
                self.performance_metrics['error_count'] += 1
                return None

        # Stream processing with memory efficiency
        async for patterns in self.stream_processor.process_stream(
            stream=tick_stream,
            processor=process_tick,
            enable_backpressure=True
        ):
            if patterns:
                yield patterns

    async def process_orderbook_stream(self, orderbook_stream: AsyncIterator[OrderBookSnapshot]) -> AsyncIterator[MarketMetrics]:
        """
        Process stream of orderbook data with memory efficiency

        Args:
            orderbook_stream: Async iterator of orderbook snapshots

        Yields:
            MarketMetrics: Market metrics from analysis
        """
        async def process_orderbook(orderbook: OrderBookSnapshot) -> Optional[MarketMetrics]:
            """Process single orderbook and return metrics"""
            if orderbook.symbol not in self.symbol_subscriptions:
                return None

            try:
                # Add to circular buffer
                self.orderbook_buffers[orderbook.symbol].append(orderbook)

                # Analyze order book
                metrics = self.orderbook_analyzer.analyze_orderbook(orderbook)

                # Update liquidity profile
                self.liquidity_profiler.update_profile(
                    orderbook.symbol,
                    pd.Timestamp(orderbook.timestamp),
                    metrics
                )

                # Update metrics
                self.performance_metrics['processed_orderbooks'] += 1

                return metrics

            except Exception as e:
                self.logger.error(f"Error processing orderbook {orderbook.symbol}: {e}")
                self.performance_metrics['error_count'] += 1
                return None

        # Stream processing with memory efficiency
        async for metrics in self.stream_processor.process_stream(
            stream=orderbook_stream,
            processor=process_orderbook,
            enable_backpressure=True
        ):
            if metrics:
                yield metrics

    async def batch_process_symbols_parallel(self, symbols: List[str], operation: str) -> Dict[str, Any]:
        """
        Process operation on multiple symbols in parallel

        Args:
            symbols: List of symbols to process
            operation: Operation type ('refresh', 'analyze', 'cleanup')

        Returns:
            Dictionary with results per symbol
        """
        if not self.enable_parallel_processing or not self.batch_processor:
            # Fall back to sequential processing
            results = {}
            for symbol in symbols:
                try:
                    if operation == 'refresh':
                        results[symbol] = await self.get_market_data(symbol, force_refresh=True)
                    elif operation == 'analyze':
                        results[symbol] = await self._analyze_symbol_comprehensive(symbol)
                    elif operation == 'cleanup':
                        results[symbol] = await self._cleanup_symbol_data(symbol)
                except Exception as e:
                    results[symbol] = {'error': str(e)}
            return results

        async def process_symbol(symbol: str) -> tuple[str, Any]:
            """Process single symbol operation"""
            try:
                if operation == 'refresh':
                    result = await self.get_market_data(symbol, force_refresh=True)
                elif operation == 'analyze':
                    result = await self._analyze_symbol_comprehensive(symbol)
                elif operation == 'cleanup':
                    result = await self._cleanup_symbol_data(symbol)
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                return symbol, result

            except Exception as e:
                return symbol, {'error': str(e)}

        # Process symbols in parallel batches
        batch_result = await self.batch_processor.process_batch(
            items=symbols,
            processor=process_symbol,
            return_exceptions=True
        )

        # Aggregate results
        results = {}
        for result in batch_result.successful:
            if isinstance(result, tuple) and len(result) == 2:
                symbol, data = result
                results[symbol] = data

        for symbol, error in batch_result.failed:
            results[symbol] = {'error': str(error)}

        self.logger.info(
            f"Batch processed {len(symbols)} symbols: "
            f"{batch_result.success_count} successful, {batch_result.failure_count} failed"
        )

        return results

    async def _analyze_symbol_comprehensive(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive analysis for a symbol"""
        if symbol not in self.symbol_subscriptions:
            return {'error': 'Symbol not subscribed'}

        # Get recent data from circular buffers
        recent_ticks = list(self.tick_buffers[symbol])
        recent_orderbooks = list(self.orderbook_buffers[symbol])

        # Analyze patterns in recent data
        tick_analyzer = self.tick_analyzers[symbol]
        patterns_summary = []

        for tick in recent_ticks[-100:]:  # Last 100 ticks
            patterns = tick_analyzer.process_tick(tick)
            if patterns:
                patterns_summary.append({
                    'timestamp': tick.timestamp,
                    'quote_stuffing': patterns.quote_stuffing,
                    'layering': patterns.layering,
                    'momentum_ignition': patterns.momentum_ignition,
                    'ping_pong': patterns.ping_pong
                })

        # Analyze market metrics from recent orderbooks
        metrics_summary = []
        for orderbook in recent_orderbooks[-50:]:  # Last 50 orderbooks
            metrics = self.orderbook_analyzer.analyze_orderbook(orderbook)
            metrics_summary.append({
                'timestamp': orderbook.timestamp,
                'spread_bps': metrics.spread_bps,
                'liquidity_score': metrics.liquidity_score,
                'imbalance': metrics.imbalance
            })

        return {
            'symbol': symbol,
            'recent_ticks_count': len(recent_ticks),
            'recent_orderbooks_count': len(recent_orderbooks),
            'patterns_detected': len(patterns_summary),
            'patterns_summary': patterns_summary[-10:],  # Last 10 patterns
            'metrics_summary': metrics_summary[-10:],    # Last 10 metrics
            'buffer_utilization': {
                'tick_buffer': self.tick_buffers[symbol].utilization,
                'orderbook_buffer': self.orderbook_buffers[symbol].utilization
            }
        }

    async def _cleanup_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Cleanup old data for a symbol"""
        if symbol not in self.symbol_subscriptions:
            return {'error': 'Symbol not subscribed'}

        # Clear circular buffers (they auto-manage size)
        tick_buffer_size_before = len(self.tick_buffers[symbol])
        orderbook_buffer_size_before = len(self.orderbook_buffers[symbol])

        # Reset aggregated data cache
        if symbol in self.aggregated_data:
            self.aggregated_data[symbol].orderbook_history.clear()
            self.aggregated_data[symbol].tick_history.clear()
            self.aggregated_data[symbol].metrics_history.clear()

        return {
            'symbol': symbol,
            'tick_buffer_size_before': tick_buffer_size_before,
            'orderbook_buffer_size_before': orderbook_buffer_size_before,
            'cleanup_completed': True
        }

    def get_memory_efficient_summary(self) -> Dict[str, Any]:
        """Get memory-efficient summary using circular buffers"""
        memory_stats = self.memory_monitor.get_current_stats()

        # Stream processor stats
        stream_stats = self.stream_processor.get_stats()

        # Buffer utilization stats
        buffer_stats = {}
        for symbol in self.symbol_subscriptions:
            if symbol in self.tick_buffers and symbol in self.orderbook_buffers:
                buffer_stats[symbol] = {
                    'tick_buffer': self.tick_buffers[symbol].get_stats(),
                    'orderbook_buffer': self.orderbook_buffers[symbol].get_stats()
                }

        # Processing performance with circular buffer
        processing_times_list = list(self.performance_metrics['processing_times'])
        avg_processing_time = (
            sum(processing_times_list) / len(processing_times_list)
            if processing_times_list else 0.0
        )

        summary = {
            'memory_stats': {
                'current_rss_mb': memory_stats.rss_mb,
                'current_percent': memory_stats.percent,
                'available_mb': memory_stats.available_mb
            },
            'stream_processing': stream_stats,
            'buffer_utilization': buffer_stats,
            'performance': {
                'avg_processing_time_ms': avg_processing_time * 1000,
                'total_processed_ticks': self.performance_metrics['processed_ticks'],
                'total_processed_orderbooks': self.performance_metrics['processed_orderbooks'],
                'error_count': self.performance_metrics['error_count']
            }
        }

        # Add batch processor stats if available
        if self.enable_parallel_processing and self.batch_processor:
            summary['batch_processing'] = self.batch_processor.get_metrics()

        # Add concurrent executor stats if available
        if self.enable_parallel_processing and self.concurrent_executor:
            summary['concurrent_execution'] = self.concurrent_executor.get_metrics()

        return summary

    async def cleanup_optimization_resources(self):
        """Cleanup Phase 8 optimization resources"""
        try:
            # Shutdown concurrent executor if enabled
            if self.enable_parallel_processing and self.concurrent_executor:
                await self.concurrent_executor.shutdown(timeout=5.0)

            # Clear all circular buffers
            for symbol in self.symbol_subscriptions:
                if symbol in self.tick_buffers:
                    self.tick_buffers[symbol].clear()
                if symbol in self.orderbook_buffers:
                    self.orderbook_buffers[symbol].clear()

            # Clear performance tracking buffer
            if hasattr(self.performance_metrics['processing_times'], 'clear'):
                self.performance_metrics['processing_times'].clear()

            self.logger.info("Phase 8 optimization resources cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Error during optimization resource cleanup: {e}")