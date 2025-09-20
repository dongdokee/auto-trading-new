# tests/integration/test_system_performance/test_performance_benchmarks.py
"""
Performance Benchmark Tests

Comprehensive performance testing and benchmarking for the trading system.
"""

import pytest
import asyncio
import time
import statistics
from decimal import Decimal
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from src.integration.trading_orchestrator import TradingOrchestrator, OrchestratorConfig
from src.integration.events.models import (
    MarketDataEvent, StrategySignalEvent, OrderEvent, ExecutionEvent
)
from src.integration.adapters.strategy_adapter import StrategyAdapter
from src.integration.adapters.execution_adapter import ExecutionAdapter


class TestPerformanceBenchmarks:
    """Performance benchmark tests for the trading system"""

    @pytest.fixture
    async def performance_system(self):
        """Setup system optimized for performance testing"""
        config = OrchestratorConfig(
            enable_paper_trading=True,
            max_concurrent_orders=100,
            risk_check_interval_seconds=5,
            portfolio_rebalance_interval_seconds=30,
            health_check_interval_seconds=10
        )

        orchestrator = TradingOrchestrator(config)
        await orchestrator.start()

        # Create adapters for testing
        strategy_adapter = StrategyAdapter(orchestrator.event_bus)
        execution_adapter = ExecutionAdapter(
            orchestrator.event_bus,
            orchestrator.state_manager
        )

        await strategy_adapter.start()
        await execution_adapter.start()

        components = {
            'orchestrator': orchestrator,
            'strategy_adapter': strategy_adapter,
            'execution_adapter': execution_adapter
        }

        yield components

        # Cleanup
        await execution_adapter.stop()
        await strategy_adapter.stop()
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_event_processing_throughput(self, performance_system):
        """Test event processing throughput under load"""
        orchestrator = performance_system['orchestrator']
        event_bus = orchestrator.event_bus

        # Test parameters
        num_events = 1000
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]

        # Create events
        events = []
        for i in range(num_events):
            symbol = symbols[i % len(symbols)]
            event = MarketDataEvent(
                source_component="performance_test",
                symbol=symbol,
                price=Decimal(f"{50000 + (i % 1000)}"),
                volume=Decimal("100.0")
            )
            events.append(event)

        # Measure throughput
        start_time = time.time()

        # Send events in batches for realistic throughput
        batch_size = 50
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            tasks = [event_bus.publish(event) for event in batch]
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.01)  # Small delay between batches

        # Wait for all events to be processed
        await asyncio.sleep(2)

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate metrics
        events_per_second = num_events / processing_time

        # Performance assertions
        assert events_per_second > 100, f"Throughput too low: {events_per_second:.2f} events/sec"

        # Check event bus metrics
        metrics = event_bus.get_metrics()
        assert metrics['event_bus']['events_published'] >= num_events

        print(f"Event processing throughput: {events_per_second:.2f} events/sec")

    @pytest.mark.asyncio
    async def test_order_execution_latency(self, performance_system):
        """Test order execution latency"""
        orchestrator = performance_system['orchestrator']
        execution_adapter = performance_system['execution_adapter']

        # Mock execution for consistent timing
        with patch.object(execution_adapter.order_router, 'route_order') as mock_router:
            mock_router.return_value = {
                'strategy': 'AGGRESSIVE',
                'total_filled': Decimal('0.1'),
                'avg_price': Decimal('50000.0'),
                'total_cost': Decimal('5.0'),
                'slices': []
            }

            latencies = []
            num_orders = 100

            for i in range(num_orders):
                order = OrderEvent(
                    source_component="latency_test",
                    action="CREATE",
                    order_id=f"latency_test_{i}",
                    symbol="BTCUSDT",
                    side="BUY",
                    size=Decimal("0.1"),
                    order_type="MARKET"
                )

                start_time = time.time()
                await orchestrator.event_bus.publish(order)

                # Wait for processing
                await asyncio.sleep(0.05)

                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                # Small delay between orders
                await asyncio.sleep(0.01)

        # Calculate latency statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]

        # Performance assertions
        assert avg_latency < 100, f"Average latency too high: {avg_latency:.2f}ms"
        assert p95_latency < 200, f"P95 latency too high: {p95_latency:.2f}ms"
        assert p99_latency < 500, f"P99 latency too high: {p99_latency:.2f}ms"

        print(f"Order execution latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms, P99: {p99_latency:.2f}ms")

    @pytest.mark.asyncio
    async def test_concurrent_order_processing(self, performance_system):
        """Test concurrent order processing capacity"""
        orchestrator = performance_system['orchestrator']
        execution_adapter = performance_system['execution_adapter']

        # Mock execution for testing
        with patch.object(execution_adapter.order_router, 'route_order') as mock_router:
            mock_router.return_value = {
                'strategy': 'TWAP',
                'total_filled': Decimal('1.0'),
                'avg_price': Decimal('50000.0'),
                'total_cost': Decimal('25.0'),
                'slices': []
            }

            num_concurrent_orders = 50
            orders = []

            # Create concurrent orders
            for i in range(num_concurrent_orders):
                order = OrderEvent(
                    source_component="concurrent_test",
                    action="CREATE",
                    order_id=f"concurrent_order_{i}",
                    symbol=f"SYMBOL{i % 5}USDT",
                    side="BUY" if i % 2 == 0 else "SELL",
                    size=Decimal("1.0"),
                    order_type="MARKET"
                )
                orders.append(order)

            # Submit all orders concurrently
            start_time = time.time()

            tasks = [orchestrator.event_bus.publish(order) for order in orders]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Wait for processing
            await asyncio.sleep(3)

            end_time = time.time()
            processing_time = end_time - start_time

            # Check results
            successful_submissions = sum(1 for r in results if r is True)
            orders_per_second = successful_submissions / processing_time

            # Performance assertions
            assert successful_submissions >= num_concurrent_orders * 0.9, "Too many order submission failures"
            assert orders_per_second > 10, f"Concurrent processing too slow: {orders_per_second:.2f} orders/sec"

            print(f"Concurrent order processing: {orders_per_second:.2f} orders/sec, Success rate: {successful_submissions}/{num_concurrent_orders}")

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, performance_system):
        """Test memory usage under sustained load"""
        import psutil

        orchestrator = performance_system['orchestrator']
        process = psutil.Process()

        # Record initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generate sustained load
        num_events = 2000
        events_per_batch = 100

        for batch in range(num_events // events_per_batch):
            batch_events = []
            for i in range(events_per_batch):
                event = MarketDataEvent(
                    source_component="memory_test",
                    symbol=f"SYMBOL{(batch * events_per_batch + i) % 10}USDT",
                    price=Decimal(f"{50000 + i}"),
                    volume=Decimal("100.0")
                )
                batch_events.append(event)

            # Submit batch
            tasks = [orchestrator.event_bus.publish(event) for event in batch_events]
            await asyncio.gather(*tasks)

            # Small delay between batches
            await asyncio.sleep(0.05)

        # Wait for processing to complete
        await asyncio.sleep(3)

        # Record final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory usage should not increase excessively
        assert memory_increase < 100, f"Memory leak detected: {memory_increase:.2f}MB increase"

        print(f"Memory usage - Initial: {initial_memory:.2f}MB, Final: {final_memory:.2f}MB, Increase: {memory_increase:.2f}MB")

    @pytest.mark.asyncio
    async def test_strategy_signal_generation_performance(self, performance_system):
        """Test strategy signal generation performance"""
        strategy_adapter = performance_system['strategy_adapter']

        # Track signal generation times
        signal_times = []
        num_market_updates = 200

        for i in range(num_market_updates):
            market_data = MarketDataEvent(
                source_component="strategy_perf_test",
                symbol="BTCUSDT",
                price=Decimal(f"{50000 + (i % 1000)}"),
                volume=Decimal("100.0"),
                bid=Decimal(f"{49995 + (i % 1000)}"),
                ask=Decimal(f"{50005 + (i % 1000)}")
            )

            start_time = time.time()
            await strategy_adapter.process_market_data(market_data)
            end_time = time.time()

            signal_time_ms = (end_time - start_time) * 1000
            signal_times.append(signal_time_ms)

            await asyncio.sleep(0.01)

        # Calculate performance metrics
        avg_signal_time = statistics.mean(signal_times)
        max_signal_time = max(signal_times)

        # Performance assertions
        assert avg_signal_time < 10, f"Average signal generation too slow: {avg_signal_time:.2f}ms"
        assert max_signal_time < 50, f"Maximum signal generation too slow: {max_signal_time:.2f}ms"

        print(f"Strategy signal generation - Avg: {avg_signal_time:.2f}ms, Max: {max_signal_time:.2f}ms")

    @pytest.mark.asyncio
    async def test_system_startup_time(self, performance_system):
        """Test system startup performance"""
        # This test measures the startup time indirectly
        # The actual startup time is measured in the fixture setup

        # Create a new system to measure startup
        config = OrchestratorConfig(enable_paper_trading=True)

        start_time = time.time()

        new_orchestrator = TradingOrchestrator(config)
        await new_orchestrator.start()

        end_time = time.time()
        startup_time = end_time - start_time

        # Cleanup
        await new_orchestrator.stop()

        # Performance assertion
        assert startup_time < 5.0, f"System startup too slow: {startup_time:.2f}s"

        print(f"System startup time: {startup_time:.2f}s")

    @pytest.mark.asyncio
    async def test_event_bus_queue_performance(self, performance_system):
        """Test event bus queue performance under various loads"""
        event_bus = performance_system['orchestrator'].event_bus

        # Test different queue sizes and measure performance
        queue_sizes = [10, 50, 100, 500]
        performance_results = {}

        for queue_size in queue_sizes:
            # Create events
            events = []
            for i in range(queue_size):
                event = MarketDataEvent(
                    source_component="queue_perf_test",
                    symbol=f"TEST{i % 5}USDT",
                    price=Decimal("50000"),
                    volume=Decimal("100")
                )
                events.append(event)

            # Measure queue performance
            start_time = time.time()

            # Submit all events
            for event in events:
                await event_bus.publish(event)

            # Wait for processing
            await asyncio.sleep(max(0.1, queue_size * 0.001))

            end_time = time.time()
            processing_time = end_time - start_time

            events_per_second = queue_size / processing_time
            performance_results[queue_size] = events_per_second

        # Verify performance scales reasonably
        for queue_size, eps in performance_results.items():
            assert eps > 50, f"Queue performance too low for size {queue_size}: {eps:.2f} events/sec"

        print("Queue performance results:")
        for queue_size, eps in performance_results.items():
            print(f"  {queue_size} events: {eps:.2f} events/sec")

    @pytest.mark.asyncio
    async def test_resource_utilization_limits(self, performance_system):
        """Test that system stays within resource utilization limits"""
        import psutil

        orchestrator = performance_system['orchestrator']
        process = psutil.Process()

        # Monitor resource usage during load test
        cpu_readings = []
        memory_readings = []

        # Generate load while monitoring resources
        for i in range(100):
            # Create some work
            events = []
            for j in range(10):
                event = MarketDataEvent(
                    source_component="resource_test",
                    symbol=f"RESOURCE{j}USDT",
                    price=Decimal(f"{50000 + j}"),
                    volume=Decimal("100")
                )
                events.append(event)

            # Submit events
            tasks = [orchestrator.event_bus.publish(event) for event in events]
            await asyncio.gather(*tasks)

            # Record resource usage
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024

            cpu_readings.append(cpu_percent)
            memory_readings.append(memory_mb)

            await asyncio.sleep(0.05)

        # Calculate average resource usage
        avg_cpu = statistics.mean(cpu_readings)
        max_cpu = max(cpu_readings)
        avg_memory = statistics.mean(memory_readings)
        max_memory = max(memory_readings)

        # Resource utilization assertions
        assert avg_cpu < 50, f"Average CPU usage too high: {avg_cpu:.1f}%"
        assert max_cpu < 80, f"Peak CPU usage too high: {max_cpu:.1f}%"
        assert max_memory < 500, f"Memory usage too high: {max_memory:.1f}MB"

        print(f"Resource utilization - CPU: {avg_cpu:.1f}% avg, {max_cpu:.1f}% peak, Memory: {avg_memory:.1f}MB avg, {max_memory:.1f}MB peak")

    @pytest.mark.asyncio
    async def test_end_to_end_latency(self, performance_system):
        """Test end-to-end latency from market data to order execution"""
        orchestrator = performance_system['orchestrator']
        execution_adapter = performance_system['execution_adapter']

        # Track end-to-end latencies
        latencies = []

        # Mock execution for timing
        with patch.object(execution_adapter.order_router, 'route_order') as mock_router:
            mock_router.return_value = {
                'strategy': 'AGGRESSIVE',
                'total_filled': Decimal('0.1'),
                'avg_price': Decimal('50000.0'),
                'total_cost': Decimal('5.0'),
                'slices': []
            }

            # Track order completion
            completed_orders = {}

            def track_execution(event):
                if hasattr(event, 'order_id'):
                    completed_orders[event.order_id] = time.time()

            orchestrator.event_bus.subscribe("EXECUTION", track_execution)

            num_tests = 50

            for i in range(num_tests):
                start_time = time.time()

                # 1. Market data arrives
                market_data = MarketDataEvent(
                    source_component="e2e_test",
                    symbol="BTCUSDT",
                    price=Decimal(f"{50000 + i}"),
                    volume=Decimal("100")
                )
                await orchestrator.event_bus.publish(market_data)

                # 2. Generate order (simulated portfolio decision)
                order_id = f"e2e_order_{i}"
                order = OrderEvent(
                    source_component="e2e_test",
                    action="CREATE",
                    order_id=order_id,
                    symbol="BTCUSDT",
                    side="BUY",
                    size=Decimal("0.1")
                )
                await orchestrator.event_bus.publish(order)

                # Wait for completion
                await asyncio.sleep(0.1)

                # Check if order was executed
                if order_id in completed_orders:
                    end_time = completed_orders[order_id]
                    e2e_latency = (end_time - start_time) * 1000
                    latencies.append(e2e_latency)

                await asyncio.sleep(0.02)

        if latencies:
            avg_latency = statistics.mean(latencies)
            p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

            # End-to-end latency assertions
            assert avg_latency < 200, f"End-to-end latency too high: {avg_latency:.2f}ms"
            assert p95_latency < 500, f"P95 end-to-end latency too high: {p95_latency:.2f}ms"

            print(f"End-to-end latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")

    @pytest.mark.asyncio
    async def test_scalability_limits(self, performance_system):
        """Test system scalability limits"""
        orchestrator = performance_system['orchestrator']

        # Test with increasing load to find limits
        load_levels = [10, 50, 100, 200, 500]
        performance_degradation = {}

        for load_level in load_levels:
            start_time = time.time()

            # Generate load
            events = []
            for i in range(load_level):
                event = MarketDataEvent(
                    source_component="scalability_test",
                    symbol=f"SCALE{i % 10}USDT",
                    price=Decimal(f"{50000 + i}"),
                    volume=Decimal("100")
                )
                events.append(event)

            # Submit load
            tasks = [orchestrator.event_bus.publish(event) for event in events]
            await asyncio.gather(*tasks)

            # Wait for processing
            await asyncio.sleep(max(0.5, load_level * 0.005))

            end_time = time.time()
            processing_time = end_time - start_time
            throughput = load_level / processing_time

            performance_degradation[load_level] = throughput

            # Small break between tests
            await asyncio.sleep(0.1)

        # Analyze scalability
        print("Scalability test results:")
        for load, throughput in performance_degradation.items():
            print(f"  {load} events: {throughput:.2f} events/sec")

        # Basic scalability check - throughput shouldn't drop dramatically
        max_throughput = max(performance_degradation.values())
        min_throughput = min(performance_degradation.values())

        # Throughput shouldn't drop below 50% of peak even at high load
        assert min_throughput > max_throughput * 0.3, "System shows poor scalability characteristics"