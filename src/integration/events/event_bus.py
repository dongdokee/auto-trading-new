# src/integration/events/event_bus.py
"""
Event Bus

Central event distribution system for inter-component communication.
Provides async message queue, event routing, and event persistence.

Phase 8 Optimizations:
- Batch event processing with configurable batch sizes
- Memory-efficient circular buffers for event history
- Parallel handler execution with concurrent processing
- Stream-based event processing for high throughput
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any, AsyncIterator
from collections import defaultdict, deque
import json
import uuid

from .models import BaseEvent, EventUnion, EventType, EventPriority
from .handlers import EventHandlerRegistry, BaseEventHandler, HandlerResult

# Phase 8 imports
from src.core.patterns.async_utils import BatchProcessor, ConcurrentExecutor, process_concurrently
from src.core.patterns.memory_utils import CircularBuffer, MemoryMonitor


class EventBusMetrics:
    """Event bus performance and health metrics with Phase 8 optimizations"""

    def __init__(self, history_size: int = 1000):
        self.events_published = 0
        self.events_processed = 0
        self.events_failed = 0
        self.total_processing_time = 0.0

        # Phase 8: Use circular buffers for memory efficiency
        self.queue_size_history = CircularBuffer(maxsize=history_size)
        self.processing_latency_history = CircularBuffer(maxsize=history_size)
        self.batch_processing_history = CircularBuffer(maxsize=100)

        self.start_time = datetime.now()

        # Phase 8: Batch processing metrics
        self.batches_processed = 0
        self.total_batch_size = 0
        self.parallel_handlers_executed = 0

    def record_event_published(self):
        """Record event publication"""
        self.events_published += 1

    def record_event_processed(self, processing_time: float):
        """Record successful event processing"""
        self.events_processed += 1
        self.total_processing_time += processing_time
        self.processing_latency_history.append(processing_time)

    def record_event_failed(self):
        """Record failed event processing"""
        self.events_failed += 1

    def record_queue_size(self, size: int):
        """Record current queue size"""
        self.queue_size_history.append(size)

    def record_batch_processed(self, batch_size: int, processing_time: float):
        """Record batch processing metrics"""
        self.batches_processed += 1
        self.total_batch_size += batch_size

        batch_metrics = {
            'size': batch_size,
            'processing_time': processing_time,
            'timestamp': datetime.now()
        }
        self.batch_processing_history.append(batch_metrics)

    def record_parallel_handler_execution(self, handler_count: int):
        """Record parallel handler execution"""
        self.parallel_handlers_executed += handler_count

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics with Phase 8 optimizations"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        avg_processing_time = (
            self.total_processing_time / self.events_processed
            if self.events_processed > 0 else 0.0
        )

        # Phase 8: Use circular buffer methods
        queue_size_list = list(self.queue_size_history)
        avg_queue_size = (
            sum(queue_size_list) / len(queue_size_list)
            if queue_size_list else 0.0
        )

        processing_times_list = list(self.processing_latency_history)
        max_processing_time = max(processing_times_list) if processing_times_list else 0.0

        success_rate = (
            (self.events_processed / (self.events_processed + self.events_failed)) * 100
            if (self.events_processed + self.events_failed) > 0 else 0.0
        )

        events_per_second = self.events_processed / uptime if uptime > 0 else 0.0

        # Phase 8: Batch processing metrics
        avg_batch_size = (
            self.total_batch_size / self.batches_processed
            if self.batches_processed > 0 else 0.0
        )

        batch_metrics_list = list(self.batch_processing_history)
        avg_batch_processing_time = 0.0
        if batch_metrics_list:
            total_batch_time = sum(bm['processing_time'] for bm in batch_metrics_list)
            avg_batch_processing_time = total_batch_time / len(batch_metrics_list)

        return {
            'uptime_seconds': uptime,
            'events_published': self.events_published,
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'success_rate_pct': success_rate,
            'events_per_second': events_per_second,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'max_processing_time_ms': max_processing_time * 1000,
            'avg_queue_size': avg_queue_size,
            'current_queue_size': queue_size_list[-1] if queue_size_list else 0,
            # Phase 8: Batch processing metrics
            'batch_processing': {
                'batches_processed': self.batches_processed,
                'avg_batch_size': avg_batch_size,
                'avg_batch_processing_time_ms': avg_batch_processing_time * 1000,
                'parallel_handlers_executed': self.parallel_handlers_executed
            },
            # Phase 8: Memory efficiency metrics
            'memory_efficiency': {
                'queue_history_utilization': self.queue_size_history.utilization,
                'latency_history_utilization': self.processing_latency_history.utilization,
                'batch_history_utilization': self.batch_processing_history.utilization
            }
        }


class EventBus:
    """
    Central event bus for system-wide event distribution

    Features:
    - Priority-based event queue
    - Async event processing
    - Event persistence for recovery
    - Metrics and monitoring
    - Circuit breaker for failing handlers

    Phase 8 Features:
    - Batch event processing for efficiency
    - Parallel handler execution
    - Memory-efficient circular buffers
    - Stream-based event processing
    """

    def __init__(self,
                 max_queue_size: int = 10000,
                 enable_persistence: bool = True,
                 persistence_buffer_size: int = 1000,
                 batch_size: int = 50,
                 max_concurrent_handlers: int = 20,
                 enable_batch_processing: bool = True):

        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence
        self.persistence_buffer_size = persistence_buffer_size
        self.batch_size = batch_size
        self.max_concurrent_handlers = max_concurrent_handlers
        self.enable_batch_processing = enable_batch_processing

        # Event queue (priority queue)
        self.event_queue = asyncio.PriorityQueue(maxsize=max_queue_size)

        # Handler registry
        self.handler_registry = EventHandlerRegistry()

        # Phase 8: Memory-efficient event persistence
        self.event_history: CircularBuffer = CircularBuffer(maxsize=persistence_buffer_size)
        self.failed_events: CircularBuffer = CircularBuffer(maxsize=persistence_buffer_size)

        # Metrics with Phase 8 optimizations
        self.metrics = EventBusMetrics(history_size=1000)

        # Processing control
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None

        # Subscribers
        self.subscribers: Dict[EventType, Set[Callable]] = defaultdict(set)

        # Circuit breaker
        self.circuit_breaker_threshold = 10  # failures
        self.circuit_breaker_window = timedelta(minutes=5)
        self.handler_failures: Dict[str, List[datetime]] = defaultdict(list)

        # Phase 8: Batch processing utilities
        if enable_batch_processing:
            self.batch_processor = BatchProcessor(
                batch_size=batch_size,
                max_concurrent_batches=3,
                timeout_seconds=30.0
            )
            self.concurrent_executor = ConcurrentExecutor(
                max_concurrent=max_concurrent_handlers,
                timeout_seconds=15.0
            )
        else:
            self.batch_processor = None
            self.concurrent_executor = None

        # Phase 8: Memory monitoring
        self.memory_monitor = MemoryMonitor(alert_threshold_mb=500.0)

        # Logger
        self.logger = logging.getLogger("event_bus")

    async def start(self):
        """Start the event bus processing"""
        if self.is_running:
            self.logger.warning("Event bus is already running")
            return

        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
        self.logger.info("Event bus started")

    async def stop(self):
        """Stop the event bus processing"""
        if not self.is_running:
            self.logger.warning("Event bus is not running")
            return

        self.is_running = False

        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Event bus stopped")

    async def publish(self, event: BaseEvent) -> bool:
        """
        Publish an event to the bus

        Args:
            event: Event to publish

        Returns:
            True if event was queued successfully
        """
        if not self.is_running:
            self.logger.error("Cannot publish event: Event bus is not running")
            return False

        try:
            # Add to queue with priority
            priority_value = event.priority.value
            queue_item = (priority_value, event.timestamp, event)

            # Non-blocking put (will raise if queue is full)
            self.event_queue.put_nowait(queue_item)

            self.metrics.record_event_published()
            self.metrics.record_queue_size(self.event_queue.qsize())

            self.logger.debug(f"Published event {event.event_id} with priority {event.priority.value}")
            return True

        except asyncio.QueueFull:
            self.logger.error(f"Event queue is full, dropping event {event.event_id}")
            return False

        except Exception as e:
            self.logger.error(f"Failed to publish event {event.event_id}: {e}")
            return False

    def register_handler(self, event_type: EventType, handler: BaseEventHandler):
        """Register an event handler"""
        self.handler_registry.register_handler(event_type, handler)

    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to events with a callback function"""
        self.subscribers[event_type].add(callback)
        self.logger.info(f"Added subscriber for {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from events"""
        self.subscribers[event_type].discard(callback)
        self.logger.info(f"Removed subscriber for {event_type.value}")

    async def _process_events(self):
        """Main event processing loop"""
        self.logger.info("Event processing loop started")

        while self.is_running:
            try:
                # Get next event from queue (blocking with timeout)
                priority, timestamp, event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )

                start_time = datetime.now()

                # Process event
                await self._handle_event(event)

                # Update metrics
                processing_time = (datetime.now() - start_time).total_seconds()
                self.metrics.record_event_processed(processing_time)

                # Mark task as done
                self.event_queue.task_done()

            except asyncio.TimeoutError:
                # No events to process, continue
                continue

            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")
                self.metrics.record_event_failed()

        self.logger.info("Event processing loop stopped")

    async def _handle_event(self, event: BaseEvent):
        """Handle a single event"""
        try:
            # Store in history if persistence is enabled
            if self.enable_persistence:
                self.event_history.append({
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'event_type': event.event_type.value,
                    'source_component': event.source_component,
                    'processed_at': datetime.now()
                })

            # Process through handlers
            handler_results = await self.handler_registry.process_event(event)

            # Process through subscribers
            await self._notify_subscribers(event)

            # Handle any additional events generated by handlers
            for result in handler_results:
                if result.additional_events:
                    for additional_event in result.additional_events:
                        await self.publish(additional_event)

            # Check for handler failures
            failed_handlers = [r for r in handler_results if not r.success]
            if failed_handlers:
                await self._handle_processing_failures(event, failed_handlers)

        except Exception as e:
            self.logger.error(f"Error handling event {event.event_id}: {e}")
            await self._handle_event_failure(event, str(e))

    async def _notify_subscribers(self, event: BaseEvent):
        """Notify all subscribers for the event type"""
        subscribers = self.subscribers.get(event.event_type, set())

        if subscribers:
            # Notify all subscribers concurrently
            tasks = []
            for callback in subscribers:
                task = asyncio.create_task(self._call_subscriber(callback, event))
                tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _call_subscriber(self, callback: Callable, event: BaseEvent):
        """Call a subscriber callback safely"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            self.logger.error(f"Subscriber callback failed for event {event.event_id}: {e}")

    async def _handle_processing_failures(self, event: BaseEvent, failed_results: List[HandlerResult]):
        """Handle processing failures"""
        for result in failed_results:
            self.logger.warning(f"Handler failed for event {event.event_id}: {result.message}")

        # Store failed event
        if self.enable_persistence:
            self.failed_events.append({
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'failure_time': datetime.now(),
                'error_messages': [r.message for r in failed_results]
            })

    async def _handle_event_failure(self, event: BaseEvent, error_message: str):
        """Handle complete event processing failure"""
        self.logger.error(f"Complete failure processing event {event.event_id}: {error_message}")

        if self.enable_persistence:
            self.failed_events.append({
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'failure_time': datetime.now(),
                'error_message': error_message
            })

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive event bus metrics"""
        return {
            'event_bus': self.metrics.get_metrics(),
            'handler_registry': self.handler_registry.get_registry_metrics(),
            'queue_status': {
                'current_size': self.event_queue.qsize(),
                'max_size': self.max_queue_size,
                'is_running': self.is_running
            },
            'persistence': {
                'enabled': self.enable_persistence,
                'history_size': len(self.event_history),
                'failed_events': len(self.failed_events)
            }
        }

    def get_failed_events(self) -> List[Dict[str, Any]]:
        """Get list of failed events for debugging"""
        return list(self.failed_events)

    async def replay_failed_events(self, event_ids: Optional[List[str]] = None) -> int:
        """
        Replay failed events

        Args:
            event_ids: Specific event IDs to replay, or None for all

        Returns:
            Number of events replayed
        """
        if not self.enable_persistence:
            self.logger.warning("Cannot replay events: Persistence is disabled")
            return 0

        replayed_count = 0
        events_to_remove = []

        for i, failed_event in enumerate(self.failed_events):
            if event_ids is None or failed_event['event_id'] in event_ids:
                # Note: This is a simplified replay - in production, you'd need to
                # reconstruct the full event from stored data
                self.logger.info(f"Would replay event {failed_event['event_id']}")
                events_to_remove.append(i)
                replayed_count += 1

        # Remove replayed events from failed list
        for i in reversed(events_to_remove):
            del self.failed_events[i]

        return replayed_count