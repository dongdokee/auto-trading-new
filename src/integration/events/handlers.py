# src/integration/events/handlers.py
"""
Event Handlers

Base classes and interfaces for handling different types of events.
Provides error handling, retry logic, and standardized event processing.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from .models import BaseEvent, EventUnion, EventType


@dataclass
class HandlerResult:
    """Result of event handler execution"""
    success: bool
    message: str = ""
    retry_after: Optional[timedelta] = None
    additional_events: List[BaseEvent] = None

    def __post_init__(self):
        if self.additional_events is None:
            self.additional_events = []


class BaseEventHandler(ABC):
    """Abstract base class for all event handlers"""

    def __init__(self, handler_name: str, max_retries: int = 3):
        self.handler_name = handler_name
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"handler.{handler_name}")

        # Performance tracking
        self.events_processed = 0
        self.events_failed = 0
        self.total_processing_time = 0.0
        self.last_processed = None

    @abstractmethod
    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """
        Handle a specific event type

        Args:
            event: Event to process

        Returns:
            HandlerResult indicating success/failure and any follow-up actions
        """
        pass

    async def process_event_with_retry(self, event: BaseEvent) -> HandlerResult:
        """
        Process event with retry logic and error handling

        Args:
            event: Event to process

        Returns:
            Final result after retries
        """
        start_time = datetime.now()

        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Processing event {event.event_id}, attempt {attempt + 1}")

                result = await self.handle_event(event)

                if result.success:
                    # Update performance metrics
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self.events_processed += 1
                    self.total_processing_time += processing_time
                    self.last_processed = datetime.now()

                    self.logger.debug(f"Successfully processed event {event.event_id}")
                    return result

                else:
                    self.logger.warning(f"Event {event.event_id} failed: {result.message}")

                    # Check if we should retry
                    if attempt < self.max_retries and result.retry_after:
                        await asyncio.sleep(result.retry_after.total_seconds())
                        continue
                    else:
                        # Final failure
                        self.events_failed += 1
                        return result

            except Exception as e:
                self.logger.error(f"Exception in handler {self.handler_name}: {e}")

                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = min(2 ** attempt, 30)  # Max 30 seconds
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final failure
                    self.events_failed += 1
                    return HandlerResult(
                        success=False,
                        message=f"Handler failed after {self.max_retries} retries: {str(e)}"
                    )

        # Should not reach here
        return HandlerResult(success=False, message="Unexpected error in retry logic")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get handler performance metrics"""
        avg_processing_time = (
            self.total_processing_time / self.events_processed
            if self.events_processed > 0 else 0.0
        )

        success_rate = (
            (self.events_processed / (self.events_processed + self.events_failed)) * 100
            if (self.events_processed + self.events_failed) > 0 else 0.0
        )

        return {
            'handler_name': self.handler_name,
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'success_rate_pct': success_rate,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'total_processing_time_s': self.total_processing_time,
            'last_processed': self.last_processed
        }


class EventHandlerRegistry:
    """Registry for managing event handlers by event type"""

    def __init__(self):
        self.handlers: Dict[EventType, List[BaseEventHandler]] = {}
        self.logger = logging.getLogger("handler_registry")

    def register_handler(self, event_type: EventType, handler: BaseEventHandler):
        """Register a handler for a specific event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []

        self.handlers[event_type].append(handler)
        self.logger.info(f"Registered handler {handler.handler_name} for {event_type.value}")

    def unregister_handler(self, event_type: EventType, handler: BaseEventHandler):
        """Unregister a handler for a specific event type"""
        if event_type in self.handlers:
            try:
                self.handlers[event_type].remove(handler)
                self.logger.info(f"Unregistered handler {handler.handler_name} for {event_type.value}")
            except ValueError:
                self.logger.warning(f"Handler {handler.handler_name} not found for {event_type.value}")

    def get_handlers(self, event_type: EventType) -> List[BaseEventHandler]:
        """Get all handlers for a specific event type"""
        return self.handlers.get(event_type, [])

    async def process_event(self, event: BaseEvent) -> List[HandlerResult]:
        """Process event through all registered handlers"""
        handlers = self.get_handlers(event.event_type)

        if not handlers:
            self.logger.warning(f"No handlers registered for event type {event.event_type.value}")
            return []

        # Process event through all handlers concurrently
        tasks = [handler.process_event_with_retry(event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Handler {handlers[i].handler_name} raised exception: {result}")
                processed_results.append(HandlerResult(
                    success=False,
                    message=f"Handler exception: {str(result)}"
                ))
            else:
                processed_results.append(result)

        return processed_results

    def get_registry_metrics(self) -> Dict[str, Any]:
        """Get metrics for all registered handlers"""
        metrics = {
            'total_handlers': 0,
            'handlers_by_type': {},
            'handler_performance': []
        }

        for event_type, handlers in self.handlers.items():
            metrics['total_handlers'] += len(handlers)
            metrics['handlers_by_type'][event_type.value] = len(handlers)

            for handler in handlers:
                metrics['handler_performance'].append(handler.get_performance_metrics())

        return metrics


# Predefined handler classes for common event types

class MarketDataHandler(BaseEventHandler):
    """Handler for market data events"""

    def __init__(self, callback: Optional[Callable] = None):
        super().__init__("market_data_handler")
        self.callback = callback

    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """Process market data event"""
        if self.callback:
            try:
                await self.callback(event)
                return HandlerResult(success=True)
            except Exception as e:
                return HandlerResult(
                    success=False,
                    message=f"Market data callback failed: {str(e)}",
                    retry_after=timedelta(seconds=1)
                )

        return HandlerResult(success=True, message="No callback registered")


class SignalHandler(BaseEventHandler):
    """Handler for strategy signal events"""

    def __init__(self, callback: Optional[Callable] = None):
        super().__init__("signal_handler")
        self.callback = callback

    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """Process strategy signal event"""
        if self.callback:
            try:
                additional_events = await self.callback(event)
                return HandlerResult(
                    success=True,
                    additional_events=additional_events or []
                )
            except Exception as e:
                return HandlerResult(
                    success=False,
                    message=f"Signal callback failed: {str(e)}",
                    retry_after=timedelta(seconds=2)
                )

        return HandlerResult(success=True, message="No callback registered")


class OrderHandler(BaseEventHandler):
    """Handler for order events"""

    def __init__(self, callback: Optional[Callable] = None):
        super().__init__("order_handler")
        self.callback = callback

    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """Process order event"""
        if self.callback:
            try:
                result = await self.callback(event)
                return HandlerResult(success=True)
            except Exception as e:
                return HandlerResult(
                    success=False,
                    message=f"Order callback failed: {str(e)}",
                    retry_after=timedelta(seconds=1)
                )

        return HandlerResult(success=True, message="No callback registered")


class RiskHandler(BaseEventHandler):
    """Handler for risk events"""

    def __init__(self, callback: Optional[Callable] = None):
        super().__init__("risk_handler")
        self.callback = callback

    async def handle_event(self, event: BaseEvent) -> HandlerResult:
        """Process risk event"""
        if self.callback:
            try:
                await self.callback(event)
                return HandlerResult(success=True)
            except Exception as e:
                return HandlerResult(
                    success=False,
                    message=f"Risk callback failed: {str(e)}",
                    retry_after=timedelta(seconds=0.5)
                )

        return HandlerResult(success=True, message="No callback registered")