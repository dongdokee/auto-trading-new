# src/integration/adapters/execution_adapter.py
"""
Execution Adapter

Bridges the order execution system with the event-driven integration system.
Handles order routing, execution monitoring, and result reporting.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
import uuid

from src.integration.events.event_bus import EventBus
from src.integration.events.models import (
    OrderEvent, ExecutionEvent, SystemEvent, RiskEvent,
    EventType, EventPriority
)
from src.integration.events.handlers import BaseEventHandler, HandlerResult
from src.integration.state.manager import StateManager

# Import execution engine components
from src.execution.order_router import SmartOrderRouter
from src.execution.order_manager import OrderManager
from src.execution.execution_algorithms import ExecutionAlgorithms
from src.execution.slippage_controller import SlippageController
from src.execution.models import Order, OrderSide, OrderUrgency

# Import API integration
from src.api.binance.executor import BinanceExecutor
from src.core.config.models import ExchangeConfig


class ExecutionAdapter:
    """
    Adapter for the order execution system

    Responsibilities:
    - Process order events and route to execution engine
    - Monitor order execution and generate execution events
    - Integrate with exchange APIs
    - Track execution performance and slippage
    """

    def __init__(self,
                 event_bus: EventBus,
                 state_manager: StateManager,
                 exchange_config: Optional[ExchangeConfig] = None):

        self.event_bus = event_bus
        self.state_manager = state_manager

        # Initialize execution components
        self.order_router = SmartOrderRouter()
        self.order_manager = OrderManager()
        self.execution_algorithms = ExecutionAlgorithms()
        self.slippage_controller = SlippageController()

        # Initialize exchange executor
        self.exchange_config = exchange_config or self._get_default_config()
        self.exchange_executor = BinanceExecutor(self.exchange_config)

        # Adapter state
        self.is_active = False
        self.is_connected = False
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.execution_history = []

        # Performance tracking
        self.orders_processed = 0
        self.orders_executed = 0
        self.orders_failed = 0
        self.total_slippage = 0.0
        self.execution_times = []

        # Logger
        self.logger = logging.getLogger("execution_adapter")

        # Register event handlers
        self._register_handlers()

    def _get_default_config(self) -> ExchangeConfig:
        """Get default exchange configuration for paper trading"""
        return ExchangeConfig(
            name="BINANCE",
            api_key="paper_trading_key",
            api_secret="paper_trading_secret",
            testnet=True,
            paper_trading=True
        )

    def _register_handlers(self):
        """Register event handlers with the event bus"""
        order_handler = ExecutionOrderHandler(self)
        self.event_bus.register_handler(EventType.ORDER, order_handler)

        system_handler = ExecutionSystemHandler(self)
        self.event_bus.register_handler(EventType.SYSTEM, system_handler)

    async def start(self):
        """Start the execution adapter"""
        try:
            self.is_active = True

            # Connect to exchange
            await self.exchange_executor.connect()
            self.is_connected = True

            # Send startup event
            startup_event = SystemEvent(
                source_component="execution_adapter",
                system_action="START",
                status="RUNNING",
                message="Execution adapter started"
            )
            await self.event_bus.publish(startup_event)

            self.logger.info("Execution adapter started")

        except Exception as e:
            self.logger.error(f"Failed to start execution adapter: {e}")
            await self._generate_system_error("Failed to start execution adapter", str(e))

    async def stop(self):
        """Stop the execution adapter"""
        try:
            self.is_active = False

            # Cancel all active orders
            await self._cancel_all_active_orders()

            # Disconnect from exchange
            if self.is_connected:
                await self.exchange_executor.disconnect()
                self.is_connected = False

            # Send shutdown event
            shutdown_event = SystemEvent(
                source_component="execution_adapter",
                system_action="STOP",
                status="STOPPED",
                message="Execution adapter stopped"
            )
            await self.event_bus.publish(shutdown_event)

            self.logger.info("Execution adapter stopped")

        except Exception as e:
            self.logger.error(f"Error stopping execution adapter: {e}")

    async def process_order(self, order_event: OrderEvent):
        """Process order event"""
        if not self.is_active or not self.is_connected:
            self.logger.warning("Cannot process order: Execution adapter not ready")
            return

        try:
            self.orders_processed += 1

            if order_event.action == "CREATE":
                await self._create_order(order_event)
            elif order_event.action == "CANCEL":
                await self._cancel_order(order_event)
            elif order_event.action == "MODIFY":
                await self._modify_order(order_event)

        except Exception as e:
            self.logger.error(f"Error processing order {order_event.order_id}: {e}")
            await self._generate_execution_failure(order_event, str(e))

    async def _create_order(self, order_event: OrderEvent):
        """Create and execute new order"""
        try:
            start_time = datetime.now()

            # Convert event to order object
            order = self._convert_event_to_order(order_event)

            # Store order in active orders
            order_id = order_event.order_id or str(uuid.uuid4())
            self.active_orders[order_id] = {
                'order': order,
                'event': order_event,
                'created_at': start_time,
                'status': 'PENDING'
            }

            # Update state manager
            await self.state_manager.add_active_order(order_id, {
                'symbol': order.symbol,
                'side': order.side.value,
                'size': float(order.size),
                'order_type': order_event.order_type,
                'status': 'PENDING'
            })

            # Route order through smart router
            execution_result = await self.order_router.route_order(order)

            # Process execution result
            await self._process_execution_result(order_id, execution_result)

            # Track execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_times.append(execution_time)

            if len(self.execution_times) > 1000:
                self.execution_times = self.execution_times[-1000:]

        except Exception as e:
            self.logger.error(f"Error creating order {order_event.order_id}: {e}")
            await self._generate_execution_failure(order_event, str(e))

    async def _cancel_order(self, order_event: OrderEvent):
        """Cancel existing order"""
        try:
            order_id = order_event.order_id

            if order_id not in self.active_orders:
                self.logger.warning(f"Cannot cancel order {order_id}: Order not found")
                return

            # Update order status
            self.active_orders[order_id]['status'] = 'CANCELLING'

            # Cancel through order manager
            success = await self.order_manager.cancel_order(order_id)

            if success:
                # Generate cancellation event
                cancellation_event = ExecutionEvent(
                    source_component="execution_adapter",
                    order_id=order_id,
                    symbol=order_event.symbol,
                    side=order_event.side,
                    executed_qty=Decimal('0'),
                    avg_price=Decimal('0'),
                    status="CANCELLED"
                )
                await self.event_bus.publish(cancellation_event)

                # Remove from active orders
                del self.active_orders[order_id]
                await self.state_manager.remove_active_order(order_id)

                self.logger.info(f"Order {order_id} cancelled successfully")

            else:
                self.logger.error(f"Failed to cancel order {order_id}")

        except Exception as e:
            self.logger.error(f"Error cancelling order {order_event.order_id}: {e}")

    async def _modify_order(self, order_event: OrderEvent):
        """Modify existing order"""
        try:
            order_id = order_event.order_id

            if order_id not in self.active_orders:
                self.logger.warning(f"Cannot modify order {order_id}: Order not found")
                return

            # For simplicity, we'll cancel and recreate the order
            await self._cancel_order(order_event)

            # Create new order with modified parameters
            new_order_event = OrderEvent(
                source_component=order_event.source_component,
                action="CREATE",
                order_id=str(uuid.uuid4()),  # New order ID
                symbol=order_event.symbol,
                side=order_event.side,
                size=order_event.size,
                order_type=order_event.order_type,
                price=order_event.price
            )

            await self._create_order(new_order_event)

        except Exception as e:
            self.logger.error(f"Error modifying order {order_event.order_id}: {e}")

    async def _process_execution_result(self, order_id: str, execution_result: Dict):
        """Process execution result and generate events"""
        try:
            order_info = self.active_orders.get(order_id)
            if not order_info:
                return

            order = order_info['order']
            total_filled = execution_result.get('total_filled', Decimal('0'))
            avg_price = execution_result.get('avg_price', Decimal('0'))
            strategy = execution_result.get('strategy', 'UNKNOWN')

            # Determine execution status
            if total_filled == order.size:
                status = "FILLED"
                self.orders_executed += 1
            elif total_filled > 0:
                status = "PARTIALLY_FILLED"
            else:
                status = "REJECTED"
                self.orders_failed += 1

            # Calculate slippage if applicable
            slippage_bps = None
            if order.price and avg_price > 0:
                slippage = abs(float(avg_price) - float(order.price)) / float(order.price)
                slippage_bps = slippage * 10000  # Convert to basis points
                self.total_slippage += slippage

                # Record slippage
                await self.slippage_controller.record_slippage(
                    order,
                    float(order.price),
                    float(avg_price),
                    float(total_filled)
                )

            # Generate execution event
            execution_event = ExecutionEvent(
                source_component="execution_adapter",
                order_id=order_id,
                symbol=order.symbol,
                side=order.side.value,
                executed_qty=total_filled,
                avg_price=avg_price,
                commission=execution_result.get('total_cost', Decimal('0')),
                status=status,
                execution_time=datetime.now(),
                slippage_bps=slippage_bps,
                execution_strategy=strategy
            )

            await self.event_bus.publish(execution_event)

            # Update order status
            if status in ["FILLED", "REJECTED"]:
                # Remove from active orders
                del self.active_orders[order_id]
                await self.state_manager.remove_active_order(order_id)
            else:
                # Update status for partial fills
                self.active_orders[order_id]['status'] = status
                await self.state_manager.update_active_order(order_id, {'status': status})

            # Store in execution history
            self.execution_history.append({
                'timestamp': datetime.now(),
                'order_id': order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'size': float(order.size),
                'filled': float(total_filled),
                'avg_price': float(avg_price),
                'status': status,
                'strategy': strategy,
                'slippage_bps': slippage_bps
            })

            # Limit history size
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]

            self.logger.info(f"Order {order_id} executed: {status}, "
                           f"Filled: {total_filled}, Price: {avg_price}")

        except Exception as e:
            self.logger.error(f"Error processing execution result for {order_id}: {e}")

    async def _cancel_all_active_orders(self):
        """Cancel all active orders"""
        order_ids = list(self.active_orders.keys())

        for order_id in order_ids:
            try:
                order_info = self.active_orders[order_id]
                cancel_event = OrderEvent(
                    source_component="execution_adapter",
                    action="CANCEL",
                    order_id=order_id,
                    symbol=order_info['order'].symbol,
                    side=order_info['order'].side.value,
                    size=order_info['order'].size
                )
                await self._cancel_order(cancel_event)

            except Exception as e:
                self.logger.error(f"Error cancelling order {order_id}: {e}")

    def _convert_event_to_order(self, order_event: OrderEvent) -> Order:
        """Convert order event to execution order object"""
        side = OrderSide.BUY if order_event.side == "BUY" else OrderSide.SELL

        urgency_map = {
            "IMMEDIATE": OrderUrgency.IMMEDIATE,
            "HIGH": OrderUrgency.HIGH,
            "MEDIUM": OrderUrgency.MEDIUM,
            "LOW": OrderUrgency.LOW
        }
        urgency = urgency_map.get(order_event.urgency, OrderUrgency.MEDIUM)

        return Order(
            symbol=order_event.symbol,
            side=side,
            size=order_event.size,
            urgency=urgency,
            price=order_event.price
        )

    async def _generate_execution_failure(self, order_event: OrderEvent, error_message: str):
        """Generate execution failure event"""
        failure_event = ExecutionEvent(
            source_component="execution_adapter",
            order_id=order_event.order_id or "unknown",
            symbol=order_event.symbol,
            side=order_event.side,
            executed_qty=Decimal('0'),
            avg_price=Decimal('0'),
            status="REJECTED",
            execution_time=datetime.now()
        )

        await self.event_bus.publish(failure_event)
        self.orders_failed += 1

    async def _generate_system_error(self, message: str, details: str):
        """Generate system error event"""
        error_event = SystemEvent(
            source_component="execution_adapter",
            system_action="ERROR",
            status="ERROR",
            message=message,
            error_details={'details': details},
            priority=EventPriority.CRITICAL
        )

        await self.event_bus.publish(error_event)

    def get_adapter_metrics(self) -> Dict[str, Any]:
        """Get execution adapter metrics"""
        avg_execution_time = (
            sum(self.execution_times) / len(self.execution_times)
            if self.execution_times else 0.0
        )

        avg_slippage = (
            self.total_slippage / self.orders_executed
            if self.orders_executed > 0 else 0.0
        )

        success_rate = (
            self.orders_executed / self.orders_processed * 100
            if self.orders_processed > 0 else 0.0
        )

        return {
            'is_active': self.is_active,
            'is_connected': self.is_connected,
            'orders_processed': self.orders_processed,
            'orders_executed': self.orders_executed,
            'orders_failed': self.orders_failed,
            'success_rate_pct': success_rate,
            'active_orders_count': len(self.active_orders),
            'avg_execution_time_ms': avg_execution_time * 1000,
            'avg_slippage_bps': avg_slippage * 10000,
            'execution_history_size': len(self.execution_history)
        }


# Event handlers for execution adapter

class ExecutionOrderHandler(BaseEventHandler):
    """Handler for order events in execution adapter"""

    def __init__(self, adapter: ExecutionAdapter):
        super().__init__("execution_order_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle order event"""
        try:
            await self.adapter.process_order(event)
            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling order event: {e}")
            return HandlerResult(
                success=False,
                message=f"Order processing failed: {str(e)}"
            )


class ExecutionSystemHandler(BaseEventHandler):
    """Handler for system events in execution adapter"""

    def __init__(self, adapter: ExecutionAdapter):
        super().__init__("execution_system_handler")
        self.adapter = adapter

    async def handle_event(self, event) -> HandlerResult:
        """Handle system event"""
        try:
            if event.system_action == "PAUSE":
                # Cancel all active orders and pause
                await self.adapter._cancel_all_active_orders()
                self.adapter.is_active = False
                self.logger.info("Execution adapter paused")

            elif event.system_action == "RESUME":
                self.adapter.is_active = True
                self.logger.info("Execution adapter resumed")

            elif event.system_action == "STOP":
                await self.adapter.stop()

            return HandlerResult(success=True)

        except Exception as e:
            self.logger.error(f"Error handling system event: {e}")
            return HandlerResult(
                success=False,
                message=f"System event processing failed: {str(e)}"
            )