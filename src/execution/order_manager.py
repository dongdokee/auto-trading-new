# src/execution/order_manager.py
import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import heapq
from src.execution.models import Order, OrderStatus
from src.core.patterns import BaseManager, LoggerFactory


@dataclass
class OrderInfo:
    """Order tracking information"""
    id: str
    order: Order
    status: OrderStatus
    submitted_at: datetime
    filled_qty: Decimal = field(default_factory=lambda: Decimal('0'))
    avg_price: Decimal = field(default_factory=lambda: Decimal('0'))
    attempts: int = 0
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    timeout_reason: Optional[str] = None


class OrderManager(BaseManager):
    """Order lifecycle management and tracking"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("OrderManager", config)

        # Order tracking
        self.active_orders: Dict[str, OrderInfo] = {}
        self.order_history: List[OrderInfo] = []

        # Configuration with defaults
        self.max_order_age: int = self.config.get('max_order_age', 300)  # 5 minutes default
        self.max_active_orders: int = self.config.get('max_active_orders', 100)  # Default limit
        self.max_attempts: int = self.config.get('max_attempts', 5)  # Default max attempts
        self.max_history_size: int = self.config.get('max_history_size', 1000)  # Default history limit

        self._lock = asyncio.Lock()
        self.logger = LoggerFactory.get_execution_logger()

    async def _do_initialize(self) -> None:
        """Initialize order manager"""
        self.logger.info("Initializing order manager")
        # Any initialization logic can go here

    async def _do_start(self) -> None:
        """Start order manager"""
        self.logger.info("Starting order manager")
        # Start background tasks like cleanup if needed

    async def _do_stop(self) -> None:
        """Stop order manager"""
        self.logger.info("Stopping order manager")
        # Clean up any background tasks

    async def submit_order(self, order: Order) -> str:
        """Submit a new order and return unique order ID"""
        # Validate order
        if not order.symbol or order.size <= 0:
            raise ValueError("Invalid order: symbol cannot be empty and size must be positive")

        async with self._lock:
            # Check active order limit
            if len(self.active_orders) >= self.max_active_orders:
                raise ValueError("Maximum active orders limit reached")

            # Generate unique ID
            order_id = str(uuid.uuid4())

            # Create order info
            order_info = OrderInfo(
                id=order_id,
                order=order,
                status=OrderStatus.PENDING,
                submitted_at=datetime.now()
            )

            # Add to active orders
            self.active_orders[order_id] = order_info

            return order_id

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        async with self._lock:
            if order_id not in self.active_orders:
                return False

            order_info = self.active_orders[order_id]

            # Cannot cancel already filled orders
            if order_info.status == OrderStatus.FILLED:
                return False

            # Update status and move to history
            order_info.status = OrderStatus.CANCELLED
            order_info.cancelled_at = datetime.now()

            # Move to history
            self.order_history.append(order_info)
            del self.active_orders[order_id]

            return True

    async def update_order_status(self, order_id: str, filled_qty: Decimal, avg_price: Decimal):
        """Update order fill status"""
        async with self._lock:
            if order_id not in self.active_orders:
                return

            order_info = self.active_orders[order_id]
            order_info.filled_qty = filled_qty
            order_info.avg_price = avg_price

            # Determine status based on fill
            if filled_qty >= order_info.order.size:
                # Fully filled (or overfilled)
                order_info.status = OrderStatus.FILLED
                order_info.filled_at = datetime.now()

                # Move to history
                self.order_history.append(order_info)
                del self.active_orders[order_id]
            elif filled_qty > 0:
                # Partially filled
                order_info.status = OrderStatus.PARTIALLY_FILLED

    async def increment_attempts(self, order_id: str):
        """Increment order attempt counter and check max attempts"""
        async with self._lock:
            if order_id not in self.active_orders:
                return

            order_info = self.active_orders[order_id]
            order_info.attempts += 1

            # Check if max attempts exceeded
            if order_info.attempts > self.max_attempts:
                order_info.status = OrderStatus.REJECTED

                # Move to history
                self.order_history.append(order_info)
                del self.active_orders[order_id]

    async def check_stale_orders(self) -> int:
        """Check and cancel stale orders"""
        stale_count = 0
        current_time = datetime.now()
        stale_orders = []

        async with self._lock:
            for order_id, order_info in self.active_orders.items():
                age = (current_time - order_info.submitted_at).total_seconds()
                if age > self.max_order_age:
                    stale_orders.append(order_id)

            # Cancel stale orders
            for order_id in stale_orders:
                order_info = self.active_orders[order_id]
                order_info.status = OrderStatus.CANCELLED
                order_info.cancelled_at = current_time

                # Move to history
                self.order_history.append(order_info)
                del self.active_orders[order_id]
                stale_count += 1

        return stale_count

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get current status of an order"""
        if order_id in self.active_orders:
            return self.active_orders[order_id].status
        return None

    def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Get complete order information"""
        return self.active_orders.get(order_id)

    def get_order_statistics(self) -> Dict:
        """Calculate order statistics from history"""
        total_orders = len(self.order_history)
        if total_orders == 0:
            return {
                'total_orders': 0,
                'filled_orders': 0,
                'cancelled_orders': 0,
                'fill_rate': 0,
                'total_volume': Decimal('0'),
                'average_price': Decimal('0')
            }

        filled_orders = sum(1 for order in self.order_history if order.status == OrderStatus.FILLED)
        cancelled_orders = sum(1 for order in self.order_history if order.status == OrderStatus.CANCELLED)

        total_volume = Decimal('0')
        total_value = Decimal('0')

        for order in self.order_history:
            if order.status == OrderStatus.FILLED:
                total_volume += order.filled_qty
                total_value += order.filled_qty * order.avg_price

        avg_price = total_value / total_volume if total_volume > 0 else Decimal('0')

        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'fill_rate': filled_orders / total_orders,
            'total_volume': total_volume,
            'average_price': avg_price
        }

    def get_orders_by_priority(self) -> List[Tuple[int, str]]:
        """Get orders sorted by priority (urgency)"""
        priority_queue = []

        for order_id, order_info in self.active_orders.items():
            # Higher urgency = lower priority number for heapq
            urgency_priority = {
                'IMMEDIATE': 1,
                'HIGH': 2,
                'MEDIUM': 3,
                'LOW': 4
            }
            priority = urgency_priority.get(order_info.order.urgency.value, 3)
            heapq.heappush(priority_queue, (priority, order_id))

        return priority_queue

    def get_performance_metrics(self) -> Dict:
        """Get order execution performance metrics"""
        if not self.order_history:
            return {
                'average_execution_time': 0,
                'fill_rate': 0,
                'success_rate': 0,
                'total_processed': 0
            }

        total_execution_time = 0
        filled_count = 0
        successful_count = 0

        for order in self.order_history:
            if order.filled_at and order.submitted_at:
                execution_time = (order.filled_at - order.submitted_at).total_seconds()
                total_execution_time += execution_time

            if order.status == OrderStatus.FILLED:
                filled_count += 1
                successful_count += 1
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                successful_count += 1

        avg_execution_time = total_execution_time / filled_count if filled_count > 0 else 0

        return {
            'average_execution_time': avg_execution_time,
            'fill_rate': filled_count / len(self.order_history),
            'success_rate': successful_count / len(self.order_history),
            'total_processed': len(self.order_history)
        }

    async def modify_order(self, order_id: str, new_size: Optional[Decimal] = None,
                          new_price: Optional[Decimal] = None) -> bool:
        """Modify an existing order"""
        async with self._lock:
            if order_id not in self.active_orders:
                return False

            order_info = self.active_orders[order_id]

            # Cannot modify filled or cancelled orders
            if order_info.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                return False

            # Update order parameters
            if new_size is not None:
                order_info.order.size = new_size
            if new_price is not None:
                order_info.order.price = new_price

            return True

    async def cleanup_old_orders(self):
        """Clean up old orders from history"""
        if len(self.order_history) > self.max_history_size:
            # Keep only the most recent orders
            self.order_history = self.order_history[-self.max_history_size:]

    async def handle_order_timeout(self, order_id: str):
        """Handle order timeout scenario"""
        async with self._lock:
            if order_id not in self.active_orders:
                return

            order_info = self.active_orders[order_id]
            order_info.status = OrderStatus.CANCELLED
            order_info.cancelled_at = datetime.now()
            order_info.timeout_reason = "Order timed out"

            # Move to history
            self.order_history.append(order_info)
            del self.active_orders[order_id]