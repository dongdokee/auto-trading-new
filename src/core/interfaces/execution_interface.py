# src/core/interfaces/execution_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from decimal import Decimal

from src.core.models import Order


class IOrderManager(ABC):
    """
    Abstract interface for order management.

    Defines the contract for managing order lifecycle,
    enabling different order management implementations.
    """

    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """
        Submit an order for execution.

        Args:
            order: Order to submit

        Returns:
            Order ID for tracking
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancellation successful
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current order status.

        Args:
            order_id: Order ID to check

        Returns:
            Order status information or None if not found
        """
        pass

    @abstractmethod
    async def update_order_status(
        self,
        order_id: str,
        filled_qty: Decimal,
        avg_price: Decimal
    ) -> None:
        """
        Update order status with execution information.

        Args:
            order_id: Order ID
            filled_qty: Quantity filled
            avg_price: Average fill price
        """
        pass

    @abstractmethod
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """
        Get all currently active orders.

        Returns:
            List of active order information
        """
        pass

    @abstractmethod
    def get_order_statistics(self) -> Dict[str, Any]:
        """
        Get order management statistics.

        Returns:
            Dict containing performance metrics
        """
        pass


class IExecutionEngine(ABC):
    """
    Abstract interface for execution engines.

    Defines the contract for executing orders using various
    execution algorithms and strategies.
    """

    @abstractmethod
    async def execute_order(
        self,
        order: Order,
        execution_strategy: str = "AGGRESSIVE"
    ) -> Dict[str, Any]:
        """
        Execute an order using specified strategy.

        Args:
            order: Order to execute
            execution_strategy: Strategy to use (AGGRESSIVE, PASSIVE, TWAP, VWAP)

        Returns:
            Execution result information
        """
        pass

    @abstractmethod
    async def execute_twap(
        self,
        order: Order,
        duration_minutes: int = 30,
        slice_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute order using Time-Weighted Average Price strategy.

        Args:
            order: Order to execute
            duration_minutes: Execution duration
            slice_count: Number of slices (auto-calculated if None)

        Returns:
            TWAP execution result
        """
        pass

    @abstractmethod
    async def execute_vwap(
        self,
        order: Order,
        participation_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Execute order using Volume-Weighted Average Price strategy.

        Args:
            order: Order to execute
            participation_rate: Market participation rate (0-1)

        Returns:
            VWAP execution result
        """
        pass

    @abstractmethod
    async def estimate_execution_cost(
        self,
        order: Order,
        strategy: str = "AGGRESSIVE"
    ) -> Dict[str, Any]:
        """
        Estimate execution cost for an order.

        Args:
            order: Order to estimate
            strategy: Execution strategy

        Returns:
            Cost estimation information
        """
        pass

    @abstractmethod
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution performance statistics.

        Returns:
            Dict containing execution metrics
        """
        pass