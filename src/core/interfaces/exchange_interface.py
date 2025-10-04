# src/core/interfaces/exchange_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from decimal import Decimal

from src.core.models import Order


class IExchangeClient(ABC):
    """
    Abstract interface for exchange API clients.

    Defines the contract that all exchange implementations must follow,
    enabling the system to work with multiple exchanges without tight coupling.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the exchange.

        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to the exchange.
        """
        pass

    @abstractmethod
    async def submit_order(self, order: Order) -> Dict[str, Any]:
        """
        Submit an order to the exchange.

        Args:
            order: Order to submit

        Returns:
            Dict containing order execution result
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: ID of order to cancel

        Returns:
            bool: True if cancellation successful
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get current status of an order.

        Args:
            order_id: ID of order to check

        Returns:
            Dict containing order status information
        """
        pass

    @abstractmethod
    async def get_account_balance(self) -> Dict[str, Decimal]:
        """
        Get current account balance.

        Returns:
            Dict mapping currency to balance amount
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.

        Returns:
            List of position dictionaries
        """
        pass

    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict containing market data
        """
        pass


class IWebSocketManager(ABC):
    """
    Abstract interface for WebSocket managers.

    Defines the contract for real-time data streaming from exchanges.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish WebSocket connection.

        Returns:
            bool: True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close WebSocket connection.
        """
        pass

    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to orderbook updates.

        Args:
            symbol: Trading symbol
            callback: Function to call with orderbook updates
        """
        pass

    @abstractmethod
    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to trade updates.

        Args:
            symbol: Trading symbol
            callback: Function to call with trade updates
        """
        pass

    @abstractmethod
    async def unsubscribe(self, stream_name: str) -> None:
        """
        Unsubscribe from a stream.

        Args:
            stream_name: Name of stream to unsubscribe from
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if WebSocket is connected.

        Returns:
            bool: True if connected
        """
        pass