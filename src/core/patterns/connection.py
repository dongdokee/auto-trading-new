"""
Base connection manager pattern for standardized connection handling.

Provides abstract base class for connection management with:
- Automatic reconnection logic
- Connection state tracking
- Error handling and recovery
- Configurable retry behavior
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from datetime import datetime, timedelta


class ConnectionError(Exception):
    """Exception raised when connection operations fail"""
    pass


class BaseConnectionManager(ABC):
    """
    Abstract base class for connection management.

    Provides standardized connection lifecycle management with automatic
    reconnection, state tracking, and error recovery.
    """

    def __init__(self, name: str = "Connection", max_reconnect_attempts: int = 5):
        """
        Initialize connection manager.

        Args:
            name: Name for logging and identification
            max_reconnect_attempts: Maximum number of reconnection attempts
        """
        self.name = name
        self._connected = False
        self._connection: Optional[Any] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = max_reconnect_attempts
        self._connection_time: Optional[datetime] = None
        self._last_error: Optional[Exception] = None
        self._reconnect_delay_base = 2  # Base delay in seconds
        self._reconnect_task: Optional[asyncio.Task] = None

        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def is_connected(self) -> bool:
        """Check if currently connected"""
        return self._connected

    @property
    def connection_time(self) -> Optional[datetime]:
        """Get connection establishment time"""
        return self._connection_time

    @property
    def last_error(self) -> Optional[Exception]:
        """Get last connection error"""
        return self._last_error

    @property
    def reconnect_attempts(self) -> int:
        """Get current reconnection attempts count"""
        return self._reconnect_attempts

    @abstractmethod
    async def _create_connection(self) -> Any:
        """
        Create the actual connection object.

        Must be implemented by subclasses to create the specific
        connection type (HTTP session, websocket, database, etc.).

        Returns:
            The connection object

        Raises:
            Exception: If connection creation fails
        """
        pass

    @abstractmethod
    async def _close_connection(self, connection: Any) -> None:
        """
        Close the connection object.

        Must be implemented by subclasses to properly clean up
        the specific connection type.

        Args:
            connection: The connection object to close
        """
        pass

    async def _test_connection(self, connection: Any) -> bool:
        """
        Test if connection is valid and working.

        Default implementation returns True. Override in subclasses
        to implement specific connection testing logic.

        Args:
            connection: The connection object to test

        Returns:
            True if connection is working, False otherwise
        """
        return True

    async def connect(self) -> None:
        """
        Establish connection with automatic retry logic.

        Raises:
            ConnectionError: If connection fails after all retry attempts
        """
        if self._connected:
            self.logger.debug(f"{self.name}: Already connected")
            return

        self.logger.info(f"{self.name}: Connecting...")

        try:
            # Create connection
            self._connection = await self._create_connection()

            # Test connection if it was created
            if self._connection and await self._test_connection(self._connection):
                self._connected = True
                self._connection_time = datetime.now()
                self._reconnect_attempts = 0
                self._last_error = None
                self.logger.info(f"{self.name}: Connected successfully")
            else:
                raise ConnectionError("Connection test failed")

        except Exception as e:
            self._last_error = e
            self.logger.error(f"{self.name}: Connection failed: {e}")
            await self._handle_connection_error(e)

    async def disconnect(self) -> None:
        """
        Disconnect and clean up resources.
        """
        # Cancel any pending reconnect task
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        self.logger.info(f"{self.name}: Disconnecting...")

        try:
            if self._connection:
                await self._close_connection(self._connection)
        except Exception as e:
            self.logger.warning(f"{self.name}: Error during disconnect: {e}")
        finally:
            self._connected = False
            self._connection = None
            self._connection_time = None

        self.logger.info(f"{self.name}: Disconnected")

    async def reconnect(self) -> None:
        """
        Force reconnection by disconnecting and connecting again.
        """
        self.logger.info(f"{self.name}: Forcing reconnection...")
        await self.disconnect()
        await self.connect()

    async def _handle_connection_error(self, error: Exception) -> None:
        """
        Handle connection errors with automatic retry logic.

        Args:
            error: The error that occurred

        Raises:
            ConnectionError: If max reconnection attempts exceeded
        """
        self._reconnect_attempts += 1

        if self._reconnect_attempts <= self._max_reconnect_attempts:
            # Calculate exponential backoff delay
            delay = self._reconnect_delay_base ** self._reconnect_attempts

            self.logger.warning(
                f"{self.name}: Connection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} "
                f"failed, retrying in {delay}s: {error}"
            )

            # Schedule reconnection attempt
            self._reconnect_task = asyncio.create_task(self._retry_connection(delay))
        else:
            error_msg = (
                f"{self.name}: Max reconnection attempts ({self._max_reconnect_attempts}) "
                f"exceeded. Last error: {error}"
            )
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)

    async def _retry_connection(self, delay: float) -> None:
        """
        Retry connection after delay.

        Args:
            delay: Delay in seconds before retry
        """
        try:
            await asyncio.sleep(delay)
            await self.connect()
        except asyncio.CancelledError:
            self.logger.debug(f"{self.name}: Reconnection cancelled")
            raise
        except Exception as e:
            self.logger.error(f"{self.name}: Reconnection failed: {e}")
            # Error will be handled by connect() method

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection status information.

        Returns:
            Dictionary with connection status details
        """
        return {
            'name': self.name,
            'connected': self._connected,
            'connection_time': self._connection_time,
            'reconnect_attempts': self._reconnect_attempts,
            'max_reconnect_attempts': self._max_reconnect_attempts,
            'last_error': str(self._last_error) if self._last_error else None
        }