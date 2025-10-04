"""
Tests for BaseConnectionManager pattern.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.core.patterns.connection import BaseConnectionManager, ConnectionError


class MockConnectionManager(BaseConnectionManager):
    """Mock implementation for testing"""

    def __init__(self, name="Test", max_reconnect_attempts=3):
        super().__init__(name, max_reconnect_attempts)
        self.create_connection_called = 0
        self.close_connection_called = 0
        self.test_connection_called = 0
        self.should_fail_creation = False
        self.should_fail_test = False

    async def _create_connection(self):
        self.create_connection_called += 1
        if self.should_fail_creation:
            raise Exception("Mock connection creation failed")
        return "mock_connection"

    async def _close_connection(self, connection):
        self.close_connection_called += 1

    async def _test_connection(self, connection):
        self.test_connection_called += 1
        if self.should_fail_test:
            return False
        return True


class TestBaseConnectionManager:
    """Test cases for BaseConnectionManager"""

    @pytest.mark.asyncio
    async def test_successful_connection(self):
        """Test successful connection establishment"""
        manager = MockConnectionManager()

        assert not manager.is_connected
        assert manager.connection_time is None

        await manager.connect()

        assert manager.is_connected
        assert manager.connection_time is not None
        assert manager.create_connection_called == 1
        assert manager.test_connection_called == 1
        assert manager.reconnect_attempts == 0

    @pytest.mark.asyncio
    async def test_connection_already_connected(self):
        """Test connecting when already connected"""
        manager = MockConnectionManager()

        await manager.connect()
        create_calls_before = manager.create_connection_called

        # Try to connect again
        await manager.connect()

        # Should not create new connection
        assert manager.create_connection_called == create_calls_before

    @pytest.mark.asyncio
    async def test_successful_disconnection(self):
        """Test successful disconnection"""
        manager = MockConnectionManager()

        await manager.connect()
        assert manager.is_connected

        await manager.disconnect()

        assert not manager.is_connected
        assert manager.connection_time is None
        assert manager.close_connection_called == 1

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """Test disconnecting when not connected"""
        manager = MockConnectionManager()

        await manager.disconnect()

        # Should not fail, and close_connection should not be called since no connection exists
        assert not manager.is_connected
        assert manager.close_connection_called == 0

    @pytest.mark.asyncio
    async def test_connection_creation_failure(self):
        """Test connection creation failure with retry"""
        manager = MockConnectionManager(max_reconnect_attempts=1)
        manager.should_fail_creation = True

        # Connect will start the retry process in background
        await manager.connect()

        # Wait a bit for retry to complete
        await asyncio.sleep(0.1)

        assert not manager.is_connected
        assert manager.reconnect_attempts > 0

    @pytest.mark.asyncio
    async def test_connection_test_failure(self):
        """Test connection test failure"""
        manager = MockConnectionManager(max_reconnect_attempts=1)
        manager.should_fail_test = True

        # Connect will start the retry process in background
        await manager.connect()

        # Wait a bit for retry to complete
        await asyncio.sleep(0.1)

        assert not manager.is_connected
        assert manager.create_connection_called >= 1

    @pytest.mark.asyncio
    async def test_reconnect(self):
        """Test force reconnection"""
        manager = MockConnectionManager()

        await manager.connect()
        assert manager.is_connected

        await manager.reconnect()

        assert manager.is_connected
        assert manager.create_connection_called == 2  # One for connect, one for reconnect
        assert manager.close_connection_called == 1

    @pytest.mark.asyncio
    async def test_connection_info(self):
        """Test connection info retrieval"""
        manager = MockConnectionManager()

        info = manager.get_connection_info()

        assert info['name'] == 'Test'
        assert info['connected'] is False
        assert info['reconnect_attempts'] == 0
        assert info['max_reconnect_attempts'] == 3

        await manager.connect()

        info = manager.get_connection_info()
        assert info['connected'] is True
        assert info['connection_time'] is not None

    @pytest.mark.asyncio
    async def test_retry_logic_exponential_backoff(self):
        """Test exponential backoff in retry logic"""
        manager = MockConnectionManager(max_reconnect_attempts=1)
        manager.should_fail_creation = True

        # Connect starts the retry process in background
        await manager.connect()

        # Allow some time for retry attempts
        await asyncio.sleep(0.2)

        # Should have attempted reconnection
        assert manager.reconnect_attempts > 0

    def test_initial_state(self):
        """Test initial connection manager state"""
        manager = MockConnectionManager("TestManager", 5)

        assert manager.name == "TestManager"
        assert not manager.is_connected
        assert manager.connection_time is None
        assert manager.last_error is None
        assert manager.reconnect_attempts == 0
        assert manager._max_reconnect_attempts == 5

    @pytest.mark.asyncio
    async def test_last_error_tracking(self):
        """Test that last error is properly tracked"""
        manager = MockConnectionManager()
        manager.should_fail_creation = True

        try:
            await manager.connect()
        except ConnectionError:
            pass

        assert manager.last_error is not None
        assert "Mock connection creation failed" in str(manager.last_error)

    @pytest.mark.asyncio
    async def test_connection_properties(self):
        """Test connection properties"""
        manager = MockConnectionManager()

        # Test initial properties
        assert not manager.is_connected
        assert manager.connection_time is None
        assert manager.last_error is None
        assert manager.reconnect_attempts == 0

        # Test after connection
        await manager.connect()

        assert manager.is_connected
        assert manager.connection_time is not None
        assert manager.last_error is None  # Should be cleared on successful connection
        assert manager.reconnect_attempts == 0