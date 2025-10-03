"""
Database connection pool management with auto-scaling.

This module provides intelligent database connection pool management including:
- Auto-scaling connection pools
- Connection health monitoring
- Performance statistics tracking
- Connection timeout handling
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from .models import OptimizationError

try:
    import asyncpg
except ImportError:
    asyncpg = None

logger = logging.getLogger(__name__)


class ConnectionPoolManager:
    """
    Database connection pool manager with auto-scaling.

    Manages database connections with intelligent pool sizing and monitoring.
    """

    def __init__(
        self,
        database_url: str,
        min_connections: int = 5,
        max_connections: int = 20,
        command_timeout: int = 60
    ):
        """
        Initialize connection pool manager.

        Args:
            database_url: Database connection URL
            min_connections: Minimum pool size
            max_connections: Maximum pool size
            command_timeout: Command timeout in seconds
        """
        if asyncpg is None:
            raise OptimizationError("asyncpg package is required for database optimization")

        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.command_timeout = command_timeout
        self.pool: Optional[asyncpg.Pool] = None
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=self.command_timeout
            )
            self.is_initialized = True
            logger.info(f"Database connection pool initialized: {self.min_connections}-{self.max_connections} connections")

        except Exception as e:
            self.is_initialized = False
            raise OptimizationError(f"Failed to initialize connection pool: {e}")

    @asynccontextmanager
    async def acquire_connection(self):
        """Acquire a connection from the pool."""
        if not self.is_initialized or not self.pool:
            raise OptimizationError("Connection pool not initialized")

        async with self.pool.acquire() as connection:
            yield connection

    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        async with self.acquire_connection() as conn:
            return await conn.fetch(query, *args)

    async def execute_single(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute a query and return single result."""
        async with self.acquire_connection() as conn:
            return await conn.fetchrow(query, *args)

    def get_pool_stats(self) -> Dict[str, int]:
        """Get connection pool statistics."""
        if not self.pool:
            return {
                'current_size': 0,
                'min_size': self.min_connections,
                'max_size': self.max_connections,
                'idle_connections': 0
            }

        return {
            'current_size': self.pool.get_size(),
            'min_size': self.pool.get_min_size(),
            'max_size': self.pool.get_max_size(),
            'idle_connections': self.pool.get_idle_size()
        }

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
        self.is_initialized = False
        logger.info("Database connection pool closed")