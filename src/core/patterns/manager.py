"""
Base manager pattern for standardized manager lifecycle management.

Provides abstract base class for manager objects with:
- Standardized initialization/start/stop lifecycle
- State tracking and error handling
- Template method pattern for customization
- Logging integration
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum

from .logging import LoggerFactory


class ManagerState(Enum):
    """Manager lifecycle states"""
    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BaseManager(ABC):
    """
    Abstract base class for manager objects.

    Provides standardized lifecycle management with template method pattern
    for initialization, start, and stop operations.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize manager.

        Args:
            name: Manager name for identification and logging
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._state = ManagerState.CREATED
        self._created_at = datetime.now()
        self._started_at: Optional[datetime] = None
        self._stopped_at: Optional[datetime] = None
        self._last_error: Optional[Exception] = None

        # Initialize logger
        self.logger = LoggerFactory.get_logger(f"manager.{name}")

        # State change callbacks
        self._state_callbacks: Dict[ManagerState, list] = {}

    @property
    def state(self) -> ManagerState:
        """Get current manager state"""
        return self._state

    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized"""
        return self._state in [
            ManagerState.INITIALIZED,
            ManagerState.STARTING,
            ManagerState.RUNNING,
            ManagerState.STOPPING
        ]

    @property
    def is_running(self) -> bool:
        """Check if manager is running"""
        return self._state == ManagerState.RUNNING

    @property
    def is_stopped(self) -> bool:
        """Check if manager is stopped"""
        return self._state == ManagerState.STOPPED

    @property
    def has_error(self) -> bool:
        """Check if manager is in error state"""
        return self._state == ManagerState.ERROR

    @property
    def last_error(self) -> Optional[Exception]:
        """Get last error that occurred"""
        return self._last_error

    @property
    def uptime(self) -> Optional[float]:
        """Get uptime in seconds if running"""
        if self._started_at and self._state == ManagerState.RUNNING:
            return (datetime.now() - self._started_at).total_seconds()
        return None

    def add_state_callback(self, state: ManagerState, callback):
        """
        Add callback for state changes.

        Args:
            state: State to monitor
            callback: Function to call when state is reached
        """
        if state not in self._state_callbacks:
            self._state_callbacks[state] = []
        self._state_callbacks[state].append(callback)

    def _set_state(self, new_state: ManagerState, error: Optional[Exception] = None) -> None:
        """
        Update manager state and trigger callbacks.

        Args:
            new_state: New state to set
            error: Error that caused state change (if any)
        """
        old_state = self._state
        self._state = new_state

        if error:
            self._last_error = error

        # Update timestamps
        if new_state == ManagerState.RUNNING:
            self._started_at = datetime.now()
        elif new_state == ManagerState.STOPPED:
            self._stopped_at = datetime.now()

        self.logger.info(f"{self.name}: State changed from {old_state.value} to {new_state.value}")

        # Trigger callbacks for new state
        if new_state in self._state_callbacks:
            for callback in self._state_callbacks[new_state]:
                try:
                    callback(self, old_state, new_state)
                except Exception as e:
                    self.logger.error(f"{self.name}: State callback error: {e}")

    async def initialize(self) -> None:
        """
        Initialize manager with template method pattern.

        Raises:
            Exception: If initialization fails
        """
        if self.is_initialized:
            self.logger.debug(f"{self.name}: Already initialized")
            return

        self._set_state(ManagerState.INITIALIZING)

        try:
            self.logger.info(f"{self.name}: Initializing...")

            await self._before_initialize()
            await self._do_initialize()
            await self._after_initialize()

            self._set_state(ManagerState.INITIALIZED)
            self.logger.info(f"{self.name}: Initialized successfully")

        except Exception as e:
            self.logger.error(f"{self.name}: Initialization failed: {e}")
            self._set_state(ManagerState.ERROR, e)
            raise

    async def start(self) -> None:
        """
        Start manager with template method pattern.

        Raises:
            Exception: If start fails
        """
        if not self.is_initialized:
            await self.initialize()

        if self.is_running:
            self.logger.debug(f"{self.name}: Already running")
            return

        self._set_state(ManagerState.STARTING)

        try:
            self.logger.info(f"{self.name}: Starting...")

            await self._before_start()
            await self._do_start()
            await self._after_start()

            self._set_state(ManagerState.RUNNING)
            self.logger.info(f"{self.name}: Started successfully")

        except Exception as e:
            self.logger.error(f"{self.name}: Start failed: {e}")
            self._set_state(ManagerState.ERROR, e)
            raise

    async def stop(self) -> None:
        """
        Stop manager with template method pattern.

        Raises:
            Exception: If stop fails
        """
        if not self.is_running:
            self.logger.debug(f"{self.name}: Not running")
            return

        self._set_state(ManagerState.STOPPING)

        try:
            self.logger.info(f"{self.name}: Stopping...")

            await self._before_stop()
            await self._do_stop()
            await self._after_stop()

            self._set_state(ManagerState.STOPPED)
            self.logger.info(f"{self.name}: Stopped successfully")

        except Exception as e:
            self.logger.error(f"{self.name}: Stop failed: {e}")
            self._set_state(ManagerState.ERROR, e)
            raise

    async def restart(self) -> None:
        """
        Restart manager by stopping and starting again.

        Raises:
            Exception: If restart fails
        """
        self.logger.info(f"{self.name}: Restarting...")
        await self.stop()
        await self.start()

    async def reset(self) -> None:
        """
        Reset manager to initial state.

        Stops if running, then resets to created state.
        """
        if self.is_running:
            await self.stop()

        self._set_state(ManagerState.CREATED)
        self._started_at = None
        self._stopped_at = None
        self._last_error = None
        self.logger.info(f"{self.name}: Reset to initial state")

    @abstractmethod
    async def _do_initialize(self) -> None:
        """
        Perform actual initialization logic.

        Must be implemented by subclasses to define specific
        initialization behavior.

        Raises:
            Exception: If initialization fails
        """
        pass

    @abstractmethod
    async def _do_start(self) -> None:
        """
        Perform actual start logic.

        Must be implemented by subclasses to define specific
        start behavior.

        Raises:
            Exception: If start fails
        """
        pass

    @abstractmethod
    async def _do_stop(self) -> None:
        """
        Perform actual stop logic.

        Must be implemented by subclasses to define specific
        stop behavior.

        Raises:
            Exception: If stop fails
        """
        pass

    # Hook methods (optional override)
    async def _before_initialize(self) -> None:
        """Hook called before initialization. Override if needed."""
        pass

    async def _after_initialize(self) -> None:
        """Hook called after initialization. Override if needed."""
        pass

    async def _before_start(self) -> None:
        """Hook called before start. Override if needed."""
        pass

    async def _after_start(self) -> None:
        """Hook called after start. Override if needed."""
        pass

    async def _before_stop(self) -> None:
        """Hook called before stop. Override if needed."""
        pass

    async def _after_stop(self) -> None:
        """Hook called after stop. Override if needed."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive manager status.

        Returns:
            Dictionary with manager status information
        """
        return {
            'name': self.name,
            'state': self._state.value,
            'created_at': self._created_at.isoformat(),
            'started_at': self._started_at.isoformat() if self._started_at else None,
            'stopped_at': self._stopped_at.isoformat() if self._stopped_at else None,
            'uptime_seconds': self.uptime,
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'is_stopped': self.is_stopped,
            'has_error': self.has_error,
            'last_error': str(self._last_error) if self._last_error else None,
            'config': self.config
        }

    def __str__(self) -> str:
        """String representation of manager"""
        return f"{self.__class__.__name__}(name='{self.name}', state='{self._state.value}')"

    def __repr__(self) -> str:
        """Detailed representation of manager"""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"state='{self._state.value}', uptime={self.uptime})")


class AsyncTaskManager(BaseManager):
    """
    Extended manager for managing async tasks.

    Provides task lifecycle management with proper cleanup.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._tasks: Dict[str, asyncio.Task] = {}

    async def _do_initialize(self) -> None:
        """Initialize task manager"""
        self.logger.info("Initializing async task manager")

    async def _do_start(self) -> None:
        """Start task manager"""
        self.logger.info("Starting async task manager")

    async def _do_stop(self) -> None:
        """Stop all running tasks"""
        await self._stop_all_tasks()

    async def _stop_all_tasks(self) -> None:
        """Cancel and clean up all running tasks"""
        if not self._tasks:
            return

        self.logger.info(f"{self.name}: Stopping {len(self._tasks)} tasks...")

        # Cancel all tasks
        for task_name, task in self._tasks.items():
            if not task.done():
                self.logger.debug(f"{self.name}: Cancelling task {task_name}")
                task.cancel()

        # Wait for tasks to complete or timeout
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks.values(), return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"{self.name}: Some tasks did not complete within timeout")

        # Clear task references
        self._tasks.clear()

    def add_task(self, name: str, coro) -> asyncio.Task:
        """
        Add a managed task.

        Args:
            name: Task name for tracking
            coro: Coroutine to run as task

        Returns:
            Created task
        """
        if name in self._tasks:
            self.logger.warning(f"{self.name}: Task {name} already exists, replacing")

        task = asyncio.create_task(coro)
        self._tasks[name] = task

        # Add done callback to clean up completed tasks
        task.add_done_callback(lambda t: self._task_done_callback(name, t))

        self.logger.debug(f"{self.name}: Added task {name}")
        return task

    def _task_done_callback(self, name: str, task: asyncio.Task) -> None:
        """Clean up completed task"""
        if name in self._tasks:
            del self._tasks[name]

        if task.cancelled():
            self.logger.debug(f"{self.name}: Task {name} was cancelled")
        elif task.exception():
            self.logger.error(f"{self.name}: Task {name} failed: {task.exception()}")
        else:
            self.logger.debug(f"{self.name}: Task {name} completed successfully")

    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all managed tasks"""
        task_info = {}
        for name, task in self._tasks.items():
            exception_str = None
            try:
                if task.done() and not task.cancelled():
                    exception = task.exception()
                    if exception:
                        exception_str = str(exception)
            except asyncio.InvalidStateError:
                # Task not done yet
                pass

            task_info[name] = {
                'done': task.done(),
                'cancelled': task.cancelled(),
                'exception': exception_str
            }
        return task_info