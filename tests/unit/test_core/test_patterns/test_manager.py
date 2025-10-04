"""
Tests for BaseManager pattern.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.core.patterns.manager import BaseManager, ManagerState, AsyncTaskManager


class MockManager(BaseManager):
    """Mock implementation for testing"""

    def __init__(self, name="TestManager", config=None):
        super().__init__(name, config)
        self.initialize_called = 0
        self.start_called = 0
        self.stop_called = 0
        self.should_fail_initialize = False
        self.should_fail_start = False
        self.should_fail_stop = False

    async def _do_initialize(self):
        self.initialize_called += 1
        if self.should_fail_initialize:
            raise Exception("Mock initialization failed")

    async def _do_start(self):
        self.start_called += 1
        if self.should_fail_start:
            raise Exception("Mock start failed")

    async def _do_stop(self):
        self.stop_called += 1
        if self.should_fail_stop:
            raise Exception("Mock stop failed")


class TestBaseManager:
    """Test cases for BaseManager"""

    def test_initial_state(self):
        """Test initial manager state"""
        manager = MockManager("TestManager", {"key": "value"})

        assert manager.name == "TestManager"
        assert manager.config == {"key": "value"}
        assert manager.state == ManagerState.CREATED
        assert not manager.is_initialized
        assert not manager.is_running
        assert manager.is_stopped is False  # Not explicitly stopped yet
        assert not manager.has_error
        assert manager.last_error is None
        assert manager.uptime is None

    @pytest.mark.asyncio
    async def test_successful_initialization(self):
        """Test successful initialization"""
        manager = MockManager()

        assert manager.state == ManagerState.CREATED
        assert not manager.is_initialized

        await manager.initialize()

        assert manager.state == ManagerState.INITIALIZED
        assert manager.is_initialized
        assert manager.initialize_called == 1
        assert manager.last_error is None

    @pytest.mark.asyncio
    async def test_initialization_idempotent(self):
        """Test that initialization is idempotent"""
        manager = MockManager()

        await manager.initialize()
        initialize_calls_before = manager.initialize_called

        await manager.initialize()

        # Should not initialize again
        assert manager.initialize_called == initialize_calls_before

    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test initialization failure"""
        manager = MockManager()
        manager.should_fail_initialize = True

        with pytest.raises(Exception, match="Mock initialization failed"):
            await manager.initialize()

        assert manager.state == ManagerState.ERROR
        assert manager.has_error
        assert manager.last_error is not None
        assert "Mock initialization failed" in str(manager.last_error)

    @pytest.mark.asyncio
    async def test_successful_start(self):
        """Test successful start"""
        manager = MockManager()

        await manager.start()

        assert manager.state == ManagerState.RUNNING
        assert manager.is_running
        assert manager.initialize_called == 1  # Auto-initialized
        assert manager.start_called == 1
        assert manager.uptime is not None

    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        """Test starting when already running"""
        manager = MockManager()

        await manager.start()
        start_calls_before = manager.start_called

        await manager.start()

        # Should not start again
        assert manager.start_called == start_calls_before

    @pytest.mark.asyncio
    async def test_start_failure(self):
        """Test start failure"""
        manager = MockManager()
        manager.should_fail_start = True

        with pytest.raises(Exception, match="Mock start failed"):
            await manager.start()

        assert manager.state == ManagerState.ERROR
        assert manager.has_error

    @pytest.mark.asyncio
    async def test_successful_stop(self):
        """Test successful stop"""
        manager = MockManager()

        await manager.start()
        assert manager.is_running

        await manager.stop()

        assert manager.state == ManagerState.STOPPED
        assert not manager.is_running
        assert manager.is_stopped
        assert manager.stop_called == 1

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """Test stopping when not running"""
        manager = MockManager()

        await manager.stop()

        # Should not fail, but should not call stop implementation
        assert not manager.is_running

    @pytest.mark.asyncio
    async def test_stop_failure(self):
        """Test stop failure"""
        manager = MockManager()
        manager.should_fail_stop = True

        await manager.start()

        with pytest.raises(Exception, match="Mock stop failed"):
            await manager.stop()

        assert manager.state == ManagerState.ERROR
        assert manager.has_error

    @pytest.mark.asyncio
    async def test_restart(self):
        """Test restart functionality"""
        manager = MockManager()

        await manager.start()
        start_calls_before = manager.start_called
        stop_calls_before = manager.stop_called

        await manager.restart()

        assert manager.is_running
        assert manager.start_called == start_calls_before + 1
        assert manager.stop_called == stop_calls_before + 1

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test reset functionality"""
        manager = MockManager()

        await manager.start()
        assert manager.is_running

        await manager.reset()

        assert manager.state == ManagerState.CREATED
        assert not manager.is_running
        assert manager.last_error is None

    def test_state_callbacks(self):
        """Test state change callbacks"""
        manager = MockManager()
        callback_calls = []

        def callback(mgr, old_state, new_state):
            callback_calls.append((old_state, new_state))

        manager.add_state_callback(ManagerState.RUNNING, callback)

        # Manually trigger state change
        manager._set_state(ManagerState.RUNNING)

        assert len(callback_calls) == 1
        assert callback_calls[0] == (ManagerState.CREATED, ManagerState.RUNNING)

    def test_get_status(self):
        """Test status information retrieval"""
        config = {"test": "value"}
        manager = MockManager("TestManager", config)

        status = manager.get_status()

        assert isinstance(status, dict)
        assert status['name'] == "TestManager"
        assert status['state'] == ManagerState.CREATED.value
        assert status['is_initialized'] is False
        assert status['is_running'] is False
        assert status['is_stopped'] is False
        assert status['has_error'] is False
        assert status['config'] == config

    @pytest.mark.asyncio
    async def test_status_after_start(self):
        """Test status after starting"""
        manager = MockManager()

        await manager.start()

        status = manager.get_status()

        assert status['state'] == ManagerState.RUNNING.value
        assert status['is_running'] is True
        assert status['uptime_seconds'] is not None

    def test_string_representations(self):
        """Test string representations"""
        manager = MockManager("TestManager")

        str_repr = str(manager)
        repr_repr = repr(manager)

        assert "TestManager" in str_repr
        assert "created" in str_repr.lower()
        assert "TestManager" in repr_repr


class TestAsyncTaskManager:
    """Test cases for AsyncTaskManager"""

    @pytest.mark.asyncio
    async def test_task_management(self):
        """Test async task management"""
        manager = AsyncTaskManager("TaskManager")

        # Mock coroutine
        async def mock_task():
            await asyncio.sleep(0.1)
            return "result"

        # Add task
        task = manager.add_task("test_task", mock_task())

        assert isinstance(task, asyncio.Task)
        assert len(manager._tasks) == 1

        # Wait for task completion
        result = await task
        assert result == "result"

        # Task should be auto-removed after completion
        await asyncio.sleep(0.01)  # Allow callback to execute
        assert len(manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_task_cancellation_on_stop(self):
        """Test that tasks are cancelled on stop"""
        manager = AsyncTaskManager("TaskManager")

        # Long-running task
        async def long_task():
            await asyncio.sleep(10)  # Long sleep
            return "should_not_complete"

        task = manager.add_task("long_task", long_task())

        # Stop manager (should cancel tasks)
        await manager._stop_all_tasks()

        assert task.cancelled()
        assert len(manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_task_replacement(self):
        """Test task replacement with same name"""
        manager = AsyncTaskManager("TaskManager")

        async def task1():
            await asyncio.sleep(0.1)

        async def task2():
            await asyncio.sleep(0.1)

        # Add first task
        first_task = manager.add_task("test_task", task1())

        # Add second task with same name (should replace)
        second_task = manager.add_task("test_task", task2())

        assert first_task is not second_task
        assert len(manager._tasks) == 1

    @pytest.mark.asyncio
    async def test_task_status(self):
        """Test task status retrieval"""
        manager = AsyncTaskManager("TaskManager")

        async def mock_task():
            await asyncio.sleep(0.1)

        task = manager.add_task("test_task", mock_task())

        status = manager.get_task_status()

        assert "test_task" in status
        assert "done" in status["test_task"]
        assert "cancelled" in status["test_task"]
        assert "exception" in status["test_task"]

        # Clean up
        await manager._stop_all_tasks()

    @pytest.mark.asyncio
    async def test_task_exception_handling(self):
        """Test task exception handling"""
        manager = AsyncTaskManager("TaskManager")

        async def failing_task():
            raise ValueError("Task failed")

        task = manager.add_task("failing_task", failing_task())

        # Wait for task to complete with exception
        try:
            await task
        except ValueError:
            pass

        # Allow callback to execute
        await asyncio.sleep(0.01)

        # Task should be removed even after exception
        assert len(manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_multiple_tasks(self):
        """Test managing multiple tasks"""
        manager = AsyncTaskManager("TaskManager")

        async def task(name, delay):
            await asyncio.sleep(delay)
            return f"result_{name}"

        # Add multiple tasks
        task1 = manager.add_task("task1", task("1", 0.05))
        task2 = manager.add_task("task2", task("2", 0.1))
        task3 = manager.add_task("task3", task("3", 0.15))

        assert len(manager._tasks) == 3

        # Wait for first task to complete
        result1 = await task1
        assert result1 == "result_1"

        # Stop all tasks
        await manager._stop_all_tasks()

        # All tasks should be cancelled or completed
        assert len(manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_do_stop_implementation(self):
        """Test that _do_stop calls _stop_all_tasks"""
        manager = AsyncTaskManager("TaskManager")

        async def mock_task():
            await asyncio.sleep(1)

        manager.add_task("test_task", mock_task())
        assert len(manager._tasks) == 1

        # Initialize and start to set up proper state
        await manager.initialize()
        await manager.start()

        # Stop should cancel all tasks
        await manager.stop()

        assert len(manager._tasks) == 0