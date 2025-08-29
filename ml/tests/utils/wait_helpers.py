"""
Utility functions for test synchronization.

This module provides event-based waiting patterns to replace flaky sleep() calls
in tests, following ml/docs/development/CODING_STANDARDS.md Section F.
"""

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any, Optional


class TestTimeout(Exception):
    """Raised when a test wait operation times out."""



def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    poll_interval: float = 0.1,
    error_message: str | None = None,
) -> None:
    """
    Wait for a condition to become true.

    Args:
        condition: A callable that returns True when the condition is met
        timeout: Maximum time to wait in seconds (default 5.0)
        poll_interval: Time between condition checks in seconds (default 0.1)
        error_message: Custom error message on timeout

    Raises:
        TestTimeout: If the condition is not met within the timeout period

    Example:
        >>> def is_ready():
        ...     return some_state.ready
        >>> wait_for_condition(is_ready, timeout=10)
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition():
            return
        time.sleep(poll_interval)

    msg = error_message or f"Condition not met within {timeout} seconds"
    raise TestTimeout(msg)


async def async_wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    poll_interval: float = 0.1,
    error_message: str | None = None,
) -> None:
    """
    Asynchronously wait for a condition to become true.

    Args:
        condition: A callable that returns True when the condition is met
        timeout: Maximum time to wait in seconds (default 5.0)
        poll_interval: Time between condition checks in seconds (default 0.1)
        error_message: Custom error message on timeout

    Raises:
        TestTimeout: If the condition is not met within the timeout period

    Example:
        >>> async def test_something():
        ...     await async_wait_for_condition(lambda: result.ready)
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition():
            return
        await asyncio.sleep(poll_interval)

    msg = error_message or f"Condition not met within {timeout} seconds"
    raise TestTimeout(msg)


async def async_wait_for_coroutine_condition(
    condition: Callable[[], Any],
    timeout: float = 5.0,
    poll_interval: float = 0.1,
    error_message: str | None = None,
) -> None:
    """
    Wait for an async condition (coroutine) to become true.

    Args:
        condition: An async callable that returns True when condition is met
        timeout: Maximum time to wait in seconds (default 5.0)
        poll_interval: Time between condition checks in seconds (default 0.1)
        error_message: Custom error message on timeout

    Raises:
        TestTimeout: If the condition is not met within the timeout period

    Example:
        >>> async def is_ready():
        ...     return await check_async_state()
        >>> await async_wait_for_coroutine_condition(is_ready)
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if await condition():
            return
        await asyncio.sleep(poll_interval)

    msg = error_message or f"Async condition not met within {timeout} seconds"
    raise TestTimeout(msg)


class EventWaiter:
    """
    Thread-safe event waiter for synchronous tests.

    Example:
        >>> waiter = EventWaiter()
        >>> def callback():
        ...     waiter.set()
        >>> register_callback(callback)
        >>> waiter.wait(timeout=5)
    """

    def __init__(self) -> None:
        """Initialize the event waiter."""
        self._event = threading.Event()

    def set(self) -> None:
        """Signal that the event has occurred."""
        self._event.set()

    def wait(self, timeout: float = 5.0) -> bool:
        """
        Wait for the event to be set.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if event was set, False if timeout occurred
        """
        return self._event.wait(timeout=timeout)

    def clear(self) -> None:
        """Reset the event for reuse."""
        self._event.clear()

    def is_set(self) -> bool:
        """Check if the event is currently set."""
        return self._event.is_set()


class AsyncEventWaiter:
    """
    Async event waiter for asynchronous tests.

    Example:
        >>> waiter = AsyncEventWaiter()
        >>> async def callback():
        ...     waiter.set()
        >>> await register_async_callback(callback)
        >>> await waiter.wait(timeout=5)
    """

    def __init__(self) -> None:
        """Initialize the async event waiter."""
        self._event = asyncio.Event()

    def set(self) -> None:
        """Signal that the event has occurred."""
        self._event.set()

    async def wait(self, timeout: float = 5.0) -> bool:
        """
        Wait for the event to be set.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if event was set, False if timeout occurred
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    def clear(self) -> None:
        """Reset the event for reuse."""
        self._event.clear()

    def is_set(self) -> bool:
        """Check if the event is currently set."""
        return self._event.is_set()


def wait_with_timeout(
    func: Callable[[], Any],
    timeout: float = 5.0,
    expected: Any = None,
    poll_interval: float = 0.1,
) -> Any:
    """
    Wait for a function to return an expected value.

    Args:
        func: Function to call repeatedly
        timeout: Maximum time to wait in seconds
        expected: Expected return value (if None, any truthy value)
        poll_interval: Time between function calls

    Returns:
        The function's return value

    Raises:
        TestTimeout: If expected value not returned within timeout

    Example:
        >>> result = wait_with_timeout(lambda: get_status(), expected="ready")
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = func()
        if expected is None:
            if result:
                return result
        elif result == expected:
            return result
        time.sleep(poll_interval)

    raise TestTimeout(
        f"Function did not return expected value within {timeout} seconds. "
        f"Expected: {expected}, Last result: {func()}"
    )


async def async_wait_with_timeout(
    func: Callable[[], Any],
    timeout: float = 5.0,
    expected: Any = None,
    poll_interval: float = 0.1,
) -> Any:
    """
    Asynchronously wait for a function to return an expected value.

    Args:
        func: Async function to call repeatedly
        timeout: Maximum time to wait in seconds
        expected: Expected return value (if None, any truthy value)
        poll_interval: Time between function calls

    Returns:
        The function's return value

    Raises:
        TestTimeout: If expected value not returned within timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if asyncio.iscoroutinefunction(func):
            result = await func()
        else:
            result = func()

        if expected is None:
            if result:
                return result
        elif result == expected:
            return result
        await asyncio.sleep(poll_interval)

    last_result = await func() if asyncio.iscoroutinefunction(func) else func()
    raise TestTimeout(
        f"Function did not return expected value within {timeout} seconds. "
        f"Expected: {expected}, Last result: {last_result}"
    )
