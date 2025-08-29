#!/usr/bin/env python3
"""Test the wait_helpers module to ensure it works correctly."""

import asyncio
import time

from ml.tests.utils.wait_helpers import AsyncEventWaiter
from ml.tests.utils.wait_helpers import EventWaiter
from ml.tests.utils.wait_helpers import TestTimeout
from ml.tests.utils.wait_helpers import async_wait_for_condition
from ml.tests.utils.wait_helpers import wait_for_condition


def test_wait_for_condition():
    """Test synchronous wait_for_condition."""
    counter = {"value": 0}

    def increment():
        counter["value"] += 1

    # Start incrementing in background
    import threading
    def background_work():
        for _ in range(5):
            time.sleep(0.01)
            increment()

    thread = threading.Thread(target=background_work)
    thread.start()

    # Wait for counter to reach 3
    wait_for_condition(
        lambda: counter["value"] >= 3,
        timeout=1.0,
        poll_interval=0.01
    )

    assert counter["value"] >= 3
    thread.join()
    print("✓ wait_for_condition works")


def test_event_waiter():
    """Test EventWaiter."""
    waiter = EventWaiter()

    def set_event():
        time.sleep(0.05)
        waiter.set()

    import threading
    thread = threading.Thread(target=set_event)
    thread.start()

    result = waiter.wait(timeout=1.0)
    assert result is True
    thread.join()
    print("✓ EventWaiter works")


async def test_async_wait():
    """Test async wait_for_condition."""
    counter = {"value": 0}

    async def increment():
        await asyncio.sleep(0.01)
        counter["value"] += 1

    # Start incrementing
    task = asyncio.create_task(increment())

    # Wait for condition
    await async_wait_for_condition(
        lambda: counter["value"] >= 1,
        timeout=1.0,
        poll_interval=0.01
    )

    await task
    assert counter["value"] >= 1
    print("✓ async_wait_for_condition works")


if __name__ == "__main__":
    # Test synchronous functions
    test_wait_for_condition()
    test_event_waiter()

    # Test async functions
    asyncio.run(test_async_wait())

    print("\nAll wait_helpers tests passed! ✓")
