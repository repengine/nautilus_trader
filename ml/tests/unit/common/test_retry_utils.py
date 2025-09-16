#!/usr/bin/env python3
from __future__ import annotations

import pytest

from ml.common.retry_utils import retry_with_backoff


def test_retry_succeeds_after_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class TransientError(RuntimeError):
        pass

    # Fail twice, then succeed
    it = iter([TransientError("boom"), TransientError("boom"), 42])

    def call() -> int:
        calls.append(1)
        v = next(it)
        if isinstance(v, Exception):
            raise v
        return v

    # Deterministic sleep collector (don’t actually sleep)
    sleeps: list[float] = []

    def fake_sleep(d: float) -> None:
        sleeps.append(d)

    # Make jitter deterministic
    monkeypatch.setattr("random.uniform", lambda a, b: 0.0)

    result = retry_with_backoff(
        call,
        max_attempts=5,
        initial_delay=0.01,
        multiplier=2.0,
        max_delay=1.0,
        jitter=0.1,
        sleep_fn=fake_sleep,
        retry_on=(TransientError,),
    )

    assert result == 42
    # Two failures → two sleeps
    assert len(sleeps) == 2
    # Exponential: 0.01, 0.02 (jitter 0)
    assert sleeps[0] == pytest.approx(0.01)
    assert sleeps[1] == pytest.approx(0.02)
    assert len(calls) == 3


def test_retry_gives_up_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom(RuntimeError):
        pass

    def call() -> int:
        raise Boom("always")

    # Don’t sleep in tests
    sleeps: list[float] = []
    fake_sleep = lambda d: sleeps.append(d)  # noqa: E731

    with pytest.raises(Boom):
        retry_with_backoff(
            call,
            max_attempts=3,
            initial_delay=0.0,
            multiplier=2.0,
            max_delay=0.01,
            jitter=0.0,
            sleep_fn=fake_sleep,
            retry_on=(Boom,),
        )

    # Two retries attempted before raising (attempts=3)
    assert len(sleeps) == 2


def test_on_exception_callback_is_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    class Oops(RuntimeError):
        pass

    called: list[tuple[int, Exception]] = []

    def call() -> int:
        if len(called) < 1:
            raise Oops("first")
        return 7

    def noisy(attempt: int, exc: BaseException) -> None:
        called.append((attempt, exc))
        # Raise inside callback; should be swallowed
        raise RuntimeError("log failure")

    # No actual sleep
    result = retry_with_backoff(
        call,
        max_attempts=2,
        initial_delay=0,
        multiplier=1,
        sleep_fn=lambda _: None,
        retry_on=(Oops,),
        on_exception=noisy,
    )

    assert result == 7
    assert len(called) == 1

