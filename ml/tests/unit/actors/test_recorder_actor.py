from __future__ import annotations

import asyncio
from typing import Any

from ml.actors.recorder import RecorderActor


class _Recorder:
    def __init__(self) -> None:
        self.flush_calls: list[Any] = []

    def on_bar(self, _bar) -> None:  # pragma: no cover - hot path only
        return

    def on_quote(self, _tick) -> None:  # pragma: no cover - hot path only
        return

    def on_trade(self, _tick) -> None:  # pragma: no cover - hot path only
        return

    async def flush_all(self) -> None:
        self.flush_calls.append("flush")


class _Loop:
    def __init__(self, running: bool) -> None:
        self._running = running
        self.created: list[object] = []

    def is_running(self) -> bool:
        return self._running

    def create_task(self, coro) -> object:
        self.created.append(coro)
        return object()


def test_on_stop_schedules_flush_when_loop_running(monkeypatch) -> None:
    recorder = _Recorder()
    actor = RecorderActor(recorder=recorder)
    loop = _Loop(running=True)
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: loop)

    actor.on_stop()

    assert len(loop.created) == 1
    assert asyncio.iscoroutine(loop.created[0])
    loop.created[0].close()


def test_on_stop_noop_when_loop_not_running(monkeypatch) -> None:
    recorder = _Recorder()
    actor = RecorderActor(recorder=recorder)
    loop = _Loop(running=False)
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: loop)

    actor.on_stop()

    assert loop.created == []
