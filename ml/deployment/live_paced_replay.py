"""
Live-paced replay harness for message bus events.

Replays JSONL bus payloads at real-time pace (or accelerated) using the
configured MessagePublisherProtocol.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ml.common.events_util import validate_bus_payload
from ml.common.message_bus import MessagePublisherProtocol
from ml.config.replay import LiveReplayConfig


logger = logging.getLogger(__name__)


class ReplayClock(Protocol):
    """Clock protocol for pacing replays."""

    def monotonic(self) -> float:
        """Return monotonic time in seconds."""
        ...

    def sleep(self, seconds: float) -> None:
        """Sleep for the requested number of seconds."""
        ...


@dataclass(slots=True, frozen=True)
class ReplayEvent:
    """Replayable event with topic, payload, and timestamp."""

    topic: str
    payload: dict[str, object]
    timestamp_ns: int


@dataclass(slots=True, frozen=True)
class ReplayLoadResult:
    """Result of loading replay events from disk."""

    events: tuple[ReplayEvent, ...]
    skipped: int


@dataclass(slots=True, frozen=True)
class ReplaySummary:
    """Summary statistics for a replay run."""

    loaded: int
    published: int
    failed: int
    skipped: int
    duration_seconds: float


class RealTimeClock:
    """Real-time clock implementation."""

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _extract_timestamp_ns(payload: Mapping[str, object], *, primary_key: str) -> int | None:
    candidates = [
        primary_key,
        "ts_max",
        "ts_min",
        "created_at",
    ]
    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        raw = payload.get(key)
        if raw is None:
            continue
        value = _coerce_int(raw)
        if value is not None:
            return value

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        raw = metadata.get("ts_event")
        if raw is not None:
            return _coerce_int(raw)

    return None


def load_replay_events(
    path: Path,
    *,
    config: LiveReplayConfig,
) -> ReplayLoadResult:
    """
    Load replay events from a JSONL file.

    Parameters
    ----------
    path : Path
        JSONL file path containing {"topic": ..., "payload": ...} entries.
    config : LiveReplayConfig
        Replay configuration with pacing options.

    Returns
    -------
    ReplayLoadResult
        Loaded events and count of skipped records.
    """
    events: list[ReplayEvent] = []
    skipped = 0
    if not path.exists():
        raise FileNotFoundError(f"Replay input not found: {path}")

    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped += 1
                logger.debug("Skipping malformed JSONL line: %s", exc)
                continue

            if not isinstance(record, Mapping):
                skipped += 1
                logger.debug("Skipping non-object JSONL record")
                continue

            topic = record.get("topic")
            payload = record.get("payload")
            if not isinstance(topic, str) or not isinstance(payload, Mapping):
                skipped += 1
                logger.debug("Skipping record with missing topic or payload")
                continue

            is_valid, errors = validate_bus_payload(payload)
            if not is_valid:
                skipped += 1
                logger.debug("Skipping invalid payload: %s", errors)
                continue

            timestamp_ns = _extract_timestamp_ns(payload, primary_key=config.timestamp_field)
            if timestamp_ns is None:
                skipped += 1
                logger.debug("Skipping payload without timestamp fields")
                continue

            events.append(
                ReplayEvent(
                    topic=topic,
                    payload=dict(payload),
                    timestamp_ns=timestamp_ns,
                )
            )
            if config.max_events is not None and len(events) >= int(config.max_events):
                break

    return ReplayLoadResult(events=tuple(events), skipped=skipped)


def replay_events(
    events: tuple[ReplayEvent, ...],
    publisher: MessagePublisherProtocol,
    *,
    config: LiveReplayConfig,
    clock: ReplayClock | None = None,
    skipped: int = 0,
) -> ReplaySummary:
    """
    Replay events with live pacing.

    Parameters
    ----------
    events : tuple[ReplayEvent, ...]
        Events to replay in the order provided.
    publisher : MessagePublisherProtocol
        Publisher used for emitting events.
    config : LiveReplayConfig
        Replay configuration.
    clock : ReplayClock | None, optional
        Clock implementation for timing (defaults to real time).
    skipped : int, default 0
        Number of skipped records reported during loading.

    Returns
    -------
    ReplaySummary
        Summary statistics for the run.
    """
    if not events:
        return ReplaySummary(
            loaded=0,
            published=0,
            failed=0,
            skipped=skipped,
            duration_seconds=0.0,
        )

    replay_clock = clock or RealTimeClock()
    start_event_ns = events[0].timestamp_ns
    start_wall = replay_clock.monotonic()
    published = 0
    failed = 0

    for event in events:
        delta_ns = event.timestamp_ns - start_event_ns
        offset_seconds = max(0.0, float(delta_ns) / 1_000_000_000.0)
        target_time = start_wall + offset_seconds / float(config.speed_multiplier)
        now = replay_clock.monotonic()
        sleep_seconds = target_time - now
        if sleep_seconds > 0:
            replay_clock.sleep(sleep_seconds)

        try:
            ok = publisher.publish(event.topic, event.payload)
        except Exception as exc:
            failed += 1
            logger.error(
                "Replay publish failed for %s",
                event.topic,
                exc_info=True,
            )
            logger.debug("Replay publish error detail: %s", exc)
        else:
            if ok:
                published += 1
            else:
                failed += 1

    duration_seconds = max(0.0, replay_clock.monotonic() - start_wall)
    return ReplaySummary(
        loaded=len(events),
        published=published,
        failed=failed,
        skipped=skipped,
        duration_seconds=duration_seconds,
    )


def run_live_paced_replay(
    path: Path,
    publisher: MessagePublisherProtocol,
    *,
    config: LiveReplayConfig,
    clock: ReplayClock | None = None,
) -> ReplaySummary:
    """
    Load events from disk and replay them at live pace.
    """
    load_result = load_replay_events(path, config=config)
    return replay_events(
        load_result.events,
        publisher,
        config=config,
        clock=clock,
        skipped=load_result.skipped,
    )


__all__ = [
    "LiveReplayConfig",
    "RealTimeClock",
    "ReplayClock",
    "ReplayEvent",
    "ReplayLoadResult",
    "ReplaySummary",
    "load_replay_events",
    "replay_events",
    "run_live_paced_replay",
]
