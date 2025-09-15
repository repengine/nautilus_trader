"""
Monotonic, watermark-gated aggregator with idempotent replay.

The aggregator buffers incoming envelopes per instrument and flushes them in timestamp
order when a watermark is advanced. It enforces idempotency via event id tracking and
optionally forwards flushed envelopes to a downstream publisher by prefixing topics.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Final

from ml.common.message_bus import MessagePublisherProtocol
from ml.common.message_topics import build_topic_for_stage
from ml.common.metrics import aggregator_buffer_size
from ml.common.metrics import aggregator_duplicates_total
from ml.common.metrics import aggregator_flushed_total
from ml.common.metrics import aggregator_watermark_lag_seconds
from ml.config.events import Stage
from ml.consumers.protocols import Envelope


TopicMapper = Callable[[str], str]


@dataclass(slots=True)
class AggregatingConsumer:
    """
    Aggregate and emit envelopes in timestamp order under watermark gating.

    Parameters
    ----------
    downstream : MessagePublisherProtocol | None
        Optional publisher to forward flushed messages to.
    topic_mapper : Callable[[str], str]
        Function mapping input topics to downstream topics. Defaults to
        using canonical builders with "aggregated." prefix.
    scheme : str
        Topic scheme to use for canonical building ("domain_op" or "stage_first").
    prefix : str
        Topic prefix for canonical building.

    """

    downstream: MessagePublisherProtocol | None = None
    topic_mapper: TopicMapper | None = None
    scheme: str = "domain_op"
    prefix: str = "aggregated.ml"

    _buffer: dict[str, list[Envelope]] = field(default_factory=dict)
    _last_emitted_ts: dict[str, int] = field(default_factory=dict)
    _processed_ids: set[str] = field(default_factory=set)

    _DEFAULT_PREFIX: Final[str] = "aggregated."

    def handle(self, topic: str, envelope: Envelope) -> None:
        """
        Buffer envelope for its instrument; ignore duplicates by id.
        """
        eid = envelope["id"]
        if eid in self._processed_ids:
            aggregator_duplicates_total.inc()
            return
        inst = envelope["instrument_id"]
        buf = self._buffer.setdefault(inst, [])
        buf.append(envelope)
        # Update buffer size gauge
        aggregator_buffer_size.labels(instrument=inst).set(len(buf))

    def advance_watermark(self, instrument_id: str, watermark_ns: int) -> list[Envelope]:
        """
        Advance instrument watermark and flush eligible envelopes in order.

        Returns the flushed envelopes in strictly non-decreasing timestamp order.
        Enforces idempotency (same id is never emitted twice).

        """
        buf = self._buffer.get(instrument_id, [])
        if not buf:
            return []
        buf.sort(key=lambda e: e["ts_event"])  # stable
        flushed: list[Envelope] = []
        last_ts = self._last_emitted_ts.get(instrument_id, -1)
        i = 0
        while i < len(buf):
            e = buf[i]
            if e["ts_event"] > watermark_ns:
                break
            if e["id"] not in self._processed_ids:
                # Enforce monotonic non-decreasing timestamps per instrument
                if e["ts_event"] < last_ts:
                    # Skip until watermark progresses; this should be rare
                    i += 1
                    continue
                flushed.append(e)
                self._processed_ids.add(e["id"])
                last_ts = e["ts_event"]
            i += 1

        # Remove flushed items from buffer
        if i > 0:
            del buf[:i]
        self._last_emitted_ts[instrument_id] = last_ts
        # Update metrics after flush
        aggregator_buffer_size.labels(instrument=instrument_id).set(len(buf))
        if flushed:
            aggregator_flushed_total.labels(instrument=instrument_id).inc(len(flushed))
            if last_ts >= 0:
                lag_sec = max(0.0, (float(watermark_ns) - float(last_ts)) / 1e9)
                aggregator_watermark_lag_seconds.labels(instrument=instrument_id).set(lag_sec)

        # Forward to downstream if configured
        if self.downstream is not None and flushed:
            for ev in flushed:
                # Use canonical topic builders or custom mapper
                if self.topic_mapper:
                    out_topic = self.topic_mapper(ev.get("stage", "events"))
                else:
                    # Use canonical builders with proper Stage enum handling
                    stage_str = ev.get("stage", "")
                    instrument_id = ev["instrument_id"]

                    # Try to parse stage string to Stage enum
                    try:
                        stage = Stage(stage_str) if stage_str else None
                    except ValueError:
                        # Fallback for invalid stage strings
                        stage = None

                    if stage:
                        out_topic = build_topic_for_stage(
                            stage=stage,
                            instrument_id=instrument_id,
                            scheme=self.scheme,
                            prefix=self.prefix,
                        )
                    else:
                        # Fallback for events without valid stage
                        from ml.common.message_topics import build_topic
                        out_topic = build_topic("events", "updated", instrument_id)

                # Publish original payload wrapped with envelope metadata
                payload = {
                    "id": ev["id"],
                    "parent_id": ev["parent_id"],
                    "instrument_id": ev["instrument_id"],
                    "ts_event": ev["ts_event"],
                    "stage": ev["stage"],
                    "correlation_id": ev["correlation_id"],
                    "payload": ev["payload"],
                }
                try:
                    self.downstream.publish(out_topic, payload)
                except Exception:
                    # Forwarding is best-effort; aggregator state already updated
                    pass

        return flushed
