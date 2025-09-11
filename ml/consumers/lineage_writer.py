"""
Lineage writer consumer which persists correlation/lineage to ObservabilityService.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from ml.consumers.protocols import Envelope
from ml.observability.service import ObservabilityService


@dataclass(slots=True)
class LineageWriter:
    """
    Write correlation/lineage entries from envelopes to ObservabilityService.

    Idempotent: deduplicates by event id to avoid duplicate writes on replay.

    """

    service: ObservabilityService
    _seen_ids: set[str] = field(default_factory=set)

    def handle(self, _topic: str, envelope: Envelope) -> None:
        eid = envelope["id"]
        if eid in self._seen_ids:
            return
        self._seen_ids.add(eid)
        # Map envelope to correlation row
        self.service.add_correlation(
            correlation_id=envelope["correlation_id"],
            event_id=envelope["id"],
            parent_event_id=envelope["parent_id"],
            instrument_id=envelope["instrument_id"],
            domain="ml",  # unified ML domain
            lineage_depth=0 if envelope["parent_id"] is None else 1,
            ts_event=int(envelope["ts_event"]),
            propagation_path=[envelope["stage"]],
        )


__all__ = ["LineageWriter"]
