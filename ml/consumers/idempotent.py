"""
Idempotent consumer template with correlation and watermark gating.

This module provides a simple in-memory implementation of an idempotent
consumer that:

- Drops duplicates by correlation_id
- Applies watermark gating to ensure non-decreasing processing per key
  (dataset_id, instrument_id, source)

It is intended as a template/example for building robust consumers.

"""

from __future__ import annotations

# ruff: noqa: I001

from typing import Any, Final
from dataclasses import dataclass, field
from collections.abc import Mapping, MutableMapping


ConsumerKey = tuple[str, str, str]


@dataclass
class IdempotentConsumer:
    """
    In-memory idempotent consumer template.

    Attributes
    ----------
    seen : set[str]
        Set of processed correlation IDs.
    watermarks : dict[ConsumerKey, int]
        High-watermark per (dataset_id, instrument_id, source) key.

    """

    seen: set[str] = field(default_factory=set)
    watermarks: MutableMapping[ConsumerKey, int] = field(default_factory=dict)

    def should_process(self, payload: Mapping[str, Any]) -> bool:
        """
        Return True if the event should be processed under idempotency/watermarks.

        Expects payload to contain keys: dataset_id, instrument_id, source,
        ts_max, metadata.correlation_id.

        """
        try:
            metadata = payload.get("metadata", {})
            correlation_id = str(metadata.get("correlation_id", ""))
            if not correlation_id or correlation_id in self.seen:
                return False

            dataset_id = str(payload.get("dataset_id", ""))
            instrument_id = str(payload.get("instrument_id", ""))
            source = str(payload.get("source", ""))
            ts_max = int(payload.get("ts_max", 0))

            key: ConsumerKey = (dataset_id, instrument_id, source)
            last = int(self.watermarks.get(key, -1))
            if ts_max < last:
                return False
            return True
        except Exception:
            return False

    def process(self, payload: Mapping[str, Any]) -> bool:
        """
        Apply idempotency + watermark gating and update state if accepted.
        """
        if not self.should_process(payload):
            return False

        metadata = payload.get("metadata", {})
        correlation_id = str(metadata.get("correlation_id", ""))
        dataset_id = str(payload.get("dataset_id", ""))
        instrument_id = str(payload.get("instrument_id", ""))
        source = str(payload.get("source", ""))
        ts_max = int(payload.get("ts_max", 0))

        key: ConsumerKey = (dataset_id, instrument_id, source)
        self.seen.add(correlation_id)
        self.watermarks[key] = ts_max
        return True


__all__: Final[list[str]] = ["ConsumerKey", "IdempotentConsumer"]
