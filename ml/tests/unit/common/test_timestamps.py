#!/usr/bin/env python3
from __future__ import annotations

import logging

import pytest

from ml.common.timestamps import normalize_timestamp_ns, sanitize_timestamp_ns


def test_normalize_timestamp_ns_variants() -> None:
    # seconds
    v, changed = normalize_timestamp_ns(1_600_000_000)
    assert changed and v == 1_600_000_000 * 1_000_000_000
    # milliseconds
    v, changed = normalize_timestamp_ns(1_600_000_000_000)
    assert changed and v == 1_600_000_000_000 * 1_000_000
    # microseconds
    v, changed = normalize_timestamp_ns(1_600_000_000_000_000)
    assert changed and v == 1_600_000_000_000_000 * 1_000
    # nanoseconds (no change)
    v, changed = normalize_timestamp_ns(1_600_000_000_000_000_000)
    assert not changed and v == 1_600_000_000_000_000_000


def test_sanitize_modes_warn_and_reject(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # warn mode with logger logs a warning
    logger = logging.getLogger("ml.tests.timestamps")
    with caplog.at_level(logging.WARNING):
        out = sanitize_timestamp_ns(1_600_000_000, mode="warn", logger=logger, context="unit")
    assert out != 1_600_000_000
    assert any("Normalized timestamp" in r.message for r in caplog.records)

    # normalize mode silently normalizes
    out2 = sanitize_timestamp_ns(1_600_000_000, mode="normalize", logger=logger)
    assert out2 != 1_600_000_000

    # reject mode raises
    with pytest.raises(ValueError):
        sanitize_timestamp_ns(1_600_000_000, mode="reject", logger=logger, context="ctx")


def test_sanitize_preserves_store_context_small_values() -> None:
    preserved = sanitize_timestamp_ns(1_000, context="ModelStore.write_prediction")
    assert preserved == 1_000
    preserved_strategy = sanitize_timestamp_ns(500, context="StrategyStore.write_signal")
    assert preserved_strategy == 500
