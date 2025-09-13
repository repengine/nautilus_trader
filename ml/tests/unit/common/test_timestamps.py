from __future__ import annotations

import logging

from ml.common.timestamps import normalize_timestamp_ns, sanitize_timestamp_ns


def test_normalize_timestamp_units() -> None:
    # seconds
    ns, changed = normalize_timestamp_ns(123)
    assert changed and ns == 123 * 1_000_000_000
    # milliseconds (~1e11)
    ns, changed = normalize_timestamp_ns(123_000_000_000)
    assert changed and ns == 123_000_000_000 * 1_000_000
    # microseconds (~1e14)
    ns, changed = normalize_timestamp_ns(123_000_000_000_000)
    assert changed and ns == 123_000_000_000_000 * 1_000
    # nanoseconds (~1e17)
    ns, changed = normalize_timestamp_ns(123_000_000_000_000_000)
    assert not changed and ns == 123_000_000_000_000_000


def test_sanitize_reject_and_warn_modes(caplog) -> None:
    # reject mode raises when normalization needed
    try:
        sanitize_timestamp_ns(1, mode="reject")
        assert False, "expected ValueError"
    except ValueError:
        pass

    # warn mode logs and normalizes
    logger = logging.getLogger("mltest")
    with caplog.at_level(logging.WARNING):
        out = sanitize_timestamp_ns(2, mode="warn", logger=logger, context="unit")
        assert out == 2_000_000_000
        assert any("Normalized timestamp" in rec.message for rec in caplog.records)
