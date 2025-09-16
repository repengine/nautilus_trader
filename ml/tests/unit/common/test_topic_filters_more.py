#!/usr/bin/env python3
from __future__ import annotations

import pytest

from ml.common.topic_filters import match_topic


@pytest.mark.parametrize(
    "pattern,topic,expected",
    [
        ("events.ml.SIGNAL.#", "events.ml.SIGNAL", True),
        ("events.ml.SIGNAL.#", "events.ml.SIGNAL.EURUSD.SIM", True),
        ("ml.features.updated.*.*", "ml.features.updated.EURUSD.SIM", True),
        ("ml.features.updated.*", "ml.features.updated", False),
        ("#", "anything.really", True),
        ("a.*.c", "a.b.c", True),
        ("a.*.c", "a.b.d", False),
        ("a.#.c", "a.b.d.c", True),
        ("a.b", "a.b.c", False),
        ("", "", True),
        ("", "a", False),
    ],
)
def test_match_topic_cases(pattern: str, topic: str, expected: bool) -> None:
    assert match_topic(pattern, topic) is expected
