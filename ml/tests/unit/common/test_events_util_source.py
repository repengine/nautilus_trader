#!/usr/bin/env python3
from __future__ import annotations

import pytest

from ml.common.events_util import to_source_enum, to_source_str
from ml.config.events import Source


def test_to_source_enum_and_str_roundtrip() -> None:
    for s in ("live", "historical", "backfill"):
        enum_val = to_source_enum(s)
        assert isinstance(enum_val, Source)
        assert to_source_str(enum_val) == s


def test_to_source_str_invalid_raises() -> None:
    with pytest.raises(ValueError):
        to_source_str("invalid")  # type: ignore[arg-type]
