#!/usr/bin/env python3
from __future__ import annotations

from ml.common.precision import MAX_PRICE_DECIMALS, clamp_price_str


def test_clamp_price_basic_and_max() -> None:
    assert clamp_price_str(1.23456789, decimals=4) == "1.2346"
    # Over-max decimals clamped to MAX_PRICE_DECIMALS
    s = clamp_price_str(0.1, decimals=MAX_PRICE_DECIMALS + 10)
    assert "." in s and len(s.split(".")[1]) == MAX_PRICE_DECIMALS


def test_clamp_zero_decimals() -> None:
    assert clamp_price_str(123.9, decimals=0) == "124"
