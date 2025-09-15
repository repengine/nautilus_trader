from __future__ import annotations

import math
import pytest

try:  # optional dependency
    from hypothesis import given
    from hypothesis import strategies as st
    from hypothesis.strategies import composite
except Exception:  # pragma: no cover - hypothesis optional
    pytest.skip("hypothesis not available", allow_module_level=True)

from typing import Literal, Tuple


def _pnl_naive(
    prices: list[float],
    signals: list[Literal["BUY", "SELL", "HOLD"]],
    strengths: list[float],
) -> float:
    """
    Compute a simple, directionally consistent PnL proxy.

    - BUY contributes +strength * price_delta
    - SELL contributes -abs(strength) * price_delta
    - HOLD contributes 0
    This is a test-local proxy used to validate directional consistency properties.

    """
    pnl = 0.0
    for i in range(len(prices) - 1):
        delta = prices[i + 1] - prices[i]
        sig = signals[i]
        s = strengths[i]
        if sig == "BUY":
            pnl += float(max(s, 0.0)) * delta
        elif sig == "SELL":
            pnl -= float(abs(min(s, 0.0))) * delta
        # HOLD has no effect
    return float(pnl)


@composite
def uptrend_case(
    draw: st.DrawFn,
    length_min: int = 2,
    length_max: int = 64,
) -> tuple[list[float], list[Literal["BUY", "HOLD"]], list[float]]:  # type: ignore[name-defined]
    n = draw(st.integers(min_value=length_min, max_value=length_max))
    base = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    step = draw(st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False))
    prices = [float(base + i * step) for i in range(n)]
    # BUY or HOLD per step (n-1 deltas)
    sigs: list[Literal["BUY", "HOLD"]] = []
    strengths: list[float] = []
    for _ in range(n - 1):
        sig = draw(st.sampled_from(["BUY", "HOLD"]))
        sigs.append(sig)
        if sig == "BUY":
            strengths.append(
                draw(
                    st.floats(min_value=0.0, max_value=1.0, allow_infinity=False, allow_nan=False),
                ),
            )
        else:
            strengths.append(0.0)
    return prices, sigs, strengths


@composite
def downtrend_case(
    draw: st.DrawFn,
    length_min: int = 2,
    length_max: int = 64,
) -> tuple[list[float], list[Literal["SELL", "HOLD"]], list[float]]:  # type: ignore[name-defined]
    n = draw(st.integers(min_value=length_min, max_value=length_max))
    base = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    step = draw(st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False))
    prices = [float(base - i * step) for i in range(n)]
    sigs: list[Literal["SELL", "HOLD"]] = []
    strengths: list[float] = []
    for _ in range(n - 1):
        sig = draw(st.sampled_from(["SELL", "HOLD"]))
        sigs.append(sig)
        if sig == "SELL":
            strengths.append(
                draw(
                    st.floats(min_value=-1.0, max_value=0.0, allow_infinity=False, allow_nan=False),
                ),
            )
        else:
            strengths.append(0.0)
    return prices, sigs, strengths


@given(case=uptrend_case())
def test_pnl_nonnegative_for_buy_only_on_uptrend(
    case: tuple[list[float], list[Literal["BUY", "HOLD"]], list[float]],
) -> None:
    prices, signals, strengths = case
    pnl = _pnl_naive(prices, signals, strengths)
    assert pnl >= -1e-12  # numerical tolerance


@given(case=downtrend_case())
def test_pnl_nonnegative_for_sell_only_on_downtrend(
    case: tuple[list[float], list[Literal["SELL", "HOLD"]], list[float]],
) -> None:
    prices, signals, strengths = case
    pnl = _pnl_naive(prices, signals, strengths)
    assert pnl >= -1e-12
