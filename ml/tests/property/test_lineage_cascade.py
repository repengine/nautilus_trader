from __future__ import annotations

from typing import Any, cast

from hypothesis import given, strategies as st

from ml.common.cascade import emit_cascade, EventDict


def _event_strategy() -> st.SearchStrategy[dict[str, Any]]:
    return st.fixed_dictionaries(
        {
            "domain": st.text(min_size=1),
            "event_type": st.text(min_size=0),
            "correlation_id": st.text(min_size=1),
            "instrument_id": st.text(min_size=1),
            "ts_event": st.integers(min_value=0, max_value=2**62),
            "event_id": st.text(min_size=1),
            "payload": st.dictionaries(st.text(min_size=1), st.integers() | st.text()),
        },
    )


@given(
    src=_event_strategy(),
    target=st.text(min_size=1),
    delay=st.integers(min_value=0, max_value=10_000),
)
def test_emit_cascade_preserves_correlation_and_monotonic_ts(
    src: dict[str, Any],
    target: str,
    delay: int,
) -> None:
    """
    emit_cascade preserves correlation_id and yields non-decreasing timestamps for non-
    negative delay.
    """
    out = emit_cascade(cast(EventDict, src), target_domain=target, delay_ns=delay)
    assert out["correlation_id"] == src["correlation_id"]
    assert out["domain"] == target
    assert out["ts_event"] >= src["ts_event"]


@given(src=_event_strategy(), target=st.text(min_size=1))
def test_emit_cascade_payload_is_copied(src: dict[str, Any], target: str) -> None:
    """
    Returned payload is a copy (mutations do not affect source).
    """
    out = emit_cascade(cast(EventDict, src), target_domain=target, delay_ns=0)
    if "payload" in src:
        # Mutate returned payload
        payload_out = out["payload"]
        payload_out["__mutated__"] = True
        payload_src = cast(dict[str, Any], src["payload"])
        assert "__mutated__" not in payload_src
