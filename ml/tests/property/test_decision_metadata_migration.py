"""
Property-based tests for decision metadata normalization.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.common import normalize_decision_metadata


@pytest.mark.property
@given(
    policy=st.text(min_size=1, max_size=32),
    horizon_minutes=st.integers(min_value=1, max_value=10_000),
)
@settings(max_examples=200, deadline=2000)
def test_normalize_decision_metadata_rejects_legacy_fields(
    policy: str,
    horizon_minutes: int,
) -> None:
    """
    Legacy payloads are rejected in strict mode.
    """
    legacy = {
        "decision_policy": policy,
        "horizon_minutes": horizon_minutes,
    }
    with pytest.raises(ValueError, match="decision_metadata"):
        normalize_decision_metadata(legacy)


@pytest.mark.property
@given(
    policy=st.text(min_size=1, max_size=32),
    label=st.text(min_size=1, max_size=64),
)
@settings(max_examples=200, deadline=2000)
def test_normalize_decision_metadata_preserves_direct_v1_payload(
    policy: str,
    label: str,
) -> None:
    """
    Direct v1 payloads are preserved.
    """
    payload = normalize_decision_metadata(
        {
            "version": "v1",
            "policy": policy,
            "label": label,
        }
    )

    assert payload["version"] == "v1"
    assert payload["policy"] == policy
    assert payload["label"] == label


@pytest.mark.property
def test_normalize_decision_metadata_rejects_nested_payload() -> None:
    with pytest.raises(ValueError, match="nested"):
        normalize_decision_metadata({"decision_metadata": {"version": "v1"}})
