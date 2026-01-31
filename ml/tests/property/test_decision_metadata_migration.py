"""
Property-based tests for decision metadata normalization.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.common.decision_metadata import normalize_decision_metadata


@pytest.mark.property
@given(
    policy=st.text(min_size=1, max_size=32),
    horizon_minutes=st.integers(min_value=1, max_value=10_000),
)
@settings(max_examples=200, deadline=2000)
def test_normalize_decision_metadata_from_legacy_fields(
    policy: str,
    horizon_minutes: int,
) -> None:
    """
    Legacy payloads map to v1 schema with policy + horizon preserved.
    """
    legacy = {
        "decision_policy": policy,
        "horizon_minutes": horizon_minutes,
    }
    payload = normalize_decision_metadata(legacy)

    assert payload["version"] == "v1"
    assert payload["policy"] == policy
    assert payload["horizon"]["value"] == horizon_minutes
    assert payload["horizon"]["unit"] == "minutes"


@pytest.mark.property
@given(
    policy=st.text(min_size=1, max_size=32),
    label=st.text(min_size=1, max_size=64),
)
@settings(max_examples=200, deadline=2000)
def test_normalize_decision_metadata_preserves_v1_payload(
    policy: str,
    label: str,
) -> None:
    """
    v1 payloads are preserved when nested under decision_metadata.
    """
    payload = normalize_decision_metadata(
        {
            "decision_metadata": {
                "version": "v1",
                "policy": policy,
                "label": label,
            }
        }
    )

    assert payload["version"] == "v1"
    assert payload["policy"] == policy
    assert payload["label"] == label
