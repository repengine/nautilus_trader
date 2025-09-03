from __future__ import annotations

import re

from hypothesis import given, strategies as st

from ml.common.message_topics import build_topic


@given(
    domain=st.sampled_from(["data", "features", "models", "strategies"]),
    operation=st.sampled_from(["created", "updated", "deprecated"]),
    instrument=st.text(min_size=3, max_size=30),
)
def test_build_topic_fuzz(domain: str, operation: str, instrument: str) -> None:
    topic = build_topic(domain, operation, instrument)
    # Pattern: ml.{domain}.{operation}.{instrument}
    assert topic.startswith(f"ml.{domain}.{operation}.")
    # No reserved characters in instrument segment
    assert not re.search(r"[/*#+$]", topic)
    # Domain and operation segments are lowercase
    parts = topic.split(".")
    assert parts[1] == domain and parts[2] == operation
