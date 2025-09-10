from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from ml.common.topic_filters import match_topic


@given(
    topic=st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=5).map(
        lambda parts: ".".join(parts),
    ),
)
def test_hash_matches_any(topic: str) -> None:
    assert match_topic("#", topic)


@given(
    parts=st.lists(st.text(min_size=1, max_size=5), min_size=2, max_size=5),
)
def test_star_matches_exactly_one_token(parts: list[str]) -> None:
    topic = ".".join(parts)
    pattern = ".".join(parts[:-1] + ["*"])
    assert match_topic(pattern, topic)
    # Adding an extra token should not match
    longer = topic + ".EXTRA"
    assert not match_topic(pattern, longer)
