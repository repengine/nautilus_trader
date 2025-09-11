from __future__ import annotations

from typing import Any

from hypothesis import given, strategies as st

from ml.common.in_memory_bus import InMemoryPublisher
from ml.common.message_topics import build_stage_topic
from ml.common.message_topics import map_stage_to_topic_segments
from ml.config.events import Stage


def _instrument_strategy() -> st.SearchStrategy[str]:
    # Include reserved characters to exercise normalization
    chars = st.characters(min_codepoint=33, max_codepoint=126)
    return st.text(alphabet=chars, min_size=1, max_size=20)


@given(
    stage=st.sampled_from(list(Stage)),
    instrument=_instrument_strategy(),
)
def test_stage_topic_scheme_parity_routes_with_wildcards(stage: Stage, instrument: str) -> None:
    """
    For any stage/instrument, topics generated under both schemes route with appropriate
    wildcard patterns in the in-memory bus.
    """
    # Build topics for both schemes
    topic_domain_op = build_stage_topic(stage, instrument, prefix="events.ml")
    topic_stage_first = build_stage_topic(stage, instrument, prefix="events.ml")

    bus = InMemoryPublisher()
    hits: dict[str, int] = {"domain_op": 0, "stage_first": 0}

    # Stage-first pattern
    bus.subscribe(
        f"events.ml.{stage.value}.#",
        lambda t, p: hits.__setitem__("stage_first", hits["stage_first"] + 1),
    )
    # Domain-op equivalent pattern
    domain, op = map_stage_to_topic_segments(stage)
    bus.subscribe(
        f"ml.{domain}.{op}.#", lambda t, p: hits.__setitem__("domain_op", hits["domain_op"] + 1)
    )

    # Publish under canonical builder for both schemes
    bus.publish(topic_stage_first, {})
    # For domain/op scheme, we need to construct via mapping
    # Use ml.common.message_topics.build_topic to build domain_op explicitly
    from ml.common.message_topics import build_topic

    topic2 = build_topic(domain, op, instrument)
    bus.publish(topic2, {})

    assert hits["stage_first"] >= 1
    assert hits["domain_op"] >= 1
