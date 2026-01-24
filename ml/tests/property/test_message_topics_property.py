from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from ml.common.message_topics import build_stage_topic
from ml.common.message_topics import build_topic
from ml.common.message_topics import build_topic_for_stage
from ml.config.events import Stage


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


@given(
    stage=st.sampled_from(
        [
            Stage.DATA_INGESTED,
            Stage.CATALOG_WRITTEN,
            Stage.FEATURE_COMPUTED,
            Stage.PREDICTION_EMITTED,
            Stage.SIGNAL_EMITTED,
            Stage.ORDER_EVENT_EMITTED,
        ],
    ),
    instrument=st.text(min_size=0, max_size=30),
)
def test_build_stage_topic_fuzz(stage: Stage, instrument: str) -> None:
    topic = build_stage_topic(stage, instrument)
    assert topic.startswith(f"events.ml.{stage.value}")
    # When instrument provided, ensure no reserved characters remain
    if instrument:
        assert not re.search(r"[/*#+$]", topic)
        assert topic.count(".") >= 2  # prefix + STAGE + maybe normalized instrument


@given(
    stage=st.sampled_from(
        [
            Stage.DATA_INGESTED,
            Stage.CATALOG_WRITTEN,
            Stage.FEATURE_COMPUTED,
            Stage.PREDICTION_EMITTED,
            Stage.SIGNAL_EMITTED,
            Stage.ORDER_EVENT_EMITTED,
        ],
    ),
    instrument=st.text(min_size=3, max_size=30),
)
def test_build_topic_for_stage_schemes_property(stage: Stage, instrument: str) -> None:
    # Domain-op scheme should follow canonical layout
    t1 = build_topic_for_stage(stage, instrument, scheme="domain_op")
    assert t1.startswith("ml.")
    assert len(t1.split(".")) >= 4
    # Stage-first scheme should prefix with events.ml
    t2 = build_topic_for_stage(stage, instrument, scheme="stage_first", prefix="events.ml")
    assert t2.startswith("events.ml.")
