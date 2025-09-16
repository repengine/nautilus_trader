from __future__ import annotations

from ml.common.message_topics import build_topic, build_topic_for_stage, map_stage_to_topic_segments
from ml.config.events import Stage


def test_normalize_instrument_id_reserved_chars() -> None:
    t = build_topic("data", "created", "EURUSD/SIM*#")
    assert t.endswith("EURUSD.SIM")


def test_build_topic_for_stage_schemes_unit() -> None:
    d, op = map_stage_to_topic_segments(Stage.PREDICTION_EMITTED)
    assert d == "models" and op == "created"
    t1 = build_topic_for_stage(Stage.SIGNAL_EMITTED, "EUR/USD", scheme="domain_op")
    t2 = build_topic_for_stage(
        Stage.SIGNAL_EMITTED,
        "EUR/USD",
        scheme="stage_first",
        prefix="events.ml",
    )
    assert ".strategies.created." in t1
    assert t2.startswith("events.ml.SIGNAL_EMITTED")
