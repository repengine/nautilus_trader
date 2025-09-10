from __future__ import annotations

import pytest

from ml.common.message_topics import build_stage_topic
from ml.common.message_topics import build_topic
from ml.common.message_topics import build_topic_for_stage
from ml.config.events import Stage


class TestMessageTopics:
    def test_build_topic_happy_path(self) -> None:
        topic = build_topic("data", "created", "EURUSD.SIM")
        assert topic == "ml.data.created.EURUSD.SIM"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("EURUSD/SIM", "EURUSD.SIM"),
            ("BTCUSDT#BINANCE", "BTCUSDT.BINANCE"),
            ("FOO+BAR$BAZ", "FOO.BAR.BAZ"),
            ("__ABC__", "__ABC__"),
            ("..ABC..", "ABC"),
            ("", "UNKNOWN"),
        ],
    )
    def test_instrument_normalization(self, raw: str, expected: str) -> None:
        topic = build_topic("features", "updated", raw)
        assert topic.endswith(expected)
        assert topic.startswith("ml.features.updated.")

    @pytest.mark.parametrize("bad_domain", ["Data", "data1", "data-", ""])
    def test_invalid_domain_rejected(self, bad_domain: str) -> None:
        with pytest.raises(ValueError):
            build_topic(bad_domain, "created", "EURUSD.SIM")

    @pytest.mark.parametrize("bad_op", ["Created", "create-1", "", "created!"])
    def test_invalid_operation_rejected(self, bad_op: str) -> None:
        with pytest.raises(ValueError):
            build_topic("data", bad_op, "EURUSD.SIM")

    def test_build_stage_topic_with_and_without_instrument(self) -> None:
        # Without instrument suffix
        t1 = build_stage_topic(Stage.CATALOG_WRITTEN)
        assert t1 == "events.ml.CATALOG_WRITTEN"

        # With instrument suffix and normalization
        t2 = build_stage_topic(Stage.FEATURE_COMPUTED, "EUR/USD")
        assert t2 == "events.ml.FEATURE_COMPUTED.EUR.USD"
        assert not any(ch in t2 for ch in "/*#+$")

    def test_build_topic_for_stage_schemes(self) -> None:
        # Canonical domain_op scheme
        topic1 = build_topic_for_stage(
            Stage.PREDICTION_EMITTED,
            "BTCUSDT#BINANCE",
            scheme="domain_op",
        )
        assert topic1.startswith("ml.models.created.")
        assert topic1.endswith("BTCUSDT.BINANCE")

        # Stage-first scheme with custom prefix
        topic2 = build_topic_for_stage(
            Stage.SIGNAL_EMITTED,
            "ETH-USD/Coinbase",
            scheme="stage_first",
            prefix="events.ml",
        )
        assert topic2.startswith("events.ml.SIGNAL_EMITTED.")
        assert not any(ch in topic2 for ch in "/*#+$")
