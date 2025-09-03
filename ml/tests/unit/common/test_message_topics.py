from __future__ import annotations

import pytest

from ml.common.message_topics import build_topic


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

