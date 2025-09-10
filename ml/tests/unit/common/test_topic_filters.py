from __future__ import annotations

from ml.common.topic_filters import match_topic


class TestTopicFilters:
    def test_literal_match(self) -> None:
        assert match_topic("ml.features.updated.EURUSD.SIM", "ml.features.updated.EURUSD.SIM")
        assert not match_topic("ml.features.updated.EURUSD.SIM", "ml.features.updated.EURUSD")

    def test_single_wildcard(self) -> None:
        assert match_topic("ml.features.updated.*.*", "ml.features.updated.EURUSD.SIM")
        assert not match_topic("ml.features.updated.*", "ml.features.updated")

    def test_multi_wildcard(self) -> None:
        assert match_topic("events.ml.FEATURE_COMPUTED.#", "events.ml.FEATURE_COMPUTED")
        assert match_topic("events.ml.FEATURE_COMPUTED.#", "events.ml.FEATURE_COMPUTED.EURUSD.SIM")
        assert match_topic("ml.models.created.#", "ml.models.created.BTCUSDT.BINANCE.foo")
