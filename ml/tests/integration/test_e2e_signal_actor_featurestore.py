"""
End-to-end function test for MLSignalActor + FeatureStore integration.

This test validates that when `use_feature_store=True` the actor delegates
feature computation to FeatureStore.compute_realtime and returns the features
from the store (with optional persistence).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import SignalStrategy
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from ml.tests.fixtures.database_fixtures import TestDatabase
import pytest


class _FakeBar:
    def __init__(self, instrument_id: InstrumentId) -> None:
        self.bar_type = MagicMock()
        self.bar_type.instrument_id = instrument_id
        self.open = Price.from_str("1.1000")
        self.high = Price.from_str("1.1010")
        self.low = Price.from_str("1.0990")
        self.close = Price.from_str("1.1005")
        self.volume = Quantity.from_int(1_000_000)
        self.ts_event = int(datetime.utcnow().timestamp() * 1e9)
        self.ts_init = int(datetime.utcnow().timestamp() * 1e9)


def _make_bar() -> _FakeBar:
    instrument_id = InstrumentId(Symbol("EURUSD"), Venue("IDEALPRO"))
    return _FakeBar(instrument_id)


class TestE2EActorFeatureStore:
    @pytest.mark.usefixtures("clean_postgres_db")
    def test_actor_delegates_to_feature_store_when_enabled(self, test_database: TestDatabase) -> None:
        # Build config with FeatureStore enabled
        cfg = MLSignalActorConfig(
            model_path="./dummy.onnx",
            model_id="model-1",
            # BarType is required by type, but we only need an object with instrument_id
            bar_type=MagicMock(),
            instrument_id=InstrumentId(Symbol("EURUSD"), Venue("IDEALPRO")),
            use_feature_store=True,
            persist_features=True,
            prediction_threshold=0.1,
            signal_strategy=SignalStrategy.THRESHOLD.value,
            db_connection=test_database.connection_string,
        )

        # Prevent actual model loading
        with patch("ml.actors.signal.MLSignalActor._load_model_with_metadata"):
            actor = MLSignalActor(cfg)

        # Stub FeatureStore.compute_realtime to return deterministic features
        expected = np.arange(10, dtype=np.float32)
        actor._feature_store.compute_realtime = MagicMock(return_value=expected)

        bar = _make_bar()
        features = actor._compute_features(bar)

        # Assert features returned from FeatureStore
        assert isinstance(features, np.ndarray)
        assert np.array_equal(features, expected)

        # Assert store was called with expected API
        actor._feature_store.compute_realtime.assert_called_once()
        kwargs = actor._feature_store.compute_realtime.call_args.kwargs
        assert "bar" in kwargs and kwargs["bar"] is bar
        assert kwargs.get("store") is True
