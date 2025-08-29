"""
Integration tests for FeatureStore canonical dataset IDs and JSONB writes.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
from sqlalchemy import select

from ml.stores.feature_store import FeatureStore


class _FakeIndicatorManager:
    def __init__(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial
        self._initialized = True

    def update_from_bar(self, _bar: Any) -> None:  # pragma: no cover - trivial
        return None

    def all_initialized(self) -> bool:  # pragma: no cover - trivial
        return True


class _FakeBar:
    def __init__(self, instrument: str, ts_ns: int) -> None:
        self.bar_type = type("BT", (), {"instrument_id": instrument})()
        self.instrument_id = instrument
        self.ts_event = ts_ns
        self.ts_init = ts_ns
        self.close = 100.0
        self.high = 101.0
        self.low = 99.0
        self.volume = 1000.0


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
def test_feature_store_realtime_event_and_jsonb(test_database: Any, monkeypatch: Any) -> None:
    """
    compute_realtime(store=True) writes JSONB dict and emits canonical dataset_id.
    """
    store = FeatureStore(connection_string=test_database.connection_string)

    # Patch feature names and online features to a small, deterministic vector
    monkeypatch.setattr(store, "_get_feature_names", lambda: ["f1", "f2"])  # type: ignore[misc]
    monkeypatch.setattr(
        store.feature_engineer,
        "calculate_features_online",
        lambda current_bar, indicator_manager, scaler=None: np.array([1.23, 4.56], dtype=np.float32),
    )

    # Mock DataRegistry to capture events
    class _MockRegistry:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def emit_event(self, **kwargs: Any) -> None:
            self.events.append(kwargs)

        def update_watermark(self, **kwargs: Any) -> None:  # pragma: no cover - smoke only
            self.events.append({"wm": kwargs})

    registry = _MockRegistry()
    monkeypatch.setattr(store, "_get_data_registry", lambda: registry)  # type: ignore[misc]

    # Compute and store realtime features
    ts_ns = int(time.time() * 1e9)
    bar = _FakeBar("EUR/USD", ts_ns)
    features = store.compute_realtime(bar, store=True, indicator_manager=_FakeIndicatorManager())
    assert isinstance(features, np.ndarray)
    assert features.size == 2

    # Verify DB contains JSONB values as a mapping
    with store.engine.connect() as conn:
        result = conn.execute(
            select(store.feature_values_table.c.values).where(
                (store.feature_values_table.c.instrument_id == "EUR/USD")
                & (store.feature_values_table.c.ts_event == ts_ns),
            ),
        ).fetchone()

    assert result is not None
    values = result[0]
    # SQLAlchemy may deserialize JSONB to dict already
    assert isinstance(values, dict)
    assert values == {"f1": 1.23, "f2": 4.56}

    # Verify canonical dataset_id in emitted events
    emitted = [e for e in registry.events if isinstance(e, dict) and e.get("dataset_id")]
    assert emitted, "No events emitted"
    assert any(e["dataset_id"] == "features" and e["stage"] == "FEATURE_COMPUTED" for e in emitted)

