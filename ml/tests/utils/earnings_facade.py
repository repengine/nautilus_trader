"""Test helpers for constructing DataStore-backed earnings facades."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from ml.stores.adapters import DataStoreEarningsAdapter
from ml.stores.data_store import DataStore
from ml.stores.data_store import QualityReport
from ml.stores.earnings_store import DummyEarningsStore


def build_test_data_store(
    connection_string: str = "postgresql://unused",
    *,
    registry: Any | None = None,
    feature_store: Any | None = None,
    model_store: Any | None = None,
    strategy_store: Any | None = None,
    earnings_store: Any | None = None,
    data_processor: Any | None = None,
) -> DataStore:
    """Construct a DataStore with validation disabled for lightweight tests."""

    store = DataStore(
        connection_string=connection_string,
        registry=registry or MagicMock(),
        feature_store=feature_store or MagicMock(),
        model_store=model_store or MagicMock(),
        strategy_store=strategy_store or MagicMock(),
        earnings_store=earnings_store or DummyEarningsStore(),
        data_processor=data_processor or MagicMock(),
    )

    def _noop_validate(dataset_id: str, data: Any, *, strict_mode: bool = False) -> QualityReport:
        length = len(data) if hasattr(data, "__len__") else 0
        return QualityReport(
            dataset_id=dataset_id,
            total_records=length,
            passed_records=length,
            failed_records=0,
            quality_score=1.0,
            violations=[],
            validation_time_ms=0.0,
        )

    store.validate_batch = _noop_validate  # type: ignore[assignment]
    store._enforce_quality_report = lambda **_: None  # type: ignore[assignment]
    store._get_contract = lambda dataset_id: SimpleNamespace(  # type: ignore[assignment]
        enforcement_mode="monitor_only",
        validation_rules=[],
        quality_thresholds={},
    )
    store._emit_success_event_and_update = lambda **_: None  # type: ignore[assignment]
    store._ensure_dataset_registered = lambda **_: None  # type: ignore[assignment]
    return store


def build_test_earnings_adapter(connection_string: str = "postgresql://unused") -> DataStoreEarningsAdapter:
    """Return a DataStoreEarningsAdapter backed by the lightweight DataStore."""

    store = build_test_data_store(connection_string=connection_string)
    return DataStoreEarningsAdapter(store)
