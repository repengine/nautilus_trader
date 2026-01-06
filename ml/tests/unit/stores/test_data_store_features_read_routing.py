from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from ml.registry.dataclasses import DatasetManifest, DatasetType, StorageKind
from ml.stores.data_store_facade import DataStore


def test_read_range_features_routes_to_feature_store_with_datetimes() -> None:
    # Prepare manifest for FEATURES type
    manifest = DatasetManifest(
        dataset_id="features_basic",
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.POSTGRES,
        location="ml_features",
        partitioning={"by": "ts_event"},
        retention_days=30,
        schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64", "v": "float64"},
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="hash",
        constraints={},
        lineage=[],
        pipeline_signature="unit",
        version="1.0.0",
    )

    # DataStore with mocked feature_store and registry accessors
    mock_feature_store = MagicMock()
    mock_feature_store.get_training_data = MagicMock(return_value="DF")

    store = DataStore(
        connection_string="sqlite:///:memory:",
        registry=MagicMock(),
        feature_store=mock_feature_store,  # avoid DB
        model_store=MagicMock(),
        strategy_store=MagicMock(),
    )

    # Monkeypatch manifest retrieval
    store._manifest_cache["features_basic"] = manifest

    start_ns = 1_000_000_000  # 1s in ns
    end_ns = 3_000_000_000  # 3s in ns

    result = store.read_range("features_basic", "EUR/USD", start_ns, end_ns)
    assert result == "DF"

    # Assert start/end converted to datetimes
    assert mock_feature_store.get_training_data.called
    kwargs = mock_feature_store.get_training_data.call_args.kwargs
    assert isinstance(kwargs["start"], datetime)
    assert isinstance(kwargs["end"], datetime)
    assert kwargs["start"] == datetime.fromtimestamp(start_ns / 1e9)
    assert kwargs["end"] == datetime.fromtimestamp(end_ns / 1e9)
