from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import polars as pl

from ml.stores.data_store import DataStore


def test_read_range_features_routes_to_feature_store_with_datetimes() -> None:
    # DataStore with mocked feature_store and registry accessors
    mock_feature_store = MagicMock()
    frame = pl.DataFrame({"ts_event": [1_000_000_000], "instrument_id": ["EUR/USD"]})
    mock_feature_store.get_training_data = MagicMock(return_value=frame)

    store = DataStore(
        connection_string="sqlite:///:memory:",
        registry=MagicMock(),
        feature_store=mock_feature_store,  # avoid DB
        model_store=MagicMock(),
        strategy_store=MagicMock(),
        earnings_store=MagicMock(),
    )

    start_ns = 1_000_000_000  # 1s in ns
    end_ns = 3_000_000_000  # 3s in ns

    result = store.read_range(
        dataset_id="features_basic",
        instrument_id="EUR/USD",
        start_ns=start_ns,
        end_ns=end_ns,
    )
    assert result is frame

    # Assert start/end converted to datetimes
    assert mock_feature_store.get_training_data.called
    kwargs = mock_feature_store.get_training_data.call_args.kwargs
    assert isinstance(kwargs["start"], datetime)
    assert isinstance(kwargs["end"], datetime)
    assert kwargs["start"] == datetime.fromtimestamp(start_ns / 1e9)
    assert kwargs["end"] == datetime.fromtimestamp(end_ns / 1e9)
