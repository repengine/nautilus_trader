from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.data.loaders.supplementary import DEFAULT_BASE_SYMBOLS
from ml.data.loaders.supplementary import PopulateSupplementaryTaskConfig
from ml.data.loaders.supplementary import PopulateYahooDataTaskConfig
from ml.data.loaders.supplementary import SUPPLEMENTARY_SYMBOLS
from ml.data.loaders.supplementary import SupplementaryDataConfig
from ml.data.loaders.supplementary import calculate_correlations
from ml.data.loaders.supplementary import calculate_spreads
from ml.data.loaders.supplementary import create_synthetic_supplementary_data
from ml.data.loaders.supplementary import populate_supplementary_data
from ml.data.loaders.supplementary import populate_yahoo_data
from ml.data.loaders.supplementary import write_supplementary_outputs


def test_create_synthetic_data_has_expected_columns(tmp_path: Path) -> None:
    config = SupplementaryDataConfig(output_dir=tmp_path, synthetic_years=1)
    data = create_synthetic_supplementary_data(config)
    assert {"timestamp", "symbol", "open", "close"}.issubset(data.columns)
    correlations = calculate_correlations(data, DEFAULT_BASE_SYMBOLS)
    spreads = calculate_spreads(data)
    outputs = write_supplementary_outputs(data, correlations, spreads, config)
    assert outputs.ohlcv_path.exists()
    assert outputs.metadata_path.exists()


def test_calculate_correlations_handles_missing_symbols() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = pd.DataFrame(
        {
            "timestamp": list(dates) * 2,
            "symbol": ["AAA"] * 5 + ["BBB"] * 5,
            "returns": [0.01, -0.02, 0.005, 0.01, 0.0] * 2,
        },
    )
    correlations = calculate_correlations(data, ("AAA", "SPY"))
    assert not correlations.empty


def _sample_supplementary_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(["2024-01-01", "2024-01-02"])
    return pd.DataFrame(
        {
            "timestamp": list(timestamps) * 2,
            "symbol": ["^GSPC", "^GSPC", "XLK", "XLK"],
            "open": [100.0, 101.0, 50.0, 51.0],
            "high": [101.0, 102.0, 51.0, 52.0],
            "low": [99.0, 100.0, 49.0, 50.0],
            "close": [100.5, 101.5, 50.5, 51.5],
            "volume": [1000, 1100, 900, 950],
            "returns": [0.0, 0.01, 0.0, 0.02],
        },
    )


def test_populate_supplementary_data_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ml.data.loaders.supplementary.create_synthetic_supplementary_data",
        lambda _cfg: _sample_supplementary_frame(),
    )

    outputs = populate_supplementary_data(PopulateSupplementaryTaskConfig(output_dir=tmp_path))

    assert outputs.ohlcv_path.exists()
    assert outputs.metadata_path.exists()


def test_populate_yahoo_data_filters_requested_categories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ml.data.loaders.supplementary.create_synthetic_supplementary_data",
        lambda _cfg: _sample_supplementary_frame(),
    )

    outputs = populate_yahoo_data(
        PopulateYahooDataTaskConfig(
            output_dir=tmp_path,
            categories=("indices",),
        ),
    )

    symbols = set(pd.read_parquet(outputs.ohlcv_path)["symbol"].unique())
    assert symbols.issubset(set(SUPPLEMENTARY_SYMBOLS["indices"]))


def test_populate_yahoo_data_rejects_unknown_categories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown Yahoo categories"):
        populate_yahoo_data(
            PopulateYahooDataTaskConfig(
                output_dir=tmp_path,
                categories=("unknown",),
            ),
        )
