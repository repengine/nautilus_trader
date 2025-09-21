from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.data.loaders.supplementary import DEFAULT_BASE_SYMBOLS
from ml.data.loaders.supplementary import SupplementaryDataConfig
from ml.data.loaders.supplementary import calculate_correlations
from ml.data.loaders.supplementary import calculate_spreads
from ml.data.loaders.supplementary import create_synthetic_supplementary_data
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
