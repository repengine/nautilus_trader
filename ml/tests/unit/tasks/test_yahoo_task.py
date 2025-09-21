from __future__ import annotations

from pathlib import Path

import pytest

from ml.tasks.ingest import PopulateYahooDataTaskConfig
from ml.tasks.ingest import populate_yahoo_data


def test_populate_yahoo_data_all_categories(tmp_path: Path) -> None:
    config = PopulateYahooDataTaskConfig(output_dir=tmp_path, synthetic_years=1, categories=None)
    outputs = populate_yahoo_data(config)
    assert outputs.ohlcv_path.exists()
    assert outputs.record_count > 0


def test_populate_yahoo_data_category_validation(tmp_path: Path) -> None:
    config = PopulateYahooDataTaskConfig(output_dir=tmp_path, categories=("invalid",))
    with pytest.raises(ValueError):
        populate_yahoo_data(config)
