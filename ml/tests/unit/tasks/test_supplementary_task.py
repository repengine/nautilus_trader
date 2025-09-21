from __future__ import annotations

from pathlib import Path

from ml.tasks.ingest import PopulateSupplementaryTaskConfig
from ml.tasks.ingest import populate_supplementary_data


def test_populate_supplementary_data_creates_outputs(tmp_path: Path) -> None:
    config = PopulateSupplementaryTaskConfig(output_dir=tmp_path, synthetic_years=1)
    outputs = populate_supplementary_data(config)
    assert outputs.ohlcv_path.exists()
    assert outputs.record_count > 0
    assert outputs.symbol_count > 0
