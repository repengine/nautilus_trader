from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from ml.data.loaders.supplementary import PopulateSupplementaryTaskConfig
from ml.data.loaders.supplementary import populate_supplementary_data


def test_populate_supplementary_data_creates_outputs(tmp_path: Path) -> None:
    config = PopulateSupplementaryTaskConfig(output_dir=tmp_path, synthetic_years=1)
    outputs = populate_supplementary_data(config)
    assert outputs.ohlcv_path.exists()
    assert outputs.record_count > 0
    assert outputs.symbol_count > 0


def test_task_supplementary_module_is_retired() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ml.tasks.ingest.supplementary")
