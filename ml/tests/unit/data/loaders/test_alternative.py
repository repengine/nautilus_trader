from __future__ import annotations

from pathlib import Path
import json

from ml.data.loaders.alternative import AlternativeDataConfig
from ml.data.loaders.alternative import AlternativeDataResult
from ml.data.loaders.alternative import AlternativeSource
from ml.data.loaders.alternative import load_tier1_symbols
from ml.data.loaders.alternative import populate_alternative_data
from ml.data.loaders.alternative import save_alternative_data


def test_populate_alternative_data_returns_frames(tmp_path: Path) -> None:
    config = AlternativeDataConfig(
        symbols=("SPY", "AAPL"),
        sources=(AlternativeSource.CBOE, AlternativeSource.SHORT_INTEREST),
    )
    result = populate_alternative_data(config)
    assert isinstance(result, AlternativeDataResult)
    assert "put_call_ratio" in result.frames
    assert "short_interest" in result.frames
    saved = save_alternative_data(result, tmp_path)
    assert saved
    for path in saved:
        assert path.exists()


def test_load_tier1_symbols_reads_progress(tmp_path: Path) -> None:
    payload = {"completed_bbo": ["spy", "aapl"]}
    path = tmp_path / "tier.json"
    path.write_text(json.dumps(payload))
    symbols = load_tier1_symbols(path)
    assert symbols == ("AAPL", "SPY")
