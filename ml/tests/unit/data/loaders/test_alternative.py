from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from ml.data.loaders import alternative as alternative_loader
from ml.data.loaders.alternative import AlternativeDataConfig
from ml.data.loaders.alternative import AlternativeDataResult
from ml.data.loaders.alternative import AlternativeSource
from ml.data.loaders.alternative import PopulateAlternativeDataTaskConfig
from ml.data.loaders.alternative import load_tier1_symbols
from ml.data.loaders.alternative import populate_alternative_data
from ml.data.loaders.alternative import populate_alternative_data_task
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


def test_populate_alternative_data_all_sources_are_covered() -> None:
    result = populate_alternative_data(
        AlternativeDataConfig(
            symbols=("SPY", "AAPL"),
            sources=tuple(AlternativeSource),
        ),
    )

    assert "aaii_sentiment" in result.frames
    assert "cot_reports" in result.frames
    assert "microstructure" in result.frames
    assert "news_sentiment" in result.frames
    assert "earnings_calendar" in result.frames
    assert "sector_industry" in result.frames
    assert set(result.non_empty_sources) == set(result.frames.keys())


def test_save_alternative_data_skips_empty_frames(tmp_path: Path) -> None:
    result = AlternativeDataResult(
        frames={
            "empty": pl.DataFrame(),
            "filled": pl.DataFrame({"symbol": ["SPY"], "value": [1.0]}),
        },
    )

    saved_paths = save_alternative_data(result, tmp_path)

    assert saved_paths == (tmp_path / "filled.parquet",)
    assert not (tmp_path / "empty.parquet").exists()
    assert saved_paths[0].exists()


def test_load_tier1_symbols_handles_missing_invalid_and_unexpected_payload(tmp_path: Path) -> None:
    missing = load_tier1_symbols(tmp_path / "missing.json")
    assert missing == tuple()

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{broken}", encoding="utf-8")
    invalid = load_tier1_symbols(invalid_path)
    assert invalid == tuple()

    malformed_payload_path = tmp_path / "malformed.json"
    malformed_payload_path.write_text(json.dumps({"completed_bbo": "SPY"}), encoding="utf-8")
    malformed = load_tier1_symbols(malformed_payload_path)
    assert malformed == tuple()


def test_resolve_task_sources_variants() -> None:
    all_sources = alternative_loader._resolve_task_sources(
        PopulateAlternativeDataTaskConfig(
            output_dir=Path("unused"),
            populate_all=True,
        ),
    )
    assert all_sources == tuple(AlternativeSource)

    with pytest.raises(ValueError, match="No sources specified"):
        alternative_loader._resolve_task_sources(
            PopulateAlternativeDataTaskConfig(
                output_dir=Path("unused"),
                populate_all=False,
                sources=None,
            ),
        )

    with pytest.raises(ValueError, match="Unsupported alternative data source"):
        alternative_loader._resolve_task_sources(
            PopulateAlternativeDataTaskConfig(
                output_dir=Path("unused"),
                populate_all=False,
                sources=("unknown-source",),
            ),
        )


def test_resolve_task_symbols_prefers_explicit_then_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = alternative_loader._resolve_task_symbols(
        PopulateAlternativeDataTaskConfig(
            output_dir=Path("unused"),
            symbols=("spy", "SPY", "aapl"),
        ),
    )
    assert set(explicit) == {"SPY", "AAPL"}

    monkeypatch.setattr(alternative_loader, "load_tier1_symbols", lambda _path: ("QQQ", "SPY"))
    fallback = alternative_loader._resolve_task_symbols(
        PopulateAlternativeDataTaskConfig(
            output_dir=Path("unused"),
            symbols=None,
        ),
    )
    assert fallback == ("QQQ", "SPY")


def test_resolve_task_symbols_raises_when_no_symbols_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alternative_loader, "load_tier1_symbols", lambda _path: tuple())

    with pytest.raises(ValueError, match="No symbols provided"):
        alternative_loader._resolve_task_symbols(
            PopulateAlternativeDataTaskConfig(
                output_dir=Path("unused"),
                symbols=None,
            ),
        )


def test_populate_alternative_data_task_executes_end_to_end(tmp_path: Path) -> None:
    config = PopulateAlternativeDataTaskConfig(
        output_dir=tmp_path,
        symbols=("spy",),
        sources=("aaii",),
    )

    result = populate_alternative_data_task(config)

    assert isinstance(result, AlternativeDataResult)
    assert set(result.frames.keys()) == {"aaii_sentiment"}
    assert (tmp_path / "aaii_sentiment.parquet").exists()


def test_populate_alternative_data_task_delegates_expected_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    expected_result = AlternativeDataResult(frames={"aaii_sentiment": pl.DataFrame({"x": [1]})})

    monkeypatch.setattr(alternative_loader, "_resolve_task_symbols", lambda _cfg: ("SPY",))
    monkeypatch.setattr(alternative_loader, "_resolve_task_sources", lambda _cfg: (AlternativeSource.AAII,))

    def _populate(config: AlternativeDataConfig) -> AlternativeDataResult:
        captured["symbols"] = config.symbols
        captured["sources"] = config.sources
        return expected_result

    monkeypatch.setattr(alternative_loader, "populate_alternative_data", _populate)
    monkeypatch.setattr(
        alternative_loader,
        "save_alternative_data",
        lambda result, output_dir: captured.update({"saved_result": result, "output_dir": output_dir}) or tuple(),
    )

    config = PopulateAlternativeDataTaskConfig(output_dir=tmp_path, symbols=("spy",), sources=("aaii",))
    result = populate_alternative_data_task(config)

    assert result is expected_result
    assert captured["symbols"] == ("SPY",)
    assert captured["sources"] == (AlternativeSource.AAII,)
    assert captured["saved_result"] is expected_result
    assert captured["output_dir"] == tmp_path
