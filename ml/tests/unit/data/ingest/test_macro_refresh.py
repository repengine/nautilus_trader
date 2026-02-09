"""Tests for :mod:`ml.data.ingest.macro_refresh`."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from types import ModuleType
from types import SimpleNamespace
from dataclasses import dataclass

import polars as pl
import pytest

from ml.config.dataset_ids import MACRO_OBSERVATIONS_DATASET_ID
from ml.config.dataset_ids import MACRO_RELEASES_DATASET_ID
from ml.config.events import Source
from ml.config import WatermarkWindowConfig
from ml.data.ingest import macro_refresh as macro
from ml.data.ingest.macro_refresh import MacroRefreshResult
from ml.data.ingest.macro_refresh import ensure_macro_ready
from ml.data.ingest.macro_refresh import refresh_alfred_if_stale
from ml.data.ingest.macro_refresh import refresh_fred_if_stale
from ml.registry.watermark import Watermark

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

class _StubFredLoader:
    def __init__(self, target: Path) -> None:
        self.target = target
        self.calls: list[dict[str, Any]] = []

    def fetch_all_indicators(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        use_cache: bool = False,
        **_: object,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "use_cache": use_cache,
            },
        )
        return {"SERIES": {}}

    def export_ml_parquet(
        self,
        data: object = None,
        out_path: Path | None = None,
        **_: object,
    ) -> Path:
        assert isinstance(data, dict)
        assert out_path is not None
        out_path.write_text("parquet", encoding="utf-8")
        return out_path

class _StubAlfredLoader:
    def __init__(self) -> None:
        self.calls = 0

    def refresh(self) -> None:
        self.calls += 1

def test_refresh_fred_if_stale_refreshes_missing(tmp_path: Path) -> None:
    target = tmp_path / "fred.parquet"
    loader = _StubFredLoader(target)
    refreshed, error = refresh_fred_if_stale(
        parquet_path=target,
        max_age=timedelta(hours=1),
        loader_factory=lambda _series_ids: loader,
    )
    assert refreshed
    assert error is None
    assert target.exists()
    assert loader.calls == [{"start_date": None, "end_date": None, "use_cache": False}]

def test_refresh_fred_if_stale_skips_when_fresh(tmp_path: Path) -> None:
    target = tmp_path / "fred.parquet"
    target.write_text("parquet", encoding="utf-8")
    loader = _StubFredLoader(target)
    refreshed, error = refresh_fred_if_stale(
        parquet_path=target,
        max_age=timedelta(hours=24),
        loader_factory=lambda _series_ids: loader,
    )
    assert not refreshed
    assert error is None
    assert loader.calls == []

def test_refresh_alfred_if_stale_invokes_loader(tmp_path: Path) -> None:
    base_dir = tmp_path / "vintages"
    loader = _StubAlfredLoader()
    refreshed, error = refresh_alfred_if_stale(
        base_dir=base_dir,
        max_age=timedelta(hours=1),
        series_ids=("CPI",),
        loader_factory=lambda series: loader,
    )
    assert refreshed
    assert error is None
    assert loader.calls == 1

def test_ensure_macro_ready_combines_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fred_path = tmp_path / "macro" / "fred.parquet"
    vintage_dir = tmp_path / "macro" / "vintages"
    calls: dict[str, Any] = {"fred": 0, "alfred": 0}

    def _fake_fred(**_: Any) -> tuple[bool, Exception | None]:
        calls["fred"] += 1
        return True, None

    def _fake_alfred(**_: Any) -> tuple[bool, Exception | None]:
        calls["alfred"] += 1
        return False, RuntimeError("boom")

    monkeypatch.setattr("ml.data.ingest.macro_refresh.refresh_fred_if_stale", _fake_fred)
    monkeypatch.setattr("ml.data.ingest.macro_refresh.refresh_alfred_if_stale", _fake_alfred)

    result = ensure_macro_ready(
        fred_path=fred_path,
        vintage_dir=vintage_dir,
        max_age=timedelta(hours=12),
        series_ids=("DGS10",),
    )
    assert calls == {"fred": 1, "alfred": 1}
    assert isinstance(result, MacroRefreshResult)
    assert result.fred_refreshed is True
    assert not result.alfred_refreshed
    assert isinstance(result.alfred_error, RuntimeError)


def test_ensure_macro_ready_ingests_sql_when_data_store_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fred_path = tmp_path / "macro" / "fred.parquet"
    fred_path.parent.mkdir(parents=True, exist_ok=True)
    vintage_dir = tmp_path / "macro" / "vintages"
    series_id = "CPI"
    fred_frame = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)],
            "series_id": [series_id],
            "value": [1.5],
        },
    )
    fred_frame.write_parquet(fred_path)

    release_dir = vintage_dir / series_id
    release_dir.mkdir(parents=True, exist_ok=True)
    release_frame = pl.DataFrame(
        {
            "series_id": [series_id],
            "observation_ts": [datetime(2024, 1, 1)],
            "release_ts": [datetime(2024, 2, 1)],
            "release_end_ts": [datetime(2024, 2, 2)],
            "value": [2.0],
        },
    )
    release_frame.write_parquet(release_dir / "release_calendar.parquet")

    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.refresh_fred_if_stale",
        lambda **_: (False, None),
    )
    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.refresh_alfred_if_stale",
        lambda **_: (False, None),
    )

    calls: list[dict[str, Any]] = []

    class _StubStore:
        def write_ingestion(
            self,
            dataset_id: str,
            records: object,
            source: str,
            run_id: str,
            instrument_id: str | None = None,
        ) -> None:
            calls.append(
                {
                    "dataset_id": dataset_id,
                    "instrument_id": instrument_id,
                    "source": source,
                    "run_id": run_id,
                    "height": getattr(records, "height", None),
                },
            )

    result = ensure_macro_ready(
        fred_path=fred_path,
        vintage_dir=vintage_dir,
        max_age=timedelta(hours=12),
        series_ids=(series_id,),
        data_store=_StubStore(),
    )

    assert result.fred_error is None
    assert result.alfred_error is None
    assert any(call["dataset_id"] == MACRO_OBSERVATIONS_DATASET_ID for call in calls)
    assert any(call["dataset_id"] == MACRO_RELEASES_DATASET_ID for call in calls)


def test_ensure_macro_ready_filters_sql_using_watermark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fred_path = tmp_path / "macro" / "fred.parquet"
    fred_path.parent.mkdir(parents=True, exist_ok=True)
    series_id = "CPI"
    fred_frame = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
            "series_id": [series_id, series_id],
            "value": [1.0, 2.0],
        },
    )
    fred_frame.write_parquet(fred_path)

    vintage_dir = tmp_path / "macro" / "vintages"
    release_dir = vintage_dir / series_id
    release_dir.mkdir(parents=True, exist_ok=True)
    release_frame = pl.DataFrame(
        {
            "series_id": [series_id, series_id],
            "observation_ts": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
            "release_ts": [datetime(2024, 1, 15), datetime(2024, 2, 15)],
            "release_end_ts": [datetime(2024, 1, 16), datetime(2024, 2, 16)],
            "value": [1.0, 2.0],
        },
    )
    release_frame.write_parquet(release_dir / "release_calendar.parquet")

    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.refresh_fred_if_stale",
        lambda **_: (False, None),
    )
    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.refresh_alfred_if_stale",
        lambda **_: (False, None),
    )

    watermark_ns = int(datetime(2024, 1, 20, tzinfo=UTC).timestamp() * 1_000_000_000)

    class _Registry:
        def get_watermark(
            self,
            dataset_id: str,
            instrument_id: str,
            source: Source | str,
        ) -> Watermark | None:
            return Watermark(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                source=str(source),
                last_success_ns=watermark_ns,
                last_attempt_ns=watermark_ns,
                last_count=1,
                completeness_pct=100.0,
                updated_at=0.0,
            )

    calls: list[dict[str, Any]] = []

    class _StubStore:
        registry = _Registry()

        def write_ingestion(
            self,
            dataset_id: str,
            records: object,
            source: str,
            run_id: str,
            instrument_id: str | None = None,
        ) -> None:
            calls.append(
                {
                    "dataset_id": dataset_id,
                    "instrument_id": instrument_id,
                    "height": getattr(records, "height", None),
                },
            )

    result = ensure_macro_ready(
        fred_path=fred_path,
        vintage_dir=vintage_dir,
        max_age=timedelta(hours=12),
        series_ids=(series_id,),
        data_store=_StubStore(),
        watermark_config=WatermarkWindowConfig(
            use_watermark=True,
            lookback_days=0,
            max_window_days=None,
            fallback_start_days=None,
        ),
    )

    assert result.fred_error is None
    assert result.alfred_error is None
    macro_observation_calls = [
        call for call in calls if call["dataset_id"] == MACRO_OBSERVATIONS_DATASET_ID
    ]
    macro_release_calls = [
        call for call in calls if call["dataset_id"] == MACRO_RELEASES_DATASET_ID
    ]
    assert macro_observation_calls and macro_observation_calls[0]["height"] == 1
    assert macro_release_calls and macro_release_calls[0]["height"] == 1


def test_require_polars_checks_dependencies_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency_calls: list[list[str]] = []

    def _record(deps: list[str]) -> None:
        dependency_calls.append(deps)

    monkeypatch.setattr(macro, "HAS_POLARS", False)
    monkeypatch.setattr(macro, "pl", SimpleNamespace())
    monkeypatch.setattr(macro, "check_ml_dependencies", _record)

    result = macro._require_polars()

    assert isinstance(result, SimpleNamespace)
    assert dependency_calls == [["polars"]]


def test_ingest_macro_observations_handles_missing_and_filtered_files(tmp_path: Path) -> None:
    class _Store:
        def write_ingestion(
            self,
            dataset_id: str,
            records: object,
            source: str,
            run_id: str,
            instrument_id: str | None = None,
        ) -> None:
            raise AssertionError("write_ingestion should not be called")

    missing_count = macro._ingest_macro_observations(
        data_store=_Store(),
        fred_path=tmp_path / "missing.parquet",
        run_id="run-a",
        series_ids=None,
    )
    assert missing_count == 0

    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)],
            "series_id": ["CPI"],
            "value": [1.0],
        },
    )
    fred_path = tmp_path / "fred.parquet"
    frame.write_parquet(fred_path)
    filtered_count = macro._ingest_macro_observations(
        data_store=_Store(),
        fred_path=fred_path,
        run_id="run-b",
        series_ids=("UNUSED",),
    )
    assert filtered_count == 0


def test_ingest_macro_observations_writes_expected_rows(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    class _Store:
        def write_ingestion(
            self,
            dataset_id: str,
            records: object,
            source: str,
            run_id: str,
            instrument_id: str | None = None,
        ) -> None:
            calls.append(
                {
                    "dataset_id": dataset_id,
                    "source": source,
                    "run_id": run_id,
                    "instrument_id": instrument_id,
                    "height": getattr(records, "height", 0),
                },
            )

    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "series_id": ["CPI", "CPI"],
            "value": [1.0, 1.2],
        },
    )
    fred_path = tmp_path / "fred.parquet"
    frame.write_parquet(fred_path)

    written = macro._ingest_macro_observations(
        data_store=_Store(),
        fred_path=fred_path,
        run_id="run-c",
        series_ids=("CPI",),
    )

    assert written == 2
    assert calls[0]["dataset_id"] == MACRO_OBSERVATIONS_DATASET_ID
    assert calls[0]["instrument_id"] == "CPI"


def test_ingest_macro_release_calendar_handles_missing_and_writes(tmp_path: Path) -> None:
    class _Store:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def write_ingestion(
            self,
            dataset_id: str,
            records: object,
            source: str,
            run_id: str,
            instrument_id: str | None = None,
        ) -> None:
            self.calls.append(
                {
                    "dataset_id": dataset_id,
                    "instrument_id": instrument_id,
                    "source": source,
                    "run_id": run_id,
                    "height": getattr(records, "height", 0),
                },
            )

    store = _Store()
    missing = macro._ingest_macro_release_calendar(
        data_store=store,
        vintage_dir=tmp_path / "missing",
        run_id="run-d",
        series_ids=None,
    )
    assert missing == 0

    series_dir = tmp_path / "vintages" / "CPI"
    series_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "series_id": ["CPI"],
            "observation_ts": [datetime(2024, 1, 1)],
            "release_ts": [datetime(2024, 1, 15)],
            "release_end_ts": [datetime(2024, 1, 16)],
            "value": [2.0],
        },
    ).write_parquet(series_dir / "release_calendar.parquet")

    written = macro._ingest_macro_release_calendar(
        data_store=store,
        vintage_dir=tmp_path / "vintages",
        run_id="run-e",
        series_ids=("CPI",),
    )

    assert written == 1
    assert store.calls[0]["dataset_id"] == MACRO_RELEASES_DATASET_ID
    assert store.calls[0]["instrument_id"] == "CPI"


def test_refresh_fred_if_stale_returns_error_on_loader_failure(tmp_path: Path) -> None:
    target = tmp_path / "fred.parquet"

    class _BrokenLoader:
        def fetch_all_indicators(self, **_: object) -> object:
            raise RuntimeError("failed")

        def export_ml_parquet(self, data: object = None, out_path: Path | None = None, **_: object) -> Path:
            raise AssertionError("export should not be called")

    refreshed, error = refresh_fred_if_stale(
        parquet_path=target,
        max_age=timedelta(seconds=0),
        loader_factory=lambda _series: _BrokenLoader(),
    )

    assert refreshed is False
    assert isinstance(error, RuntimeError)


def test_refresh_alfred_if_stale_skips_without_series_and_when_fresh(tmp_path: Path) -> None:
    skipped, error = refresh_alfred_if_stale(
        base_dir=tmp_path / "vintages",
        max_age=timedelta(hours=1),
        series_ids=(),
        loader_factory=lambda _series: _StubAlfredLoader(),
    )
    assert skipped is False
    assert error is None

    base_dir = tmp_path / "fresh" / "vintages"
    series_dir = base_dir / "CPI"
    series_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "series_id": ["CPI"],
            "observation_ts": [datetime(2024, 1, 1)],
            "release_ts": [datetime(2024, 1, 15)],
            "release_end_ts": [datetime(2024, 1, 16)],
            "value": [1.0],
        },
    ).write_parquet(series_dir / "release_calendar.parquet")

    fresh, fresh_error = refresh_alfred_if_stale(
        base_dir=base_dir,
        max_age=timedelta(days=365),
        series_ids=("CPI",),
        loader_factory=lambda _series: _StubAlfredLoader(),
    )
    assert fresh is False
    assert fresh_error is None


def test_refresh_alfred_if_stale_returns_error_on_loader_failure(tmp_path: Path) -> None:
    class _BrokenLoader:
        def refresh(self) -> object:
            raise RuntimeError("alfred failed")

    refreshed, error = refresh_alfred_if_stale(
        base_dir=tmp_path / "vintages",
        max_age=timedelta(seconds=0),
        series_ids=("CPI",),
        loader_factory=lambda _series: _BrokenLoader(),
    )

    assert refreshed is False
    assert isinstance(error, RuntimeError)


def test_ensure_macro_ready_swallows_sql_ingestion_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fred_path = tmp_path / "macro" / "fred.parquet"
    vintage_dir = tmp_path / "macro" / "vintages"
    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.refresh_fred_if_stale",
        lambda **_: (False, None),
    )
    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.refresh_alfred_if_stale",
        lambda **_: (False, None),
    )

    def _raise_obs(**_: Any) -> int:
        raise RuntimeError("sql failed")

    monkeypatch.setattr(macro, "_ingest_macro_observations", _raise_obs)

    result = ensure_macro_ready(
        fred_path=fred_path,
        vintage_dir=vintage_dir,
        max_age=timedelta(hours=1),
        data_store=SimpleNamespace(),
    )

    assert result.fred_error is None
    assert result.alfred_error is None


def test_build_loader_helpers_construct_expected_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    fred_module = ModuleType("ml.data.loaders.fred_loader")

    class _FREDConfig:
        pass

    @dataclass(slots=True, frozen=True)
    class _FREDIndicator:
        series_id: str
        name: str
        category: str

    class _FREDLoader:
        DEFAULT_INDICATORS = (_FREDIndicator("CPI", "CPI", "macro"),)

        def __init__(self, config: _FREDConfig, indicators: list[_FREDIndicator] | None) -> None:
            self.config = config
            self.indicators = indicators

    fred_module.FREDConfig = _FREDConfig
    fred_module.FREDIndicator = _FREDIndicator
    fred_module.FREDDataLoader = _FREDLoader

    alfred_module = ModuleType("ml.data.loaders.alfred_loader")

    @dataclass(slots=True, frozen=True)
    class _ALFREDConfig:
        series_ids: tuple[str, ...]
        start_date: str | None
        end_date: str | None
        window_days: int
        fallback_to_fred_series: tuple[str, ...]

    class _ALFREDLoader:
        def __init__(self, cfg: _ALFREDConfig) -> None:
            self.cfg = cfg

    alfred_module.ALFREDConfig = _ALFREDConfig
    alfred_module.ALFREDDataLoader = _ALFREDLoader

    monkeypatch.setitem(sys.modules, "ml.data.loaders.fred_loader", fred_module)
    monkeypatch.setitem(sys.modules, "ml.data.loaders.alfred_loader", alfred_module)

    fred_loader = macro._build_fred_loader(series_ids=("CPI", "NEW_SERIES"))
    assert isinstance(fred_loader, _FREDLoader)
    assert fred_loader.indicators is not None
    assert [item.series_id for item in fred_loader.indicators] == ["CPI", "NEW_SERIES"]

    default_fred_loader = macro._build_fred_loader(series_ids=None)
    assert isinstance(default_fred_loader, _FREDLoader)
    assert default_fred_loader.indicators is None

    alfred_loader = macro._build_alfred_loader(
        series_ids=("CPI",),
        realtime_start="2020-01-01",
        realtime_end="2024-01-01",
        window_days=120,
        fallback_series=("NEW_SERIES",),
    )
    assert isinstance(alfred_loader, _ALFREDLoader)
    assert alfred_loader.cfg.fallback_to_fred_series == ("NEW_SERIES",)


def test_protocol_method_placeholders_are_callable() -> None:
    assert macro._FREDLoaderProtocol.fetch_all_indicators(object()) is None
    assert macro._FREDLoaderProtocol.export_ml_parquet(object()) is None
    assert macro._ALFREDLoaderProtocol.refresh(object()) is None


def test_refresh_fred_if_stale_uses_default_loader_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "nested" / "fred.parquet"
    call_state: dict[str, Any] = {}

    class _Loader:
        def fetch_all_indicators(self, use_cache: bool = False, **_: object) -> object:
            call_state["use_cache"] = use_cache
            return {"series": {}}

        def export_ml_parquet(self, data: object = None, out_path: Path | None = None, **_: object) -> Path:
            assert data == {"series": {}}
            assert out_path is not None
            out_path.write_text("ok", encoding="utf-8")
            return out_path

    def _builder(series_ids: tuple[str, ...] | None) -> _Loader:
        call_state["series_ids"] = series_ids
        return _Loader()

    monkeypatch.setattr(macro, "_build_fred_loader", _builder)

    refreshed, error = refresh_fred_if_stale(
        parquet_path=target,
        max_age=timedelta(hours=1),
        series_ids=("CPI",),
    )

    assert refreshed is True
    assert error is None
    assert target.exists()
    assert call_state == {"series_ids": ("CPI",), "use_cache": False}


def test_refresh_alfred_if_stale_stale_file_uses_default_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = tmp_path / "vintages"
    series_dir = base_dir / "CPI"
    series_dir.mkdir(parents=True, exist_ok=True)
    release_path = series_dir / "release_calendar.parquet"
    release_path.write_text("stale", encoding="utf-8")
    os.utime(release_path, (1.0, 1.0))

    calls: list[tuple[tuple[str, ...], str | None, str | None, int]] = []
    refresh_calls: list[int] = []

    class _Loader:
        def refresh(self) -> None:
            refresh_calls.append(1)

    def _build(
        series_ids: tuple[str, ...],
        realtime_start: str | None,
        realtime_end: str | None,
        window_days: int,
        fallback_series: tuple[str, ...] | None = None,
    ) -> _Loader:
        assert fallback_series is None
        calls.append((series_ids, realtime_start, realtime_end, window_days))
        return _Loader()

    monkeypatch.setattr(macro, "_build_alfred_loader", _build)

    refreshed, error = refresh_alfred_if_stale(
        base_dir=base_dir,
        max_age=timedelta(microseconds=1),
        series_ids=("CPI",),
        loader_factory=None,
    )

    assert refreshed is True
    assert error is None
    assert calls == [(("CPI",), None, None, 365)]
    assert refresh_calls == [1]


def test_ensure_macro_ready_skips_alfred_when_vintage_dir_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fred_path = tmp_path / "macro" / "fred.parquet"
    calls: list[str] = []

    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.refresh_fred_if_stale",
        lambda **_: (False, None),
    )
    monkeypatch.setattr(
        macro,
        "_ingest_macro_observations",
        lambda **_: calls.append("obs") or 1,
    )
    monkeypatch.setattr(
        macro,
        "_ingest_macro_release_calendar",
        lambda **_: calls.append("release") or 1,
    )

    result = ensure_macro_ready(
        fred_path=fred_path,
        vintage_dir=None,
        max_age=timedelta(hours=1),
        data_store=SimpleNamespace(registry=None),
    )

    assert result.alfred_refreshed is False
    assert result.alfred_error is None
    assert calls == ["obs"]


def test_ensure_macro_ready_uses_provided_alfred_loader_factory_and_fallback_series(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fred_path = tmp_path / "macro" / "fred.parquet"
    vintage_dir = tmp_path / "macro" / "vintages"

    captured: dict[str, Any] = {}

    def sentinel_factory(_series: object) -> _StubAlfredLoader:
        return _StubAlfredLoader()

    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.refresh_fred_if_stale",
        lambda **_: (False, None),
    )

    def _fake_refresh(
        *,
        base_dir: Path,
        max_age: timedelta,
        series_ids: tuple[str, ...],
        loader_factory: Any,
    ) -> tuple[bool, Exception | None]:
        captured["base_dir"] = base_dir
        captured["max_age"] = max_age
        captured["series_ids"] = series_ids
        captured["loader_factory"] = loader_factory
        return True, None

    monkeypatch.setattr("ml.data.ingest.macro_refresh.refresh_alfred_if_stale", _fake_refresh)

    result = ensure_macro_ready(
        fred_path=fred_path,
        vintage_dir=vintage_dir,
        max_age=timedelta(hours=2),
        series_ids=("CPI",),
        alfred_loader_factory=sentinel_factory,
        alfred_fallback_series=("BYPASS",),
    )

    assert result.alfred_refreshed is True
    assert result.alfred_error is None
    assert captured["base_dir"] == vintage_dir
    assert captured["series_ids"] == ("CPI",)
    assert captured["loader_factory"] is sentinel_factory


def test_ensure_macro_ready_handles_missing_fallback_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fred_path = tmp_path / "macro" / "fred.parquet"
    vintage_dir = tmp_path / "macro" / "vintages"
    fallback_path = Path("ml/config/macro_alfred_fallback_series.txt")
    original_exists = Path.exists

    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.refresh_fred_if_stale",
        lambda **_: (False, None),
    )
    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.refresh_alfred_if_stale",
        lambda **_: (False, None),
    )

    def _patched_exists(path: Path) -> bool:
        if path == fallback_path:
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", _patched_exists)

    result = ensure_macro_ready(
        fred_path=fred_path,
        vintage_dir=vintage_dir,
        max_age=timedelta(hours=2),
        series_ids=("CPI",),
    )

    assert result.fred_error is None
    assert result.alfred_error is None


def test_ensure_macro_ready_reads_fallback_series_file_and_skips_comments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fred_path = tmp_path / "macro" / "fred.parquet"
    vintage_dir = tmp_path / "vintages"
    fallback_path = Path("ml/config/macro_alfred_fallback_series.txt")
    original_exists = Path.exists
    original_read_text = Path.read_text

    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.refresh_fred_if_stale",
        lambda **_: (False, None),
    )

    build_calls: list[dict[str, Any]] = []

    class _Loader:
        def refresh(self) -> None:
            build_calls.append({"refreshed": True})

    def _build(
        series_ids: tuple[str, ...],
        realtime_start: str | None,
        realtime_end: str | None,
        window_days: int,
        fallback_series: tuple[str, ...] | None = None,
    ) -> _Loader:
        build_calls.append(
            {
                "series_ids": series_ids,
                "start": realtime_start,
                "end": realtime_end,
                "window_days": window_days,
                "fallback_series": fallback_series,
            },
        )
        return _Loader()

    def _patched_exists(path: Path) -> bool:
        if path == fallback_path:
            return True
        return original_exists(path)

    def _patched_read_text(path: Path, encoding: str = "utf-8") -> str:
        if path == fallback_path:
            return "# ignore\n\nBAMLX\nBAMLY\n"
        return original_read_text(path, encoding=encoding)

    monkeypatch.setattr(macro, "_build_alfred_loader", _build)
    monkeypatch.setattr(Path, "exists", _patched_exists)
    monkeypatch.setattr(Path, "read_text", _patched_read_text)

    result = ensure_macro_ready(
        fred_path=fred_path,
        vintage_dir=vintage_dir,
        max_age=timedelta(seconds=0),
        series_ids=("CPI",),
        alfred_realtime_start="2020-01-01",
        alfred_realtime_end="2020-12-31",
        alfred_window_days=45,
    )

    assert result.alfred_refreshed is True
    assert result.alfred_error is None
    build_call = build_calls[0]
    assert build_call["series_ids"] == ("CPI",)
    assert build_call["fallback_series"] == ["BAMLX", "BAMLY"]


def test_ingest_macro_observations_handles_empty_drop_null_and_invalid_series(
    tmp_path: Path,
) -> None:
    class _Store:
        def write_ingestion(
            self,
            dataset_id: str,
            records: object,
            source: str,
            run_id: str,
            instrument_id: str | None = None,
        ) -> None:
            raise AssertionError("write_ingestion should not be called")

    empty_path = tmp_path / "empty.parquet"
    pl.DataFrame(schema={"timestamp": pl.Datetime, "series_id": pl.String, "value": pl.Float64}).write_parquet(
        empty_path,
    )
    assert macro._ingest_macro_observations(
        data_store=_Store(),
        fred_path=empty_path,
        run_id="run-empty",
        series_ids=None,
    ) == 0

    null_path = tmp_path / "null_ts.parquet"
    pl.DataFrame(
        {
            "timestamp": [None],
            "series_id": ["CPI"],
            "value": [1.0],
        },
    ).write_parquet(null_path)
    assert macro._ingest_macro_observations(
        data_store=_Store(),
        fred_path=null_path,
        run_id="run-null",
        series_ids=None,
    ) == 0

    invalid_series_path = tmp_path / "invalid_series.parquet"
    pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)],
            "series_id": [""],
            "value": [1.0],
        },
    ).write_parquet(invalid_series_path)
    assert macro._ingest_macro_observations(
        data_store=_Store(),
        fred_path=invalid_series_path,
        run_id="run-invalid",
        series_ids=None,
    ) == 0


def test_ingest_macro_observations_watermark_start_none_and_filtered_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    class _Store:
        def write_ingestion(
            self,
            dataset_id: str,
            records: object,
            source: str,
            run_id: str,
            instrument_id: str | None = None,
        ) -> None:
            calls.append(getattr(records, "height", 0))

    fred_path = tmp_path / "watermark.parquet"
    pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "series_id": ["CPI", "CPI"],
            "value": [1.0, 2.0],
        },
    ).write_parquet(fred_path)

    def _start_none(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(start=None)

    monkeypatch.setattr(macro, "resolve_watermark_start_datetime", _start_none)
    written_without_filter = macro._ingest_macro_observations(
        data_store=_Store(),
        fred_path=fred_path,
        run_id="run-keep",
        series_ids=None,
        watermark_registry=SimpleNamespace(),
        watermark_config=WatermarkWindowConfig(use_watermark=True, lookback_days=7),
    )
    assert written_without_filter == 2
    assert calls == [2]

    def _future_start(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(start=datetime(2040, 1, 1, tzinfo=UTC))

    monkeypatch.setattr(macro, "resolve_watermark_start_datetime", _future_start)
    written_filtered = macro._ingest_macro_observations(
        data_store=_Store(),
        fred_path=fred_path,
        run_id="run-filtered",
        series_ids=None,
        watermark_registry=SimpleNamespace(),
        watermark_config=WatermarkWindowConfig(use_watermark=True, lookback_days=7),
    )
    assert written_filtered == 0
    assert calls == [2]


def test_ingest_macro_release_calendar_iterdir_missing_empty_drop_null_and_invalid_series(
    tmp_path: Path,
) -> None:
    class _Store:
        def write_ingestion(
            self,
            dataset_id: str,
            records: object,
            source: str,
            run_id: str,
            instrument_id: str | None = None,
        ) -> None:
            raise AssertionError("write_ingestion should not be called")

    vintage_dir = tmp_path / "vintages"
    (vintage_dir / "missing_release").mkdir(parents=True, exist_ok=True)

    empty_dir = vintage_dir / "empty_release"
    empty_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        schema={
            "series_id": pl.String,
            "observation_ts": pl.Datetime,
            "release_ts": pl.Datetime,
            "release_end_ts": pl.Datetime,
            "value": pl.Float64,
        },
    ).write_parquet(empty_dir / "release_calendar.parquet")

    drop_null_dir = vintage_dir / "drop_null"
    drop_null_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "series_id": ["CPI"],
            "observation_ts": [None],
            "release_ts": [datetime(2024, 1, 1)],
            "release_end_ts": [datetime(2024, 1, 2)],
            "value": [1.0],
        },
    ).write_parquet(drop_null_dir / "release_calendar.parquet")

    invalid_dir = vintage_dir / "invalid_series"
    invalid_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "series_id": [""],
            "observation_ts": [datetime(2024, 1, 1)],
            "release_ts": [datetime(2024, 1, 2)],
            "release_end_ts": [datetime(2024, 1, 3)],
            "value": [2.0],
        },
    ).write_parquet(invalid_dir / "release_calendar.parquet")

    written = macro._ingest_macro_release_calendar(
        data_store=_Store(),
        vintage_dir=vintage_dir,
        run_id="release-skip",
        series_ids=None,
    )

    assert written == 0


def test_ingest_macro_release_calendar_watermark_start_none_and_filtered_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_heights: list[int] = []

    class _Store:
        def write_ingestion(
            self,
            dataset_id: str,
            records: object,
            source: str,
            run_id: str,
            instrument_id: str | None = None,
        ) -> None:
            call_heights.append(getattr(records, "height", 0))

    vintage_dir = tmp_path / "vintages"
    series_dir = vintage_dir / "CPI"
    series_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "series_id": ["CPI", "CPI"],
            "observation_ts": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "release_ts": [datetime(2024, 1, 3), datetime(2024, 1, 4)],
            "release_end_ts": [datetime(2024, 1, 5), datetime(2024, 1, 6)],
            "value": [1.0, 2.0],
        },
    ).write_parquet(series_dir / "release_calendar.parquet")

    def _start_none(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(start=None)

    monkeypatch.setattr(macro, "resolve_watermark_start_datetime", _start_none)
    unfiltered = macro._ingest_macro_release_calendar(
        data_store=_Store(),
        vintage_dir=vintage_dir,
        run_id="release-run-a",
        series_ids=("CPI",),
        watermark_registry=SimpleNamespace(),
        watermark_config=WatermarkWindowConfig(use_watermark=True, lookback_days=7),
    )
    assert unfiltered == 2
    assert call_heights == [2]

    def _future_start(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(start=datetime(2040, 1, 1, tzinfo=UTC))

    monkeypatch.setattr(macro, "resolve_watermark_start_datetime", _future_start)
    filtered = macro._ingest_macro_release_calendar(
        data_store=_Store(),
        vintage_dir=vintage_dir,
        run_id="release-run-b",
        series_ids=("CPI",),
        watermark_registry=SimpleNamespace(),
        watermark_config=WatermarkWindowConfig(use_watermark=True, lookback_days=7),
    )
    assert filtered == 0
    assert call_heights == [2]
