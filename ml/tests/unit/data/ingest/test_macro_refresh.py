"""Tests for :mod:`ml.data.ingest.macro_refresh`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from ml.config.dataset_ids import MACRO_OBSERVATIONS_DATASET_ID
from ml.config.dataset_ids import MACRO_RELEASES_DATASET_ID
from ml.config.events import Source
from ml.config.ingestion_windows import WatermarkWindowConfig
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
