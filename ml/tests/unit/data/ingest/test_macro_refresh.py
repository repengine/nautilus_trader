"""Tests for :mod:`ml.data.ingest.macro_refresh`."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ml.data.ingest.macro_refresh import MacroRefreshResult
from ml.data.ingest.macro_refresh import ensure_macro_ready
from ml.data.ingest.macro_refresh import refresh_alfred_if_stale
from ml.data.ingest.macro_refresh import refresh_fred_if_stale

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
