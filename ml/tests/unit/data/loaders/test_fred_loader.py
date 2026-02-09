"""Unit tests for :mod:`ml.data.loaders.fred_loader`."""

from __future__ import annotations

import json
import math
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import polars as pl
import pytest

from ml.data.loaders import fred_loader as fred_mod
from ml.data.loaders.fred_loader import FREDConfig
from ml.data.loaders.fred_loader import FREDDataLoader
from ml.data.loaders.fred_loader import FREDIndicator


class _FakeFredClient:
    """Small fredapi client stub backed by an in-memory response map."""

    def __init__(
        self,
        responses: dict[str, pd.Series],
        *,
        failures_before_success: int = 0,
        always_fail: bool = False,
    ) -> None:
        self._responses = responses
        self._failures_before_success = failures_before_success
        self._always_fail = always_fail
        self.calls: list[dict[str, object]] = []

    def get_series(
        self,
        series_id: str,
        observation_start: datetime | None = None,
        observation_end: datetime | None = None,
    ) -> pd.Series:
        self.calls.append(
            {
                "series_id": series_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
            },
        )
        if self._always_fail:
            raise RuntimeError("upstream-down")
        if self._failures_before_success > 0:
            self._failures_before_success -= 1
            raise RuntimeError("temporary")
        return self._responses.get(series_id, pd.Series(dtype=float))


class _RegistryStub:
    """Capture registry registrations."""

    def __init__(self) -> None:
        self.manifests: list[object] = []

    def register_dataset(self, manifest: object) -> None:
        self.manifests.append(manifest)


class _StoreStub:
    """Capture store ingestion calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
                "records": records,
                "source": source,
                "run_id": run_id,
                "instrument_id": instrument_id,
            },
        )


def _make_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    client: _FakeFredClient | None = None,
    indicators: list[FREDIndicator] | None = None,
    config: FREDConfig | None = None,
) -> FREDDataLoader:
    """Build a loader with fredapi patched to a local fake client."""
    active_client = client or _FakeFredClient({})
    monkeypatch.setattr(fred_mod, "HAS_FREDAPI", True)
    monkeypatch.setattr(fred_mod, "_fredapi", SimpleNamespace(Fred=lambda api_key: active_client))
    cfg = config or FREDConfig(
        api_key="dummy",
        cache_dir=tmp_path / "cache",
        cache_ttl_hours=24,
        max_retries=3,
        retry_delay_seconds=0.0,
    )
    return FREDDataLoader(config=cfg, indicators=indicators)


def _sample_indicator_series() -> pd.Series:
    """Return a deterministic daily series with one null value."""
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    return pd.Series([1.0, None, 3.0], index=index)


def test_fred_config_reads_env_and_creates_cache_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "env-key")
    cfg = FREDConfig(api_key=None, cache_dir=tmp_path / "cache")

    assert cfg.api_key == "env-key"
    assert cfg.cache_dir.exists()


def test_fred_config_requires_api_key_when_missing_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(ValueError, match="FRED API key not provided"):
        FREDConfig(api_key=None, cache_dir=tmp_path / "cache")


def test_loader_init_raises_when_fredapi_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fred_mod, "HAS_FREDAPI", False)
    monkeypatch.setattr(fred_mod, "_fredapi", None)

    with pytest.raises(ImportError, match="fredapi package required"):
        FREDDataLoader(config=FREDConfig(api_key="dummy", cache_dir=tmp_path / "cache"))


def test_loader_init_and_cache_path_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)

    assert loader._get_cache_path("CPIAUCSL") == loader.config.cache_dir / "CPIAUCSL.parquet"
    assert loader._get_cache_metadata_path("CPIAUCSL") == (
        loader.config.cache_dir / "CPIAUCSL_metadata.json"
    )


def test_rate_limit_wait_and_reset_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    loader.config = FREDConfig(
        api_key="dummy",
        cache_dir=tmp_path / "cache2",
        rate_limit_calls=1,
        max_retries=1,
        retry_delay_seconds=0.0,
    )

    sleeps: list[float] = []
    time_values = iter([120.0, 180.0, 300.0])
    monkeypatch.setattr(fred_mod.time, "time", lambda: next(time_values))
    monkeypatch.setattr(fred_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    loader._last_call_time = 100.0
    loader._call_count = loader.config.rate_limit_calls
    loader._rate_limit()  # waits because still inside window

    loader._last_call_time = 100.0
    loader._call_count = loader.config.rate_limit_calls
    loader._rate_limit()  # resets because window elapsed

    assert sleeps and sleeps[0] > 0


def test_cache_validity_and_roundtrip_save_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    series_id = "CPIAUCSL"

    assert not loader._is_cache_valid(series_id)

    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "series_id": [series_id, series_id],
            "value": [1.0, 2.0],
            "timestamp_ns": [1, 2],
        },
    )
    loader._save_to_cache(series_id, frame)

    assert loader._is_cache_valid(series_id)
    loaded = loader._load_from_cache(series_id)
    assert loaded is not None
    assert loaded.height == 2


def test_cache_validity_handles_invalid_metadata_and_load_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    series_id = "CPIAUCSL"
    cache_path = loader._get_cache_path(series_id)
    metadata_path = loader._get_cache_metadata_path(series_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("broken parquet", encoding="utf-8")
    metadata_path.write_text("{bad json", encoding="utf-8")

    assert not loader._is_cache_valid(series_id)

    metadata_path.write_text(json.dumps({"timestamp": 0}), encoding="utf-8")
    assert loader._load_from_cache(series_id) is None


def test_cache_validity_handles_non_mapping_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    series_id = "CPIAUCSL"
    cache_path = loader._get_cache_path(series_id)
    metadata_path = loader._get_cache_metadata_path(series_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"PAR1")
    metadata_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    assert loader._is_cache_valid(series_id) is False


def test_load_from_cache_handles_parquet_read_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    series_id = "CPIAUCSL"
    cache_path = loader._get_cache_path(series_id)
    metadata_path = loader._get_cache_metadata_path(series_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("not a parquet", encoding="utf-8")
    metadata_path.write_text(json.dumps({"timestamp": datetime.now().timestamp()}), encoding="utf-8")

    assert loader._load_from_cache(series_id) is None


def test_save_to_cache_swallows_write_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)],
            "series_id": ["CPI"],
            "value": [1.0],
            "timestamp_ns": [1],
        },
    )

    def _raise(_path: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(frame, "write_parquet", _raise)
    loader._save_to_cache("CPI", frame)


def test_fetch_indicator_uses_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    cached = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "series_id": ["CPI"], "value": [1.0]})
    monkeypatch.setattr(loader, "_load_from_cache", lambda _series_id: cached)

    frame = loader.fetch_indicator("CPI", use_cache=True)

    assert frame.equals(cached)


def test_fetch_indicator_retries_filters_nulls_and_saves_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indicator = FREDIndicator(series_id="CPI", name="cpi", category="macro")
    client = _FakeFredClient({"CPI": _sample_indicator_series()}, failures_before_success=1)
    loader = _make_loader(
        tmp_path,
        monkeypatch,
        client=client,
        indicators=[indicator],
        config=FREDConfig(
            api_key="dummy",
            cache_dir=tmp_path / "cache",
            max_retries=3,
            retry_delay_seconds=0.0,
        ),
    )

    def _retry(
        fn: Any,
        *,
        max_attempts: int,
        initial_delay: float,
        multiplier: float,
        max_delay: float,
        jitter: float,
        sleep_fn: Any,
        on_exception: Any,
    ) -> object:
        del initial_delay, multiplier, max_delay, jitter, sleep_fn
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - branch under test
                on_exception(attempt, exc)
                attempt += 1
                if attempt >= max_attempts:
                    raise

    monkeypatch.setattr("ml.common.retry_utils.retry_with_backoff", _retry)

    frame = loader.fetch_indicator("CPI", use_cache=True)

    assert frame.height == 3
    assert frame.columns == ["timestamp", "series_id", "value", "timestamp_ns"]
    assert frame["value"].null_count() == 0
    assert math.isnan(frame["value"].to_list()[1])
    assert loader._get_cache_path("CPI").exists()


def test_fetch_indicator_with_explicit_dates_and_no_cache_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeFredClient({"CPI": _sample_indicator_series()})
    loader = _make_loader(
        tmp_path,
        monkeypatch,
        client=client,
        config=FREDConfig(
            api_key="dummy",
            cache_dir=tmp_path / "cache",
            max_retries=1,
            retry_delay_seconds=0.0,
        ),
    )
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 5)
    frame = loader.fetch_indicator(
        "CPI",
        start_date=start,
        end_date=end,
        use_cache=False,
    )

    assert frame.height == 3
    assert client.calls[-1]["observation_start"] == start
    assert client.calls[-1]["observation_end"] == end
    assert not loader._get_cache_path("CPI").exists()


def test_fetch_indicator_raises_after_retries_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeFredClient({}, always_fail=True)
    loader = _make_loader(
        tmp_path,
        monkeypatch,
        client=client,
        config=FREDConfig(
            api_key="dummy",
            cache_dir=tmp_path / "cache",
            max_retries=2,
            retry_delay_seconds=0.0,
        ),
    )

    def _retry_raise(
        fn: Any,
        *,
        max_attempts: int,
        initial_delay: float,
        multiplier: float,
        max_delay: float,
        jitter: float,
        sleep_fn: Any,
        on_exception: Any,
    ) -> object:
        del initial_delay, multiplier, max_delay, jitter, sleep_fn
        attempt = 0
        while attempt < max_attempts:
            try:
                return fn()
            except Exception as exc:
                on_exception(attempt, exc)
                attempt += 1
        raise RuntimeError("still failing")

    monkeypatch.setattr("ml.common.retry_utils.retry_with_backoff", _retry_raise)

    with pytest.raises(RuntimeError, match="Failed to fetch CPI"):
        loader.fetch_indicator("CPI", use_cache=False)


def test_fetch_all_indicators_continues_when_one_series_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indicators = [
        FREDIndicator(series_id="GOOD", name="good", category="macro"),
        FREDIndicator(series_id="BAD", name="bad", category="macro"),
    ]
    loader = _make_loader(tmp_path, monkeypatch, indicators=indicators)

    def _fetch(
        series_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        use_cache: bool = True,
    ) -> pl.DataFrame:
        del start_date, end_date, use_cache
        if series_id == "BAD":
            raise RuntimeError("boom")
        return pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "series_id": [series_id], "value": [1.0]})

    monkeypatch.setattr(loader, "fetch_indicator", _fetch)
    data = loader.fetch_all_indicators()

    assert set(data.keys()) == {"GOOD"}


def test_combine_indicators_handles_empty_and_joined_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    empty = loader.combine_indicators({})
    assert empty.is_empty()

    left = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "value": [1.0, 2.0],
        },
    )
    right = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "value": [10.0, 20.0],
        },
    )
    combined = loader.combine_indicators({"L": left, "R": right})

    assert "timestamp_ns" in combined.columns
    assert combined.height == 3
    assert combined["timestamp"].to_list() == [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ]


def test_store_indicators_handles_no_data_and_empty_combined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    store = _StoreStub()
    registry = _RegistryStub()

    monkeypatch.setattr(loader, "fetch_all_indicators", dict)
    loader.store_indicators(store, registry, data=None)
    assert registry.manifests == []
    assert store.calls == []

    monkeypatch.setattr(
        loader,
        "combine_indicators",
        lambda _data: pl.DataFrame(),
    )
    loader.store_indicators(
        store,
        registry,
        data={"CPI": pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1.0]})},
    )
    assert registry.manifests == []
    assert store.calls == []


def test_store_indicators_registers_manifest_and_writes_non_nan_series(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    store = _StoreStub()
    registry = _RegistryStub()

    combined = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "timestamp_ns": [1_704_067_200_000_000_000, 1_704_153_600_000_000_000],
            "CPI": [1.0, 2.0],
            "ALL_NAN": [math.nan, math.nan],
        },
    )
    monkeypatch.setattr(loader, "combine_indicators", lambda _data: combined)

    loader.store_indicators(
        store,
        registry,
        data={"CPI": pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1.0]})},
    )

    assert len(registry.manifests) == 1
    assert len(store.calls) == 1
    assert store.calls[0]["dataset_id"] == "fred_economic_indicators"
    assert store.calls[0]["source"] == "fred"
    assert str(store.calls[0]["instrument_id"]) == "FRED.CPI"


def test_update_realtime_delegates_to_fetch_and_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    store = _StoreStub()
    registry = _RegistryStub()
    captured: dict[str, object] = {}
    expected = {"CPI": pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1.0]})}

    def _fetch(
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        use_cache: bool = True,
    ) -> dict[str, pl.DataFrame]:
        captured["start"] = start_date
        captured["end"] = end_date
        captured["use_cache"] = use_cache
        return expected

    def _store(
        data_store: object,
        data_registry: object,
        data: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        captured["store"] = (data_store, data_registry, data)

    monkeypatch.setattr(loader, "fetch_all_indicators", _fetch)
    monkeypatch.setattr(loader, "store_indicators", _store)

    loader.update_realtime(store, registry)

    assert captured["use_cache"] is False
    assert isinstance(captured["start"], datetime)
    assert isinstance(captured["end"], datetime)
    assert captured["store"] == (store, registry, expected)


def test_export_ml_parquet_handles_empty_and_timestamp_ns_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)
    output_empty = tmp_path / "empty.parquet"
    path_empty = loader.export_ml_parquet(data={}, out_path=output_empty)
    assert path_empty == output_empty
    assert output_empty.exists()
    assert pl.read_parquet(output_empty).columns == ["timestamp", "series_id", "value"]

    ts_ns = [1_704_067_200_000_000_000, 1_704_153_600_000_000_000]
    data = {
        "CPI": pl.DataFrame({"timestamp_ns": ts_ns, "value": [1.0, 2.0]}),
        "SKIP": pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "other": [5.0]}),
    }
    output_non_empty = tmp_path / "non_empty.parquet"
    path = loader.export_ml_parquet(data=data, out_path=output_non_empty)

    assert path == output_non_empty
    exported = pl.read_parquet(output_non_empty)
    assert exported.height == 2
    assert set(exported["series_id"].to_list()) == {"CPI"}
    assert exported["value"].to_list() == [1.0, 2.0]


def test_export_ml_parquet_fetches_when_data_none_and_skips_empty_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(tmp_path, monkeypatch)

    monkeypatch.setattr(
        loader,
        "fetch_all_indicators",
        lambda use_cache=True: {
            "EMPTY": pl.DataFrame(),
            "CPI": pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1)],
                    "value": [1.0],
                },
            ),
        },
    )
    out_path = tmp_path / "from_none.parquet"
    written = loader.export_ml_parquet(data=None, out_path=out_path)

    assert written == out_path
    frame = pl.read_parquet(out_path)
    assert frame.height == 1
    assert frame["series_id"].to_list() == ["CPI"]
