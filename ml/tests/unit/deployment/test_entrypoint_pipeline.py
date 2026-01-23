from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID
from ml.config.dataset_ids import EVENTS_CALENDAR_DATASET_ID
from ml.config.dataset_ids import FEATURE_VALUES_DATASET_ID
from ml.config.dataset_ids import L2_MINUTE_DATASET_ID
from ml.config.dataset_ids import MACRO_OBSERVATIONS_DATASET_ID
from ml.config.dataset_ids import MACRO_RELEASES_DATASET_ID
from ml.config.dataset_ids import MICRO_MINUTE_DATASET_ID
from ml.config.dataset_coverage import CoverageDatasetEntry
from ml.config.market_data import MarketDatasetInput
from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.data.coverage.feature_restorer import SUPPORTED_FEATURE_DATASET_IDS
from ml.data.coverage.manager import BucketClassification
from ml.data.coverage.manager import BucketSpec
from ml.data.coverage.manager import DatasetCoverageConfig
from ml.data.rehydration import CatalogRehydrationConfig
from ml.deployment import entrypoint_pipeline
from ml.stores.migrations_runner import SchemaHealthCheckError
from ml.stores.providers import ParquetCoverageSpec

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
    "valid_env",
)


class _DummyScheduler:
    def __init__(
        self,
        *,
        catalog: Any,
        config: Any,
        use_orchestrator: bool,
        dual_write: bool,
    ) -> None:
        self.config = config
        self.run_count = 0
        self.targeted_calls: list[list[Any]] = []

    def run_daily_update(self) -> None:
        self.run_count += 1

    def stop(self) -> None:
        return None

    def run_targeted_update(self, buckets: Any) -> None:
        self.targeted_calls.append(list(buckets))


def _reset_pipeline_status() -> None:
    entrypoint_pipeline.pipeline_status.update(
        {
            "healthy": False,
            "last_run": None,
            "errors": [],
            "last_rehydrate": None,
        },
    )
    entrypoint_pipeline.pipeline_status["coverage"] = entrypoint_pipeline._default_coverage_status()


@pytest.fixture(autouse=True)
def _reset_status() -> None:
    _reset_pipeline_status()


def test_build_catalog_rehydrator_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    catalog = object()
    config = SchedulerConfig()
    instantiations = {"count": 0}

    class _TrackingRehydrator:
        def __init__(self, **_: Any) -> None:
            instantiations["count"] += 1

    monkeypatch.setattr(entrypoint_pipeline, "ParquetCatalogRehydrator", _TrackingRehydrator)
    monkeypatch.setattr(
        runner,
        "_build_catalog_rehydration_config",
        lambda: CatalogRehydrationConfig(enabled=False),
    )

    runner._build_catalog_rehydrator(catalog=catalog, scheduler_config=config)

    assert runner._rehydrator is None
    assert runner._rehydrator_config is not None
    assert runner._rehydrator_config.enabled is False
    assert instantiations["count"] == 0


def test_build_catalog_rehydrator_missing_db_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    config = SchedulerConfig()

    monkeypatch.setattr(
        runner,
        "_build_catalog_rehydration_config",
        lambda: CatalogRehydrationConfig(enabled=True),
    )
    monkeypatch.setattr(runner, "_resolve_db_connection", lambda _: None)

    def _raise_if_called(**_: Any) -> None:
        raise AssertionError("rehydrator should not instantiate")

    monkeypatch.setattr(entrypoint_pipeline, "ParquetCatalogRehydrator", _raise_if_called)

    runner._build_catalog_rehydrator(catalog=object(), scheduler_config=config)

    assert runner._rehydrator is None
    assert runner._rehydrator_config is None


def test_build_catalog_rehydrator_enabled_initializes_rehydrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    catalog = object()
    config = SchedulerConfig()
    captured: dict[str, Any] = {}

    class _RecordingRehydrator:
        def __init__(
            self,
            *,
            catalog: Any,
            db_connection: str,
            config: CatalogRehydrationConfig,
            registry: Any | None = None,
        ) -> None:
            captured["catalog"] = catalog
            captured["db_connection"] = db_connection
            captured["config"] = config
            captured["registry"] = registry
            captured["instance"] = self

    monkeypatch.setattr(entrypoint_pipeline, "ParquetCatalogRehydrator", _RecordingRehydrator)
    rehydration_config = CatalogRehydrationConfig(enabled=True, lookback_days=7)
    monkeypatch.setattr(runner, "_build_catalog_rehydration_config", lambda: rehydration_config)
    monkeypatch.setattr(runner, "_resolve_db_connection", lambda _: "postgresql://example")

    runner._build_catalog_rehydrator(catalog=catalog, scheduler_config=config)

    assert runner._rehydrator is captured["instance"]
    assert runner._rehydrator_config is rehydration_config
    assert captured["catalog"] is catalog
    assert captured["db_connection"] == "postgresql://example"


def test_select_rehydration_instruments_filters_fresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    db_path = tmp_path / "rehydrate.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE market_data (
                instrument_id TEXT NOT NULL,
                ts_event BIGINT NOT NULL,
                ts_init BIGINT NOT NULL
            )
            """,
        )
        now_ns = int(datetime.now(tz=UTC).timestamp() * 1_000_000_000)
        conn.executemany(
            "INSERT INTO market_data (instrument_id, ts_event, ts_init) VALUES (?, ?, ?)",
            [
                ("FRESH.EQUS", now_ns, now_ns),
                ("STALE.EQUS", now_ns - (48 * 3_600 * 1_000_000_000), now_ns),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    runner._rehydrator_config = CatalogRehydrationConfig(enabled=True)
    monkeypatch.setattr(
        runner,
        "_resolve_db_connection",
        lambda _: f"sqlite:///{db_path}",
    )
    monkeypatch.setenv("CATALOG_REHYDRATE_STALENESS_HOURS", "6")
    config = SchedulerConfig(
        symbols=["FRESH", "STALE"],
        databento=DatabentoConfig(dataset="EQUS.MINI"),
    )

    selected = runner._select_rehydration_instruments(config)

    assert "STALE.EQUS" in selected
    assert "FRESH.EQUS" not in selected


def test_select_rehydration_instruments_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    runner._rehydrator_config = CatalogRehydrationConfig(enabled=True)
    config = SchedulerConfig(
        symbols=["AAPL"],
        databento=DatabentoConfig(dataset="EQUS.MINI"),
    )
    monkeypatch.setenv("CATALOG_REHYDRATE_STALE_ONLY", "0")

    selected = runner._select_rehydration_instruments(config)

    assert selected == ["AAPL.EQUS"]


def test_load_feature_coverage_entries_uses_env_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    isolated_prometheus_registry: Any,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    manifest = tmp_path / "coverage.toml"
    manifest.write_text(
        """
[[datasets]]
dataset_id = "ml.macro_release_calendar"
schema = "macro_release_calendar"
entities = "CPIAUCSL"
entity_field = "series_id"
[datasets.parquet]
path = "data/features/macro/fred/vintages"
partition_field = "series_id"
partition_template = "{value}/release_calendar.parquet"
timestamp_field = "release_ts"
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("COVERAGE_DATASETS_FILE", str(manifest))

    entries = runner._load_feature_coverage_entries()

    assert entries
    assert entries[0].dataset.dataset_id == MACRO_RELEASES_DATASET_ID
    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "nautilus_ml_coverage_manifest_events_total",
        labels={"event": "loaded"},
    )
    assert metric_value is not None and metric_value >= 1.0


def test_load_feature_coverage_entries_handles_missing_manifest(
    monkeypatch: pytest.MonkeyPatch,
    isolated_prometheus_registry: Any,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    monkeypatch.setenv("COVERAGE_DATASETS_FILE", "does-not-exist.toml")

    entries = runner._load_feature_coverage_entries()

    assert entries == tuple()
    assert entrypoint_pipeline.pipeline_status["errors"]
    assert "feature_manifest_missing" in entrypoint_pipeline.pipeline_status["errors"][0]
    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "nautilus_ml_coverage_manifest_events_total",
        labels={"event": "missing"},
    )
    assert metric_value is not None and metric_value >= 1.0


def test_load_feature_coverage_entries_handles_invalid_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    isolated_prometheus_registry: Any,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    manifest = tmp_path / "invalid.toml"
    manifest.write_text("this is not toml", encoding="utf-8")
    monkeypatch.setenv("COVERAGE_DATASETS_FILE", str(manifest))

    entries = runner._load_feature_coverage_entries()

    assert entries == tuple()
    assert entrypoint_pipeline.pipeline_status["errors"]
    assert "feature_manifest_invalid" in entrypoint_pipeline.pipeline_status["errors"][0]
    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "nautilus_ml_coverage_manifest_events_total",
        labels={"event": "invalid"},
    )
    assert metric_value is not None and metric_value >= 1.0


def test_load_feature_coverage_entries_requires_default_manifest_when_coverage_enabled(
    monkeypatch: pytest.MonkeyPatch,
    isolated_prometheus_registry: Any,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    monkeypatch.delenv("COVERAGE_DATASETS_FILE", raising=False)
    monkeypatch.setenv("COVERAGE_RESTORE_ENABLED", "1")

    original_exists = Path.exists

    def _fake_exists(self: Path) -> bool:  # type: ignore[override]
        if str(self).endswith("ml/config/coverage_datasets_tier1.toml"):
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", _fake_exists)

    entries = runner._load_feature_coverage_entries()

    assert entries == tuple()
    assert entrypoint_pipeline.pipeline_status["errors"]
    assert "feature_manifest_missing" in entrypoint_pipeline.pipeline_status["errors"][0]
    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "nautilus_ml_coverage_manifest_events_total",
        labels={"event": "missing"},
    )
    assert metric_value is not None and metric_value >= 1.0


def test_load_feature_coverage_entries_includes_feature_datasets(
    monkeypatch: pytest.MonkeyPatch,
    isolated_prometheus_registry: Any,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    monkeypatch.delenv("COVERAGE_DATASETS_FILE", raising=False)

    entries = runner._load_feature_coverage_entries()

    dataset_ids = {entry.dataset.dataset_id for entry in entries}
    expected = {
        EARNINGS_ACTUALS_DATASET_ID,
        EARNINGS_ESTIMATES_DATASET_ID,
        FEATURE_VALUES_DATASET_ID,
        MACRO_RELEASES_DATASET_ID,
        MACRO_OBSERVATIONS_DATASET_ID,
        EVENTS_CALENDAR_DATASET_ID,
        MICRO_MINUTE_DATASET_ID,
        L2_MINUTE_DATASET_ID,
    }
    assert expected == SUPPORTED_FEATURE_DATASET_IDS
    assert expected.issubset(dataset_ids)
    for entry in entries:
        if entry.dataset.dataset_id in expected:
            assert entry.parquet_spec is not None
    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "nautilus_ml_coverage_manifest_events_total",
        labels={"event": "loaded"},
    )
    assert metric_value is not None and metric_value >= 1.0


def test_load_feature_coverage_entries_rejects_unsupported_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    isolated_prometheus_registry: Any,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    manifest = tmp_path / "unsupported.toml"
    manifest.write_text(
        """
[[datasets]]
dataset_id = "ml.unsupported"
schema = "unsupported_schema"
entities = "AAPL"
[datasets.parquet]
path = "data/unsupported.parquet"
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("COVERAGE_DATASETS_FILE", str(manifest))

    entries = runner._load_feature_coverage_entries()

    assert entries == tuple()
    errors = entrypoint_pipeline.pipeline_status["errors"]
    assert any("feature_manifest_invalid" in error for error in errors)
    coverage_status = entrypoint_pipeline.pipeline_status["coverage"]
    assert coverage_status["last_error"] == "feature_manifest_invalid"
    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "nautilus_ml_coverage_manifest_events_total",
        labels={"event": "invalid"},
    )
    assert metric_value is not None and metric_value >= 1.0


def test_create_config_parses_market_dataset_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    monkeypatch.setenv(
        "MARKET_DATASET_INPUTS",
        '[{"descriptor_id":"EQUS.MINI_TBBO","symbols":["SPY","QQQ"]},{"descriptor_id":"EQUS.MINI_MBP1"},{"descriptor_id":"XNAS.ITCH_MBP10","start":"2024-01-01"}]',
    )
    monkeypatch.setenv("MARKET_BACKFILL_LOOKBACK_DAYS", "5")
    config = runner._create_config()

    assert config.market_inputs is not None
    assert len(config.market_inputs) == 3
    first = config.market_inputs[0]
    assert first.descriptor_id == "EQUS.MINI_TBBO"
    assert first.symbols == ("SPY", "QQQ")
    second = config.market_inputs[1]
    assert second.descriptor_id == "EQUS.MINI_MBP1"
    third = config.market_inputs[2]
    assert third.descriptor_id == "XNAS.ITCH_MBP10"
    assert third.start == "2024-01-01"
    assert config.market_backfill_lookback_days == 5


def test_parse_market_dataset_inputs_rejects_unknown_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    monkeypatch.setenv("MARKET_DATASET_INPUTS", '["DBEQ.MINI"]')
    with pytest.raises(ValueError):
        runner._parse_market_dataset_inputs()


def test_parse_market_dataset_inputs_rejects_invalid_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    monkeypatch.setenv(
        "MARKET_DATASET_INPUTS",
        '[{"descriptor_id":"EQUS.MINI","schema_override":"mbp-10"}]',
    )
    with pytest.raises(ValueError):
        runner._parse_market_dataset_inputs()


def test_build_dataset_coverage_configs_resolves_descriptor_dataset() -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    config = SchedulerConfig(
        symbols=["SPY.XNAS"],
        databento=DatabentoConfig(dataset="EQUS.MINI", schema="ohlcv-1m"),
        market_inputs=(MarketDatasetInput(descriptor_id="EQUS.MINI_TBBO"),),
    )

    entries = runner._build_dataset_coverage_configs(config)

    assert len(entries) == 1
    dataset = entries[0].dataset
    assert dataset.dataset_id == "EQUS.MINI_TBBO"
    assert dataset.schema == "tbbo"
    assert dataset.instruments == ("SPY.XNAS",)


def test_run_coverage_restoration_routes_classifications(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    scheduler_config = SchedulerConfig(
        symbols=["AAPL.XNAS", "META.XNAS"],
        databento=DatabentoConfig(dataset="EQUS.MINI", schema="ohlcv-1m"),
        feature_store_connection="postgresql://feature",
    )
    runner.scheduler = _DummyScheduler(
        catalog=None,
        config=scheduler_config,
        use_orchestrator=False,
        dual_write=False,
    )
    runner._catalog_path = tmp_path
    monkeypatch.setenv("COVERAGE_RESTORE_ENABLED", "1")

    catalog_spec = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="META.XNAS",
        bucket_start_ns=1_700_000_000_000_000_000,
    )
    source_spec = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="AAPL.XNAS",
        bucket_start_ns=1_700_086_400_000_000_000,
    )

    manager_init: dict[str, Any] = {}

    class _RecordingCoverageManager:
        def __init__(
            self,
            *,
            config: Any,
            sql_provider: Any,
            catalog_provider: Any,
            schema_auditor: Any,
        ) -> None:
            manager_init["config"] = config
            manager_init["sql_provider"] = sql_provider
            manager_init["catalog_provider"] = catalog_provider
            manager_init["schema_auditor"] = schema_auditor

        def restore_all(self) -> tuple[BucketClassification, ...]:
            return (
                BucketClassification(spec=catalog_spec, has_sql=False, has_catalog=True),
                BucketClassification(spec=source_spec, has_sql=False, has_catalog=False),
            )

    class _RecordingSqlProvider:
        def __init__(
            self,
            *,
            connection_string: str,
            dataset_overrides: dict[str, object] | None = None,
        ) -> None:
            self.connection_string = connection_string
            self.dataset_overrides = dataset_overrides or {}

    class _RecordingCatalogProvider:
        def __init__(self, *, catalog_path: str, identifier_template: str | None = None) -> None:
            self.catalog_path = catalog_path
            self.identifier_template = identifier_template

    class _RecordingSchemaAuditor:
        def __init__(self, *, db_url: str) -> None:
            self.db_url = db_url

    restored_specs: list[list[BucketSpec]] = []

    def _record_restore(
        self: entrypoint_pipeline.PipelineRunner,
        specs: list[BucketSpec],
        scheduler_cfg: SchedulerConfig,
    ) -> None:
        assert scheduler_cfg is scheduler_config
        restored_specs.append(list(specs))

    monkeypatch.setattr(entrypoint_pipeline, "CoverageManager", _RecordingCoverageManager)
    monkeypatch.setattr("ml.stores.providers.SqlCoverageProvider", _RecordingSqlProvider)
    monkeypatch.setattr("ml.stores.providers.CatalogCoverageProvider", _RecordingCatalogProvider)
    monkeypatch.setattr(entrypoint_pipeline, "SchemaAuditor", _RecordingSchemaAuditor)
    monkeypatch.setattr(
        entrypoint_pipeline.PipelineRunner,
        "_restore_catalog_buckets",
        _record_restore,
    )

    runner._run_coverage_restoration(scheduler_config)

    assert restored_specs == [[catalog_spec]]
    assert runner.scheduler is not None
    assert runner.scheduler.targeted_calls == [[source_spec]]
    cfg = manager_init["config"]
    assert cfg.datasets
    dataset_cfg = cfg.datasets[0]
    assert dataset_cfg.dataset_id == scheduler_config.databento.dataset
    assert dataset_cfg.schema == scheduler_config.databento.schema
    assert dataset_cfg.instruments == tuple(scheduler_config.symbols)
    assert manager_init["sql_provider"].connection_string == "postgresql://feature"
    catalog_provider = manager_init["catalog_provider"]
    catalog_path = getattr(catalog_provider, "catalog_path", None)
    if catalog_path is None:
        providers = getattr(catalog_provider, "providers", ())
        catalog_backend = next(
            (provider for provider in providers if hasattr(provider, "catalog_path")),
            None,
        )
        assert catalog_backend is not None
        catalog_path = catalog_backend.catalog_path
    assert catalog_path == str(tmp_path)
    assert manager_init["schema_auditor"].db_url == "postgresql://feature"
    coverage = entrypoint_pipeline.pipeline_status["coverage"]
    assert coverage["buckets_total"] == 2
    assert coverage["buckets_restore_catalog"] == 1
    assert coverage["buckets_reingest_source"] == 1
    assert coverage["buckets_healthy"] == 0
    assert coverage["last_error"] is None
    assert coverage["last_success"] is not None


def test_run_coverage_restoration_skips_source_reingest_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    scheduler_config = SchedulerConfig(
        symbols=["AAPL.XNAS"],
        databento=DatabentoConfig(dataset="EQUS.MINI", schema="ohlcv-1m"),
        feature_store_connection="postgresql://feature",
    )
    runner.scheduler = _DummyScheduler(
        catalog=None,
        config=scheduler_config,
        use_orchestrator=False,
        dual_write=False,
    )
    runner._catalog_path = tmp_path
    monkeypatch.setenv("COVERAGE_RESTORE_ENABLED", "1")
    monkeypatch.setenv("COVERAGE_RESTORE_REINGEST_ENABLED", "0")

    source_spec = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="AAPL.XNAS",
        bucket_start_ns=1_700_086_400_000_000_000,
    )

    class _CoverageManagerStub:
        def __init__(
            self,
            *,
            config: Any,
            sql_provider: Any,
            catalog_provider: Any,
            schema_auditor: Any,
        ) -> None:
            return

        def restore_all(self) -> tuple[BucketClassification, ...]:
            return (BucketClassification(spec=source_spec, has_sql=False, has_catalog=False),)

    monkeypatch.setattr(entrypoint_pipeline, "CoverageManager", _CoverageManagerStub)
    monkeypatch.setattr("ml.stores.providers.SqlCoverageProvider", lambda **_: object())
    monkeypatch.setattr("ml.stores.providers.CatalogCoverageProvider", lambda **_: object())
    monkeypatch.setattr(entrypoint_pipeline, "SchemaAuditor", lambda **_: object())

    runner._run_coverage_restoration(scheduler_config)

    assert runner.scheduler is not None
    assert runner.scheduler.targeted_calls == []


def test_run_coverage_restoration_dry_run_skips_restores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    scheduler_config = SchedulerConfig(
        symbols=["SPY.XNAS"],
        databento=DatabentoConfig(dataset="EQUS.MINI", schema="ohlcv-1m"),
        feature_store_connection="postgresql://feature",
    )
    runner.scheduler = _DummyScheduler(
        catalog=None,
        config=scheduler_config,
        use_orchestrator=False,
        dual_write=False,
    )
    runner._catalog_path = tmp_path
    monkeypatch.setenv("COVERAGE_RESTORE_ENABLED", "1")

    feature_spec = BucketSpec(
        dataset_id=FEATURE_VALUES_DATASET_ID,
        schema="feature_values",
        instrument_id="SPY.XNAS",
        bucket_start_ns=1_700_000_000_000_000_000,
    )
    market_spec = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="SPY.XNAS",
        bucket_start_ns=1_700_086_400_000_000_000_000,
    )

    class _DryRunCoverageManager:
        def __init__(
            self,
            *,
            config: Any,
            sql_provider: Any,
            catalog_provider: Any,
            schema_auditor: Any,
        ) -> None:
            return

        def restore_all(self) -> tuple[BucketClassification, ...]:
            return (
                BucketClassification(spec=feature_spec, has_sql=False, has_catalog=True),
                BucketClassification(spec=market_spec, has_sql=False, has_catalog=False),
            )

    feature_entry = CoverageDatasetEntry(
        dataset=DatasetCoverageConfig(
            dataset_id=FEATURE_VALUES_DATASET_ID,
            schema="feature_values",
            instruments=("SPY.XNAS",),
        ),
        parquet_spec=ParquetCoverageSpec(
            dataset_id=FEATURE_VALUES_DATASET_ID,
            base_path=str(tmp_path / "feature_values"),
            partition_field="instrument_id",
            timestamp_field="ts_event",
            partition_template="{value}",
        ),
    )
    market_entry = CoverageDatasetEntry(
        dataset=DatasetCoverageConfig(
            dataset_id="EQUS.MINI",
            schema="ohlcv-1m",
            instruments=("SPY.XNAS",),
        ),
        parquet_spec=None,
    )

    monkeypatch.setattr(entrypoint_pipeline, "CoverageManager", _DryRunCoverageManager)
    monkeypatch.setattr("ml.stores.providers.SqlCoverageProvider", lambda **_: object())
    monkeypatch.setattr("ml.stores.providers.CatalogCoverageProvider", lambda **_: object())
    monkeypatch.setattr(entrypoint_pipeline, "SchemaAuditor", lambda **_: object())
    monkeypatch.setattr(
        entrypoint_pipeline.PipelineRunner,
        "_compose_coverage_entries",
        lambda self, _: (feature_entry, market_entry),
    )

    restored_features: list[list[BucketSpec]] = []
    restored_catalog: list[list[BucketSpec]] = []

    def _record_feature_restore(
        self: entrypoint_pipeline.PipelineRunner,
        *,
        specs: list[BucketSpec],
        scheduler_config: SchedulerConfig,
        parquet_specs: Any,
    ) -> None:
        restored_features.append(list(specs))

    def _record_catalog_restore(
        self: entrypoint_pipeline.PipelineRunner,
        specs: list[BucketSpec],
        scheduler_config: SchedulerConfig,
    ) -> None:
        restored_catalog.append(list(specs))

    monkeypatch.setattr(
        entrypoint_pipeline.PipelineRunner,
        "_restore_feature_buckets",
        _record_feature_restore,
    )
    monkeypatch.setattr(
        entrypoint_pipeline.PipelineRunner,
        "_restore_catalog_buckets",
        _record_catalog_restore,
    )

    runner._run_coverage_restoration(scheduler_config, dry_run=True)

    assert restored_features == []
    assert restored_catalog == []
    assert runner.scheduler is not None
    assert runner.scheduler.targeted_calls == []


def test_run_coverage_restoration_triggers_feature_reingest_when_bucket_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    scheduler_config = SchedulerConfig(
        symbols=["SPY.XNAS"],
        databento=DatabentoConfig(dataset="EQUS.MINI", schema="ohlcv-1m"),
        feature_store_connection="postgresql://feature",
    )
    runner.scheduler = _DummyScheduler(
        catalog=None,
        config=scheduler_config,
        use_orchestrator=False,
        dual_write=False,
    )
    runner._catalog_path = tmp_path
    monkeypatch.setenv("COVERAGE_RESTORE_ENABLED", "1")

    feature_spec = BucketSpec(
        dataset_id=FEATURE_VALUES_DATASET_ID,
        schema="feature_values",
        instrument_id="SPY.XNAS",
        bucket_start_ns=0,
    )

    class _CoverageManagerStub:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def restore_all(self) -> tuple[BucketClassification, ...]:
            return (BucketClassification(spec=feature_spec, has_sql=False, has_catalog=False),)

    feature_entry = CoverageDatasetEntry(
        dataset=DatasetCoverageConfig(
            dataset_id=FEATURE_VALUES_DATASET_ID,
            schema="feature_values",
            instruments=("SPY.XNAS",),
        ),
        parquet_spec=ParquetCoverageSpec(
            dataset_id=FEATURE_VALUES_DATASET_ID,
            base_path=str(tmp_path / "feature_values"),
            partition_field="instrument_id",
            timestamp_field="ts_event",
            partition_template="{value}",
        ),
    )

    monkeypatch.setattr(entrypoint_pipeline, "CoverageManager", _CoverageManagerStub)
    monkeypatch.setattr("ml.stores.providers.SqlCoverageProvider", lambda **_: object())
    monkeypatch.setattr("ml.stores.providers.CatalogCoverageProvider", lambda **_: object())
    monkeypatch.setattr(entrypoint_pipeline, "SchemaAuditor", lambda **_: object())
    monkeypatch.setattr(
        entrypoint_pipeline.PipelineRunner,
        "_compose_coverage_entries",
        lambda self, _: (feature_entry,),
    )

    reingest_calls: list[list[BucketSpec]] = []

    def _record_feature_reingest(
        self: entrypoint_pipeline.PipelineRunner,
        *,
        specs: list[BucketSpec],
        scheduler_config: SchedulerConfig,
    ) -> None:
        reingest_calls.append(list(specs))

    monkeypatch.setattr(
        entrypoint_pipeline.PipelineRunner,
        "_reingest_feature_buckets",
        _record_feature_reingest,
    )
    monkeypatch.setattr(
        entrypoint_pipeline.PipelineRunner,
        "_restore_catalog_buckets",
        lambda *args, **kwargs: None,
    )

    runner._run_coverage_restoration(scheduler_config)

    assert reingest_calls == [[feature_spec]]


def test_run_coverage_restoration_applies_bucket_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    scheduler_config = SchedulerConfig(
        symbols=["AAPL.XNAS", "MSFT.XNAS"],
        databento=DatabentoConfig(dataset="EQUS.MINI", schema="ohlcv-1m"),
        feature_store_connection="postgresql://feature",
    )
    runner.scheduler = _DummyScheduler(
        catalog=None,
        config=scheduler_config,
        use_orchestrator=False,
        dual_write=False,
    )
    runner._catalog_path = tmp_path
    monkeypatch.setenv("COVERAGE_RESTORE_ENABLED", "1")
    monkeypatch.setenv("COVERAGE_MAX_BUCKETS_PER_RUN", "1")

    catalog_spec_primary = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="AAPL.XNAS",
        bucket_start_ns=1,
    )
    catalog_spec_secondary = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="MSFT.XNAS",
        bucket_start_ns=2,
    )
    source_spec_primary = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="SPY.XNAS",
        bucket_start_ns=3,
    )

    class _CapCoverageManager:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return

        def restore_all(self) -> tuple[BucketClassification, ...]:
            return (
                BucketClassification(spec=catalog_spec_primary, has_sql=False, has_catalog=True),
                BucketClassification(spec=catalog_spec_secondary, has_sql=False, has_catalog=True),
                BucketClassification(spec=source_spec_primary, has_sql=False, has_catalog=False),
            )

    restored_specs: list[list[BucketSpec]] = []

    def _record_restore(
        self: entrypoint_pipeline.PipelineRunner,
        specs: list[BucketSpec],
        scheduler_cfg: SchedulerConfig,
    ) -> None:
        assert scheduler_cfg is scheduler_config
        restored_specs.append(list(specs))

    monkeypatch.setattr(entrypoint_pipeline, "CoverageManager", _CapCoverageManager)
    monkeypatch.setattr("ml.stores.providers.SqlCoverageProvider", lambda **_: object())
    monkeypatch.setattr("ml.stores.providers.CatalogCoverageProvider", lambda **_: object())
    monkeypatch.setattr(entrypoint_pipeline, "SchemaAuditor", lambda **_: object())
    monkeypatch.setattr(
        entrypoint_pipeline.PipelineRunner,
        "_restore_catalog_buckets",
        _record_restore,
    )

    runner._run_coverage_restoration(scheduler_config)

    assert restored_specs == [[catalog_spec_primary]]
    assert runner.scheduler is not None
    assert runner.scheduler.targeted_calls == []
    errors = entrypoint_pipeline.pipeline_status["errors"]
    assert errors
    assert any(entry.startswith("coverage_cap:") for entry in errors)


def test_run_coverage_restoration_raises_when_scheduler_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    scheduler_config = SchedulerConfig(
        symbols=["SPY.XNAS"],
        databento=DatabentoConfig(dataset="EQUS.MINI", schema="ohlcv-1m"),
        feature_store_connection="postgresql://feature",
    )
    monkeypatch.setenv("COVERAGE_RESTORE_ENABLED", "1")
    with pytest.raises(entrypoint_pipeline.CoverageRestorationError):
        runner._run_coverage_restoration(scheduler_config)


def test_run_coverage_restoration_once_initializes_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = SchedulerConfig(symbols=["AAPL.XNAS"])
    runner = entrypoint_pipeline.PipelineRunner()
    runner._create_config = lambda: config  # type: ignore[assignment]
    bootstrap_calls: list[SchedulerConfig] = []
    runner._bootstrap_database = lambda cfg: bootstrap_calls.append(cfg)  # type: ignore[assignment]

    catalog_obj = object()

    def _init_catalog(_: SchedulerConfig) -> object:
        runner._catalog_path = tmp_path
        return catalog_obj

    runner._initialize_catalog = _init_catalog  # type: ignore[assignment]

    rehydrator_calls = {"count": 0}
    runner._build_catalog_rehydrator = lambda *, catalog, scheduler_config: rehydrator_calls.update(
        count=rehydrator_calls["count"] + 1,
    )  # type: ignore[assignment]

    coverage_template = {
        "last_run": "2024-01-01T00:00:00Z",
        "last_success": "2024-01-01T00:00:00Z",
        "buckets_total": 1,
        "buckets_restore_catalog": 1,
        "buckets_reingest_source": 0,
        "buckets_healthy": 0,
        "last_error": None,
    }

    def _fake_run(
        self: entrypoint_pipeline.PipelineRunner,
        scheduler_cfg: SchedulerConfig,
        *,
        dry_run: bool = False,
    ) -> None:
        assert scheduler_cfg is config
        assert dry_run is False
        entrypoint_pipeline.pipeline_status["coverage"].update(coverage_template)

    monkeypatch.setattr(entrypoint_pipeline.PipelineRunner, "_run_coverage_restoration", _fake_run)

    scheduler_state: dict[str, object] = {}
    feature_engineer_stub = object()

    class _SchedulerStub:
        def __init__(
            self,
            *,
            catalog: object,
            config: SchedulerConfig,
            feature_engineer: object | None,
            use_orchestrator: bool,
            dual_write: bool,
            dual_write_dataset_types: Any = None,
            dataset_type_identifier_templates: Any = None,
        ) -> None:
            del dual_write_dataset_types
            del dataset_type_identifier_templates
            assert catalog is catalog_obj
            assert config is config
            assert feature_engineer is feature_engineer_stub
            scheduler_state["instance"] = self
            self.targeted_calls: list[list[object]] = []
            self.stop_calls = 0

        def run_targeted_update(self, buckets: Any) -> None:
            self.targeted_calls.append(list(buckets))

        def stop(self) -> None:
            self.stop_calls += 1

    monkeypatch.setattr(entrypoint_pipeline, "DataScheduler", _SchedulerStub)
    monkeypatch.setattr(
        entrypoint_pipeline.PipelineRunner,
        "_build_feature_engineer",
        lambda self: feature_engineer_stub,
    )
    monkeypatch.setenv("DB_CONNECTION", "postgresql://example")

    summary = runner.run_coverage_restoration_once()

    assert bootstrap_calls == [config]
    assert rehydrator_calls["count"] == 1
    scheduler_instance = scheduler_state["instance"]
    assert isinstance(scheduler_instance, _SchedulerStub)
    assert scheduler_instance.stop_calls == 1
    assert runner.scheduler is None
    assert summary == entrypoint_pipeline.pipeline_status["coverage"]
    assert summary["buckets_restore_catalog"] == 1


def test_run_backfill_logs_error_when_update_fails() -> None:
    runner = entrypoint_pipeline.PipelineRunner()

    class _FailingScheduler:
        def run_daily_update(self) -> None:
            raise RuntimeError("update failed")

    runner.scheduler = _FailingScheduler()

    runner._run_backfill()

    errors = entrypoint_pipeline.pipeline_status["errors"]
    assert errors
    assert any("update failed" in entry for entry in errors)


def test_bootstrap_database_verifies_instrumentation_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    config = SchedulerConfig()
    engine = object()
    summary = SimpleNamespace(
        applied_count=0,
        already_applied_count=0,
        profile=SimpleNamespace(value="auto"),
    )

    def _apply_profiled_migrations(*, db_url: str) -> SimpleNamespace:
        assert db_url == "postgresql://example"
        return summary

    monkeypatch.setattr(entrypoint_pipeline, "apply_profiled_migrations", _apply_profiled_migrations)
    monkeypatch.setattr(entrypoint_pipeline.EngineManager, "get_engine", lambda _: engine)
    monkeypatch.setattr(entrypoint_pipeline, "is_postgres_url", lambda _: True)
    monkeypatch.setattr(
        entrypoint_pipeline.PipelineRunner,
        "_resolve_db_connection",
        lambda self, cfg: "postgresql://example",
    )
    monkeypatch.setattr(
        entrypoint_pipeline,
        "verify_market_data_schema",
        lambda _: SimpleNamespace(
            profile=SimpleNamespace(value="legacy"),
            tables=[SimpleNamespace(table_name="market_data")],
        ),
    )
    instrumentation_calls: dict[str, Any] = {"count": 0, "engine": None}

    def _record(engine_obj: object) -> None:
        instrumentation_calls["count"] += 1
        instrumentation_calls["engine"] = engine_obj

    monkeypatch.setattr(entrypoint_pipeline, "verify_instrumentation_tables", _record)

    runner._bootstrap_database(config)

    assert instrumentation_calls["count"] == 1
    assert instrumentation_calls["engine"] is engine


def test_bootstrap_database_raises_when_instrumentation_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    config = SchedulerConfig()
    engine = object()
    summary = SimpleNamespace(
        applied_count=0,
        already_applied_count=0,
        profile=SimpleNamespace(value="auto"),
    )

    def _apply_profiled_migrations(*, db_url: str) -> SimpleNamespace:
        assert db_url == "postgresql://example"
        return summary

    monkeypatch.setattr(entrypoint_pipeline, "apply_profiled_migrations", _apply_profiled_migrations)
    monkeypatch.setattr(entrypoint_pipeline.EngineManager, "get_engine", lambda _: engine)
    monkeypatch.setattr(entrypoint_pipeline, "is_postgres_url", lambda _: True)
    monkeypatch.setattr(
        entrypoint_pipeline.PipelineRunner,
        "_resolve_db_connection",
        lambda self, cfg: "postgresql://example",
    )
    monkeypatch.setattr(
        entrypoint_pipeline,
        "verify_market_data_schema",
        lambda _: SimpleNamespace(
            profile=SimpleNamespace(value="legacy"),
            tables=[SimpleNamespace(table_name="market_data")],
        ),
    )

    def _raise(_: object) -> None:
        raise SchemaHealthCheckError("instrumentation missing")

    monkeypatch.setattr(entrypoint_pipeline, "verify_instrumentation_tables", _raise)

    with pytest.raises(SchemaHealthCheckError):
        runner._bootstrap_database(config)
