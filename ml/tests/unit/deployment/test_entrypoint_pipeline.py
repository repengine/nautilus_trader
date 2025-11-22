from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sqlite3

import pytest

from ml.config.market_data import MarketDatasetInput
from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.data.coverage.manager import BucketClassification
from ml.data.coverage.manager import BucketSpec
from ml.data.rehydration import CatalogRehydrationConfig
from ml.deployment import entrypoint_pipeline
from ml.stores.migrations_runner import SchemaHealthCheckError

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
        ) -> None:
            captured["catalog"] = catalog
            captured["db_connection"] = db_connection
            captured["config"] = config
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
                ("FRESH.XNAS", now_ns, now_ns),
                ("STALE.XNAS", now_ns - (48 * 3_600 * 1_000_000_000), now_ns),
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
        symbols=["FRESH.XNAS", "STALE.XNAS"],
        databento=DatabentoConfig(dataset="EQUS.MINI"),
    )

    selected = runner._select_rehydration_instruments(config)

    assert "STALE.XNAS" in selected
    assert "FRESH.XNAS" not in selected


def test_select_rehydration_instruments_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    runner._rehydrator_config = CatalogRehydrationConfig(enabled=True)
    config = SchedulerConfig(
        symbols=["AAPL.XNAS"],
        databento=DatabentoConfig(dataset="EQUS.MINI"),
    )
    monkeypatch.setenv("CATALOG_REHYDRATE_STALE_ONLY", "0")

    selected = runner._select_rehydration_instruments(config)

    assert selected == ["AAPL.XNAS"]


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
dataset_id = "ml.test_dataset"
schema = "test_schema"
entities = "AAPL"
[datasets.parquet]
path = "data/test.parquet"
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("COVERAGE_DATASETS_FILE", str(manifest))

    entries = runner._load_feature_coverage_entries()

    assert entries
    assert entries[0].dataset.dataset_id == "ml.test_dataset"
    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "nautilus_ml_coverage_manifest_events_total",
        labels={"event": "loaded"},
    )
    assert metric_value == 1.0


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
    assert metric_value == 1.0


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
    assert metric_value == 1.0


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
    assert dataset.dataset_id == "EQUS.MINI"
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
        def __init__(self, *, catalog_path: str) -> None:
            self.catalog_path = catalog_path

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

    def _fake_run(self: entrypoint_pipeline.PipelineRunner, scheduler_cfg: SchedulerConfig) -> None:
        assert scheduler_cfg is config
        entrypoint_pipeline.pipeline_status["coverage"].update(coverage_template)

    monkeypatch.setattr(entrypoint_pipeline.PipelineRunner, "_run_coverage_restoration", _fake_run)

    scheduler_state: dict[str, object] = {}

    class _SchedulerStub:
        def __init__(
            self,
            *,
            catalog: object,
            config: SchedulerConfig,
            use_orchestrator: bool,
            dual_write: bool,
        ) -> None:
            assert catalog is catalog_obj
            assert config is config
            scheduler_state["instance"] = self
            self.targeted_calls: list[list[object]] = []
            self.stop_calls = 0

        def run_targeted_update(self, buckets: Any) -> None:
            self.targeted_calls.append(list(buckets))

        def stop(self) -> None:
            self.stop_calls += 1

    monkeypatch.setattr(entrypoint_pipeline, "DataScheduler", _SchedulerStub)
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


def test_bootstrap_database_verifies_instrumentation_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = entrypoint_pipeline.PipelineRunner()
    config = SchedulerConfig()
    engine = object()
    summary = SimpleNamespace(applied_count=0, already_applied_count=0)

    class _RunnerStub:
        def __init__(self, *, db_url: str) -> None:
            assert db_url == "postgresql://example"
            self.engine = engine

        def apply_pending_migrations(self) -> SimpleNamespace:
            return summary

    monkeypatch.setattr(entrypoint_pipeline, "MigrationRunner", _RunnerStub)
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
            table_name="market_data",
            schema="public",
            primary_key=("instrument_id", "ts_event"),
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
    summary = SimpleNamespace(applied_count=0, already_applied_count=0)

    class _RunnerStub:
        def __init__(self, *, db_url: str) -> None:
            assert db_url == "postgresql://example"
            self.engine = engine

        def apply_pending_migrations(self) -> SimpleNamespace:
            return summary

    monkeypatch.setattr(entrypoint_pipeline, "MigrationRunner", _RunnerStub)
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
            table_name="market_data",
            schema="public",
            primary_key=("instrument_id", "ts_event"),
        ),
    )

    def _raise(_: object) -> None:
        raise SchemaHealthCheckError("instrumentation missing")

    monkeypatch.setattr(entrypoint_pipeline, "verify_instrumentation_tables", _raise)

    with pytest.raises(SchemaHealthCheckError):
        runner._bootstrap_database(config)
