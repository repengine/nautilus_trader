from __future__ import annotations

import builtins
import json
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from ml.orchestration import pipeline_runner
from ml.orchestration.pipeline_runner import MLPipelineRunner
from ml.orchestration.pipeline_runner import PipelineRunConfig
from ml.orchestration.pipeline_runner import _execute_pipeline_mode
from ml.orchestration.pipeline_runner import _validate_backfill_dates
from ml.orchestration.pipeline_runner import load_config
from ml.orchestration.pipeline_runner import run_pipeline
from ml.orchestration.pipeline_runner import setup_logging

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


def _new_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> MLPipelineRunner:
    monkeypatch.setattr(pipeline_runner.signal, "signal", lambda _sig, _handler: None)
    return MLPipelineRunner(config or {}, dry_run=dry_run)


def test_run_pipeline_initialises_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "ml.orchestration.pipeline_runner.load_config",
        lambda _: {"catalog_path": "./data"},
    )

    class _DummyRunner:
        def __init__(self, config: dict[str, Any], dry_run: bool) -> None:
            captured["config"] = config
            captured["dry_run"] = dry_run

        def setup_ml_system(self) -> Any:
            captured["setup_called"] = True
            return SimpleNamespace(config=SimpleNamespace(symbols=["SPY"]))

    def _fake_execute(runner: Any, mode: str, start: str | None, end: str | None) -> None:
        captured["mode"] = mode
        captured["start"] = start
        captured["end"] = end
        captured["runner"] = runner

    monkeypatch.setattr("ml.orchestration.pipeline_runner.MLPipelineRunner", _DummyRunner)
    monkeypatch.setattr("ml.orchestration.pipeline_runner._execute_pipeline_mode", _fake_execute)

    run_pipeline(
        PipelineRunConfig(
            mode="daily",
            dry_run=True,
            config_path="config.json",
        ),
    )

    assert captured["config"] == {"catalog_path": "./data"}
    assert captured["dry_run"] is True
    assert captured["setup_called"] is True
    assert captured["mode"] == "daily"


def test_setup_signal_handlers_registers_callbacks_and_sets_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks: dict[int, Any] = {}

    def _capture_signal(sig_num: int, handler: Any) -> None:
        callbacks[sig_num] = handler

    monkeypatch.setattr(pipeline_runner.signal, "signal", _capture_signal)
    runner = MLPipelineRunner(config={}, dry_run=True)

    assert pipeline_runner.signal.SIGINT in callbacks
    assert pipeline_runner.signal.SIGTERM in callbacks
    callbacks[pipeline_runner.signal.SIGINT](pipeline_runner.signal.SIGINT, None)
    assert runner.shutdown_requested is True


def test_validate_environment_requires_databento_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(monkeypatch, config={"enable_features": False}, dry_run=False)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.setattr(pipeline_runner, "HAS_POLARS", True)
    monkeypatch.setattr(pipeline_runner, "HAS_DATABENTO", True)

    with pytest.raises(ValueError, match="DATABENTO_API_KEY"):
        runner._validate_environment()


def test_validate_environment_checks_polars_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(monkeypatch, config={"enable_features": False}, dry_run=True)
    checker = MagicMock()
    monkeypatch.setattr(pipeline_runner, "HAS_POLARS", False)
    monkeypatch.setattr(pipeline_runner, "check_ml_dependencies", checker)

    runner._validate_environment()

    checker.assert_called_once_with(["polars"])


def test_validate_environment_requires_databento_package_when_not_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(monkeypatch, config={"enable_features": False}, dry_run=False)
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    monkeypatch.setattr(pipeline_runner, "HAS_POLARS", True)
    monkeypatch.setattr(pipeline_runner, "HAS_DATABENTO", False)

    with pytest.raises(ImportError, match="databento package is required"):
        runner._validate_environment()


def test_create_scheduler_config_uses_universe_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(
        monkeypatch,
        config={
            "universe_mode": "moderate",
            "enable_features": True,
        },
        dry_run=True,
    )
    monkeypatch.setenv("DB_CONNECTION", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setattr(
        pipeline_runner.UniverseConfig,
        "get_full_universe",
        lambda self: ["SPY", "QQQ"],
    )

    scheduler_config = runner._create_scheduler_config()

    assert scheduler_config.symbols == ["SPY", "QQQ"]
    assert scheduler_config.feature_store_connection == "postgresql://user:pass@localhost:5432/test"


def test_initialize_feature_engineer_returns_none_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(monkeypatch, config={}, dry_run=True)
    original_import = builtins.__import__

    def _fake_import(
        name: str,
        globals_obj: dict[str, object] | None = None,
        locals_obj: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "ml.features":
            raise ImportError("ml.features unavailable")
        return original_import(name, globals_obj, locals_obj, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    assert runner._initialize_feature_engineer() is None


def test_initialize_feature_engineer_constructs_feature_engineer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(
        monkeypatch,
        config={
            "enable_microstructure_features": True,
            "enable_trade_flow_features": True,
            "return_periods": [1, 2],
            "momentum_periods": [3, 4],
        },
        dry_run=True,
    )
    fake_module = ModuleType("ml.features")

    class _FakeFeatureConfig:
        def __init__(
            self,
            *,
            include_microstructure: bool,
            include_trade_flow: bool,
            return_periods: list[int],
            momentum_periods: list[int],
        ) -> None:
            self.include_microstructure = include_microstructure
            self.include_trade_flow = include_trade_flow
            self.return_periods = return_periods
            self.momentum_periods = momentum_periods

    class _FakeFeatureEngineer:
        def __init__(self, *, config: _FakeFeatureConfig) -> None:
            self.config = config

    fake_module.FeatureConfig = _FakeFeatureConfig
    fake_module.FeatureEngineer = _FakeFeatureEngineer
    monkeypatch.setitem(sys.modules, "ml.features", fake_module)

    feature_engineer = runner._initialize_feature_engineer()

    assert isinstance(feature_engineer, _FakeFeatureEngineer)
    assert feature_engineer.config.include_microstructure is True
    assert feature_engineer.config.return_periods == [1, 2]


def test_run_health_checks_uses_catalog_and_database_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(monkeypatch, config={"enable_features": True}, dry_run=True)
    runner.catalog = SimpleNamespace(instruments=lambda: ["SPY.XNAS"])
    monkeypatch.setenv("DB_CONNECTION", "postgresql://postgres:postgres@localhost:5432/nautilus")
    close_called: dict[str, bool] = {"value": False}

    class _Conn:
        def close(self) -> None:
            close_called["value"] = True

    fake_psycopg2 = ModuleType("psycopg2")
    fake_psycopg2.connect = lambda _conn_str: _Conn()
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)

    runner._run_health_checks()

    assert close_called["value"] is True


def test_run_health_checks_ignores_database_connection_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(monkeypatch, config={"enable_features": True}, dry_run=True)
    runner.catalog = SimpleNamespace(instruments=lambda: ["SPY.XNAS"])
    fake_psycopg2 = ModuleType("psycopg2")

    def _raise_connect(_conn_str: str) -> Any:
        raise RuntimeError("cannot connect")

    fake_psycopg2.connect = _raise_connect
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)

    runner._run_health_checks()


def test_run_backfill_requires_scheduler_when_not_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(monkeypatch, config={}, dry_run=False)
    start = datetime(2026, 1, 6, tzinfo=UTC)
    end = datetime(2026, 1, 6, tzinfo=UTC)

    with pytest.raises(RuntimeError, match="Scheduler not initialized"):
        runner.run_backfill(start, end)


def test_run_backfill_skips_weekends_and_continues_after_non_fatal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(monkeypatch, config={"stop_on_error": False}, dry_run=False)
    scheduler = MagicMock()
    scheduler.run_daily_update.side_effect = [RuntimeError("boom"), None]
    runner.scheduler = scheduler
    monkeypatch.setattr(pipeline_runner.time, "sleep", lambda _seconds: None)
    start = datetime(2026, 1, 9, tzinfo=UTC)  # Friday
    end = datetime(2026, 1, 12, tzinfo=UTC)  # Monday

    runner.run_backfill(start, end)

    assert scheduler.run_daily_update.call_count == 2


def test_run_backfill_raises_when_stop_on_error_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(monkeypatch, config={"stop_on_error": True}, dry_run=False)
    scheduler = MagicMock()
    scheduler.run_daily_update.side_effect = RuntimeError("boom")
    runner.scheduler = scheduler
    start = datetime(2026, 1, 8, tzinfo=UTC)
    end = datetime(2026, 1, 8, tzinfo=UTC)

    with pytest.raises(RuntimeError, match="boom"):
        runner.run_backfill(start, end)


def test_run_daily_respects_dry_run_and_scheduler_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_runner = _new_runner(monkeypatch, config={}, dry_run=True)
    dry_runner.run_daily()

    active_runner = _new_runner(monkeypatch, config={}, dry_run=False)
    with pytest.raises(RuntimeError, match="Scheduler not initialized"):
        active_runner.run_daily()


def test_run_daily_executes_scheduler_update(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _new_runner(monkeypatch, config={}, dry_run=False)
    scheduler = MagicMock()
    runner.scheduler = scheduler

    runner.run_daily()

    scheduler.run_daily_update.assert_called_once_with()


def test_run_realtime_loops_until_shutdown_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _new_runner(monkeypatch, config={}, dry_run=False)
    sleep_calls: dict[str, int] = {"count": 0}

    def _sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        runner.shutdown_requested = True

    monkeypatch.setattr(pipeline_runner.time, "sleep", _sleep)

    runner.run_realtime()

    assert sleep_calls["count"] == 1


def test_run_realtime_handles_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _new_runner(monkeypatch, config={}, dry_run=False)

    def _raise_interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_runner.time, "sleep", _raise_interrupt)

    runner.run_realtime()


def test_load_config_defaults_and_errors(tmp_path: Path) -> None:
    defaults = load_config(None)
    assert defaults["catalog_path"] == "./data"

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(str(tmp_path / "missing.json"))

    unsupported = tmp_path / "config.toml"
    unsupported.write_text("x = 1", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported config format"):
        load_config(str(unsupported))


def test_load_config_reads_json_mapping(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"catalog_path": "/tmp/catalog"}), encoding="utf-8")

    loaded = load_config(str(config_file))

    assert loaded == {"catalog_path": "/tmp/catalog"}


def test_load_config_rejects_non_mapping_payload(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration must be a mapping"):
        load_config(str(config_file))


def test_load_config_reads_yaml_mapping_via_importlib(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("catalog_path: /tmp/catalog\n", encoding="utf-8")
    fake_yaml = ModuleType("yaml")
    fake_yaml.safe_load = lambda _handle: {"catalog_path": "/tmp/catalog"}
    importlib_module = ModuleType("importlib")
    importlib_module.import_module = lambda _name: fake_yaml
    monkeypatch.setitem(sys.modules, "importlib", importlib_module)

    loaded = load_config(str(config_file))

    assert loaded == {"catalog_path": "/tmp/catalog"}


def test_setup_logging_configures_debug_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _configure_logging(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(pipeline_runner, "configure_logging", _configure_logging)

    setup_logging(verbose=True)
    setup_logging(verbose=False)

    assert calls == [{"level": "DEBUG"}, {}]
    assert pipeline_runner.logging.getLogger("urllib3").level == pipeline_runner.logging.WARNING
    assert pipeline_runner.logging.getLogger("databento").level == pipeline_runner.logging.WARNING


def test_validate_backfill_dates_validates_required_and_order() -> None:
    with pytest.raises(ValueError, match="Backfill mode requires --start-date and --end-date"):
        _validate_backfill_dates(None, None)

    with pytest.raises(ValueError, match="Start date must be before end date"):
        _validate_backfill_dates("2026-01-02", "2026-01-01")

    start_dt, end_dt = _validate_backfill_dates("2026-01-01", "2026-01-02")
    assert start_dt.tzinfo is UTC
    assert end_dt.tzinfo is UTC


def test_execute_pipeline_mode_routes_to_expected_runner_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyRunner:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.backfill_range: tuple[datetime, datetime] | None = None

        def run_backfill(self, start: datetime, end: datetime) -> None:
            self.calls.append("backfill")
            self.backfill_range = (start, end)

        def run_daily(self) -> None:
            self.calls.append("daily")

        def run_realtime(self) -> None:
            self.calls.append("realtime")

    runner = _DummyRunner()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    monkeypatch.setattr(pipeline_runner, "_validate_backfill_dates", lambda _s, _e: (start, end))

    _execute_pipeline_mode(runner, "backfill", "2026-01-01", "2026-01-02")
    _execute_pipeline_mode(runner, "daily", None, None)
    _execute_pipeline_mode(runner, "realtime", None, None)

    assert runner.calls == ["backfill", "daily", "realtime"]
    assert runner.backfill_range == (start, end)

    with pytest.raises(ValueError, match="Unsupported pipeline mode"):
        _execute_pipeline_mode(runner, "invalid", None, None)


def test_setup_ml_system_initializes_catalog_scheduler_and_feature_engineer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(
        monkeypatch,
        config={"catalog_path": "/tmp/catalog", "enable_features": True},
        dry_run=False,
    )
    catalog_obj = SimpleNamespace()
    scheduler = SimpleNamespace(config=SimpleNamespace(symbols=["SPY", "QQQ"]))

    monkeypatch.setattr(runner, "_validate_environment", lambda: None)
    monkeypatch.setattr(pipeline_runner, "ParquetDataCatalog", lambda _path: catalog_obj)
    monkeypatch.setattr(runner, "_create_scheduler_config", lambda: "scheduler-config")
    monkeypatch.setattr(pipeline_runner, "DataCollector", lambda: "collector")
    monkeypatch.setattr(runner, "_initialize_feature_engineer", lambda: "feature-engineer")

    def _build_scheduler(*, catalog: Any, config: Any, collector: Any, feature_engineer: Any) -> Any:
        assert catalog is catalog_obj
        assert config == "scheduler-config"
        assert collector == "collector"
        assert feature_engineer == "feature-engineer"
        return scheduler

    monkeypatch.setattr(pipeline_runner, "DataScheduler", _build_scheduler)
    run_health_checks_called: dict[str, bool] = {"value": False}
    monkeypatch.setattr(
        runner,
        "_run_health_checks",
        lambda: run_health_checks_called.__setitem__("value", True),
    )

    resolved_scheduler = runner.setup_ml_system()

    assert resolved_scheduler is scheduler
    assert runner.catalog is catalog_obj
    assert runner.scheduler is scheduler
    assert run_health_checks_called["value"] is True


def test_setup_ml_system_dry_run_disables_collector_and_feature_engineer_when_features_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(
        monkeypatch,
        config={"catalog_path": "/tmp/catalog", "enable_features": False},
        dry_run=True,
    )
    monkeypatch.setattr(runner, "_validate_environment", lambda: None)
    monkeypatch.setattr(pipeline_runner, "ParquetDataCatalog", lambda _path: SimpleNamespace())
    monkeypatch.setattr(runner, "_create_scheduler_config", lambda: "scheduler-config")
    monkeypatch.setattr(
        runner,
        "_initialize_feature_engineer",
        lambda: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    captured: dict[str, Any] = {}

    def _build_scheduler(*, catalog: Any, config: Any, collector: Any, feature_engineer: Any) -> Any:
        captured["catalog"] = catalog
        captured["config"] = config
        captured["collector"] = collector
        captured["feature_engineer"] = feature_engineer
        return SimpleNamespace(config=SimpleNamespace(symbols=[]))

    monkeypatch.setattr(pipeline_runner, "DataScheduler", _build_scheduler)
    monkeypatch.setattr(runner, "_run_health_checks", lambda: None)

    runner.setup_ml_system()

    assert captured["collector"] is None
    assert captured["feature_engineer"] is None


def test_validate_environment_warns_when_db_connection_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(monkeypatch, config={"enable_features": True}, dry_run=True)
    monkeypatch.delenv("DB_CONNECTION", raising=False)
    monkeypatch.setattr(pipeline_runner, "HAS_POLARS", True)
    monkeypatch.setattr(pipeline_runner, "HAS_DATABENTO", True)

    runner._validate_environment()


def test_run_health_checks_handles_psycopg2_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _new_runner(monkeypatch, config={"enable_features": True}, dry_run=True)
    original_import = builtins.__import__

    def _fake_import(
        name: str,
        globals_obj: dict[str, object] | None = None,
        locals_obj: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "psycopg2":
            raise ImportError("psycopg2 unavailable")
        return original_import(name, globals_obj, locals_obj, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    runner._run_health_checks()


def test_run_health_checks_skips_optional_paths_when_catalog_missing_and_features_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _new_runner(monkeypatch, config={"enable_features": False}, dry_run=True)
    runner.catalog = None

    runner._run_health_checks()


def test_run_backfill_dry_run_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _new_runner(monkeypatch, config={}, dry_run=True)
    start = datetime(2026, 1, 6, tzinfo=UTC)
    end = datetime(2026, 1, 7, tzinfo=UTC)

    runner.run_backfill(start, end)


def test_run_backfill_logs_interrupted_when_shutdown_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _new_runner(monkeypatch, config={}, dry_run=False)
    runner.shutdown_requested = True
    runner.scheduler = MagicMock()
    start = datetime(2026, 1, 6, tzinfo=UTC)
    end = datetime(2026, 1, 7, tzinfo=UTC)

    runner.run_backfill(start, end)

    runner.scheduler.run_daily_update.assert_not_called()


def test_run_realtime_dry_run_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _new_runner(monkeypatch, config={}, dry_run=True)

    runner.run_realtime()
