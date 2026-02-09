from __future__ import annotations

import json
import subprocess
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Self
from typing import cast

import pytest

import ml.data as _ml_data
import ml.pipelines.build_runner as build_runner
from ml.common.subprocess_utils import SubprocessExecutionError
from ml.pipelines.build_runner import BuildConfig
from ml.pipelines.build_runner import BuildTask
from ml.pipelines.build_runner import BuildWindow
from ml.pipelines.build_runner import execute
from ml.pipelines.build_runner import load_config
from ml.pipelines.build_runner import plan_tasks
from ml.tests.utils.targets import build_default_target_semantics_payload

_ = _ml_data


class _MetricProbe:
    def __init__(self) -> None:
        self.label_calls: list[dict[str, Any]] = []
        self.observe_calls: list[float] = []
        self.inc_calls = 0

    def labels(self, **labels: Any) -> _MetricProbe:
        self.label_calls.append(labels)
        return self

    def observe(self, value: float) -> None:
        self.observe_calls.append(float(value))

    def inc(self) -> None:
        self.inc_calls += 1


class _FrozenDateTime:
    @staticmethod
    def now(tz: object | None = None) -> datetime:
        _ = tz
        return datetime(2026, 2, 8, tzinfo=UTC)

    @staticmethod
    def fromisoformat(value: str) -> datetime:
        return datetime.fromisoformat(value)


class _FakeFuture:
    def __init__(self, result_value: int | Exception) -> None:
        self._result_value = result_value

    def result(self) -> int:
        if isinstance(self._result_value, Exception):
            raise self._result_value
        return self._result_value


class _FakeProcessPool:
    def __init__(self, max_workers: int, outcomes: dict[str, int | Exception]) -> None:
        self.max_workers = max_workers
        self._outcomes = outcomes

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        _ = (exc_type, exc, tb)
        return None

    def submit(
        self,
        fn: Any,
        cfg: BuildConfig,
        task: BuildTask,
    ) -> _FakeFuture:
        _ = (fn, cfg)
        return _FakeFuture(self._outcomes[task.symbol])


def test_load_config_and_plan(tmp_path: Path) -> None:
    target_semantics = build_default_target_semantics_payload()
    cfg_obj = {
        "data_dir": str(tmp_path / "data"),
        "out_dir": str(tmp_path / "out"),
        "symbols": ["spy", "qqq"],
        "include_macro": True,
        "macro_lag_days": 1,
        "target_semantics": target_semantics,
        "workers": 1,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg_obj), encoding="utf-8")

    cfg = load_config(cfg_path)
    assert isinstance(cfg, BuildConfig)
    tasks = plan_tasks(cfg)
    assert [t.symbol for t in tasks] == ["SPY", "QQQ"]


def test_to_text_and_parse_target_semantics_validation() -> None:
    payload = build_default_target_semantics_payload()

    assert build_runner._to_text(b"abc\xff") == "abc"
    assert build_runner._to_text("xyz") == "xyz"
    assert build_runner._to_text(None) == ""

    cfg = BuildConfig.from_mapping(
        {
            "data_dir": "data",
            "out_dir": "out",
            "symbols": ["spy"],
            "target_semantics": json.dumps(payload),
        },
    )
    assert cfg.symbols == ["SPY"]
    assert cfg.target_semantics["version"] == payload["version"]
    assert cfg.target_semantics["horizons"][0]["minutes"] == 15

    with pytest.raises(ValueError, match="non-empty 'symbols'"):
        BuildConfig.from_mapping({"symbols": [], "target_semantics": payload})

    with pytest.raises(ValueError, match="target_semantics is required"):
        BuildConfig.from_mapping({"symbols": ["SPY"]})

    with pytest.raises(ValueError, match="target_semantics must be JSON object or dict"):
        BuildConfig.from_mapping({"symbols": ["SPY"], "target_semantics": "not-json"})

    with pytest.raises(ValueError, match="target_semantics must be JSON object or dict"):
        BuildConfig.from_mapping({"symbols": ["SPY"], "target_semantics": json.dumps([1, 2])})

    with pytest.raises(ValueError, match="target_semantics must be JSON object or dict"):
        BuildConfig.from_mapping({"symbols": ["SPY"], "target_semantics": 1.23})


def test_load_config_dependency_guard_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")

    invalid_path = tmp_path / "cfg.yaml"
    invalid_path.write_text("x: y", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported config format"):
        load_config(invalid_path)

    toml_path = tmp_path / "cfg.toml"
    toml_path.write_text("symbols=['SPY']", encoding="utf-8")
    calls: list[list[str]] = []

    def _fake_check_ml_dependencies(packages: list[str]) -> None:
        calls.append(packages)

    monkeypatch.setattr(build_runner, "_tomli", None)
    monkeypatch.setattr(build_runner, "check_ml_dependencies", _fake_check_ml_dependencies)
    with pytest.raises(RuntimeError, match="TOML parsing not available"):
        load_config(toml_path)
    assert calls == [["pandas"]]


def test_apply_vintage_conversion_updates_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_parquet = tmp_path / "dataset.parquet"
    dataset_parquet.write_bytes(b"pq")
    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_path.write_text(json.dumps({"base": 1}), encoding="utf-8")

    conversion_calls: list[tuple[Path, Path]] = []
    metadata_updates: list[dict[str, Any]] = []
    writes: list[tuple[Path, dict[str, Any]]] = []

    def _fake_convert(source: Path, destination: Path) -> Any:
        conversion_calls.append((source, destination))
        return SimpleNamespace(vintage_columns=("v1",), age_columns=("a1",))

    def _fake_update_metadata(
        metadata: dict[str, Any],
        *,
        vintage_columns: tuple[str, ...],
        age_columns: tuple[str, ...],
    ) -> dict[str, Any]:
        metadata_updates.append(metadata)
        assert vintage_columns == ("v1",)
        assert age_columns == ("a1",)
        return {"base": metadata["base"], "converted": True}

    def _fake_write_metadata(path: Path, metadata: dict[str, Any]) -> None:
        writes.append((path, metadata))

    monkeypatch.setattr(build_runner, "convert_vintage_timestamps_to_age", _fake_convert)
    monkeypatch.setattr(build_runner, "update_metadata_with_vintage_age", _fake_update_metadata)
    monkeypatch.setattr(build_runner, "write_metadata", _fake_write_metadata)

    build_runner._apply_vintage_conversion(dataset_parquet)

    assert conversion_calls == [
        (dataset_parquet, dataset_parquet.with_name("dataset_with_vintage_age.parquet")),
    ]
    assert metadata_updates == [{"base": 1}]
    assert writes == [(metadata_path, {"base": 1, "converted": True})]

    with pytest.raises(FileNotFoundError):
        build_runner._apply_vintage_conversion(tmp_path / "missing.parquet")

    orphan_dataset = tmp_path / "nested" / "orphan.parquet"
    orphan_dataset.parent.mkdir(parents=True, exist_ok=True)
    orphan_dataset.write_bytes(b"pq")
    with pytest.raises(FileNotFoundError):
        build_runner._apply_vintage_conversion(orphan_dataset)


def test_run_single_builds_cli_args_and_days_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_args: list[str] = []

    def _fake_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        captured_args[:] = argv
        return 0

    import ml.cli.build_tft_dataset as build_cli

    monkeypatch.setattr(build_runner, "datetime", _FrozenDateTime)
    monkeypatch.setattr(build_cli, "main", _fake_main)

    cfg = BuildConfig(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        target_semantics=build_default_target_semantics_payload(),
        window=BuildWindow(days_back=2),
        include_macro=True,
        include_micro=True,
        include_l2=True,
        include_macro_deltas=True,
        include_calendar_lags=True,
        include_clustering_tags=True,
        include_context_features=True,
        chunk_days=3,
        register_features=True,
        feature_registry_dir=tmp_path / "registry",
        convert_vintage_to_age=True,
    )

    rc = build_runner._run_single(cfg, BuildTask(symbol="SPY"))

    assert rc == 0
    assert captured_args[captured_args.index("--start") + 1] == "2026-02-06"
    assert captured_args[captured_args.index("--end") + 1] == "2026-02-08"
    assert "--include_macro" in captured_args
    assert "--include_micro" in captured_args
    assert "--include_l2" in captured_args
    assert "--include_macro_deltas" in captured_args
    assert "--include_calendar_lags" in captured_args
    assert "--include_clustering_tags" in captured_args
    assert "--include_context_features" in captured_args
    assert "--chunk_days" in captured_args
    assert "--register_features" in captured_args
    assert "--feature_registry_dir" in captured_args
    assert "--convert-vintage-age" in captured_args


def test_run_single_subprocess_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_events: list[dict[str, Any]] = []

    def _fake_progress(_out_dir: Path, event: dict[str, Any]) -> None:
        progress_events.append(event)

    cfg = BuildConfig(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        target_semantics=build_default_target_semantics_payload(),
        use_subprocess=True,
        subprocess_timeout=12.0,
        register_features=False,
    )

    def _run_ok(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        _ = args
        _ = kwargs
        return subprocess.CompletedProcess(
            args=["uv", "run"],
            returncode=0,
            stdout="success-output",
            stderr="",
        )

    monkeypatch.setattr(build_runner, "_log_progress", _fake_progress)
    monkeypatch.setattr(build_runner, "run_command", _run_ok)

    rc_ok = build_runner._run_single(cfg, BuildTask(symbol="SPY"))
    assert rc_ok == 0
    assert progress_events[-1]["event"] == "subprocess_log"
    assert progress_events[-1]["output"] == "success-output"

    def _run_fail(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        _ = args
        _ = kwargs
        raise SubprocessExecutionError(
            command=("uv", "run"),
            returncode=7,
            stdout=b"failed-output",
            stderr=b"boom",
        )

    monkeypatch.setattr(build_runner, "run_command", _run_fail)
    rc_fail = build_runner._run_single(cfg, BuildTask(symbol="QQQ"))
    assert rc_fail == 7
    assert progress_events[-1]["output"] == "failed-output"


def test_run_single_prefer_api_success_and_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml.cli.build_tft_dataset as build_cli
    import ml.common.metrics_manager as metrics_module
    import ml.data as data_module

    api_cfg_records: list[dict[str, Any]] = []
    convert_calls: list[Path] = []
    metric_calls: list[dict[str, Any]] = []
    cli_calls: list[list[str]] = []

    class _FakeAPICfg:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _FakeMetrics:
        def inc(
            self,
            metric_name: str,
            metric_help: str,
            *,
            labels: dict[str, str],
            labelnames: tuple[str, str, str],
        ) -> None:
            metric_calls.append(
                {
                    "metric_name": metric_name,
                    "metric_help": metric_help,
                    "labels": labels,
                    "labelnames": labelnames,
                },
            )

    class _FakeMetricsManager:
        @staticmethod
        def default() -> _FakeMetrics:
            return _FakeMetrics()

    def _fake_api_build(api_cfg: Any) -> Any:
        kwargs = cast(dict[str, Any], api_cfg.kwargs)
        api_cfg_records.append(kwargs)
        dataset_path = tmp_path / "out" / "SPY" / "dataset.parquet"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_bytes(b"pq")
        return SimpleNamespace(dataset_parquet=dataset_path)

    def _fake_convert(dataset_parquet: Path) -> None:
        convert_calls.append(dataset_parquet)

    def _fake_cli_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        cli_calls.append(argv)
        return 3

    monkeypatch.setattr(data_module, "DatasetBuildConfig", _FakeAPICfg)
    monkeypatch.setattr(data_module, "build_tft_dataset", _fake_api_build)
    monkeypatch.setattr(build_runner, "_apply_vintage_conversion", _fake_convert)
    monkeypatch.setattr(metrics_module, "MetricsManager", _FakeMetricsManager)
    monkeypatch.setattr(build_cli, "main", _fake_cli_main)

    success_cfg = BuildConfig(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        target_semantics=build_default_target_semantics_payload(),
        window=BuildWindow(start="2026-02-01", end="2026-02-02"),
        prefer_api=True,
        convert_vintage_to_age=True,
    )
    assert build_runner._run_single(success_cfg, BuildTask(symbol="SPY")) == 0
    assert api_cfg_records[-1]["symbols"] == ["SPY"]
    assert api_cfg_records[-1]["start"] == datetime.fromisoformat("2026-02-01")
    assert api_cfg_records[-1]["end"] == datetime.fromisoformat("2026-02-02")
    assert convert_calls and convert_calls[-1].name == "dataset.parquet"

    def _failing_api_build(_api_cfg: Any) -> Any:
        raise RuntimeError("api failure")

    monkeypatch.setattr(data_module, "build_tft_dataset", _failing_api_build)
    fallback_cfg = BuildConfig(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "out",
        symbols=["QQQ"],
        target_semantics=build_default_target_semantics_payload(),
        prefer_api=True,
    )
    assert build_runner._run_single(fallback_cfg, BuildTask(symbol="QQQ")) == 3
    assert metric_calls
    assert metric_calls[-1]["labels"]["op"] == "api_build_fail_fallback_cli"
    assert cli_calls


def test_run_single_prefer_api_fallback_tolerates_metric_logging_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml.cli.build_tft_dataset as build_cli
    import ml.common.metrics_manager as metrics_module
    import ml.data as data_module

    cli_calls: list[list[str]] = []

    class _FakeAPICfg:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _FailingMetricsManager:
        @staticmethod
        def default() -> Any:
            raise RuntimeError("metric init failure")

    def _failing_api_build(_api_cfg: Any) -> Any:
        raise RuntimeError("api failed")

    def _fake_cli_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        cli_calls.append(argv)
        return 5

    monkeypatch.setattr(data_module, "DatasetBuildConfig", _FakeAPICfg)
    monkeypatch.setattr(data_module, "build_tft_dataset", _failing_api_build)
    monkeypatch.setattr(metrics_module, "MetricsManager", _FailingMetricsManager)
    monkeypatch.setattr(build_cli, "main", _fake_cli_main)

    cfg = BuildConfig(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "out",
        symbols=["DIA"],
        target_semantics=build_default_target_semantics_payload(),
        prefer_api=True,
    )
    assert build_runner._run_single(cfg, BuildTask(symbol="DIA")) == 5
    assert cli_calls


def test_execute_sequential_and_parallel_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, Any]] = []
    run_metric = _MetricProbe()
    duration_metric = _MetricProbe()

    def _fake_log_progress(_out_dir: Path, event: dict[str, Any]) -> None:
        events.append(event)

    monkeypatch.setattr(build_runner, "_log_progress", _fake_log_progress)
    monkeypatch.setattr(build_runner, "_RUNS_TOTAL", run_metric)
    monkeypatch.setattr(build_runner, "_RUN_DURATION", duration_metric)

    def _run_single_sequential(_cfg: BuildConfig, task: BuildTask) -> int:
        if task.symbol == "SPY":
            return 0
        if task.symbol == "QQQ":
            return 2
        raise RuntimeError("boom")

    monkeypatch.setattr(build_runner, "_run_single", _run_single_sequential)
    sequential_cfg = BuildConfig(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "seq",
        symbols=["SPY", "QQQ", "DIA"],
        target_semantics=build_default_target_semantics_payload(),
        workers=1,
    )
    sequential_result = execute(sequential_cfg)
    assert sequential_result == {"total": 3, "succeeded": 1, "failed": 2}
    assert any(call.get("status") == "success" for call in run_metric.label_calls)
    assert any(call.get("status") == "failure" for call in run_metric.label_calls)
    assert any(call.get("status") == "exception" for call in run_metric.label_calls)
    assert len(duration_metric.observe_calls) == 2
    assert any(event["event"] == "exception" for event in events)

    outcomes: dict[str, int | Exception] = {
        "SPY": 0,
        "QQQ": 3,
        "IWM": RuntimeError("parallel boom"),
    }

    def _fake_pool_factory(max_workers: int) -> _FakeProcessPool:
        return _FakeProcessPool(max_workers=max_workers, outcomes=outcomes)

    def _fake_as_completed(futures: Any) -> list[_FakeFuture]:
        return list(futures)

    monkeypatch.setattr(build_runner, "ProcessPoolExecutor", _fake_pool_factory)
    monkeypatch.setattr(build_runner, "as_completed", _fake_as_completed)

    parallel_cfg = BuildConfig(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "par",
        symbols=["SPY", "QQQ", "IWM"],
        target_semantics=build_default_target_semantics_payload(),
        workers=3,
    )
    parallel_result = execute(parallel_cfg)
    assert parallel_result == {"total": 3, "succeeded": 1, "failed": 2}
    assert any(event["event"] == "failure" for event in events)


def test_execute_monkeypatched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub builder main to write a marker file and succeed
    calls: list[list[str]] = []

    def _fake_build_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        calls.append(list(argv))
        # Find --out_dir value
        out_idx = argv.index("--out_dir") + 1
        out_dir = Path(argv[out_idx])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dataset.parquet").write_bytes(b"")
        return 0

    import ml.cli.build_tft_dataset as build_cli

    monkeypatch.setattr(build_cli, "main", _fake_build_main)

    cfg = BuildConfig(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        target_semantics=build_default_target_semantics_payload(),
    )
    res = execute(cfg)
    assert res["total"] == 1 and res["succeeded"] == 1 and res["failed"] == 0
    # Verify artifact and progress
    assert (tmp_path / "out" / "SPY" / "dataset.parquet").exists()
    progress = tmp_path / "out" / "progress.jsonl"
    lines = progress.read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(line).get("event") == "success" for line in lines)


def test_main_reports_result_and_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []

    def _fake_print(message: object) -> None:
        printed.append(str(message))

    cfg = BuildConfig(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        target_semantics=build_default_target_semantics_payload(),
    )

    monkeypatch.setattr("builtins.print", _fake_print)
    monkeypatch.setattr(build_runner, "load_config", lambda _path: cfg)
    monkeypatch.setattr(
        build_runner,
        "execute",
        lambda _cfg: {"total": 1, "succeeded": 1, "failed": 0},
    )
    assert build_runner.main(["--config", str(tmp_path / "cfg.json")]) == 0
    assert '"failed": 0' in printed[-1]

    monkeypatch.setattr(
        build_runner,
        "execute",
        lambda _cfg: {"total": 1, "succeeded": 0, "failed": 1},
    )
    assert build_runner.main(["--config", str(tmp_path / "cfg.json")]) == 1
