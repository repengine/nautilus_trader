#!/usr/bin/env python3
"""
Dataset build runner with simple planning and local concurrency.

Reads a JSON/TOML config describing symbols, time windows, and flags, then
invokes the lightweight dataset build CLI per symbol. Designed to orchestrate
per-symbol runs with optional parallelism and resumable progress logging.

Notes
-----
- This runner intentionally calls the Python main() of build_tft_dataset to
  avoid subprocess overhead and to keep integration simple.
- For large builds, consider slicing by day/month partitions and using an
  external scheduler; this module focuses on single-node orchestration.

"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ml._imports import check_ml_dependencies


try:
    from ml.common.metrics_bootstrap import get_counter
    from ml.common.metrics_bootstrap import get_histogram

    _RUNS_TOTAL = get_counter(
        "nautilus_ml_build_runner_runs_total",
        "Total dataset build tasks executed",
        ["status"],
    )
    _RUN_DURATION = get_histogram(
        "nautilus_ml_build_runner_task_duration_seconds",
        "Duration of per-symbol dataset build tasks",
        ["symbol"],
    )
except Exception:  # pragma: no cover - metrics optional
    _RUNS_TOTAL = None  # type: ignore[assignment]
    _RUN_DURATION = None  # type: ignore[assignment]


try:  # Python 3.11+
    import tomllib as _tomli
except Exception:  # pragma: no cover - older Pythons
    _tomli = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BuildWindow:
    start: str | None = None  # ISO date (inclusive)
    end: str | None = None  # ISO date (inclusive)
    days_back: int | None = None  # if set, overrides start/end


@dataclass(frozen=True)
class BuildConfig:
    data_dir: Path
    out_dir: Path
    symbols: list[str]
    window: BuildWindow = BuildWindow()
    include_macro: bool = False
    macro_lag_days: int = 1
    include_micro: bool = False
    include_l2: bool = False
    horizon_minutes: int = 15
    threshold: float = 0.001
    lookback_periods: int = 60
    workers: int = 1
    use_subprocess: bool = False

    @staticmethod
    def from_mapping(obj: dict[str, Any]) -> BuildConfig:
        def _p(key: str, default: Any | None = None) -> Any:
            return obj.get(key, default)

        win_obj = _p("window", {}) or {}
        window = BuildWindow(
            start=win_obj.get("start"),
            end=win_obj.get("end"),
            days_back=win_obj.get("days_back"),
        )
        symbols = [str(s).upper() for s in (_p("symbols", []) or [])]
        if not symbols:
            raise ValueError("Config requires non-empty 'symbols' list")
        cfg = BuildConfig(
            data_dir=Path(str(_p("data_dir", "data/tier1"))),
            out_dir=Path(str(_p("out_dir", "./tft_ds"))),
            symbols=symbols,
            window=window,
            include_macro=bool(_p("include_macro", False)),
            macro_lag_days=int(_p("macro_lag_days", 1)),
            include_micro=bool(_p("include_micro", False)),
            include_l2=bool(_p("include_l2", False)),
            horizon_minutes=int(_p("horizon_minutes", 15)),
            threshold=float(_p("threshold", 0.001)),
            lookback_periods=int(_p("lookback_periods", 60)),
            workers=max(1, int(_p("workers", 1))),
            use_subprocess=bool(_p("use_subprocess", False)),
        )
        return cfg


def load_config(path: Path) -> BuildConfig:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".json"}:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return BuildConfig.from_mapping(obj)
    if path.suffix.lower() in {".toml", ".tml"}:
        if _tomli is None:  # pragma: no cover - guard
            # tomllib is stdlib in 3.11+; otherwise require tomli
            check_ml_dependencies(["pandas"])  # Provide a helpful message path
            raise RuntimeError("TOML parsing not available: install Python 3.11+ or 'tomli'")
        obj = _tomli.loads(path.read_text(encoding="utf-8"))
        return BuildConfig.from_mapping(obj)
    raise ValueError(f"Unsupported config format: {path.suffix}")


@dataclass(frozen=True)
class BuildTask:
    symbol: str
    # Optional: future partitioning fields (e.g., year/month)


def plan_tasks(cfg: BuildConfig) -> list[BuildTask]:
    # Minimal plan: one task per symbol (can be expanded to per-day/month)
    return [BuildTask(symbol=s) for s in cfg.symbols]


def _progress_log_path(out_dir: Path) -> Path:
    return out_dir / "progress.jsonl"


def _log_progress(out_dir: Path, event: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, separators=(",", ":"))
    _progress_log_path(out_dir).open("a", encoding="utf-8").write(line + "\n")


def _run_single(cfg: BuildConfig, task: BuildTask) -> int:
    symbol_out = cfg.out_dir / task.symbol
    symbol_out.mkdir(parents=True, exist_ok=True)
    args = [
        "--data_dir",
        str(cfg.data_dir),
        "--symbols",
        task.symbol,
        "--out_dir",
        str(symbol_out),
        "--horizon_minutes",
        str(cfg.horizon_minutes),
        "--threshold",
        str(cfg.threshold),
        "--lookback_periods",
        str(cfg.lookback_periods),
    ]
    if cfg.include_macro:
        args += ["--include_macro", "--macro_lag_days", str(cfg.macro_lag_days)]
    if cfg.include_micro:
        args += ["--include_micro"]
    if cfg.include_l2:
        args += ["--include_l2"]

    if cfg.use_subprocess:
        import subprocess

        cmd = [
            "uv",
            "run",
            "--active",
            "--no-sync",
            "python",
            "-m",
            "ml.scripts.build_tft_dataset",
            *args,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        _log_progress(
            cfg.out_dir,
            {"event": "subprocess_log", "symbol": task.symbol, "output": proc.stdout[-5000:]},
        )
        return int(proc.returncode)
    else:
        # Import here to avoid import-time overhead for consumers
        from ml.scripts.build_tft_dataset import main as build_main

        return int(build_main(args))


def execute(
    cfg: BuildConfig,
) -> dict[str, Any]:
    tasks = plan_tasks(cfg)
    results: dict[str, Any] = {"total": len(tasks), "succeeded": 0, "failed": 0}
    import time

    if cfg.workers <= 1:
        for t in tasks:
            _log_progress(cfg.out_dir, {"event": "start", "symbol": t.symbol})
            try:
                start = time.perf_counter()
                rc = _run_single(cfg, t)
                dur = time.perf_counter() - start
                if _RUN_DURATION is not None:
                    _RUN_DURATION.labels(symbol=t.symbol).observe(dur)
                if rc == 0:
                    results["succeeded"] += 1
                    _log_progress(cfg.out_dir, {"event": "success", "symbol": t.symbol, "rc": rc})
                    if _RUNS_TOTAL is not None:
                        _RUNS_TOTAL.labels(status="success").inc()
                else:
                    results["failed"] += 1
                    _log_progress(cfg.out_dir, {"event": "failure", "symbol": t.symbol, "rc": rc})
                    if _RUNS_TOTAL is not None:
                        _RUNS_TOTAL.labels(status="failure").inc()
            except Exception as exc:  # pragma: no cover - defensive
                results["failed"] += 1
                _log_progress(
                    cfg.out_dir, {"event": "exception", "symbol": t.symbol, "error": str(exc)}
                )
                if _RUNS_TOTAL is not None:
                    _RUNS_TOTAL.labels(status="exception").inc()
    else:
        # Parallel execution per symbol
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            future_map: dict[Any, BuildTask] = {}
            for t in tasks:
                _log_progress(cfg.out_dir, {"event": "start", "symbol": t.symbol})
                future = pool.submit(_run_single, cfg, t)
                future_map[future] = t
            for fut in as_completed(future_map):
                t = future_map[fut]
                try:
                    start = time.perf_counter()
                    rc = fut.result()
                    dur = time.perf_counter() - start
                    if _RUN_DURATION is not None:
                        _RUN_DURATION.labels(symbol=t.symbol).observe(dur)
                    if int(rc) == 0:
                        results["succeeded"] += 1
                        _log_progress(
                            cfg.out_dir, {"event": "success", "symbol": t.symbol, "rc": int(rc)}
                        )
                        if _RUNS_TOTAL is not None:
                            _RUNS_TOTAL.labels(status="success").inc()
                    else:
                        results["failed"] += 1
                        _log_progress(
                            cfg.out_dir, {"event": "failure", "symbol": t.symbol, "rc": int(rc)}
                        )
                        if _RUNS_TOTAL is not None:
                            _RUNS_TOTAL.labels(status="failure").inc()
                except Exception as exc:  # pragma: no cover - defensive
                    results["failed"] += 1
                    _log_progress(
                        cfg.out_dir, {"event": "exception", "symbol": t.symbol, "error": str(exc)}
                    )
                    if _RUNS_TOTAL is not None:
                        _RUNS_TOTAL.labels(status="exception").inc()
    return results


def _today_iso() -> str:
    d = date.today()
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    ap = argparse.ArgumentParser(description="Run dataset builds per symbol from config")
    ap.add_argument("--config", required=True, help="Path to JSON/TOML config")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    res = execute(cfg)
    print(json.dumps({"result": res, "ts": _today_iso()}))
    return 0 if res.get("failed", 0) == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
