#!/usr/bin/env python3

from __future__ import annotations


"""
One-shot scheduler smoke runner (CI-friendly).

Runs a single orchestrator invocation using the provided config, forcing
coverage_mode=sql and write_mode=datastore to avoid catalog dependencies. It
sets ML_ALLOW_DUMMY=1 so integration falls back to in-memory components.
"""

# ruff: noqa: E402  (allow CLI docstring before imports)

import argparse
import os
from typing import Any, cast

from ml.cli.pipeline_orchestrator import main as _orch_main
from ml.orchestration import config_loader as _cfg


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="One-shot scheduler smoke runner")
    ap.add_argument("--config", default="ml/config/pipeline_scheduler_example.toml")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Ensure dummy fallback for integration in CI
    os.environ.setdefault("ML_ALLOW_DUMMY", "1")
    if args.dry_run:
        os.environ["ORCH_DRY_RUN"] = "1"

    cfg = _cfg.load_orchestrator_config(str(args.config))
    orch_args = _cfg.to_pipeline_args(cfg)

    # Force minimal coverage/writer for smoke
    orch_args = [
        "--coverage_mode",
        "sql",
        "--write_mode",
        "datastore",
        *orch_args,
    ]

    # Relax health checks to avoid CI env dependencies
    try:
        import ml.core.integration as _integ

        _orig_init = _integ.MLIntegrationManager.__init__

        def _init(self: Any, *a: object, **kw: Any) -> None:
            kw.setdefault("ensure_healthy", False)
            cast(Any, _orig_init)(self, *a, **kw)

        _integ.MLIntegrationManager.__init__ = _init  # type: ignore[method-assign]
    except Exception:
        pass

    # Stub heavy CLIs to make smoke fast and deterministic
    try:
        import ml.cli.build_tft_dataset as _build

        def _stub_build(argv: list[str] | None = None) -> int:
            if argv and "--out_dir" in argv:
                out = argv[argv.index("--out_dir") + 1]
                try:
                    os.makedirs(out, exist_ok=True)
                    with open(os.path.join(out, "dataset.csv"), "w", encoding="utf-8") as f:
                        f.write("id,time_index\n1,1\n")
                except Exception:
                    pass
            return 0

        _build.main = _stub_build
    except Exception:
        pass

    try:
        import ml.training.teacher.tft_cli as _tft

        _tft.main = lambda argv=None: 0
    except Exception:
        pass

    rc = int(_orch_main(orch_args))
    if rc != 0:
        print(f"SMOKE FAIL: orchestrator exited with {rc}")
        return rc
    print("SMOKE OK")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
