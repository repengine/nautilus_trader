from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from ml.config.observability import ObservabilityConfig
from ml.core.integration import MLIntegrationManager


def _seed_sample(mgr: MLIntegrationManager) -> None:
    MLIntegrationManager.initialize_observability_pipeline(mgr)
    svc = getattr(mgr, "observability_service", None)
    if svc is None:
        return
    # Add minimal rows across tables
    svc.add_latency_stage(
        correlation_id="00000000-0000-0000-0000-000000000001",
        instrument_id="EURUSD.SIM",
        pipeline_stage="data_ingestion",
        ts_stage_start=1,
        ts_stage_end=2,
    )
    svc.add_metric(
        metric_name="ml_predictions_total",
        metric_type="counter",
        value=1.0,
        timestamp=1,
        labels={"instrument_id": "EURUSD.SIM"},
    )
    svc.add_correlation(
        correlation_id="00000000-0000-0000-0000-000000000001",
        event_id="00000000-0000-0000-0000-000000000002",
        parent_event_id=None,
        instrument_id="EURUSD.SIM",
        domain="data",
        lineage_depth=0,
        ts_event=1,
        propagation_path=["data"],
    )
    svc.add_health(
        component_id="data_store",
        health_score=0.9,
        subsystem_scores={"db": 1.0},
        timestamp=2,
        measurement_window_ms=100,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Observability flush CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_jsonl = sub.add_parser("flush-jsonl", help="Flush to JSONL/CSV")
    p_jsonl.add_argument("--base-path", required=True)
    p_jsonl.add_argument("--format", default="jsonl", choices=["jsonl", "csv"])
    p_jsonl.add_argument("--seed-sample", action="store_true")

    p_db = sub.add_parser("flush-db", help="Flush to DB")
    p_db.add_argument("--db-url", required=True)
    p_db.add_argument("--seed-sample", action="store_true")

    p_start = sub.add_parser("start", help="Start background flushing (thread or async)")
    p_start.add_argument("--sink", default="file", choices=["file", "db"], help="file or db sink")
    p_start.add_argument("--base-path", default="./observability", help="base path when sink=file")
    p_start.add_argument("--format", default="jsonl", choices=["jsonl", "csv"], help="file format")
    p_start.add_argument("--db-url", default=None, help="DB URL when sink=db")
    p_start.add_argument("--interval", type=float, default=60.0, help="flush interval seconds")
    p_start.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="run duration seconds (0=until SIGTERM)",
    )
    p_start.add_argument(
        "--seed-sample",
        action="store_true",
        help="seed a sample set of rows before start",
    )
    # Async worker options
    p_start.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="use async worker instead of thread flusher",
    )
    p_start.add_argument(
        "--async-queue",
        type=int,
        default=4096,
        help="async worker bounded queue size",
    )
    p_start.add_argument(
        "--async-component",
        type=str,
        default="obs_async_worker",
        help="component label for metrics",
    )

    p_status = sub.add_parser("status", help="Show observability async worker status")

    args = parser.parse_args(argv)
    mgr = object.__new__(MLIntegrationManager)  # lightweight

    if args.cmd == "flush-jsonl":
        if args.seed_sample:
            _seed_sample(mgr)
        out = MLIntegrationManager.flush_observability_to_path(
            mgr,
            base_path=Path(args.base_path),
            file_format=args.format,
        )
        # Print summary
        for k, pth in out.items():
            print(f"{k}: {pth}")
        return 0
    elif args.cmd == "flush-db":
        if args.seed_sample:
            _seed_sample(mgr)
        out_db = MLIntegrationManager.flush_observability_to_db(
            mgr,
            connection_string=str(args.db_url),
        )
        for k, cnt in out_db.items():
            print(f"{k}: {cnt}")
        return 0
    elif args.cmd == "start":  # start
        if args.seed_sample:
            _seed_sample(mgr)
        # Async mode: construct ObservabilityConfig and start from config
        if args.async_mode:
            # Run async worker inside an event loop to avoid RuntimeError
            async def _run_async() -> int:
                cfg = ObservabilityConfig(
                    sink="db" if args.sink == "db" else "file",
                    base_path=str(args.base_path),
                    file_format=str(args.format),
                    db_connection_string=str(args.db_url) if args.db_url else None,
                    interval_seconds=float(args.interval),
                    async_enabled=True,
                    async_queue_maxsize=int(args.async_queue),
                    async_component_label=str(args.async_component),
                )
                # Start worker using integration manager
                MLIntegrationManager.start_observability_from_config(mgr, cfg)
                # Optional bounded run duration
                if args.duration and args.duration > 0:
                    await asyncio.sleep(float(args.duration))
                    # Stop worker directly to avoid calling asyncio.run inside running loop
                    worker = getattr(mgr, "_obs_async_worker", None)
                    if worker is not None:
                        try:
                            await worker.stop(drain=True, timeout=1.0)
                        except Exception:
                            pass
                return 0

            return asyncio.run(_run_async())
        # Default thread flusher
        else:
            MLIntegrationManager.start_observability_flush(
                mgr,
                base_path=Path(args.base_path),
                interval_seconds=float(args.interval),
                file_format=args.format,
                sink=args.sink,
                db_connection_string=str(args.db_url) if args.db_url else None,
            )
            if args.duration and args.duration > 0:
                time.sleep(float(args.duration))
                MLIntegrationManager.stop_observability_flush(mgr)
            return 0
    else:  # status
        status = MLIntegrationManager.get_observability_async_status(mgr)
        running_obj = status.get("running", False)
        qsize_obj = status.get("queue_size", 0)
        running = bool(running_obj) if isinstance(running_obj, (bool, int, str)) else False
        if isinstance(qsize_obj, (int, float)):
            qsize = int(qsize_obj)
        elif isinstance(qsize_obj, str):
            try:
                qsize = int(qsize_obj)
            except ValueError:
                qsize = 0
        else:
            qsize = 0
        print(f"async_worker_running={running} queue_size={qsize}")
        return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main(sys.argv[1:]))
