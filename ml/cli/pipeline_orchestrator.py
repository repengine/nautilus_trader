#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import uuid as _uuid
from collections.abc import Callable
from pathlib import Path

from ml.common.logging_config import bind_log_context
from ml.orchestration.pipeline_orchestrator import DatasetBuildConfig
from ml.orchestration.pipeline_orchestrator import HPOConfig
from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
from ml.orchestration.pipeline_orchestrator import OrchestratorConfig
from ml.orchestration.pipeline_orchestrator import TeacherTrainConfig
from ml.orchestration.pipeline_orchestrator import _CliMain as _CliMain
from ml.stores.io_raw import ParquetCatalogRawWriter
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol
from ml.stores.providers import CatalogCoverageProvider
from ml.stores.providers import SqlCoverageProvider
from ml.stores.writers import DataStoreMarketDataWriter
from ml.stores.writers import ParquetCatalogMarketDataWriter


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run end-to-end ML pipeline (cold path)")

    # Ingestion/backfill
    ap.add_argument("--ingest", action="store_true", help="Run ingestion backfill first")
    ap.add_argument("--dataset_id", default="EQUS.MINI")
    ap.add_argument("--schema", default="bars", choices=["bars", "tbbo", "trades"])
    ap.add_argument("--instruments", default="SPY.NYSE")
    ap.add_argument("--lookback_days", type=int, default=7)
    ap.add_argument("--coverage_mode", default="catalog", choices=["catalog", "sql"])
    ap.add_argument("--catalog_path", default=os.getenv("CATALOG_PATH", ""))
    ap.add_argument(
        "--db",
        default=os.getenv("NAUTILUS_DB", "postgresql://postgres:postgres@localhost:5432/nautilus"),
    )

    # Writer mode for ingestion
    ap.add_argument("--write_mode", default="parquet", choices=["parquet", "datastore"])

    # Dataset build
    ap.add_argument("--data_dir", default="data/tier1")
    ap.add_argument("--symbols", default="SPY.NYSE")
    ap.add_argument("--out_dir", default="ml_out")
    ap.add_argument("--include_macro", action="store_true")
    ap.add_argument("--macro_lag_days", type=int, default=1)
    ap.add_argument("--include_micro", action="store_true")
    ap.add_argument("--include_l2", action="store_true")
    ap.add_argument("--include_events", action="store_true")
    ap.add_argument("--include_calendar", action="store_true")
    ap.add_argument(
        "--emit_dataset_events",
        action="store_true",
        help="Emit dataset events via DataRegistry for the TFT build",
    )
    ap.add_argument("--horizon_minutes", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.001)
    ap.add_argument("--lookback_periods", type=int, default=30)
    ap.add_argument("--start_iso", default=None, help="Optional start date ISO (YYYY-MM-DD)")
    ap.add_argument("--end_iso", default=None, help="Optional end date ISO (YYYY-MM-DD)")
    ap.add_argument("--chunk_days", type=int, default=0, help="Chunk build by N days (0=disabled)")

    # HPO
    ap.add_argument("--hpo", action="store_true")
    ap.add_argument("--hpo_epochs", type=int, default=2)
    ap.add_argument("--hpo_batch_size", type=int, default=32)
    ap.add_argument("--hpo_tail_rows", type=int, default=5000)
    ap.add_argument("--hpo_limit_groups", type=int, default=50)

    # Teacher training
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--teacher_model_id", default="teacher_model")
    ap.add_argument("--feature_registry_dir", default=None)
    ap.add_argument("--dataset_register_features", action="store_true",
                    help="If set, register features during dataset build using feature_registry_dir")
    ap.add_argument("--feature_set_id", default=None)
    ap.add_argument("--max_epochs", type=int, default=5)

    # Optional promotions and feature registration
    ap.add_argument("--auto_register_model", action="store_true")
    ap.add_argument("--gates_json", default=None)
    ap.add_argument("--auto_promote", action="store_true")
    ap.add_argument("--deploy_target", default=None)

    ap.add_argument("--auto_register_features", action="store_true")
    ap.add_argument("--feature_metrics_json", default=None)

    # Optional small feature refresh phase
    ap.add_argument("--refresh_features", action="store_true")

    # Promotion stage 2 (walk-forward + cost-aware backtest)
    ap.add_argument("--promote_stage2", action="store_true")
    ap.add_argument("--stage2_gates_json", default=None)
    ap.add_argument("--stage2_cost_bps", type=float, default=0.0)
    ap.add_argument(
        "--stage2_engine",
        choices=["returns", "backtest"],
        default="returns",
        help="Stage 2 engine: returns (default) or backtest (advisory)",
    )
    ap.add_argument("--stage2_commission_bps", type=float, default=0.0)
    ap.add_argument("--stage2_slippage_bps", type=float, default=0.0)
    ap.add_argument("--final_model_id", default=None, help="Model ID to promote in stage 2 (optional)")

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _run_id: str = f"orch_{_uuid.uuid4().hex[:12]}"
    bind_log_context(run_id=_run_id, component="ml.pipeline_orchestrator")

    # Coverage provider
    coverage: CoverageProviderProtocol
    if args.coverage_mode == "catalog":
        if not args.catalog_path:
            raise SystemExit("catalog_path is required for catalog coverage mode")
        coverage = CatalogCoverageProvider(catalog_path=args.catalog_path)
    else:
        coverage = SqlCoverageProvider(connection_string=args.db)

    # Writer selection
    writer: MarketDataWriterProtocol
    raw_writer = None
    if args.write_mode == "datastore":
        # Use IntegrationManager to get DataStore with adapters (CATALOG_PATH recommended)
        from ml.core.integration import MLIntegrationManager

        mgr = MLIntegrationManager(
            db_connection=args.db,
            auto_start_postgres=False,
            auto_migrate=False,
            ensure_healthy=False,
        )
        if mgr.data_store is None:
            raise SystemExit("DataStore unavailable; use parquet write_mode or set CATALOG_PATH")
        writer = DataStoreMarketDataWriter(store=mgr.data_store)  # type: ignore[arg-type]
        registry = mgr.data_registry
    else:
        # Parquet writer backed by catalog
        try:
            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
        except Exception as exc:  # pragma: no cover - import env issue
            raise SystemExit(f"ParquetDataCatalog unavailable: {exc}")
        if not args.catalog_path:
            raise SystemExit("catalog_path is required for parquet write_mode")
        catalog = ParquetDataCatalog(args.catalog_path)
        writer = ParquetCatalogMarketDataWriter(catalog=catalog)
        # Enable dual-write of raw bars/quotes/trades directly to catalog
        raw_writer = ParquetCatalogRawWriter(catalog)
        # Registry for event emission (IntegrationManager provides one easily)
        from ml.core.integration import MLIntegrationManager

        mgr = MLIntegrationManager(
            db_connection=args.db,
            auto_start_postgres=False,
            auto_migrate=False,
            ensure_healthy=False,
        )
        registry = mgr.data_registry

    # Ingestor stub: use Databento adapter via CLI if desired; here we use the existing ingestion orchestrator contract
    ingestor = None
    if args.ingest:
        # Create Databento adapter if API key available; else skip ingestion step
        api_key = os.getenv("DATABENTO_API_KEY", "").strip()
        if api_key:
            from ml.data.ingest.databento_adapter import DatabentoAPIClient
            from ml.data.ingest.resume import DatabentoIngestor

            client = DatabentoAPIClient(api_key=api_key)
            ingestor = DatabentoIngestor(client=client)

    # Build CLI mains
    from ml.cli.build_tft_dataset import main as build_main
    from ml.training.teacher.tft_cli import main as teacher_main

    # CLI entrypoint type alias: def main(argv: list[str] | None = None) -> int
    _CliMain = Callable[[list[str] | None], int]
    hpo_main_cli: _CliMain | None
    try:
        from ml.cli.hpo_tft import main as _hpo_main

        hpo_main_cli = _hpo_main
    except Exception:
        hpo_main_cli = None

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        registry=registry,  # RegistryProtocol instance
        ingestor=ingestor if ingestor is not None else None,
        raw_writer=raw_writer,
        build_main=build_main,
        hpo_main=hpo_main_cli,
        teacher_main=teacher_main,
    )

    # Optional ingestion
    if args.ingest and ingestor is not None:
        # Map CLI schema aliases to provider schema tokens
        schema_map = {
            "bars": "ohlcv-1m",
            "tbbo": "tbbo",
            "trades": "trades",
        }
        provider_schema = schema_map.get(str(args.schema).lower(), str(args.schema))
        instruments = [s.strip() for s in str(args.instruments).split(",") if s.strip()]
        for inst in instruments:
            orch.backfill(
                dataset_id=args.dataset_id,
                schema=provider_schema,
                instrument_id=inst,
                lookback_days=int(args.lookback_days),
            )

    # Dataset build / HPO / teacher train
    # Prefer using catalog_path as the dataset source when the caller provided it,
    # unless data_dir was explicitly set to a non-default value.
    _data_dir_effective = str(args.data_dir)
    if args.catalog_path and str(args.data_dir) == "data/tier1":
        _data_dir_effective = str(args.catalog_path)

    ds_cfg = DatasetBuildConfig(
        data_dir=_data_dir_effective,
        symbols=str(args.symbols),
        out_dir=str(args.out_dir),
        include_macro=bool(args.include_macro),
        macro_lag_days=int(args.macro_lag_days),
        include_micro=bool(args.include_micro),
        include_l2=bool(args.include_l2),
        include_events=bool(args.include_events),
        include_calendar=bool(args.include_calendar),
        horizon_minutes=int(args.horizon_minutes),
        threshold=float(args.threshold),
        lookback_periods=int(args.lookback_periods),
        emit_dataset_events=bool(args.emit_dataset_events),
        start_iso=(str(args.start_iso) if args.start_iso else None),
        end_iso=(str(args.end_iso) if args.end_iso else None),
        chunk_days=int(args.chunk_days or 0),
        register_features=bool(args.dataset_register_features),
        feature_registry_dir=(str(args.feature_registry_dir) if args.feature_registry_dir else str(Path.home() / ".nautilus" / "ml" / "features")) if bool(args.dataset_register_features) else None,
        feature_role="teacher",
    )
    hpo_cfg = HPOConfig(
        enabled=bool(args.hpo),
        epochs=int(args.hpo_epochs),
        batch_size=int(args.hpo_batch_size),
        tail_rows=int(args.hpo_tail_rows),
        limit_groups=int(args.hpo_limit_groups),
    )
    teacher_cfg = TeacherTrainConfig(
        enabled=bool(args.train),
        model_id=str(args.teacher_model_id),
        feature_registry_dir=args.feature_registry_dir,
        feature_set_id=args.feature_set_id,
        max_epochs=int(args.max_epochs),
    )
    cfg = OrchestratorConfig(dataset=ds_cfg, hpo=hpo_cfg, teacher=teacher_cfg)
    rc = orch.run(cfg)
    if rc != 0:
        return rc

    # Optional promotions/feature registration
    from typing import Any

    register_and_promote_model: Any
    register_or_refresh_features: Any
    try:
        from ml.orchestration.promotions import register_and_promote_model
        from ml.orchestration.promotions import register_or_refresh_features
    except Exception:
        register_and_promote_model = None
        register_or_refresh_features = None

    # Model registration/promotion if requested
    if register_and_promote_model is not None and (
        bool(args.auto_register_model) or bool(args.gates_json) or bool(args.auto_promote)
    ):
        # Prefer explicit metrics path if provided; else look for a conventional file in out_dir
        metrics_path = None
        # Common conventions from training/evaluation flows
        for candidate in ("model_metrics.json", "teacher_meta.json"):
            p = Path(str(args.out_dir)) / candidate
            if p.exists():
                metrics_path = str(p)
                break
        if metrics_path is None and isinstance(args.gates_json, str):
            # Still allow promotion based solely on gates if tests stub the registry
            metrics_path = str(Path(str(args.out_dir)) / "model_metrics.json")

        gates: list[dict[str, object]] = []
        if args.gates_json:
            import json

            try:
                gj = json.loads(Path(str(args.gates_json)).read_text(encoding="utf-8"))
                for g in gj.get("gates", []):
                    gates.append(
                        {
                            "metric_name": str(g["metric"]),
                            "threshold": float(g["threshold"]),
                            "comparison": str(g.get("comparison", "gte")),
                            "required": bool(g.get("required", True)),
                        },
                    )
            except Exception:
                gates = []
        from typing import Any, cast

        from ml.registry.dataclasses import QualityGate

        qgates = [
            QualityGate(
                metric_name=str(g["metric_name"]),
                threshold=float(cast(Any, g["threshold"])),
                comparison=str(g.get("comparison", "gte")),
                required=bool(g.get("required", True)),
            )
            for g in gates
        ]

        try:
            # Integration for registries
            from ml.core.integration import MLIntegrationManager

            mgr2 = MLIntegrationManager(
                auto_start_postgres=False,
                auto_migrate=False,
                ensure_healthy=False,
            )
            model_id = register_and_promote_model(
                model_metrics_path=str(
                    metrics_path or (Path(str(args.out_dir)) / "model_metrics.json"),
                ),
                out_dir=str(args.out_dir),
                registry=mgr2.model_registry,
                feature_registry=mgr2.feature_registry,
                gates=qgates,
                auto_promote=bool(args.auto_promote),
                deploy_target=(str(args.deploy_target) if args.deploy_target else None),
            )
        except Exception:
            # Non-fatal for orchestrator
            model_id = None
            _ = model_id

    # Promotion stage 2 (walk-forward + cost-aware backtest)
    if bool(getattr(args, "promote_stage2", False)):
        from typing import Any

        from ml.orchestration.promotions import Stage2Config
        from ml.orchestration.promotions import run_promotion_stage2
        from ml.registry.dataclasses import QualityGate

        # Build gates from stage2_gates_json if provided, else reuse --gates_json
        gates_src = args.stage2_gates_json or args.gates_json
        sgates: list[QualityGate] = []
        if gates_src:
            try:
                import json as _json
                gj = _json.loads(Path(str(gates_src)).read_text(encoding="utf-8"))
                for g in gj.get("gates", []):
                    sgates.append(
                        QualityGate(
                            metric_name=str(g["metric"]),
                            threshold=float(g["threshold"]),
                            comparison=str(g.get("comparison", "gte")),
                            required=bool(g.get("required", True)),
                        ),
                    )
            except Exception:
                sgates = []

        from typing import Literal as _Lit
        from typing import cast as _cast
        s2_cfg = Stage2Config(
            out_dir=str(args.out_dir),
            dataset_csv=str(Path(str(args.out_dir)) / "dataset.csv"),
            data_dir=str(args.data_dir),
            horizon_minutes=int(args.horizon_minutes),
            engine_mode=_cast(_Lit["returns", "backtest"], str(args.stage2_engine)),
            cost_bps=float(args.stage2_cost_bps),
            commission_bps=float(args.stage2_commission_bps),
            slippage_bps=float(args.stage2_slippage_bps),
            model_id_hint=(str(args.final_model_id) if args.final_model_id else None),
            gates=sgates,
            auto_promote=bool(args.auto_promote),
            deploy_target=(str(args.deploy_target) if args.deploy_target else None),
        )
        _ = run_promotion_stage2(s2_cfg)

    # Feature registration/refresh (small optional phase)
    if register_or_refresh_features is not None and (
        bool(args.auto_register_features) or bool(args.feature_metrics_json)
    ):
        metrics_path = (
            str(args.feature_metrics_json)
            if args.feature_metrics_json
            else str(Path(str(args.out_dir)) / "feature_metrics.json")
        )
        try:
            from ml.core.integration import MLIntegrationManager

            mgr3 = MLIntegrationManager(
                auto_start_postgres=False,
                auto_migrate=False,
                ensure_healthy=False,
            )
            register_or_refresh_features(
                feature_metrics_path=metrics_path,
                feature_registry=mgr3.feature_registry,
                auto_register=bool(args.auto_register_features),
            )
        except Exception as exc:
            # Non-fatal: best-effort feature registration
            logger.debug("Feature registration/refresh failed: %s", exc, exc_info=True)

    # Optional small feature refresh phase (emit a marker event)
    if bool(args.refresh_features):
        try:
            from typing import cast

            from ml.common.event_emitter import emit_dataset_event
            from ml.config.events import EventStatus
            from ml.config.events import Source
            from ml.config.events import Stage
            from ml.core.integration import MLIntegrationManager
            from ml.registry.protocols import RegistryProtocol

            mgr4 = MLIntegrationManager(
                auto_start_postgres=False,
                auto_migrate=False,
                ensure_healthy=False,
            )
            reg = mgr4.data_registry
            import time as _time

            from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

            now_ns = _sanitize(
                int(_time.time_ns()),
                context="cli.pipeline_orchestrator:refresh_features.now",
            )
            emit_dataset_event(
                cast(RegistryProtocol, reg),
                dataset_id="features",
                instrument_id="GLOBAL",
                stage=Stage.FEATURE_COMPUTED,
                source=Source.HISTORICAL,
                run_id=f"refresh_{now_ns}",
                ts_min=now_ns,
                ts_max=now_ns,
                count=1,
                status=EventStatus.SUCCESS,
                metadata={"phase": "refresh_features"},
                dataset_type="features",
                component="orchestrator",
            )
        except Exception as exc:
            # Non-fatal: best-effort marker emission
            logger.debug("Emit refresh_features marker failed: %s", exc, exc_info=True)

    return 0


logger = logging.getLogger(__name__)
if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
