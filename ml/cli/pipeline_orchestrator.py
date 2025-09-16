#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from pathlib import Path

from ml.orchestration.pipeline_orchestrator import DatasetBuildConfig
from ml.orchestration.pipeline_orchestrator import HPOConfig
from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
from ml.orchestration.pipeline_orchestrator import OrchestratorConfig
from ml.orchestration.pipeline_orchestrator import TeacherTrainConfig
from ml.orchestration.pipeline_orchestrator import _CliMain as _CliMain
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
    ap.add_argument("--horizon_minutes", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.001)
    ap.add_argument("--lookback_periods", type=int, default=30)

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

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

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
    if args.write_mode == "datastore":
        # Use IntegrationManager to get DataStore with adapters (CATALOG_PATH recommended)
        from ml.core.integration import MLIntegrationManager

        mgr = MLIntegrationManager(
            db_connection=args.db,
            auto_start_postgres=False,
            auto_migrate=False,
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
        # Registry for event emission (IntegrationManager provides one easily)
        from ml.core.integration import MLIntegrationManager

        mgr = MLIntegrationManager(
            db_connection=args.db,
            auto_start_postgres=False,
            auto_migrate=False,
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
        ingestor=ingestor if ingestor is not None else object(),
        build_main=build_main,
        hpo_main=hpo_main_cli,
        teacher_main=teacher_main,
    )

    # Optional ingestion
    if args.ingest:
        instruments = [s.strip() for s in str(args.instruments).split(",") if s.strip()]
        for inst in instruments:
            orch.backfill(
                dataset_id=args.dataset_id,
                schema=args.schema,
                instrument_id=inst,
                lookback_days=int(args.lookback_days),
            )

    # Dataset build / HPO / teacher train
    ds_cfg = DatasetBuildConfig(
        data_dir=str(args.data_dir),
        symbols=str(args.symbols),
        out_dir=str(args.out_dir),
        include_macro=bool(args.include_macro),
        macro_lag_days=int(args.macro_lag_days),
        include_micro=bool(args.include_micro),
        include_l2=bool(args.include_l2),
        horizon_minutes=int(args.horizon_minutes),
        threshold=float(args.threshold),
        lookback_periods=int(args.lookback_periods),
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
