#!/usr/bin/env python3

"""
High-level ML pipeline orchestrator (cold path only).

Composes existing ingestion, dataset build, HPO, and training CLIs into a typed,
testable interface suitable for a single long-running service or a nightly batch job.
All heavy work (DataFrames, file I/O, GPU training) remains strictly off the actor hot
paths.

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid as _uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from ml.common.logging_config import bind_log_context
from ml.config.coverage import CoveragePolicy
from ml.config.coverage import get_max_lookback_days
from ml.data.ingest.databento_adapter import DatabentoAPIClient
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.service import DatabentoIngestionService
from ml.registry.protocols import RegistryProtocol
from ml.stores.io_raw import ParquetCatalogRawWriter
from ml.stores.io_raw import RawIngestionWriterProtocol
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol
from ml.stores.providers import CatalogCoverageProvider
from ml.stores.providers import SqlCoverageProvider
from ml.stores.writers import DataStoreMarketDataWriter
from ml.stores.writers import ParquetCatalogMarketDataWriter


logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from ml.config.scheduler_config import SchedulerConfig


class _CliMain(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


@dataclass(slots=True, frozen=True)
class DatasetBuildConfig:
    data_dir: str
    symbols: str
    out_dir: str
    include_macro: bool = False
    macro_lag_days: int = 1
    include_micro: bool = False
    include_l2: bool = False
    include_events: bool = False
    include_calendar: bool = False
    fred_vintage_dir: str | None = None
    events_dir: str | None = None
    student_mode: bool = False
    horizon_minutes: int = 15
    threshold: float = 0.001
    lookback_periods: int = 30
    emit_dataset_events: bool = False
    # Optional time window and chunking for memory/perf control
    start_iso: str | None = None
    end_iso: str | None = None
    chunk_days: int = 0
    # Optional feature registration
    register_features: bool = False
    feature_registry_dir: str | None = None
    feature_role: str = "teacher"


@dataclass(slots=True, frozen=True)
class HPOConfig:
    enabled: bool = False
    epochs: int = 2
    batch_size: int = 32
    tail_rows: int = 5000
    limit_groups: int = 50


@dataclass(slots=True, frozen=True)
class TeacherTrainConfig:
    enabled: bool = True
    model_id: str = "teacher_model"
    feature_registry_dir: str | None = None
    feature_set_id: str | None = None
    max_epochs: int = 5


@dataclass(slots=True, frozen=True)
class OrchestratorConfig:
    dataset: DatasetBuildConfig
    hpo: HPOConfig
    teacher: TeacherTrainConfig
    # Optional promotions/feature refresh settings (used by config-driven scheduler)
    promotions: PromotionsConfig | None = None
    # Optional data ingestion pre-stage before dataset build
    pre_ingestion: SchedulerConfig | None = None
    pre_ingestion_options: PreIngestionOptions | None = None


@dataclass(slots=True, frozen=True)
class PromotionsConfig:
    # Model promotions
    auto_register_model: bool = False
    gates_json: str | None = None
    auto_promote: bool = False
    deploy_target: str | None = None
    # Feature registration/refresh
    auto_register_features: bool = False
    feature_metrics_json: str | None = None
    refresh_features: bool = False


@dataclass(slots=True)
class MLPipelineOrchestrator:
    coverage: CoverageProviderProtocol
    writer: MarketDataWriterProtocol
    build_main: _CliMain
    teacher_main: _CliMain
    registry: object | None = None  # Backwards compatibility alias
    data_registry: object | None = (
        None  # RegistryProtocol at runtime; kept lax to avoid import cycles
    )
    ingestor: object | None = None  # DatabentoIngestor or similar
    hpo_main: _CliMain | None = None
    raw_writer: RawIngestionWriterProtocol | None = None
    service: DatabentoIngestionService | None = None
    model_registry: object | None = None
    feature_registry: object | None = None

    def __post_init__(self) -> None:
        if self.data_registry is None and self.registry is not None:
            object.__setattr__(self, "data_registry", self.registry)

    # ---------------------------------------------------------------------
    # Pre-ingestion (unified orchestrator path)
    # ---------------------------------------------------------------------

    def run_pre_ingestion(
        self,
        *,
        catalog_path: Path,
        scheduler_cfg: SchedulerConfig,
        options: PreIngestionOptions | None = None,
    ) -> None:
        """
        Run a data ingestion pre-stage using DataScheduler in orchestrator mode.

        Always cold path. Dual-write (SQL + Parquet) can be enabled in options to
        guarantee the dataset builder has catalog data while preserving SQL coverage.

        """
        from ml.data.scheduler import DataScheduler
        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

        opts = options or PreIngestionOptions()
        catalog = ParquetDataCatalog(str(catalog_path))
        scheduler = DataScheduler(
            catalog=catalog,
            config=scheduler_cfg,
            use_orchestrator=opts.use_orchestrator,
            dual_write=opts.dual_write,
            start_metrics_server=opts.start_metrics_server,
            metrics_port=opts.metrics_port,
        )
        scheduler.run_daily_update()

    def backfill(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> list[tuple[int, int]]:
        orchestrator = IngestionOrchestrator(
            coverage=self.coverage,
            writer=self.writer,
            registry=self.data_registry,  # type: ignore[arg-type]
            ingestor=self.ingestor,  # type: ignore[arg-type]
            raw_writer=self.raw_writer,
            service=self.service,
        )
        return orchestrator.backfill_gaps(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            lookback_days=lookback_days,
        )

    def backfill_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        policy: CoveragePolicy | None = None,
    ) -> list[tuple[int, int]]:
        """
        Backfill gaps bounded by subscription coverage policy.

        Determines the maximum lookback window from CoveragePolicy for the given dataset
        identifier and delegates to IngestionOrchestrator.

        """
        days = get_max_lookback_days(dataset_id, policy)
        return self.backfill(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            lookback_days=days,
        )

    def build_dataset(self, cfg: DatasetBuildConfig) -> int:
        # Prefer the public API to capture BuildResult (feature_set_id). Fallback to CLI if it fails.
        try:
            from ml.data import DatasetBuildConfig as APICfg
            from ml.data import build_tft_dataset as api_build

            symbols_list = [s.strip().upper() for s in cfg.symbols.split(",") if s.strip()]
            api_cfg = APICfg(
                data_dir=Path(cfg.data_dir),
                out_dir=Path(cfg.out_dir),
                symbols=symbols_list,
                include_macro=cfg.include_macro,
                macro_lag_days=cfg.macro_lag_days,
                include_micro=cfg.include_micro,
                include_l2=cfg.include_l2,
                include_events=cfg.include_events,
                include_calendar=cfg.include_calendar,
                fred_vintage_dir=(
                    Path(cfg.fred_vintage_dir).expanduser() if cfg.fred_vintage_dir else None
                ),
                events_base_dir=(Path(cfg.events_dir).expanduser() if cfg.events_dir else None),
                student_mode=cfg.student_mode,
                horizon_minutes=cfg.horizon_minutes,
                threshold=cfg.threshold,
                lookback_periods=cfg.lookback_periods,
                start=(
                    None
                    if not cfg.start_iso
                    else __import__("datetime").datetime.fromisoformat(cfg.start_iso)
                ),
                end=(
                    None
                    if not cfg.end_iso
                    else __import__("datetime").datetime.fromisoformat(cfg.end_iso)
                ),
                chunk_days=int(cfg.chunk_days or 0),
                emit_dataset_events=cfg.emit_dataset_events,
                register_features=cfg.register_features,
                feature_registry_dir=(
                    None if cfg.feature_registry_dir is None else Path(cfg.feature_registry_dir)
                ),
                feature_role=cfg.feature_role,
            )
            logger.info(
                "Dataset readiness | macro=%s events=%s student_mode=%s vintages=%s events_dir=%s",
                cfg.include_macro,
                cfg.include_events,
                getattr(cfg, "student_mode", False),
                bool(cfg.fred_vintage_dir),
                bool(cfg.events_dir),
            )
            result = api_build(api_cfg)
            if not result.feature_names:
                raise ValueError("API dataset build returned no features; falling back to CLI")
            # Persist feature registration metadata for HPO
            try:
                meta_path = Path(cfg.out_dir) / "feature_registration.json"
                import json as _json

                payload = {
                    "feature_set_id": result.feature_set_id,
                    "feature_registry_dir": cfg.feature_registry_dir,
                    "feature_role": cfg.feature_role,
                }
                meta_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
            except Exception:
                pass
            return 0
        except Exception as exc:  # pragma: no cover - defensive fallback to CLI path
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "API-based dataset build failed; falling back to CLI: %s",
                exc,
                exc_info=True,
            )

        # Fallback to invoking the CLI main with assembled args
        args: list[str] = [
            "--data_dir",
            cfg.data_dir,
            "--symbols",
            cfg.symbols,
            "--out_dir",
            cfg.out_dir,
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
        if getattr(cfg, "include_events", False):
            args += ["--include_events"]
        if getattr(cfg, "include_calendar", False):
            args += ["--include_calendar"]
        if cfg.fred_vintage_dir:
            args += ["--fred_vintage_dir", cfg.fred_vintage_dir]
        if cfg.events_dir:
            args += ["--events_dir", cfg.events_dir]
        if cfg.student_mode:
            args += ["--student_mode"]
        if getattr(cfg, "start_iso", None):
            args += ["--start", str(cfg.start_iso)]
        if getattr(cfg, "end_iso", None):
            args += ["--end", str(cfg.end_iso)]
        if int(getattr(cfg, "chunk_days", 0) or 0) > 0:
            args += ["--chunk_days", str(int(cfg.chunk_days))]
        if cfg.emit_dataset_events:
            args += ["--emit_dataset_events"]
        if cfg.register_features:
            args += ["--register_features"]
            reg_dir = cfg.feature_registry_dir or str(Path.home() / ".nautilus" / "ml" / "features")
            args += ["--feature_registry_dir", reg_dir]
        return self.build_main(args)

    def run_hpo(self, cfg: HPOConfig, dataset_csv: Path, out_dir: Path) -> int:
        if not cfg.enabled or self.hpo_main is None:
            return 0
        args = [
            "--train_data_csv",
            str(dataset_csv),
            "--out_dir",
            str(out_dir),
            "--epochs",
            str(cfg.epochs),
            "--batch_size",
            str(cfg.batch_size),
            "--tail_rows",
            str(cfg.tail_rows),
            "--limit_groups",
            str(cfg.limit_groups),
        ]
        return self.hpo_main(args)

    def train_teacher(self, cfg: TeacherTrainConfig, dataset_csv: Path, out_dir: Path) -> int:
        if not cfg.enabled:
            return 0
        args: list[str] = [
            "--train_data_csv",
            str(dataset_csv),
            "--out_dir",
            str(out_dir),
            "--model_id",
            cfg.model_id,
            "--max_epochs",
            str(cfg.max_epochs),
        ]
        if cfg.feature_registry_dir is not None:
            args += ["--feature_registry_dir", cfg.feature_registry_dir]
        if cfg.feature_set_id is not None:
            args += ["--feature_set_id", cfg.feature_set_id]
        return self.teacher_main(args)

    def run(self, cfg: OrchestratorConfig) -> int:
        # 0) Optional pre-ingestion stage (unified orchestrator path)
        if cfg.pre_ingestion is not None:
            # Prefer environment CATALOG_PATH to keep configs portable
            import os

            catalog_path_env = os.getenv("CATALOG_PATH")
            if catalog_path_env:
                self.run_pre_ingestion(
                    catalog_path=Path(catalog_path_env),
                    scheduler_cfg=cfg.pre_ingestion,
                    options=cfg.pre_ingestion_options,
                )

        # 1) Build dataset
        rc = self.build_dataset(cfg.dataset)
        if rc != 0:
            return rc
        out_dir = Path(cfg.dataset.out_dir)
        dataset_csv = out_dir / "dataset.csv"

        # 2) HPO (optional)
        rc = self.run_hpo(cfg.hpo, dataset_csv=dataset_csv, out_dir=out_dir)
        if rc != 0:
            return rc

        # 3) Train teacher / calibration
        rc = self.train_teacher(cfg.teacher, dataset_csv=dataset_csv, out_dir=out_dir)
        if rc != 0:
            return rc

        self._handle_promotions(cfg.promotions, out_dir=out_dir, dataset_csv=dataset_csv)
        return rc

    def _handle_promotions(
        self,
        promotions: PromotionsConfig | None,
        *,
        out_dir: Path,
        dataset_csv: Path,
    ) -> None:
        if promotions is None:
            return

        from ml.orchestration.promotions import register_and_promote_model
        from ml.orchestration.promotions import register_or_refresh_features
        from ml.registry.dataclasses import QualityGate

        logger = logging.getLogger(__name__)

        feature_registry = self.feature_registry
        feature_metrics_path = Path(
            promotions.feature_metrics_json or (Path(out_dir) / "feature_metrics.json"),
        )
        if promotions.refresh_features or promotions.auto_register_features:
            if not feature_metrics_path.exists():
                feature_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "feature_set_id": f"auto_refresh_{_uuid.uuid4().hex[:8]}",
                    "generated_ts": int(time.time()),
                    "dataset_csv": str(dataset_csv),
                }
                feature_metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            succeeded = False
            if feature_registry is not None:
                try:
                    register_or_refresh_features(
                        feature_metrics_path=str(feature_metrics_path),
                        feature_registry=feature_registry,
                        auto_register=bool(promotions.auto_register_features),
                    )
                    succeeded = True
                except Exception as exc:
                    logger.warning("Feature refresh failed: %s", exc)
            if not succeeded:
                self._emit_feature_refresh_event(feature_metrics_path)

        should_promote_model = bool(
            promotions.auto_register_model or promotions.auto_promote or promotions.deploy_target,
        )

        if not should_promote_model or self.model_registry is None:
            return

        metrics_path = Path(out_dir) / "model_metrics.json"
        if not metrics_path.exists():
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_id": f"auto_model_{_uuid.uuid4().hex[:8]}",
                "model_path": str(Path(out_dir) / "model.onnx"),
                "architecture": "unknown",
                "feature_schema_hash": f"auto_{_uuid.uuid4().hex[:8]}",
                "serveable": True,
            }
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        gates: list[QualityGate] = []
        if promotions.gates_json:
            try:
                data = json.loads(Path(promotions.gates_json).read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        metric_name = item.get("metric") or item.get("metric_name")
                        if not metric_name:
                            continue
                        comparison_value = item.get("comparison")
                        comparison: str = (
                            str(comparison_value)
                            if comparison_value is not None
                            else "gte"
                        )
                        gates.append(
                            QualityGate(
                                metric_name=str(metric_name),
                                threshold=float(item.get("threshold", 0.0)),
                                comparison=comparison,
                                required=bool(item.get("required", True)),
                            ),
                        )
            except Exception as exc:
                logger.warning("Failed to parse gates JSON %s: %s", promotions.gates_json, exc)
                gates = []

        try:
            register_and_promote_model(
                model_metrics_path=str(metrics_path),
                out_dir=str(out_dir),
                registry=self.model_registry,
                feature_registry=self.feature_registry,
                gates=gates,
                auto_promote=bool(promotions.auto_promote),
                deploy_target=promotions.deploy_target,
            )
        except Exception as exc:
            logger.warning("Model promotion failed: %s", exc)

    def _emit_feature_refresh_event(self, metrics_path: Path) -> None:
        try:
            from ml.common.event_emitter import emit_dataset_event
            from ml.config.events import EventStatus
            from ml.config.events import Source
            from ml.config.events import Stage
        except Exception:
            return

        feature_set_id = "unknown"
        metadata: dict[str, object] = {}
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    feature_set_id = str(data.get("feature_set_id", feature_set_id))
                    metadata = {k: v for k, v in data.items() if isinstance(k, str)}
            except Exception:
                pass

        meta_payload = dict(metadata)
        meta_payload["feature_set_id"] = feature_set_id

        try:
            registry_obj = self.data_registry
            if registry_obj is None:
                return
            data_registry = cast(RegistryProtocol, registry_obj)
            emit_dataset_event(
                data_registry,
                dataset_id="features",
                instrument_id="GLOBAL",
                stage=Stage.FEATURE_COMPUTED,
                source=Source.HISTORICAL,
                run_id=f"refresh_{feature_set_id}",
                ts_min=0,
                ts_max=0,
                count=1,
                status=EventStatus.SUCCESS,
                metadata=meta_payload,
                dataset_type="features",
                component="pipeline_orchestrator.refresh_features",
            )
        except Exception:
            pass


@dataclass(slots=True, frozen=True)
class PreIngestionOptions:
    """
    Options for the pre-ingestion scheduler stage.
    """

    use_orchestrator: bool = True
    dual_write: bool = True
    start_metrics_server: bool = False
    metrics_port: int | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    import os

    parser = argparse.ArgumentParser(description="Run end-to-end ML pipeline (cold path)")

    # Ingestion/backfill
    parser.add_argument("--ingest", action="store_true", help="Run ingestion backfill first")
    parser.add_argument("--dataset_id", default="EQUS.MINI")
    parser.add_argument("--schema", default="bars", choices=["bars", "tbbo", "trades"])
    parser.add_argument("--instruments", default="SPY.NYSE")
    parser.add_argument("--lookback_days", type=int, default=7)
    parser.add_argument("--coverage_mode", default="catalog", choices=["catalog", "sql"])
    parser.add_argument("--catalog_path", default=os.getenv("CATALOG_PATH", ""))
    parser.add_argument(
        "--db",
        default=os.getenv("NAUTILUS_DB", "postgresql://postgres:postgres@localhost:5432/nautilus"),
    )

    # Writer mode for ingestion
    parser.add_argument("--write_mode", default="parquet", choices=["parquet", "datastore"])

    # Dataset build
    parser.add_argument("--data_dir", default="data/tier1")
    parser.add_argument("--symbols", default="SPY.NYSE")
    parser.add_argument("--out_dir", default="ml_out")
    parser.add_argument("--include_macro", action="store_true")
    parser.add_argument("--macro_lag_days", type=int, default=1)
    parser.add_argument("--include_micro", action="store_true")
    parser.add_argument("--include_l2", action="store_true")
    parser.add_argument("--include_events", action="store_true")
    parser.add_argument("--include_calendar", action="store_true")
    parser.add_argument(
        "--fred_vintage_dir",
        default=None,
        help="Optional ALFRED vintage directory",
    )
    parser.add_argument("--events_dir", default=None, help="Optional normalized events directory")
    parser.add_argument(
        "--student_mode",
        action="store_true",
        help="Build student-mode (L1-only) dataset",
    )
    parser.add_argument(
        "--emit_dataset_events",
        action="store_true",
        help="Emit dataset events via DataRegistry for the TFT build",
    )
    parser.add_argument("--horizon_minutes", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--lookback_periods", type=int, default=30)
    parser.add_argument("--start_iso", default=None, help="Optional start date ISO (YYYY-MM-DD)")
    parser.add_argument("--end_iso", default=None, help="Optional end date ISO (YYYY-MM-DD)")
    parser.add_argument(
        "--chunk_days",
        type=int,
        default=0,
        help="Chunk build by N days (0=disabled)",
    )

    # HPO
    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--hpo_epochs", type=int, default=2)
    parser.add_argument("--hpo_batch_size", type=int, default=32)
    parser.add_argument("--hpo_tail_rows", type=int, default=5000)
    parser.add_argument("--hpo_limit_groups", type=int, default=50)

    # Teacher training
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--teacher_model_id", default="teacher_model")
    parser.add_argument("--feature_registry_dir", default=None)
    parser.add_argument(
        "--dataset_register_features",
        action="store_true",
        help="Register features during dataset build using feature_registry_dir",
    )
    parser.add_argument("--feature_set_id", default=None)
    parser.add_argument("--max_epochs", type=int, default=5)

    # Optional promotions and feature registration
    parser.add_argument("--auto_register_model", action="store_true")
    parser.add_argument("--gates_json", default=None)
    parser.add_argument("--auto_promote", action="store_true")
    parser.add_argument("--deploy_target", default=None)

    parser.add_argument("--auto_register_features", action="store_true")
    parser.add_argument("--feature_metrics_json", default=None)

    # Optional small feature refresh phase
    parser.add_argument("--refresh_features", action="store_true")

    # Promotion stage 2 (walk-forward + cost-aware backtest)
    parser.add_argument("--promote_stage2", action="store_true")
    parser.add_argument("--stage2_gates_json", default=None)
    parser.add_argument("--stage2_cost_bps", type=float, default=0.0)
    parser.add_argument(
        "--stage2_engine",
        choices=["returns", "backtest"],
        default="returns",
        help="Stage 2 engine: returns (default) or backtest (advisory)",
    )
    parser.add_argument("--stage2_commission_bps", type=float, default=0.0)
    parser.add_argument("--stage2_slippage_bps", type=float, default=0.0)
    parser.add_argument(
        "--final_model_id",
        default=None,
        help="Model ID to promote in stage 2 (optional)",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
    raw_writer: RawIngestionWriterProtocol | None = None
    if args.write_mode == "datastore":
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
        try:
            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
        except Exception as exc:  # pragma: no cover - import env issue
            raise SystemExit(f"ParquetDataCatalog unavailable: {exc}")
        if not args.catalog_path:
            raise SystemExit("catalog_path is required for parquet write_mode")
        catalog = ParquetDataCatalog(args.catalog_path)
        writer = ParquetCatalogMarketDataWriter(catalog=catalog)
        raw_writer = ParquetCatalogRawWriter(catalog)
        from ml.core.integration import MLIntegrationManager

        mgr = MLIntegrationManager(
            db_connection=args.db,
            auto_start_postgres=False,
            auto_migrate=False,
            ensure_healthy=False,
        )
        registry = mgr.data_registry

    ingestor: object | None = None
    ingestion_service: DatabentoIngestionService | None = None
    if args.ingest:
        api_key = os.getenv("DATABENTO_API_KEY", "").strip()
        if api_key:
            client = DatabentoAPIClient(api_key=api_key)
            ingestor = DatabentoIngestor(client=client)
            try:
                ingestion_service = DatabentoIngestionService.from_env()
            except Exception as exc:  # pragma: no cover - runtime warning only
                logging.getLogger(__name__).warning(
                    "Failed to initialise ingestion service: %s",
                    exc,
                )

    from ml.cli.build_tft_dataset import main as build_main
    from ml.training.teacher.tft_cli import main as teacher_main

    try:
        from ml.cli.hpo_tft import main as _hpo_main

        hpo_main_cli: _CliMain | None = _hpo_main
    except Exception:  # pragma: no cover - optional dependency
        hpo_main_cli = None

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=registry,
        ingestor=ingestor if ingestor is not None else None,
        build_main=build_main,
        hpo_main=hpo_main_cli,
        teacher_main=teacher_main,
        raw_writer=raw_writer,
        service=ingestion_service,
        model_registry=getattr(mgr, "model_registry", None),
        feature_registry=getattr(mgr, "feature_registry", None),
    )

    if args.ingest and ingestor is not None:
        if ingestion_service is None:
            logging.getLogger(__name__).warning(
                "Ingestion requested but DatabentoIngestionService unavailable; skipping",
            )
        else:
            schema_map = {"bars": "ohlcv-1m", "tbbo": "tbbo", "trades": "trades"}
            provider_schema = schema_map.get(str(args.schema).lower(), str(args.schema))
            instruments = [s.strip() for s in str(args.instruments).split(",") if s.strip()]
            for inst in instruments:
                orch.backfill(
                    dataset_id=args.dataset_id,
                    schema=provider_schema,
                    instrument_id=inst,
                    lookback_days=int(args.lookback_days),
                )

    data_dir_effective = str(args.data_dir)
    if args.catalog_path and str(args.data_dir) == "data/tier1":
        data_dir_effective = str(args.catalog_path)

    ds_cfg = DatasetBuildConfig(
        data_dir=data_dir_effective,
        symbols=str(args.symbols),
        out_dir=str(args.out_dir),
        include_macro=bool(args.include_macro),
        macro_lag_days=int(args.macro_lag_days),
        include_micro=bool(args.include_micro),
        include_l2=bool(args.include_l2),
        include_events=bool(getattr(args, "include_events", False)),
        include_calendar=bool(getattr(args, "include_calendar", False)),
        fred_vintage_dir=str(args.fred_vintage_dir) if args.fred_vintage_dir else None,
        events_dir=str(args.events_dir) if args.events_dir else None,
        student_mode=bool(args.student_mode),
        emit_dataset_events=bool(getattr(args, "emit_dataset_events", False)),
        horizon_minutes=int(args.horizon_minutes),
        threshold=float(args.threshold),
        lookback_periods=int(args.lookback_periods),
        start_iso=args.start_iso,
        end_iso=args.end_iso,
        chunk_days=int(args.chunk_days),
        register_features=bool(args.dataset_register_features),
        feature_registry_dir=args.feature_registry_dir,
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

    promotions_cfg = PromotionsConfig(
        auto_register_model=bool(args.auto_register_model),
        gates_json=args.gates_json,
        auto_promote=bool(args.auto_promote),
        deploy_target=args.deploy_target,
        auto_register_features=bool(args.auto_register_features),
        feature_metrics_json=args.feature_metrics_json,
        refresh_features=bool(args.refresh_features),
    )

    orchestrator_cfg = OrchestratorConfig(
        dataset=ds_cfg,
        hpo=hpo_cfg,
        teacher=teacher_cfg,
        promotions=promotions_cfg,
    )

    orch.run(orchestrator_cfg)
    return 0
