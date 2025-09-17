#!/usr/bin/env python3

"""
High-level ML pipeline orchestrator (cold path only).

Composes existing ingestion, dataset build, HPO, and training CLIs into a typed,
testable interface suitable for a single long-running service or a nightly batch job.
All heavy work (DataFrames, file I/O, GPU training) remains strictly off the actor hot
paths.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol


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
    horizon_minutes: int = 15
    threshold: float = 0.001
    lookback_periods: int = 30


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
    registry: object  # RegistryProtocol at runtime; kept lax to avoid import cycles
    ingestor: object  # DatabentoIngestor or similar

    build_main: _CliMain
    hpo_main: _CliMain | None
    teacher_main: _CliMain

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
        from ml.data import DataScheduler
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
            registry=self.registry,  # type: ignore[arg-type]
            ingestor=self.ingestor,  # type: ignore[arg-type]
        )
        return orchestrator.backfill_gaps(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            lookback_days=lookback_days,
        )

    def build_dataset(self, cfg: DatasetBuildConfig) -> int:
        # Prefer the public API when CLI main is not provided (keeps tests stubbing CLI intact)
        if self.build_main is None:
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
                    horizon_minutes=cfg.horizon_minutes,
                    threshold=cfg.threshold,
                    lookback_periods=cfg.lookback_periods,
                )
                api_build(api_cfg)
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
        return rc


@dataclass(slots=True, frozen=True)
class PreIngestionOptions:
    """
    Options for the pre-ingestion scheduler stage.
    """

    use_orchestrator: bool = True
    dual_write: bool = True
    start_metrics_server: bool = False
    metrics_port: int | None = None
