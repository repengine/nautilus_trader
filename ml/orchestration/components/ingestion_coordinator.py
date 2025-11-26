"""
IngestionCoordinator component - extracted from MLPipelineOrchestrator (Phase 2.2.1).

Responsibilities:
- Coordinate data ingestion from multiple sources (Databento, Yahoo, FRED, Earnings)
- Progressive fallback chains (PRIMARY → CACHED → FILE → DUMMY)
- Checkpoint creation and resume functionality
- Data validation and error handling
- Metrics and event emission
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ml.config.coverage import CoveragePolicy
    from ml.config.scheduler_config import SchedulerConfig
    from ml.data.ingest.market_bindings import ResolvedMarketBinding
    from ml.data.ingest.orchestrator import BackfillWindowList
    from ml.orchestration.config_types import PreIngestionOptions
    from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
    from ml.registry.data_registry import DataRegistry
    from ml.stores.data_store import DataStore


logger = logging.getLogger(__name__)


class IngestionCoordinator:
    """
    Coordinates data ingestion from multiple sources with progressive fallback.

    Handles:
    - Multi-source ingestion (Databento, Yahoo Finance, FRED, Earnings)
    - Progressive fallback chains (PRIMARY → CACHED → FILE → DUMMY)
    - Checkpoint/resume functionality for interrupted ingestion
    - Data validation and quality checks
    - Metrics emission and event publishing

    This component follows the delegation pattern to the parent MLPipelineOrchestrator,
    ensuring behavioral parity while providing a focused interface for ingestion operations.

    Parameters
    ----------
    orchestrator : MLPipelineOrchestrator
        Parent orchestrator instance (for delegation to preserve legacy behavior)
    data_store : DataStore
        DataStore for persisting ingested data
    data_registry : DataRegistry
        DataRegistry for dataset manifest management

    Examples
    --------
    >>> from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
    >>> from ml.orchestration.components import IngestionCoordinator
    >>> orchestrator = MLPipelineOrchestrator(config)
    >>> coordinator = IngestionCoordinator(
    ...     orchestrator=orchestrator,
    ...     data_store=orchestrator.data_store,
    ...     data_registry=orchestrator.data_registry,
    ... )
    >>> # Coordinate ingestion with progressive fallback
    >>> summary = coordinator.coordinate_ingestion(
    ...     dataset_id="databento.ohlcv-1s",
    ...     schema="ohlcv-1s",
    ...     instrument_ids=["SPY.NASDAQ"],
    ...     lookback_days=30,
    ... )
    """

    def __init__(
        self,
        orchestrator: MLPipelineOrchestrator,
        data_store: DataStore,
        data_registry: DataRegistry,
    ) -> None:
        """
        Initialize IngestionCoordinator with dependencies.

        Parameters
        ----------
        orchestrator : MLPipelineOrchestrator
            Parent orchestrator for delegation (preserves behavioral parity)
        data_store : DataStore
            DataStore instance for data persistence
        data_registry : DataRegistry
            DataRegistry instance for manifest management
        """
        self._orchestrator = orchestrator
        self._data_store = data_store
        self._data_registry = data_registry

    # =========================================================================
    # Main Coordination Method
    # =========================================================================

    def coordinate_ingestion(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_ids: list[str],
        lookback_days: int,
        policy: CoveragePolicy | None = None,
    ) -> dict[str, int | str]:
        """
        Main ingestion coordination method with progressive fallback.

        Implements the fallback chain: PRIMARY (binding) → CACHED (coverage) →
        FILE (local) → DUMMY (empty with warnings).

        Parameters
        ----------
        dataset_id : str
            Dataset identifier (e.g., "databento.ohlcv-1s")
        schema : str
            Schema name (e.g., "ohlcv-1s", "tbbo", "trades")
        instrument_ids : list[str]
            List of instrument IDs to ingest
        lookback_days : int
            Number of days to look back for historical data
        policy : CoveragePolicy | None, optional
            Coverage policy for determining lookback bounds

        Returns
        -------
        dict[str, int | str]
            Ingestion summary with keys:
            - rows_written: Number of rows persisted
            - fallback_level: "primary" | "cached" | "file" | "dummy"
            - error: Error message if applicable

        Examples
        --------
        >>> summary = coordinator.coordinate_ingestion(
        ...     dataset_id="databento.ohlcv-1s",
        ...     schema="ohlcv-1s",
        ...     instrument_ids=["SPY.NASDAQ", "QQQ.NASDAQ"],
        ...     lookback_days=30,
        ... )
        >>> assert summary["rows_written"] > 0
        >>> assert summary["fallback_level"] == "primary"
        """
        logger.info(
            "coordinate_ingestion_called",
            extra={
                "dataset_id": dataset_id,
                "schema": schema,
                "instrument_count": len(instrument_ids),
                "lookback_days": lookback_days,
            },
        )

        # Placeholder implementation - tests are skipped
        # In production, this would implement:
        # 1. Try PRIMARY ingestion (binding-based)
        # 2. On failure, try CACHED (coverage-based)
        # 3. On failure, try FILE (local files)
        # 4. On failure, activate DUMMY (empty data + warnings)
        return {
            "rows_written": 0,
            "fallback_level": "dummy",
            "error": "Component structure only - implementation pending",
        }

    # =========================================================================
    # Data Source Methods
    # =========================================================================

    def ingest_from_databento(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> BackfillWindowList:
        """
        Ingest OHLCV data from Databento API (L1).

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Schema name
        instrument_id : str
            Instrument to ingest
        lookback_days : int
            Days to look back

        Returns
        -------
        BackfillWindowList
            Backfill result with rows_written and window_count
        """
        logger.info(
            "ingest_from_databento_called",
            extra={
                "dataset_id": dataset_id,
                "schema": schema,
                "instrument_id": instrument_id,
                "lookback_days": lookback_days,
            },
        )
        # Placeholder - tests are skipped
        from ml.data.ingest.orchestrator import BackfillWindowList
        return BackfillWindowList(
            persisted=(),
            requested=(),
            frames_written=0,
            rows_written=0,
        )

    def ingest_from_yahoo(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> int:
        """
        Ingest fundamentals from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            Stock symbol
        start_date : str
            Start date (ISO format)
        end_date : str
            End date (ISO format)

        Returns
        -------
        int
            Number of rows ingested
        """
        logger.info(
            "ingest_from_yahoo_called",
            extra={"symbol": symbol, "start_date": start_date, "end_date": end_date},
        )
        # Placeholder - tests are skipped
        return 0

    def ingest_from_fred(
        self,
        *,
        series_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> int:
        """
        Ingest macro indicators from FRED.

        Parameters
        ----------
        series_ids : list[str]
            FRED series IDs (e.g., ["GDP", "UNRATE"])
        start_date : str
            Start date (ISO format)
        end_date : str
            End date (ISO format)

        Returns
        -------
        int
            Number of rows ingested
        """
        logger.info(
            "ingest_from_fred_called",
            extra={
                "series_ids": series_ids,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        # Placeholder - tests are skipped
        return 0

    def ingest_earnings_data(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> int:
        """
        Ingest earnings data (alternative data).

        Parameters
        ----------
        symbol : str
            Stock symbol
        start_date : str
            Start date (ISO format)
        end_date : str
            End date (ISO format)

        Returns
        -------
        int
            Number of rows ingested
        """
        logger.info(
            "ingest_earnings_data_called",
            extra={"symbol": symbol, "start_date": start_date, "end_date": end_date},
        )
        # Placeholder - tests are skipped
        return 0

    # =========================================================================
    # Backfill Methods (Delegate to Orchestrator)
    # =========================================================================

    def backfill(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> BackfillWindowList:
        """
        Backfill gaps for a single instrument.

        Delegates to orchestrator.backfill() for guaranteed behavioral parity.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Schema name
        instrument_id : str
            Instrument ID
        lookback_days : int
            Days to look back

        Returns
        -------
        BackfillWindowList
            Backfill result with rows_written and window_count
        """
        return self._orchestrator.backfill(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            lookback_days=lookback_days,
        )

    def backfill_binding(
        self,
        *,
        binding: ResolvedMarketBinding,
        lookback_days: int,
    ) -> dict[str, BackfillWindowList]:
        """
        Backfill using resolved market binding (PRIMARY path).

        Delegates to orchestrator.backfill_binding() for guaranteed behavioral parity.

        Parameters
        ----------
        binding : ResolvedMarketBinding
            Resolved market binding
        lookback_days : int
            Days to look back

        Returns
        -------
        dict[str, BackfillWindowList]
            Map of instrument_id to backfill results
        """
        return self._orchestrator.backfill_binding(
            binding=binding,
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
        Backfill using coverage policy (CACHED path).

        Delegates to orchestrator.backfill_coverage() for guaranteed behavioral parity.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Schema name
        instrument_id : str
            Instrument ID
        policy : CoveragePolicy | None, optional
            Coverage policy for determining lookback bounds

        Returns
        -------
        list[tuple[int, int]]
            List of (start_ns, end_ns) windows
        """
        return self._orchestrator.backfill_coverage(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            policy=policy,
        )

    def run_pre_ingestion(
        self,
        *,
        catalog_path: Path,
        scheduler_cfg: SchedulerConfig,
        options: PreIngestionOptions | None = None,
    ) -> None:
        """
        Run pre-ingestion stage using DataScheduler.

        Delegates to orchestrator.run_pre_ingestion() for guaranteed behavioral parity.

        Parameters
        ----------
        catalog_path : Path
            Path to Parquet catalog
        scheduler_cfg : SchedulerConfig
            Scheduler configuration
        options : PreIngestionOptions | None, optional
            Pre-ingestion options (dual-write, metrics, etc.)
        """
        self._orchestrator.run_pre_ingestion(
            catalog_path=catalog_path,
            scheduler_cfg=scheduler_cfg,
            options=options,
        )

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    def _handle_ingestion_fallback(
        self,
        *,
        dataset_id: str,
        instrument_ids: list[str],
        lookback_days: int,
        level: str,
    ) -> dict[str, int | str]:
        """
        Handle ingestion fallback chain logic.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_ids : list[str]
            Instruments to ingest
        lookback_days : int
            Days to look back
        level : str
            Current fallback level

        Returns
        -------
        dict[str, int | str]
            Ingestion summary
        """
        logger.warning(
            "ingestion_fallback_activated",
            extra={"dataset_id": dataset_id, "level": level},
        )
        # Placeholder - tests are skipped
        return {"rows_written": 0, "fallback_level": level, "error": "Fallback activated"}

    def _create_ingestion_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        rows_written: int,
        current_instrument_index: int,
        progress: float,
    ) -> None:
        """
        Create ingestion checkpoint for resume capability.

        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file
        rows_written : int
            Rows written so far
        current_instrument_index : int
            Current instrument index
        progress : float
            Progress fraction (0.0 to 1.0)
        """
        checkpoint_data = {
            "rows_written": rows_written,
            "current_instrument_index": current_instrument_index,
            "progress": progress,
        }
        checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))
        logger.info(
            "ingestion_checkpoint_created",
            extra={"checkpoint_path": str(checkpoint_path), "progress": progress},
        )

    def _restore_from_checkpoint(
        self,
        *,
        checkpoint_path: Path,
    ) -> dict[str, int | float]:
        """
        Restore ingestion state from checkpoint.

        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file

        Returns
        -------
        dict[str, int | float]
            Checkpoint data (rows_written, current_instrument_index, progress)
        """
        if not checkpoint_path.exists():
            logger.info("no_checkpoint_found", extra={"checkpoint_path": str(checkpoint_path)})
            return {"rows_written": 0, "current_instrument_index": 0, "progress": 0.0}

        checkpoint_data: dict[str, int | float] = json.loads(checkpoint_path.read_text())
        logger.info(
            "ingestion_checkpoint_restored",
            extra={"checkpoint_path": str(checkpoint_path), "data": checkpoint_data},
        )
        return checkpoint_data

    def _validate_ingestion_data(
        self,
        *,
        data: object,  # Could be DataFrame, dict, etc.
        instrument_id: str,
    ) -> tuple[bool, list[str]]:
        """
        Validate ingested data meets quality standards.

        Parameters
        ----------
        data : object
            Data to validate (DataFrame, dict, etc.)
        instrument_id : str
            Instrument ID for context

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, error_messages)
        """
        # Placeholder - tests are skipped
        logger.debug(
            "validate_ingestion_data_called",
            extra={"instrument_id": instrument_id},
        )
        return True, []

    def _emit_ingestion_event(
        self,
        *,
        event_type: str,
        dataset_id: str,
        rows_written: int,
    ) -> None:
        """
        Emit ingestion event to message bus.

        Parameters
        ----------
        event_type : str
            Event type (e.g., "ingestion_completed")
        dataset_id : str
            Dataset identifier
        rows_written : int
            Number of rows written
        """
        logger.info(
            "ingestion_event_emitted",
            extra={
                "event_type": event_type,
                "dataset_id": dataset_id,
                "rows_written": rows_written,
            },
        )
        # Placeholder - tests are skipped
        # In production, would publish to message bus

    def _get_ingestion_state(self) -> dict[str, object]:
        """
        Get current ingestion state.

        Returns
        -------
        dict[str, object]
            Current state
        """
        # Placeholder - tests are skipped
        return {}

    def _update_ingestion_state(
        self,
        *,
        rows_written: int,
        current_instrument: str,
    ) -> None:
        """
        Update ingestion state.

        Parameters
        ----------
        rows_written : int
            Total rows written
        current_instrument : str
            Current instrument being processed
        """
        logger.debug(
            "ingestion_state_updated",
            extra={"rows_written": rows_written, "current_instrument": current_instrument},
        )
        # Placeholder - tests are skipped
