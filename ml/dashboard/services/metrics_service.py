"""Store metrics integration service."""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from typing import TYPE_CHECKING, Any, Final

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager
from ml.dashboard.services.base_service import BaseIntegrationService


if TYPE_CHECKING:
    from ml.core.integration import MLIntegrationManager
    from ml.dashboard.store_health import StoreHealthSummary


logger = logging.getLogger(__name__)

_PERFORMANCE_WINDOW_SECONDS: Final[int] = 86_400
_INGESTION_WINDOW_SECONDS: Final[int] = 300


@dataclass(slots=True)
class IngestionRateSnapshot:
    """Aggregated ingestion metrics."""

    bars_per_sec: float = 0.0
    quotes_per_sec: float = 0.0
    l2_updates_per_sec: float = 0.0
    data_quality: float = 0.0


@dataclass(slots=True)
class PortfolioSnapshot:
    """Portfolio and exposure metrics."""

    total_value: float = 0.0
    cash: float = 0.0
    margin_used: float = 0.0
    positions: int = 0


@dataclass(slots=True)
class StoreMetricsSnapshot:
    """Comprehensive metrics consumed by the dashboard UI."""

    daily_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    active_models: int = 0
    ingestion_rate: IngestionRateSnapshot = field(default_factory=IngestionRateSnapshot)
    portfolio: PortfolioSnapshot = field(default_factory=PortfolioSnapshot)


@dataclass(slots=True)
class StoreHealthItemDetail:
    """Per-entity freshness information for a store."""

    key: str
    latest_event_iso: str | None
    age_seconds: float | None


@dataclass(slots=True)
class StoreHealthEntry:
    """High-level health status for an individual store."""

    store: str
    healthy: bool
    fallback_active: bool
    connectivity_ok: bool | None
    write_ok: bool | None
    buffer_backlog: int | None
    latest_event_iso: str | None
    age_seconds: float | None
    items: tuple[StoreHealthItemDetail, ...]
    error: str | None


@dataclass(slots=True)
class StoreHealthSummarySnapshot:
    """Composed health summary payload returned to the dashboard."""

    generated_at: str
    stores: tuple[StoreHealthEntry, ...]


@dataclass(slots=True)
class PerformanceMetricsAggregate:
    """Aggregated strategy/model metrics for dashboard KPIs."""

    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    active_models: int = 0
    active_strategies: int = 0


class StoreIntegrationService(BaseIntegrationService):
    """Integration facade exposing store metrics."""

    def __init__(
        self,
        integration_manager: MLIntegrationManager | None,
    ) -> None:
        super().__init__(integration_manager)

    def get_service_name(self) -> str:
        return "store_integration"

    async def health_check(self) -> dict[str, Any]:
        if not self._integration:
            return {"healthy": False, "reason": "No integration manager"}

        health: dict[str, Any] = {}
        for store_name in ("data_store", "model_store", "feature_store", "strategy_store"):
            store = getattr(self._integration, store_name, None)
            if store is None or not hasattr(store, "health_check"):
                health[store_name] = {"healthy": False, "reason": "Store not available"}
                continue

            try:
                health[store_name] = await self._run_async(partial(store.health_check))
            except Exception as exc:  # pragma: no cover - defensive
                health[store_name] = {"healthy": False, "error": str(exc)}

        return health

    async def get_metrics_snapshot(self) -> StoreMetricsSnapshot:
        """Return dashboard-friendly KPI metrics."""
        self._track_operation(operation="get_metrics", status="started")
        engine = self._resolve_engine()
        if engine is None:
            self._track_operation(operation="get_metrics", status="no_engine")
            return StoreMetricsSnapshot()

        try:
            snapshot = await self._run_async(partial(self._collect_metrics_snapshot, engine))
        except Exception:  # pragma: no cover - defensive
            logger.debug("store metrics aggregation failed", exc_info=True)
            self._track_operation(operation="get_metrics", status="failed")
            return StoreMetricsSnapshot()

        self._track_operation(operation="get_metrics", status="success")
        return snapshot

    async def get_store_health_summary(
        self,
        *,
        top_dataset_limit: int = 5,
    ) -> StoreHealthSummarySnapshot:
        """Collect cold-path store health data via shared dashboard helpers."""
        self._track_operation(operation="store_health", status="started")
        summaries = await self._run_async(
            partial(self._summarize_stores, top_dataset_limit=top_dataset_limit),
        )
        entries = tuple(self._convert_health_entry(summary) for summary in summaries)
        payload = StoreHealthSummarySnapshot(
            generated_at=dt.datetime.now(dt.UTC).isoformat(),
            stores=entries,
        )
        self._track_operation(operation="store_health", status="success")
        return payload

    def _summarize_stores(
        self,
        *,
        top_dataset_limit: int,
    ) -> tuple[StoreHealthSummary, ...]:
        from ml.dashboard.store_health import summarize_all_stores

        integration = self._integration
        feature_store = getattr(integration, "feature_store", None) if integration else None
        model_store = getattr(integration, "model_store", None) if integration else None
        strategy_store = getattr(integration, "strategy_store", None) if integration else None
        engine = self._resolve_engine()
        return summarize_all_stores(
            feature_store=feature_store,
            model_store=model_store,
            strategy_store=strategy_store,
            engine=engine,
            top_dataset_limit=top_dataset_limit,
            now=dt.datetime.now(dt.UTC),
        )

    def _resolve_engine(self) -> Engine | None:
        integration = self._integration
        if integration is None:
            return None
        connection = getattr(integration, "db_connection", None)
        if not connection:
            return None
        try:
            return EngineManager.get_engine(connection)
        except Exception:  # pragma: no cover - defensive
            return None

    def _convert_health_entry(self, summary: StoreHealthSummary) -> StoreHealthEntry:
        items = tuple(
            StoreHealthItemDetail(
                key=item.key,
                latest_event_iso=item.latest_event_iso,
                age_seconds=item.age_seconds,
            )
            for item in summary.items
        )
        return StoreHealthEntry(
            store=summary.store,
            healthy=summary.healthy,
            fallback_active=summary.fallback_active,
            connectivity_ok=summary.connectivity_ok,
            write_ok=summary.write_ok,
            buffer_backlog=summary.buffer_backlog,
            latest_event_iso=summary.latest_event_iso,
            age_seconds=summary.age_seconds,
            items=items,
            error=summary.error,
        )

    def _collect_metrics_snapshot(self, engine: Engine) -> StoreMetricsSnapshot:
        now_ns = time.time_ns()
        performance = self._compute_performance_metrics(engine, now_ns)
        ingestion_rate = self._compute_ingestion_metrics(engine, now_ns)
        portfolio, daily_pnl = self._compute_portfolio_metrics(engine)

        snapshot = StoreMetricsSnapshot(
            daily_pnl=daily_pnl,
            sharpe_ratio=performance.sharpe_ratio,
            win_rate=performance.win_rate,
            max_drawdown=performance.max_drawdown,
            active_models=max(performance.active_models, performance.active_strategies),
            ingestion_rate=ingestion_rate,
            portfolio=portfolio,
        )
        return snapshot

    def _compute_performance_metrics(self, engine: Engine, now_ns: int) -> PerformanceMetricsAggregate:
        aggregate = PerformanceMetricsAggregate()
        window_ns = _PERFORMANCE_WINDOW_SECONDS * 1_000_000_000
        cutoff_ns = max(now_ns - window_ns, 0)

        try:
            with engine.connect() as conn:
                signal_row = conn.execute(
                    text(
                        """
                        SELECT
                            COUNT(*) AS signal_count,
                            COUNT(DISTINCT strategy_id) AS strategy_count,
                            AVG(strength) AS avg_strength,
                            STDDEV_POP(strength) AS std_strength,
                            AVG(CASE WHEN strength > 0 THEN 1.0 ELSE 0.0 END) AS positive_ratio
                        FROM ml_strategy_signals
                        WHERE ts_event >= :cutoff
                        """
                    ),
                    {"cutoff": cutoff_ns},
                ).one_or_none()

                if signal_row is not None:
                    mapping = signal_row._mapping
                    aggregate.active_strategies = int(mapping.get("strategy_count") or 0)
                    avg_strength_val = mapping.get("avg_strength")
                    std_strength_val = mapping.get("std_strength")
                    positive_ratio_val = mapping.get("positive_ratio")
                    aggregate.sharpe_ratio = self._calculate_sharpe_ratio(
                        float(avg_strength_val) if avg_strength_val is not None else 0.0,
                        float(std_strength_val) if std_strength_val is not None else 0.0,
                    )
                    if positive_ratio_val is not None:
                        win_rate = float(positive_ratio_val)
                        aggregate.win_rate = max(0.0, min(win_rate, 1.0))

                model_row = conn.execute(
                    text(
                        """
                        SELECT COUNT(DISTINCT model_id) AS model_count
                        FROM ml_model_predictions
                        WHERE ts_event >= :cutoff
                        """
                    ),
                    {"cutoff": cutoff_ns},
                ).one_or_none()
                if model_row is not None:
                    aggregate.active_models = int(model_row._mapping.get("model_count") or 0)

                risk_row = conn.execute(
                    text(
                        "SELECT MAX(COALESCE(max_drawdown, 0)) AS max_drawdown FROM ml_risk_limits"
                    ),
                ).one_or_none()
                if risk_row is not None:
                    aggregate.max_drawdown = float(risk_row._mapping.get("max_drawdown") or 0.0)

        except Exception:  # pragma: no cover - defensive
            logger.debug("performance metrics query failed", exc_info=True)

        return aggregate

    def _compute_ingestion_metrics(self, engine: Engine, now_ns: int) -> IngestionRateSnapshot:
        snapshot = IngestionRateSnapshot()
        window_seconds = _INGESTION_WINDOW_SECONDS
        if window_seconds <= 0:
            return snapshot

        cutoff_ns = max(now_ns - window_seconds * 1_000_000_000, 0)

        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT
                            dataset_id,
                            SUM(count) AS total_count,
                            SUM(CASE WHEN status = 'success' THEN count ELSE 0 END) AS success_count
                        FROM ml_data_events
                        WHERE ts_event >= :cutoff
                          AND stage = 'INGESTED'
                        GROUP BY dataset_id
                        """
                    ),
                    {"cutoff": cutoff_ns},
                ).fetchall()
        except Exception:  # pragma: no cover - defensive
            logger.debug("ingestion metrics query failed", exc_info=True)
            return snapshot

        totals: dict[str, int] = {"bars": 0, "quotes": 0, "l2": 0}
        total_records = 0
        success_records = 0
        for row in rows:
            mapping = row._mapping
            dataset_id = str(mapping.get("dataset_id") or "")
            bucket = self._categorize_dataset(dataset_id)
            if bucket is None:
                continue
            total_count = int(mapping.get("total_count") or 0)
            success_count = int(mapping.get("success_count") or 0)
            totals[bucket] += total_count
            total_records += total_count
            success_records += success_count

        duration = float(window_seconds)
        snapshot.bars_per_sec = totals["bars"] / duration if duration else 0.0
        snapshot.quotes_per_sec = totals["quotes"] / duration if duration else 0.0
        snapshot.l2_updates_per_sec = totals["l2"] / duration if duration else 0.0
        snapshot.data_quality = (
            success_records / total_records if total_records else 1.0
        )
        return snapshot

    def _compute_portfolio_metrics(self, engine: Engine) -> tuple[PortfolioSnapshot, float]:
        portfolio = PortfolioSnapshot()
        daily_pnl = 0.0

        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT
                            COALESCE(SUM(position_value), 0) AS total_value,
                            COALESCE(SUM(exposure), 0) AS exposure,
                            COALESCE(SUM(unrealized_pnl), 0) AS unrealized_pnl,
                            COALESCE(SUM(realized_pnl), 0) AS realized_pnl,
                            COUNT(*) AS position_count
                        FROM ml_positions
                        """
                    ),
                ).one_or_none()
        except Exception:  # pragma: no cover - defensive
            logger.debug("portfolio metrics query failed", exc_info=True)
            return portfolio, 0.0

        if row is None:
            return portfolio, 0.0

        mapping = row._mapping
        total_value = float(mapping.get("total_value") or 0.0)
        exposure = float(mapping.get("exposure") or 0.0)
        unrealized = float(mapping.get("unrealized_pnl") or 0.0)
        realized = float(mapping.get("realized_pnl") or 0.0)
        position_count = int(mapping.get("position_count") or 0)

        portfolio.total_value = total_value
        portfolio.margin_used = exposure
        portfolio.positions = position_count
        portfolio.cash = max(total_value - exposure, 0.0)
        daily_pnl = unrealized + realized
        return portfolio, daily_pnl

    @staticmethod
    def _calculate_sharpe_ratio(avg: float, std: float) -> float:
        if std <= 0.0:
            return 0.0
        return avg / std

    @staticmethod
    def _categorize_dataset(dataset_id: str) -> str | None:
        lowered = dataset_id.lower()
        if "bar" in lowered:
            return "bars"
        if "quote" in lowered or "nbbo" in lowered:
            return "quotes"
        if "l2" in lowered or "book" in lowered or "depth" in lowered:
            return "l2"
        return None


__all__ = [
    "IngestionRateSnapshot",
    "PortfolioSnapshot",
    "StoreHealthEntry",
    "StoreHealthItemDetail",
    "StoreHealthSummarySnapshot",
    "StoreIntegrationService",
    "StoreMetricsSnapshot",
]
