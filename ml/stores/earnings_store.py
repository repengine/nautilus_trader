"""
Earnings data store for corporate fundamentals integration.

This module provides PostgreSQL-backed storage for earnings actuals (SEC EDGAR),
consensus estimates (Yahoo Finance), and earnings calendar with point-in-time correctness.

Key principles:
- Protocol-first design (EarningsStoreProtocol)
- Progressive fallback to DummyEarningsStore when PostgreSQL unavailable
- Point-in-time queries prevent look-ahead bias in backtesting
- All timestamps in nanoseconds (ts_event, ts_init)

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import BIGINT
from sqlalchemy import VARCHAR
from sqlalchemy import Column
from sqlalchemy import Date
from sqlalchemy import Index
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
from sqlalchemy.dialects.postgresql import INTEGER
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import CreateSchema
from sqlalchemy.sql import Select

from ml.common.metrics_bootstrap import get_counter
from ml.core.db_engine import EngineManager


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# Metrics
_earnings_writes = get_counter(
    "ml_earnings_writes_total",
    "Total earnings data writes",
    labelnames=["table"],
)
_earnings_reads = get_counter(
    "ml_earnings_reads_total",
    "Total earnings data reads",
    labelnames=["table"],
)
_earnings_fallbacks = get_counter(
    "ml_earnings_fallback_total",
    "Total fallbacks to DummyEarningsStore",
)


class EarningsStore:
    """
    PostgreSQL-backed earnings data store.

    Stores earnings actuals, estimates, and calendar with full point-in-time support.
    Implements EarningsStoreProtocol for structural typing.

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string (e.g., 'postgresql://user:pass@host/db')
    schema : str
        Database schema name (default: 'ml')

    """

    def __init__(
        self,
        connection_string: str,
        schema: str = "ml",
    ) -> None:
        self._engine: Engine = EngineManager.get_engine(connection_string)
        self._schema = schema
        self._metadata = MetaData(schema=schema)

        self._ensure_schema_exists()

        # Define earnings_actuals table
        self._actuals_table = Table(
            "earnings_actuals",
            self._metadata,
            Column("ticker", VARCHAR(20), nullable=False),
            Column("period_end", Date, nullable=False),
            Column("filing_date", Date, nullable=False),
            Column("ts_event", BIGINT, nullable=False),
            Column("ts_init", BIGINT, nullable=False),
            Column("eps_basic", DOUBLE_PRECISION),
            Column("eps_diluted", DOUBLE_PRECISION),
            Column("revenue", DOUBLE_PRECISION),
            Column("net_income", DOUBLE_PRECISION),
            Column("operating_income", DOUBLE_PRECISION),
            Column("shares_outstanding", BIGINT),
            Column("filing_type", VARCHAR(10)),
            Column("fiscal_year", INTEGER),
            Column("fiscal_quarter", INTEGER),
            Column("data_source", VARCHAR(20), server_default="EDGAR"),
            Index("idx_earnings_actuals_ts_event", "ts_event"),
            Index("idx_earnings_actuals_ticker", "ticker"),
            Index("idx_earnings_actuals_filing_date", "filing_date"),
            schema=schema,
        )

        # Define earnings_estimates table
        self._estimates_table = Table(
            "earnings_estimates",
            self._metadata,
            Column("ticker", VARCHAR(20), nullable=False),
            Column("estimate_date", Date, nullable=False),
            Column("period_end", Date, nullable=False),
            Column("ts_event", BIGINT, nullable=False),
            Column("ts_init", BIGINT, nullable=False),
            Column("eps_consensus", DOUBLE_PRECISION),
            Column("revenue_consensus", DOUBLE_PRECISION),
            Column("num_analysts", INTEGER),
            Column("data_source", VARCHAR(20), server_default="YAHOO"),
            Index("idx_earnings_estimates_ts_event", "ts_event"),
            Index("idx_earnings_estimates_ticker", "ticker"),
            Index("idx_earnings_estimates_period", "period_end"),
            schema=schema,
        )

        self._metadata.create_all(
            self._engine,
            tables=(self._actuals_table, self._estimates_table),
        )

        logger.info(
            "Initialized EarningsStore",
            extra={"schema": schema, "engine": str(self._engine.url)},
        )

    def _ensure_schema_exists(self) -> None:
        """Create the configured schema when it does not already exist."""
        try:
            with self._engine.begin() as conn:
                conn.execute(CreateSchema(self._schema))
        except SQLAlchemyError as exc:
            message = str(exc).lower()
            if "already exists" not in message:
                raise

    def _build_actuals_query(
        self,
        *,
        ticker: str,
        start_date: str | None,
        end_date: str | None,
        as_of_ts: int | None,
    ) -> Select[Any]:
        """Return the parametrized query used for fetching earnings actuals."""
        query = select(self._actuals_table).where(self._actuals_table.c.ticker == ticker)

        if start_date is not None:
            query = query.where(self._actuals_table.c.period_end >= start_date)

        if end_date is not None:
            query = query.where(self._actuals_table.c.period_end <= end_date)

        if as_of_ts is not None:
            query = query.where(self._actuals_table.c.ts_event < as_of_ts)

        return query.order_by(
            self._actuals_table.c.period_end.desc(),
            self._actuals_table.c.ts_event.desc(),
        )

    def _build_estimates_query(
        self,
        *,
        ticker: str,
        period_end: str,
        as_of_ts: int | None,
    ) -> Select[Any]:
        """Return the parametrized query used for fetching earnings estimates."""
        query = select(self._estimates_table).where(
            (self._estimates_table.c.ticker == ticker)
            & (self._estimates_table.c.period_end == period_end),
        )

        if as_of_ts is not None:
            query = query.where(self._estimates_table.c.ts_event < as_of_ts)

        return query.order_by(
            self._estimates_table.c.estimate_date.desc(),
            self._estimates_table.c.ts_event.desc(),
        )

    def write_actuals(
        self,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        eps_basic: float | None = None,
        net_income: float | None = None,
        operating_income: float | None = None,
        shares_outstanding: int | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        fiscal_quarter: int | None = None,
    ) -> None:
        """
        Write actual earnings data from SEC EDGAR.

        Uses upsert (INSERT ... ON CONFLICT DO UPDATE) to handle duplicates.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL')
        period_end : str
            Quarter end date (ISO format: 'YYYY-MM-DD')
        filing_date : str
            10-Q/10-K filing date (ISO format)
        eps_diluted : float | None
            Diluted earnings per share
        revenue : float | None
            Total revenue in dollars
        ts_event : int
            Filing date in nanoseconds
        ts_init : int
            Record creation timestamp in nanoseconds
        eps_basic : float | None
            Basic earnings per share
        net_income : float | None
            Net income in dollars
        operating_income : float | None
            Operating income in dollars
        shares_outstanding : int | None
            Weighted average shares outstanding
        filing_type : str | None
            '10-Q' or '10-K'
        fiscal_year : int | None
            Fiscal year
        fiscal_quarter : int | None
            Fiscal quarter (1-4)

        """
        values = {
            "ticker": ticker,
            "period_end": period_end,
            "filing_date": filing_date,
            "ts_event": ts_event,
            "ts_init": ts_init,
            "eps_basic": eps_basic,
            "eps_diluted": eps_diluted,
            "revenue": revenue,
            "net_income": net_income,
            "operating_income": operating_income,
            "shares_outstanding": shares_outstanding,
            "filing_type": filing_type,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
        }

        stmt = pg_insert(self._actuals_table).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "period_end"],
            set_={
                "filing_date": stmt.excluded.filing_date,
                "ts_event": stmt.excluded.ts_event,
                "ts_init": stmt.excluded.ts_init,
                "eps_basic": stmt.excluded.eps_basic,
                "eps_diluted": stmt.excluded.eps_diluted,
                "revenue": stmt.excluded.revenue,
                "net_income": stmt.excluded.net_income,
                "operating_income": stmt.excluded.operating_income,
                "shares_outstanding": stmt.excluded.shares_outstanding,
                "filing_type": stmt.excluded.filing_type,
                "fiscal_year": stmt.excluded.fiscal_year,
                "fiscal_quarter": stmt.excluded.fiscal_quarter,
            },
        )

        with self._engine.begin() as conn:
            conn.execute(stmt)

        _earnings_writes.labels(table="actuals").inc()
        logger.debug("Wrote earnings actual", extra={"ticker": ticker, "period_end": period_end})

    def write_estimates(
        self,
        ticker: str,
        estimate_date: str,
        period_end: str,
        eps_consensus: float | None,
        ts_event: int,
        ts_init: int,
        revenue_consensus: float | None = None,
        num_analysts: int | None = None,
    ) -> None:
        """
        Write consensus earnings estimates from Yahoo Finance.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        estimate_date : str
            Date estimate was recorded (ISO format)
        period_end : str
            Quarter being estimated (ISO format)
        eps_consensus : float | None
            Consensus EPS estimate
        ts_event : int
            Estimate date in nanoseconds
        ts_init : int
            Record creation timestamp in nanoseconds
        revenue_consensus : float | None
            Consensus revenue estimate
        num_analysts : int | None
            Number of analysts contributing

        """
        values = {
            "ticker": ticker,
            "estimate_date": estimate_date,
            "period_end": period_end,
            "ts_event": ts_event,
            "ts_init": ts_init,
            "eps_consensus": eps_consensus,
            "revenue_consensus": revenue_consensus,
            "num_analysts": num_analysts,
        }

        stmt = pg_insert(self._estimates_table).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "estimate_date", "period_end"],
            set_={
                "ts_event": stmt.excluded.ts_event,
                "ts_init": stmt.excluded.ts_init,
                "eps_consensus": stmt.excluded.eps_consensus,
                "revenue_consensus": stmt.excluded.revenue_consensus,
                "num_analysts": stmt.excluded.num_analysts,
            },
        )

        with self._engine.begin() as conn:
            conn.execute(stmt)

        _earnings_writes.labels(table="estimates").inc()
        logger.debug(
            "Wrote earnings estimate",
            extra={"ticker": ticker, "period_end": period_end, "estimate_date": estimate_date},
        )

    def get_actuals(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get actual earnings for a ticker with point-in-time filtering.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        start_date : str | None
            Start date filter (ISO format, inclusive)
        end_date : str | None
            End date filter (ISO format, inclusive)
        as_of_ts : int | None
            Point-in-time timestamp (only include filings with ts_event < as_of_ts)

        Returns
        -------
        list[dict[str, Any]]
            List of actual earnings records, sorted by period_end descending

        """
        query = self._build_actuals_query(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            as_of_ts=as_of_ts,
        )

        with self._engine.connect() as conn:
            result = conn.execute(query)
            rows = [dict(row._mapping) for row in result]

        _earnings_reads.labels(table="actuals").inc()
        logger.debug(
            "Read earnings actuals",
            extra={
                "ticker": ticker,
                "count": len(rows),
                "as_of_ts": as_of_ts,
            },
        )

        return rows

    def get_estimates(
        self,
        ticker: str,
        period_end: str,
        as_of_ts: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Get consensus estimate for a specific period.

        Returns the most recent estimate before as_of_ts (if specified).

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        period_end : str
            Quarter being estimated (ISO format)
        as_of_ts : int | None
            Point-in-time timestamp (get estimate with ts_event < as_of_ts)

        Returns
        -------
        dict[str, Any] | None
            Estimate record or None if not found

        """
        query = self._build_estimates_query(
            ticker=ticker,
            period_end=period_end,
            as_of_ts=as_of_ts,
        ).limit(1)

        with self._engine.connect() as conn:
            result = conn.execute(query)
            row = result.first()

        _earnings_reads.labels(table="estimates").inc()

        if row is None:
            logger.debug(
                "No earnings estimate found",
                extra={"ticker": ticker, "period_end": period_end},
            )
            return None

        logger.debug(
            "Read earnings estimate",
            extra={"ticker": ticker, "period_end": period_end},
        )
        return dict(row._mapping)

    def flush(self) -> None:
        """
        Flush any pending writes to persistent storage.

        For PostgreSQL with autocommit, this is a no-op.

        """
        # PostgreSQL commits immediately with autocommit


class DummyEarningsStore:
    """
    In-memory fallback earnings store for testing and PostgreSQL unavailability.

    Implements EarningsStoreProtocol but stores data in memory dictionaries.
    Useful for testing and when PostgreSQL is not available.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._actuals: dict[tuple[str, str], dict[str, Any]] = {}
        self._estimates: dict[tuple[str, str, str], dict[str, Any]] = {}
        logger.warning("Using DummyEarningsStore - data will not persist")
        _earnings_fallbacks.inc()

    def write_actuals(
        self,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        eps_basic: float | None = None,
        net_income: float | None = None,
        operating_income: float | None = None,
        shares_outstanding: int | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        fiscal_quarter: int | None = None,
    ) -> None:
        """Write actual earnings to in-memory store."""
        key = (ticker, period_end)
        self._actuals[key] = {
            "ticker": ticker,
            "period_end": period_end,
            "filing_date": filing_date,
            "ts_event": ts_event,
            "ts_init": ts_init,
            "eps_basic": eps_basic,
            "eps_diluted": eps_diluted,
            "revenue": revenue,
            "net_income": net_income,
            "operating_income": operating_income,
            "shares_outstanding": shares_outstanding,
            "filing_type": filing_type,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
        }

    def write_estimates(
        self,
        ticker: str,
        estimate_date: str,
        period_end: str,
        eps_consensus: float | None,
        ts_event: int,
        ts_init: int,
        revenue_consensus: float | None = None,
        num_analysts: int | None = None,
    ) -> None:
        """Write earnings estimate to in-memory store."""
        key = (ticker, estimate_date, period_end)
        self._estimates[key] = {
            "ticker": ticker,
            "estimate_date": estimate_date,
            "period_end": period_end,
            "ts_event": ts_event,
            "ts_init": ts_init,
            "eps_consensus": eps_consensus,
            "revenue_consensus": revenue_consensus,
            "num_analysts": num_analysts,
        }

    def get_actuals(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get actuals from in-memory store with filtering."""
        results = []
        for (tick, period), data in self._actuals.items():
            if tick != ticker:
                continue

            if start_date is not None and period < start_date:
                continue

            if end_date is not None and period > end_date:
                continue

            if as_of_ts is not None and data["ts_event"] >= as_of_ts:
                continue

            results.append(data)

        # Sort by period_end descending
        results.sort(key=lambda x: x["period_end"], reverse=True)
        return results

    def get_estimates(
        self,
        ticker: str,
        period_end: str,
        as_of_ts: int | None = None,
    ) -> dict[str, Any] | None:
        """Get estimate from in-memory store."""
        matching_estimates = []

        for (tick, est_date, period), data in self._estimates.items():
            if tick != ticker or period != period_end:
                continue

            if as_of_ts is not None and data["ts_event"] >= as_of_ts:
                continue

            matching_estimates.append(data)

        if not matching_estimates:
            return None

        # Return most recent estimate
        matching_estimates.sort(key=lambda x: x["estimate_date"], reverse=True)
        return matching_estimates[0]

    def flush(self) -> None:
        """No-op for in-memory store."""


__all__ = [
    "DummyEarningsStore",
    "EarningsStore",
]
