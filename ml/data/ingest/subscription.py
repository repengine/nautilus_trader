"""
Unified Databento subscription management and policy enforcement.

Consolidates subscription coverage checking, policy validation, and lookback
window management into a single, well-typed module.

This module provides:
1. SubscriptionPolicy: Unified policy combining coverage guards and lookback windows
2. SubscriptionChecker: Diagnostic tool for querying subscription details
3. Helper functions for policy construction and validation

Environment Variables:
    DATABENTO_ALLOWED_DATASETS: Comma-separated dataset names
    DATABENTO_ALLOWED_SCHEMAS: Comma-separated schemas
    DATABENTO_ALLOWED_SYMBOLS: Comma-separated symbols (optional strict allowlist)
    DATABENTO_MAX_DAYS: Maximum days per request window (integer)
    DATABENTO_MAX_DAYS_BY_SCHEMA: Per-schema max days (e.g., "ohlcv-1m:3650,mbp-1:365")
    DATABENTO_EARLIEST_DATE: ISO date (YYYY-MM-DD) earliest allowed start
    DATABENTO_LATEST_DATE: ISO date (YYYY-MM-DD) latest allowed end
    DATABENTO_MAX_SYMBOLS: Maximum number of symbols per request (integer)
    DATABENTO_POLICY_STRICT: "1" to raise on violations, "0" to clamp/filter
    ML_L0_LOOKBACK_DAYS: L0 lookback window (default 2555 ~ 7y)
    ML_L1_LOOKBACK_DAYS: L1 lookback window (default 365)
    ML_L2_LOOKBACK_DAYS: L2 lookback window (default 30)
    ML_L3_LOOKBACK_DAYS: L3 lookback window (default 30)

"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Any, TypedDict

from ml._imports import HAS_DATABENTO
from ml._imports import check_ml_dependencies


if TYPE_CHECKING:
    import databento as db


__all__ = [
    "DatasetInfo",
    "SubscriptionCheckResult",
    "SubscriptionChecker",
    "SubscriptionPolicy",
    "get_effective_policy",
    "get_max_lookback_days",
]


# Type definitions for subscription checking
class DatasetInfo(TypedDict, total=False):
    """Information about a dataset's subscription coverage."""

    start: str
    end: str
    days: int
    years: float
    schemas: list[str]


class SubscriptionCheckResult(TypedDict, total=False):
    """Results from subscription validation check."""

    available_datasets: list[str]
    datasets: dict[str, DatasetInfo]
    costs: dict[str, float]
    warnings: list[str]
    recommendations: list[str]
    date_ranges: dict[str, str]
    schemas: dict[str, list[str]]


# Helper functions for parsing environment variables
def _parse_csv(envval: str | None) -> set[str] | None:
    """Parse comma-separated environment variable into set."""
    if not envval:
        return None
    items = [s.strip() for s in envval.split(",") if s.strip()]
    return set(items) if items else None


def _parse_iso_date(envval: str | None) -> datetime | None:
    """Parse ISO date string from environment variable."""
    if not envval:
        return None
    try:
        dt = datetime.fromisoformat(envval)
        # Default to UTC if naive
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
    except Exception:
        return None


def _parse_int_env(envval: str | None, default: int | None = None) -> int | None:
    """Parse integer from environment variable."""
    if not envval:
        return default
    try:
        return int(envval)
    except ValueError:
        return default


def _parse_max_days_by_schema(envval: str | None) -> dict[str, int] | None:
    """Parse per-schema max days mapping from environment variable."""
    if not envval:
        return None
    md_map: dict[str, int] = {}
    for part in envval.split(","):
        kv = part.strip()
        if not kv or ":" not in kv:
            continue
        k, v = kv.split(":", 1)
        try:
            md_map[k.strip()] = int(v.strip())
        except ValueError:
            continue
    return md_map if md_map else None


@dataclass(frozen=True)
class SubscriptionPolicy:
    """
    Unified subscription policy for Databento coverage and lookback windows.

    Combines functionality from DatabentoCoveragePolicy and CoveragePolicy into
    a single, comprehensive policy class.

    Attributes
    ----------
    allowed_datasets : set[str] | None
        Allowlist of permitted datasets. None means all allowed.
    allowed_schemas : set[str] | None
        Allowlist of permitted schemas. None means all allowed.
    allowed_symbols : set[str] | None
        Allowlist of permitted symbols. None means all allowed.
    max_days : int | None
        Global maximum days per request window.
    max_days_by_schema : dict[str, int] | None
        Per-schema maximum days overrides.
    earliest : datetime | None
        Earliest allowed start date for requests.
    latest : datetime | None
        Latest allowed end date for requests.
    max_symbols : int | None
        Maximum number of symbols per request.
    strict : bool
        If True, raise on violations. If False, clamp/filter.
    l0_max_lookback_days : int
        Maximum lookback for L0 data (bars).
    l1_max_lookback_days : int
        Maximum lookback for L1 data (quotes, trades).
    l2_max_lookback_days : int
        Maximum lookback for L2 data (market depth).
    l3_max_lookback_days : int
        Maximum lookback for L3 data (full order book).

    """

    # Coverage guard fields (from DatabentoCoveragePolicy)
    allowed_datasets: set[str] | None = None
    allowed_schemas: set[str] | None = None
    allowed_symbols: set[str] | None = None
    max_days: int | None = None
    max_days_by_schema: dict[str, int] | None = None
    earliest: datetime | None = None
    latest: datetime | None = None
    max_symbols: int | None = None
    strict: bool = False

    # Lookback window fields (from CoveragePolicy)
    l0_max_lookback_days: int = 365 * 7  # 7 years for bars
    l1_max_lookback_days: int = 365  # 1 year for quotes/trades
    l2_max_lookback_days: int = 30  # 30 days for market depth
    l3_max_lookback_days: int = 30  # 30 days for full order book

    def allow_dataset(self, dataset: str) -> None:
        """
        Dynamically permit a dataset even when an allowlist is active.

        Parameters
        ----------
        dataset : str
            Dataset identifier to allow.

        """
        if not dataset:
            return
        if self.allowed_datasets is None:
            object.__setattr__(self, "allowed_datasets", {dataset})
            return
        updated = set(self.allowed_datasets)
        updated.add(dataset)
        object.__setattr__(self, "allowed_datasets", updated)

    def validate_dataset_schema(self, *, dataset: str, schema: str) -> None:
        """
        Validate that dataset and schema are permitted by policy.

        Parameters
        ----------
        dataset : str
            Dataset identifier.
        schema : str
            Schema identifier.

        Raises
        ------
        PermissionError
            If dataset or schema is not in allowlist.

        """
        if self.allowed_datasets is not None and dataset not in self.allowed_datasets:
            msg = f"Dataset '{dataset}' is not in allowed set: {sorted(self.allowed_datasets)}"
            raise PermissionError(msg)
        if self.allowed_schemas is not None and schema not in self.allowed_schemas:
            msg = f"Schema '{schema}' is not in allowed set: {sorted(self.allowed_schemas)}"
            raise PermissionError(msg)

    def clamp_range(
        self,
        start: datetime,
        end: datetime,
        *,
        dataset: str | None = None,
        schema: str | None = None,
    ) -> tuple[datetime, datetime]:
        """
        Clamp date range by policy constraints.

        Applies earliest/latest bounds and max_days limits based on schema.

        Parameters
        ----------
        start : datetime
            Requested start date.
        end : datetime
            Requested end date.
        dataset : str | None, optional
            Dataset identifier for error messages.
        schema : str | None, optional
            Schema identifier for schema-specific max_days.

        Returns
        -------
        tuple[datetime, datetime]
            Clamped (start, end) range.

        Raises
        ------
        PermissionError
            If strict mode is enabled and clamping results in empty window.

        """
        s = start
        e = end

        # Apply earliest/latest bounds
        if self.earliest and s < self.earliest:
            s = self.earliest
        if self.latest and e > self.latest:
            e = self.latest

        # Determine applicable max-days
        effective_max_days: int | None = None
        if schema and self.max_days_by_schema and schema in self.max_days_by_schema:
            effective_max_days = int(self.max_days_by_schema[schema])
        elif self.max_days is not None:
            effective_max_days = int(self.max_days)

        # Apply max-days constraint
        if effective_max_days is not None:
            max_delta = timedelta(days=effective_max_days)
            if (e - s) > max_delta:
                s = e - max_delta

        # Validate result
        if e <= s:
            if self.strict:
                extra = f" for {dataset}/{schema}" if dataset or schema else ""
                raise PermissionError(
                    f"Clamped window is empty{extra}: start={start}, end={end}",
                )

        return s, e

    def filter_symbols(self, symbols: Sequence[str]) -> list[str]:
        """
        Filter symbols by allowlist and max count.

        Parameters
        ----------
        symbols : Sequence[str]
            Symbols to filter.

        Returns
        -------
        list[str]
            Filtered symbol list.

        Raises
        ------
        PermissionError
            If strict mode is enabled and no symbols remain after filtering.

        """
        out: list[str] = list(symbols)

        # Apply allowlist
        if self.allowed_symbols is not None:
            out = [s for s in out if s in self.allowed_symbols]

        # Apply max symbols constraint
        if self.max_symbols is not None and len(out) > self.max_symbols:
            out = out[: int(self.max_symbols)]

        # Validate result
        if not out and self.strict:
            raise PermissionError("No symbols permitted by policy")

        return out

    def get_lookback_days_for_level(self, data_level: str) -> int:
        """
        Get maximum lookback days for a data level.

        Parameters
        ----------
        data_level : str
            Data level identifier (L0, L1, L2, L3, bars, quotes, trades, etc.).

        Returns
        -------
        int
            Maximum lookback days for the specified level.

        """
        key = data_level.strip().lower()

        # L0 mappings
        if key in {"l0", "bars", "ohlcv", "ohlcv-1m", "ohlcv-1h", "ohlcv-1d"}:
            return self.l0_max_lookback_days

        # L1 mappings
        if key in {"l1", "quotes", "trades", "tbbo", "bbo"}:
            return self.l1_max_lookback_days

        # L2 mappings
        if key in {"l2", "mbp", "mbp-1", "mbp-10", "orderbook"}:
            return self.l2_max_lookback_days

        # L3 mappings
        if key in {"l3", "mbo", "depth"}:
            return self.l3_max_lookback_days

        # Default to most conservative (L2/L3)
        return self.l2_max_lookback_days

    @staticmethod
    def from_env() -> SubscriptionPolicy:
        """
        Construct policy from environment variables.

        Returns
        -------
        SubscriptionPolicy
            Policy instance configured from environment.

        """
        # Parse coverage guard variables
        allowed_datasets = _parse_csv(os.getenv("DATABENTO_ALLOWED_DATASETS"))
        allowed_schemas = _parse_csv(os.getenv("DATABENTO_ALLOWED_SCHEMAS"))
        allowed_symbols = _parse_csv(os.getenv("DATABENTO_ALLOWED_SYMBOLS"))
        max_days = _parse_int_env(os.getenv("DATABENTO_MAX_DAYS"))
        max_days_by_schema = _parse_max_days_by_schema(
            os.getenv("DATABENTO_MAX_DAYS_BY_SCHEMA"),
        )
        earliest = _parse_iso_date(os.getenv("DATABENTO_EARLIEST_DATE"))
        latest = _parse_iso_date(os.getenv("DATABENTO_LATEST_DATE"))
        max_symbols = _parse_int_env(os.getenv("DATABENTO_MAX_SYMBOLS"))
        strict = os.getenv("DATABENTO_POLICY_STRICT", "0").strip() in {
            "1",
            "true",
            "TRUE",
        }

        # Parse lookback window variables
        l0_lookback = _parse_int_env(os.getenv("ML_L0_LOOKBACK_DAYS"), 365 * 7)
        l1_lookback = _parse_int_env(os.getenv("ML_L1_LOOKBACK_DAYS"), 365)
        l2_lookback = _parse_int_env(os.getenv("ML_L2_LOOKBACK_DAYS"), 30)
        l3_lookback = _parse_int_env(os.getenv("ML_L3_LOOKBACK_DAYS"), 30)

        return SubscriptionPolicy(
            allowed_datasets=allowed_datasets,
            allowed_schemas=allowed_schemas,
            allowed_symbols=allowed_symbols,
            max_days=max_days,
            max_days_by_schema=max_days_by_schema,
            earliest=earliest,
            latest=latest,
            max_symbols=max_symbols,
            strict=strict,
            l0_max_lookback_days=l0_lookback or 365 * 7,
            l1_max_lookback_days=l1_lookback or 365,
            l2_max_lookback_days=l2_lookback or 30,
            l3_max_lookback_days=l3_lookback or 30,
        )


class SubscriptionChecker:
    """
    Diagnostic tool for checking Databento subscription limits and coverage.

    Queries subscription to determine:
    - Available datasets
    - Date ranges included in subscription
    - Available schemas (L0/L1/L2/L3)
    - Estimated costs

    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize subscription checker.

        Parameters
        ----------
        api_key : str | None, optional
            Databento API key. If None, reads from DATABENTO_API_KEY env var.

        Raises
        ------
        RuntimeError
            If databento is not installed.
        ValueError
            If API key is not provided or found in environment.

        """
        if not HAS_DATABENTO:
            check_ml_dependencies(["databento"])

        key = api_key or os.getenv("DATABENTO_API_KEY")
        if not key:
            msg = "DATABENTO_API_KEY must be provided or set in environment"
            raise ValueError(msg)

        import databento as db

        self.client: db.Historical = db.Historical(key)
        self.results: SubscriptionCheckResult = {
            "datasets": {},
            "costs": {},
            "warnings": [],
            "recommendations": [],
        }

    def check_available_datasets(self) -> list[str]:
        """
        Get list of datasets available to this subscription.

        Returns
        -------
        list[str]
            List of available dataset identifiers.

        """
        try:
            datasets: list[str] = self.client.metadata.list_datasets()
            self.results["available_datasets"] = datasets
            return datasets
        except Exception as e:
            msg = f"Error checking datasets: {e}"
            self.results.setdefault("warnings", []).append(msg)
            return []

    def check_dataset_range(self, dataset: str) -> dict[str, Any]:
        """
        Check available date range for a dataset.

        Parameters
        ----------
        dataset : str
            Dataset identifier.

        Returns
        -------
        dict[str, Any]
            Range information including start_date and end_date.

        """
        try:
            range_info: dict[str, Any] = self.client.metadata.get_dataset_range(dataset)

            start_date = range_info.get("start_date", "Unknown")
            end_date = range_info.get("end_date", "Unknown")

            if start_date != "Unknown" and end_date != "Unknown":
                import pandas as pd

                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                days = (end - start).days
                years = days / 365.25

                self.results["datasets"][dataset] = {
                    "start": start_date,
                    "end": end_date,
                    "days": days,
                    "years": years,
                }

            return range_info

        except Exception as e:
            msg = f"Could not check range for {dataset}: {e}"
            self.results.setdefault("warnings", []).append(msg)
            return {}

    def check_available_schemas(self, dataset: str) -> list[str]:
        """
        Check what schemas are available for a dataset.

        Parameters
        ----------
        dataset : str
            Dataset identifier.

        Returns
        -------
        list[str]
            List of available schema identifiers.

        """
        try:
            schemas: list[str] = self.client.metadata.list_schemas(dataset)

            if dataset in self.results["datasets"]:
                self.results["datasets"][dataset]["schemas"] = schemas
            else:
                self.results["datasets"][dataset] = {"schemas": schemas}

            return schemas

        except Exception as e:
            msg = f"Could not check schemas for {dataset}: {e}"
            self.results.setdefault("warnings", []).append(msg)
            return []

    def get_results(self) -> SubscriptionCheckResult:
        """
        Get accumulated check results.

        Returns
        -------
        SubscriptionCheckResult
            Dictionary containing all check results.

        """
        return self.results


# Module-level helper functions
def get_effective_policy(policy: SubscriptionPolicy | None = None) -> SubscriptionPolicy:
    """
    Get effective subscription policy, defaulting to environment-based.

    Parameters
    ----------
    policy : SubscriptionPolicy | None, optional
        Explicit policy. If None, constructs from environment.

    Returns
    -------
    SubscriptionPolicy
        Effective policy instance.

    """
    return policy if policy is not None else SubscriptionPolicy.from_env()


def get_max_lookback_days(
    data_level: str,
    policy: SubscriptionPolicy | None = None,
) -> int:
    """
    Get maximum lookback days for a data level.

    Convenience function for getting lookback days without instantiating policy.

    Parameters
    ----------
    data_level : str
        Data level identifier (L0, L1, L2, L3, bars, quotes, etc.).
    policy : SubscriptionPolicy | None, optional
        Policy to use. If None, constructs from environment.

    Returns
    -------
    int
        Maximum lookback days for the specified level.

    """
    p = get_effective_policy(policy)
    return p.get_lookback_days_for_level(data_level)
