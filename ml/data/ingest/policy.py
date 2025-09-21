"""
Databento coverage policy guard.

Provides a lightweight, opt-in guard to prevent accidental API calls outside of
your subscription coverage by enforcing dataset/schema allowlists and limiting
the request window and symbol count.

Enable via environment variables (all optional; disabled by default):

- DATABENTO_ALLOWED_DATASETS: comma-separated dataset names (e.g., "EQUS.MINI,XNAS.ITCH")
- DATABENTO_ALLOWED_SCHEMAS: comma-separated schemas (e.g., "ohlcv-1m,mbp-1,tbbo,trades")
- DATABENTO_ALLOWED_SYMBOLS: comma-separated symbols (optional strict allowlist)
- DATABENTO_MAX_DAYS: maximum days per request window (integer)
- DATABENTO_MAX_DAYS_BY_SCHEMA: per-schema max days CSV (e.g.,
  "ohlcv-1m:3650,tbbo:365,trades:365,bbo:365,mbp-1:365,mbp-10:31,mbo:31,imbalance:31")
- DATABENTO_EARLIEST_DATE: ISO date (YYYY-MM-DD) earliest allowed start
- DATABENTO_LATEST_DATE: ISO date (YYYY-MM-DD) latest allowed end
- DATABENTO_MAX_SYMBOLS: maximum number of symbols per request (integer)
- DATABENTO_POLICY_STRICT: "1" to raise on violations, "0" to clamp/filter (default "0")

This module is cold-path only and safe to import from CLIs or ingest helpers.

"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta


def _parse_csv(envval: str | None) -> set[str] | None:
    if not envval:
        return None
    items = [s.strip() for s in envval.split(",") if s.strip()]
    return set(items) if items else None


def _parse_iso_date(envval: str | None) -> datetime | None:
    if not envval:
        return None
    try:
        dt = datetime.fromisoformat(envval)
        # Default to UTC if naive
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
    except Exception:
        return None


@dataclass(frozen=True)
class DatabentoCoveragePolicy:
    allowed_datasets: set[str] | None = None
    allowed_schemas: set[str] | None = None
    allowed_symbols: set[str] | None = None
    max_days: int | None = None
    earliest: datetime | None = None
    latest: datetime | None = None
    max_symbols: int | None = None
    strict: bool = False
    max_days_by_schema: dict[str, int] | None = None

    @staticmethod
    def from_env() -> DatabentoCoveragePolicy:
        """
        Construct policy from environment variables.

        See module docstring for variable names.

        """
        allowed_datasets = _parse_csv(os.getenv("DATABENTO_ALLOWED_DATASETS"))
        allowed_schemas = _parse_csv(os.getenv("DATABENTO_ALLOWED_SCHEMAS"))
        allowed_symbols = _parse_csv(os.getenv("DATABENTO_ALLOWED_SYMBOLS"))
        max_days_env = os.getenv("DATABENTO_MAX_DAYS")
        max_days = int(max_days_env) if (max_days_env and max_days_env.isdigit()) else None
        earliest = _parse_iso_date(os.getenv("DATABENTO_EARLIEST_DATE"))
        latest = _parse_iso_date(os.getenv("DATABENTO_LATEST_DATE"))
        max_symbols_env = os.getenv("DATABENTO_MAX_SYMBOLS")
        try:
            max_symbols = int(max_symbols_env) if max_symbols_env else None
        except Exception:
            max_symbols = None
        strict = os.getenv("DATABENTO_POLICY_STRICT", "0").strip() in {"1", "true", "TRUE"}
        # Parse per-schema max days map if provided
        md_map_env = os.getenv("DATABENTO_MAX_DAYS_BY_SCHEMA")
        md_map: dict[str, int] | None = None
        if md_map_env:
            md_map = {}
            for part in md_map_env.split(","):
                kv = part.strip()
                if not kv or ":" not in kv:
                    continue
                k, v = kv.split(":", 1)
                try:
                    md_map[k.strip()] = int(v.strip())
                except Exception:
                    continue
        return DatabentoCoveragePolicy(
            allowed_datasets=allowed_datasets,
            allowed_schemas=allowed_schemas,
            allowed_symbols=allowed_symbols,
            max_days=max_days,
            earliest=earliest,
            latest=latest,
            max_symbols=max_symbols,
            strict=strict,
            max_days_by_schema=md_map,
        )

    def validate_dataset_schema(self, *, dataset: str, schema: str) -> None:
        if self.allowed_datasets is not None and dataset not in self.allowed_datasets:
            msg = f"Dataset '{dataset}' is not in allowed set: {sorted(self.allowed_datasets)}"
            if self.strict:
                raise PermissionError(msg)
            raise PermissionError(msg)
        if self.allowed_schemas is not None and schema not in self.allowed_schemas:
            msg = f"Schema '{schema}' is not in allowed set: {sorted(self.allowed_schemas)}"
            if self.strict:
                raise PermissionError(msg)
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
        Clamp [start, end] by earliest/latest and max_days if configured.

        Returns an adjusted (start, end) tuple. If adjustment results in an empty
        interval and `strict` is True, raises PermissionError.

        """
        s = start
        e = end
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

        if effective_max_days is not None:
            # If window exceeds allowed days, clamp start to e - max_days
            max_delta = timedelta(days=effective_max_days)
            if (e - s) > max_delta:
                s = e - max_delta
        if e <= s:
            if self.strict:
                extra = f" for {dataset}/{schema}" if dataset or schema else ""
                raise PermissionError(f"Clamped window is empty{extra}: start={start}, end={end}")
        return s, e

    def filter_symbols(self, symbols: Sequence[str]) -> list[str]:
        out: list[str] = list(symbols)
        if self.allowed_symbols is not None:
            out = [s for s in out if s in self.allowed_symbols]
        if self.max_symbols is not None and len(out) > self.max_symbols:
            out = out[: int(self.max_symbols)]
        if not out and self.strict:
            raise PermissionError("No symbols permitted by policy")
        return out
