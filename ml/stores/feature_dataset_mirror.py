"""
Feature dataset parquet mirror refresh utilities.

Exports SQL-backed feature datasets into their parquet mirrors so the parquet
files remain complete backups of the authoritative SQL tables.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text

from ml._imports import HAS_PANDAS
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml.config import FeatureDatasetMirrorConfig
from ml.core.db_engine import EngineManager


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureDatasetMirrorExportConfig:
    """
    Configuration for exporting SQL feature datasets into parquet mirrors.
    """

    db_connection: str
    series_ids: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class FeatureDatasetMirrorExportResult:
    """
    Summary of a mirror refresh run.
    """

    macro_observations_rows: int
    macro_release_rows: int
    events_rows: int


def refresh_feature_dataset_mirrors(
    config: FeatureDatasetMirrorExportConfig,
    *,
    mirror_config: FeatureDatasetMirrorConfig | None = None,
) -> FeatureDatasetMirrorExportResult:
    """
    Refresh parquet mirrors for macro observations, macro releases, and events.

    Args:
        config: SQL export configuration.
        mirror_config: Parquet mirror configuration (defaults to env).

    Returns:
        Result counts for each mirror export.
    """
    mirror_cfg = mirror_config or FeatureDatasetMirrorConfig.from_env()
    _ensure_pandas()

    engine = EngineManager.get_engine(
        config.db_connection,
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
    )

    series_ids = config.series_ids or _load_series_ids(mirror_cfg.macro_series_path)
    macro_obs_rows = _export_macro_observations(
        engine=engine,
        series_ids=series_ids,
        out_path=mirror_cfg.macro_fred_path,
    )
    macro_release_rows = _export_macro_release_calendar(
        engine=engine,
        series_ids=series_ids,
        base_dir=mirror_cfg.macro_vintage_dir,
    )
    events_rows = _export_events_calendar(
        engine=engine,
        out_path=mirror_cfg.events_path,
    )

    logger.info(
        "Feature dataset mirrors refreshed",
        extra={
            "macro_observations_rows": macro_obs_rows,
            "macro_release_rows": macro_release_rows,
            "events_rows": events_rows,
        },
    )

    return FeatureDatasetMirrorExportResult(
        macro_observations_rows=macro_obs_rows,
        macro_release_rows=macro_release_rows,
        events_rows=events_rows,
    )


def _ensure_pandas() -> None:
    if not HAS_PANDAS or pd is None:
        check_ml_dependencies(["pandas"])


def _load_series_ids(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.strip()
        if not token or token.startswith("#"):
            continue
        tokens.append(token)
    return tuple(tokens)


def _export_macro_observations(
    *,
    engine: Any,
    series_ids: tuple[str, ...],
    out_path: Path,
) -> int:
    if pd is None:
        raise RuntimeError("pandas is required for macro observation export")
    where = ""
    params: dict[str, Any] = {}
    if series_ids:
        where = "WHERE series_id = ANY(:series_ids)"
        params["series_ids"] = list(series_ids)
    sql = f"""
        SELECT series_id, observation_ts, value
        FROM ml.macro_observations
        {where}
        ORDER BY series_id, observation_ts
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=["timestamp", "series_id", "value"])
        df.to_parquet(out_path, index=False)
        return 0
    df["timestamp"] = pd.to_datetime(df["observation_ts"], unit="ns", utc=True).dt.tz_convert(
        "UTC",
    ).dt.tz_localize(None)
    df = df[["timestamp", "series_id", "value"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


def _export_macro_release_calendar(
    *,
    engine: Any,
    series_ids: tuple[str, ...],
    base_dir: Path,
) -> int:
    if pd is None:
        raise RuntimeError("pandas is required for macro release export")
    total_rows = 0
    if not series_ids:
        return 0
    sql = text(
        """
        SELECT series_id, observation_ts, value, release_ts, release_end_ts
        FROM ml.macro_release_calendar
        WHERE series_id = :series_id
        ORDER BY observation_ts, release_ts
        """,
    )
    for series_id in series_ids:
        with engine.begin() as conn:
            df = pd.read_sql(sql, conn, params={"series_id": series_id})
        series_dir = base_dir / series_id
        series_dir.mkdir(parents=True, exist_ok=True)
        if df.empty:
            empty = pd.DataFrame(
                columns=[
                    "series_id",
                    "observation_ts",
                    "value",
                    "release_ts",
                    "release_end_ts",
                ],
            )
            empty.to_parquet(series_dir / "release_calendar.parquet", index=False)
            continue
        df["observation_ts"] = pd.to_datetime(df["observation_ts"], unit="ns", utc=True).dt.tz_convert(
            "UTC",
        ).dt.tz_localize(None)
        df["release_ts"] = pd.to_datetime(df["release_ts"], unit="ns", utc=True).dt.tz_convert(
            "UTC",
        ).dt.tz_localize(None)
        df["release_end_ts"] = pd.to_datetime(
            df["release_end_ts"],
            unit="ns",
            utc=True,
            errors="coerce",
        ).dt.tz_convert("UTC").dt.tz_localize(None)
        df = df[
            [
                "series_id",
                "observation_ts",
                "value",
                "release_ts",
                "release_end_ts",
            ]
        ]
        df.to_parquet(series_dir / "release_calendar.parquet", index=False)
        total_rows += len(df)
    return total_rows


def _export_events_calendar(
    *,
    engine: Any,
    out_path: Path,
) -> int:
    if pd is None:
        raise RuntimeError("pandas is required for events export")
    sql = """
        SELECT event_timestamp, event_type, name, instrument_id,
               importance, source, metadata
        FROM ml.events_calendar
        ORDER BY event_timestamp, event_type, name
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn)
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            columns=[
                "event_timestamp",
                "event_type",
                "name",
                "instrument_id",
                "importance",
                "source",
                "metadata",
            ],
        )
        df.to_parquet(out_path, index=False)
        return 0
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], unit="ns", utc=True).dt.tz_convert(
        "UTC",
    ).dt.tz_localize(None)
    df["instrument_id"] = df["instrument_id"].fillna("").astype(str)
    df["metadata"] = df["metadata"].apply(
        lambda value: value if isinstance(value, str) else json.dumps(value or {}),
    )
    df = df[
        [
            "event_timestamp",
            "event_type",
            "name",
            "instrument_id",
            "importance",
            "source",
            "metadata",
        ]
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


__all__ = [
    "FeatureDatasetMirrorExportConfig",
    "FeatureDatasetMirrorExportResult",
    "refresh_feature_dataset_mirrors",
]
