"""
Defaults for dataset manifest metadata reconstruction.
"""

from __future__ import annotations

from ml.registry.dataclasses import DatasetType


_DEFAULT_PRIMARY_KEYS_BY_TYPE: dict[DatasetType, tuple[str, ...]] = {
    DatasetType.EARNINGS_ACTUALS: ("ticker", "period_end"),
    DatasetType.EARNINGS_ESTIMATES: ("ticker", "period_end", "estimate_date"),
    DatasetType.MACRO_RELEASES: ("series_id", "observation_ts", "release_ts", "ts_event"),
    DatasetType.MACRO_OBSERVATIONS: ("series_id", "observation_ts", "ts_event"),
    DatasetType.EVENTS_CALENDAR: ("event_type", "event_timestamp", "instrument_id", "name", "ts_event"),
    DatasetType.MICRO_MINUTE_FEATURES: ("instrument_id", "timestamp", "ts_event"),
    DatasetType.L2_MINUTE_FEATURES: ("instrument_id", "timestamp", "ts_event"),
    DatasetType.RISK_HALT_EVENTS: ("event_id",),
    DatasetType.REPLAY_SUMMARY: ("run_id",),
}


def resolve_primary_keys(dataset_type: DatasetType, schema: dict[str, str]) -> list[str]:
    """
    Resolve default primary keys for a dataset type.

    Args:
        dataset_type: Dataset type identifier.
        schema: Schema column mapping.

    Returns:
        List of primary key column names.
    """
    defaults = _DEFAULT_PRIMARY_KEYS_BY_TYPE.get(dataset_type)
    if defaults:
        return list(defaults)
    if "instrument_id" in schema and "ts_event" in schema:
        return ["instrument_id", "ts_event"]
    if "ts_event" in schema:
        return ["ts_event"]
    return list(schema.keys())[:1]


__all__ = ["resolve_primary_keys"]
