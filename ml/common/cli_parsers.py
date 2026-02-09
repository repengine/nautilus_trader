"""
Shared CLI parser helpers for ML task entrypoints.
"""

from __future__ import annotations

import argparse
import json

from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import coerce_storage_kind


def parse_market_inputs_json(
    value: str | None,
) -> tuple[MarketDatasetInput, ...] | None:
    """
    Parse ``market_inputs_json`` payload into ``MarketDatasetInput`` objects.

    Args:
        value: JSON payload string supplied by CLI arguments.

    Returns:
        Tuple of normalized ``MarketDatasetInput`` entries, or ``None`` when
        the input payload is omitted.

    Raises:
        argparse.ArgumentTypeError: If the payload is invalid JSON or contains
            unsupported shapes/values.

    Example:
        >>> payload = '[{"descriptor_id":"feeds.eq", "symbols":"aapl,msft"}]'
        >>> result = parse_market_inputs_json(payload)
        >>> assert result is not None
        >>> assert result[0].symbols == ("AAPL", "MSFT")
    """
    if value is None:
        return None

    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("market_inputs_json must be valid JSON") from exc

    if isinstance(payload, (str, dict)):
        items: list[object] = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise argparse.ArgumentTypeError(
            "market_inputs_json must encode a descriptor string or list of descriptor objects",
        )

    parsed: list[MarketDatasetInput] = []
    for entry in items:
        if isinstance(entry, str):
            parsed.append(MarketDatasetInput(descriptor_id=entry))
            continue

        if isinstance(entry, dict):
            descriptor_id = entry.get("descriptor_id")
            dataset_id = entry.get("dataset_id")
            symbols_field = entry.get("symbols")

            symbols_tuple: tuple[str, ...] | None
            if symbols_field is None:
                symbols_tuple = None
            elif isinstance(symbols_field, str):
                symbols_tuple = tuple(
                    token.strip().upper()
                    for token in symbols_field.split(",")
                    if token.strip()
                )
            elif isinstance(symbols_field, (list, tuple)):
                symbols_tuple = tuple(
                    str(token).strip().upper()
                    for token in symbols_field
                    if str(token).strip()
                )
            else:
                raise argparse.ArgumentTypeError(
                    "symbols in market_inputs_json must be a list or comma-separated string",
                )

            schema_override = entry.get("schema") or entry.get("schema_override")
            storage_raw = entry.get("storage_kind") or entry.get("storage_kind_override")
            storage_kind = None
            if storage_raw is not None:
                try:
                    storage_kind = coerce_storage_kind(storage_raw)
                except ValueError as exc:
                    raise argparse.ArgumentTypeError(
                        f"Invalid storage_kind '{storage_raw}' in market_inputs_json",
                    ) from exc

            parsed.append(
                MarketDatasetInput(
                    descriptor_id=str(descriptor_id) if descriptor_id is not None else None,
                    dataset_id=str(dataset_id) if dataset_id is not None else None,
                    symbols=symbols_tuple,
                    schema_override=str(schema_override) if schema_override is not None else None,
                    storage_kind_override=storage_kind,
                    start=str(entry.get("start")) if entry.get("start") is not None else None,
                    end=str(entry.get("end")) if entry.get("end") is not None else None,
                ),
            )
            continue

        raise argparse.ArgumentTypeError(
            "market_inputs_json entries must be descriptor strings or mapping objects",
        )

    return tuple(parsed) if parsed else None


__all__ = ["parse_market_inputs_json"]
