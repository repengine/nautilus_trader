#!/usr/bin/env python3
"""Inspect Databento dataset availability and cost for a symbol."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, cast

import databento as db


class DatabentoClient(Protocol):
    symbology: Any
    metadata: Any


def build_client() -> DatabentoClient:
    return db.Historical()


def parse_datasets(raw: str) -> list[str]:
    datasets = [item.strip() for item in raw.split(",") if item.strip()]
    if not datasets:
        raise ValueError("At least one dataset is required")
    return datasets


def generate_report(
    *,
    client: DatabentoClient,
    symbol: str,
    datasets: Sequence[str],
    start: str,
    end: str,
) -> list[str]:
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    lines: list[str] = [f"Symbol lookup: {symbol} ({start} -> {end})", "=" * 72]
    for dataset in datasets:
        lines.append(f"Dataset: {dataset}")
        try:
            resolved = cast(
                Mapping[str, object],
                client.symbology.resolve(
                    dataset=dataset,
                    symbols=[symbol],
                    stype_in="raw_symbol",
                    stype_out="instrument_id",
                    start_date=start_dt.date().isoformat(),
                    end_date=end_dt.date().isoformat(),
                ),
            )
        except Exception as exc:  # pragma: no cover - network failure
            lines.append(f"  symbology: error ({exc})")
            lines.append("")
            continue

        result = cast(Mapping[str, object], resolved.get("result", {}))
        symbol_info = cast(tuple[Mapping[str, object], ...], result.get(symbol, ()))
        if not symbol_info:
            lines.append("  symbology: not found")
            lines.append("")
            continue

        entry = symbol_info[0]
        instrument_id = cast(str | None, entry.get("s")) or "?"
        date0 = cast(str | None, entry.get("d0")) or "?"
        date1 = cast(str | None, entry.get("d1")) or "?"
        lines.append(
            f"  symbology: resolved instrument_id={instrument_id} valid {date0} -> {date1}",
        )
        for schema in ("ohlcv-1m", "ohlcv-1d"):
            try:
                cost = client.metadata.get_cost(
                    dataset=dataset,
                    symbols=[symbol],
                    schema=schema,
                    start=start,
                    end=end,
                )
                lines.append(f"  cost[{schema}]: {cost}")
            except Exception as exc:  # pragma: no cover - remote failure
                lines.append(f"  cost[{schema}]: error ({exc})")
        lines.append("")
    return lines


def emit_report(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("--datasets", default="EQUS.MINI,EQUS.SUMMARY")
    parser.add_argument("--start", default="2024-07-01")
    parser.add_argument("--end", default="2024-07-02")
    args = parser.parse_args(argv)

    try:
        datasets = parse_datasets(args.datasets)
    except ValueError as exc:
        parser.error(str(exc))
    client = build_client()
    lines = generate_report(
        client=client,
        symbol=args.symbol,
        datasets=datasets,
        start=args.start,
        end=args.end,
    )
    emit_report(lines)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
