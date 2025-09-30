#!/usr/bin/env python3
"""Download and compare SPY OHLCV-1m bars from EQUS.MINI and XNAS.ITCH."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from datetime import date
from datetime import datetime
from datetime import timedelta

import databento as db
import pandas as pd


SCHEMA = "ohlcv-1m"
DEFAULT_DATASETS: tuple[str, str] = ("EQUS.MINI", "XNAS.ITCH")
DEFAULT_SYMBOL = "SPY"
TARGET_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    yesterday = date.today() - timedelta(days=1)

    parser = argparse.ArgumentParser(
        description="Compare SPY OHLCV-1m bars between EQUS.MINI and XNAS.ITCH datasets.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=yesterday.isoformat(),
        help="Trading date to download in YYYY-MM-DD (defaults to yesterday).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help="Ticker symbol to download (defaults to SPY).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs=2,
        default=list(DEFAULT_DATASETS),
        metavar=("PRIMARY", "SECONDARY"),
        help="Two Databento datasets to compare (default: EQUS.MINI XNAS.ITCH).",
    )
    return parser.parse_args()


def build_time_range(trading_day: str) -> tuple[str, str]:
    """Return ISO8601 range for the full trading day."""
    start = datetime.fromisoformat(f"{trading_day}T00:00:00")
    end = start + timedelta(days=1)
    return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")


def fetch_dataset(
    client: db.Historical,
    dataset: str,
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch OHLCV-1m bars for a dataset."""
    response = client.timeseries.get_range(
        dataset=dataset,
        schema=SCHEMA,
        symbols=[symbol],
        start=start,
        end=end,
    )

    frame = response.to_df()

    if frame.empty:
        raise RuntimeError(f"No data returned for {symbol} from {dataset} between {start} and {end}.")

    # Normalize column names and focus on the expected comparison fields.
    if "ts_event" not in frame.columns:
        if frame.index.name != "ts_event":
            raise KeyError("ts_event timestamp not present in Databento response.")
        frame = frame.reset_index()

    frame = frame.loc[:, ["ts_event", *TARGET_COLUMNS]].copy()
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
    return frame.set_index("ts_event").sort_index()


def summarize_overlap(columns: Iterable[str], merged: pd.DataFrame, left: str, right: str) -> None:
    """Print column-wise comparison summary."""
    for column in columns:
        left_col = f"{column}_{left}"
        right_col = f"{column}_{right}"
        if left_col not in merged.columns or right_col not in merged.columns:
            continue

        overlap = merged[[left_col, right_col]].dropna(how="any")
        if overlap.empty:
            print(f"- {column}: no overlapping bars")
            continue

        diffs = (overlap[left_col] - overlap[right_col]).abs()
        non_zero = (diffs > 0).sum()
        print(
            f"- {column}: {len(overlap)} overlapping, {non_zero} mismatched, "
            f"mean abs diff {diffs.mean():.6g}, max abs diff {diffs.max():.6g}"
        )


def main() -> None:
    args = parse_args()

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise SystemExit("DATABENTO_API_KEY must be set in the environment.")

    primary, secondary = args.datasets
    start, end = build_time_range(args.date)

    client = db.Historical(api_key)

    print(f"Downloading {args.symbol} {SCHEMA} bars for {args.date}...")
    primary_frame = fetch_dataset(client, primary, args.symbol, start, end)
    secondary_frame = fetch_dataset(client, secondary, args.symbol, start, end)

    print(f"{primary}: {len(primary_frame)} bars")
    print(f"{secondary}: {len(secondary_frame)} bars")

    merged = primary_frame.add_suffix(f"_{primary}").merge(
        secondary_frame.add_suffix(f"_{secondary}"),
        left_index=True,
        right_index=True,
        how="outer",
    )

    missing_primary = merged[f"open_{primary}"].isna().sum()
    missing_secondary = merged[f"open_{secondary}"].isna().sum()
    print(
        f"Merged bars: {len(merged)} total, "
        f"{missing_primary} only in {secondary}, {missing_secondary} only in {primary}"
    )

    summarize_overlap(TARGET_COLUMNS, merged, primary, secondary)

    # Show the first few mismatched rows for manual inspection.
    diff_mask = []
    for column in TARGET_COLUMNS:
        left_col = f"{column}_{primary}"
        right_col = f"{column}_{secondary}"
        if left_col in merged and right_col in merged:
            diff_mask.append((merged[left_col] - merged[right_col]).abs() > 0)
    if diff_mask:
        combined_mask = pd.concat(diff_mask, axis=1).any(axis=1)
        mismatches = merged[combined_mask].head(5)
        if not mismatches.empty:
            print("\nSample mismatches:")
            print(mismatches)


if __name__ == "__main__":
    main()
