#!/usr/bin/env python3
"""
Generate dataset quality report (macro null-rates, feature coverage, targets).

Reads a dataset parquet (typically produced by build_tft_dataset.py) and emits
JSON and optional Markdown summaries with the following sections:

- Macro null-rates per FRED series present in the dataset
- Feature coverage per symbol (non-null ratio by feature)
- Target distribution overall and per symbol

Examples
--------
python -m ml.scripts.dataset_report \
  --dataset /tmp/tft_ds_spy/dataset.parquet \
  --out_json /tmp/tft_ds_spy/report.json --out_md /tmp/tft_ds_spy/report.md

"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml._imports as _imports


pd = _imports.pd
pl = _imports.pl
check_ml_dependencies = _imports.check_ml_dependencies


@dataclass(frozen=True)
class ReportPaths:
    dataset: Path
    out_json: Path
    out_md: Path | None


def _infer_macro_columns(columns: Iterable[str]) -> list[str]:
    """
    Infer FRED macro columns by intersecting with known default series IDs.

    Falls back to heuristic: all-uppercase columns with letters/digits/underscore.

    """
    # Known defaults from FRED loader (keep in sync with FREDDataLoader.DEFAULT_INDICATORS)
    known = {
        "DGS1",
        "DGS2",
        "DGS10",
        "DGS30",
        "FEDFUNDS",
        "SOFR",
        "VIXCLS",
        "GDP",
        "GDPC1",
        "CPIAUCSL",
        "CPILFESL",
        "PCEPI",
        "UNRATE",
        "PAYEMS",
        "CIVPART",
        "UMCSENT",
        "RSXFS",
        "HOUST",
    }
    cols = list(columns)
    present = [c for c in cols if c in known]
    if present:
        return sorted(present)
    # Heuristic fallback
    import re

    pat = re.compile(r"^[A-Z0-9_]{3,}$")
    return sorted([c for c in cols if pat.match(c) is not None])


def _infer_feature_columns(columns: Iterable[str]) -> list[str]:
    exclude = {"y", "time_index", "timestamp", "instrument_id", "ts_event"}
    return [c for c in columns if c not in exclude]


def _compute_macro_null_rates_pl(df: Any, macro_cols: list[str]) -> dict[str, float]:
    if not macro_cols:
        return {}
    assert pl is not None
    total = df.height if df.height > 0 else 1
    rates: dict[str, float] = {}
    for c in macro_cols:
        if c not in df.columns:
            continue
        nulls = df.select(pl.col(c).is_null().sum().alias("n")).item()
        rates[c] = float(nulls) / float(total)
    return rates


def _compute_macro_null_rates_pd(df: Any, macro_cols: list[str]) -> dict[str, float]:
    if not macro_cols:
        return {}
    assert pd is not None
    total = len(df) if len(df) > 0 else 1
    rates: dict[str, float] = {}
    for c in macro_cols:
        if c not in df.columns:
            continue
        rates[c] = float(df[c].isna().sum()) / float(total)
    return rates


def _feature_coverage_pl(df: Any, feature_cols: list[str]) -> dict[str, Any]:
    assert pl is not None
    if not feature_cols:
        return {"overall": {}, "by_symbol": {}}
    total = df.height if df.height > 0 else 1
    overall = {}
    for c in feature_cols:
        if c not in df.columns:
            continue
        nn = df.select(pl.col(c).is_null().not_().sum().alias("n")).item()
        overall[c] = float(nn) / float(total)
    by_symbol: dict[str, dict[str, float]] = {}
    if "instrument_id" in df.columns:
        # Per-symbol coverage
        grouped = (
            df.lazy()
            .group_by("instrument_id")
            .agg(
                [
                    pl.col(c).is_null().not_().sum().alias(c)
                    for c in feature_cols
                    if c in df.columns
                ],
            )
            .collect()
        )
        for row in grouped.iter_rows(named=True):
            sym = str(row.get("instrument_id"))
            denom = int(df.filter(pl.col("instrument_id") == row["instrument_id"]).height) or 1
            by_symbol[sym] = {
                k: float(v) / float(denom) for k, v in row.items() if k != "instrument_id"
            }
    return {"overall": overall, "by_symbol": by_symbol}


def _feature_coverage_pd(df: Any, feature_cols: list[str]) -> dict[str, Any]:
    assert pd is not None
    if not feature_cols:
        return {"overall": {}, "by_symbol": {}}
    total = len(df) if len(df) > 0 else 1
    overall = {}
    for c in feature_cols:
        if c not in df.columns:
            continue
        nn = int(df[c].notna().sum())
        overall[c] = float(nn) / float(total)
    by_symbol: dict[str, dict[str, float]] = {}
    if "instrument_id" in df.columns:
        for sym, g in df.groupby("instrument_id"):
            denom = len(g) or 1
            by_symbol[str(sym)] = {
                c: float(g[c].notna().sum()) / float(denom) for c in feature_cols if c in g.columns
            }
    return {"overall": overall, "by_symbol": by_symbol}


def _target_stats_pl(df: Any) -> dict[str, Any]:
    assert pl is not None
    if "y" not in df.columns:
        return {}
    total = df.height
    pos = df.select(pl.col("y").sum().alias("p")).item() if total > 0 else 0
    base = {
        "total": int(total),
        "positives": int(pos),
        "positive_rate": float(pos) / float(total) if total else 0.0,
    }
    by_symbol: dict[str, Any] = {}
    if "instrument_id" in df.columns and total > 0:
        grouped = (
            df.lazy()
            .group_by("instrument_id")
            .agg([pl.col("y").sum().alias("positives"), pl.count().alias("total")])
            .collect()
        )
        for row in grouped.iter_rows(named=True):
            total_sym = int(row["total"]) or 1
            by_symbol[str(row["instrument_id"])] = {
                "total": int(row["total"]),
                "positives": int(row["positives"]),
                "positive_rate": float(row["positives"]) / float(total_sym),
            }
    return {"overall": base, "by_symbol": by_symbol}


def _target_stats_pd(df: Any) -> dict[str, Any]:
    assert pd is not None
    if "y" not in df.columns:
        return {}
    total = len(df)
    pos = int(df["y"].sum()) if total > 0 else 0
    base = {
        "total": int(total),
        "positives": int(pos),
        "positive_rate": float(pos) / float(total) if total else 0.0,
    }
    by_symbol: dict[str, Any] = {}
    if "instrument_id" in df.columns and total > 0:
        grouped = df.groupby("instrument_id")["y"].agg(["sum", "count"]).reset_index()
        for _, r in grouped.iterrows():
            total_sym = int(r["count"]) or 1
            by_symbol[str(r["instrument_id"])] = {
                "total": int(r["count"]),
                "positives": int(r["sum"]),
                "positive_rate": float(r["sum"]) / float(total_sym),
            }
    return {"overall": base, "by_symbol": by_symbol}


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Dataset Report"]
    # Macro
    macro = report.get("macro_null_rates", {})
    if macro:
        lines.append("\n## Macro Null Rates")
        for k, v in sorted(macro.items()):
            lines.append(f"- {k}: {v:.4f}")
    # Target
    tgt = report.get("target", {})
    if tgt:
        ov = tgt.get("overall", {})
        lines.append("\n## Target Distribution")
        lines.append(
            f"- total: {ov.get('total', 0)}; positives: {ov.get('positives', 0)}; rate: {ov.get('positive_rate', 0.0):.4f}",
        )
    return "\n".join(lines) + "\n"


def _resolve_paths(dataset_path: str, out_json: str | None, out_md: str | None) -> ReportPaths:
    ds = Path(dataset_path)
    if not ds.exists():
        raise SystemExit(f"Dataset not found: {ds}")
    default_dir = ds.parent
    json_path = Path(out_json) if out_json else default_dir / "dataset_report.json"
    md_path = Path(out_md) if out_md else None
    return ReportPaths(dataset=ds, out_json=json_path, out_md=md_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate dataset quality report")
    ap.add_argument("--dataset", required=True, help="Path to dataset parquet or CSV")
    ap.add_argument("--out_json", required=False, help="Path to write JSON report")
    ap.add_argument("--out_md", required=False, help="Optional path to write Markdown summary")
    args = ap.parse_args(argv)

    paths = _resolve_paths(args.dataset, args.out_json, args.out_md)

    # Load data (prefer Polars)
    df: Any
    if pl is not None and paths.dataset.suffix.lower() == ".parquet":
        df = pl.read_parquet(str(paths.dataset))
    elif pl is not None and paths.dataset.suffix.lower() == ".csv":  # pragma: no cover
        df = pl.read_csv(str(paths.dataset))
    elif pd is not None:  # pragma: no cover
        if paths.dataset.suffix.lower() == ".parquet":
            df = pd.read_parquet(str(paths.dataset))
        else:
            df = pd.read_csv(str(paths.dataset))
    else:  # pragma: no cover
        check_ml_dependencies(["polars", "pandas"])  # fail with clear message
        raise SystemExit("Unable to load dataset: no dataframe engine available")

    if pl is not None and isinstance(df, pl.DataFrame):
        cols = list(df.columns)
        macro_cols = _infer_macro_columns(cols)
        feature_cols = _infer_feature_columns(cols)
        report = {
            "shape": [int(df.height), len(df.columns)],
            "macro_null_rates": _compute_macro_null_rates_pl(df, macro_cols),
            "feature_coverage": _feature_coverage_pl(df, feature_cols),
            "target": _target_stats_pl(df),
        }
    else:  # pandas path
        assert pd is not None
        cols = list(df.columns)
        macro_cols = _infer_macro_columns(cols)
        feature_cols = _infer_feature_columns(cols)
        report = {
            "shape": [len(df), len(df.columns)],
            "macro_null_rates": _compute_macro_null_rates_pd(df, macro_cols),
            "feature_coverage": _feature_coverage_pd(df, feature_cols),
            "target": _target_stats_pd(df),
        }

    # Write JSON
    paths.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report_json": str(paths.out_json)}, separators=(",", ":")))

    # Optional Markdown
    if paths.out_md is not None:
        md = _to_markdown(report)
        paths.out_md.write_text(md, encoding="utf-8")
        print(json.dumps({"report_md": str(paths.out_md)}, separators=(",", ":")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
