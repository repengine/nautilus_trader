"""
Dataset quality reporting tasks.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import ml._imports as _imports


pd = _imports.pd
pl = _imports.pl
check_ml_dependencies = _imports.check_ml_dependencies


class SupportsToDict(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(slots=True, frozen=True)
class DatasetReportConfig:
    dataset_path: Path
    output_json: Path | None = None
    output_markdown: Path | None = None


@dataclass(slots=True)
class DatasetReport:
    data: dict[str, Any]
    markdown: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.data, indent=2, sort_keys=True)


def _infer_macro_columns(columns: Iterable[str]) -> list[str]:
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
    import re

    pattern = re.compile(r"^[A-Z0-9_]{3,}$")
    return sorted(c for c in cols if pattern.match(c))


def _infer_feature_columns(columns: Iterable[str]) -> list[str]:
    exclude = {"y", "time_index", "timestamp", "instrument_id", "ts_event"}
    return [c for c in columns if c not in exclude]


def _compute_macro_null_rates(df: Any, macro_cols: list[str]) -> dict[str, float]:
    if not macro_cols:
        return {}
    if pl is not None and isinstance(df, pl.DataFrame):
        total = df.height or 1
        return {
            col: float(df.select(pl.col(col).is_null().sum().alias("n")).item()) / float(total)
            for col in macro_cols
            if col in df.columns
        }
    if pd is not None:
        total = len(df) or 1
        return {
            col: float(df[col].isna().sum()) / float(total)
            for col in macro_cols
            if col in df.columns
        }
    check_ml_dependencies(["polars", "pandas"])
    raise RuntimeError("No dataframe engine available")


def _feature_coverage(df: Any, feature_cols: list[str]) -> dict[str, Any]:
    if not feature_cols:
        return {"overall": {}, "by_symbol": {}}
    by_symbol: dict[str, dict[str, float]] = {}
    if pl is not None and isinstance(df, pl.DataFrame):
        total = df.height or 1
        overall = {
            col: float(df.select(pl.col(col).is_null().not_().sum().alias("n")).item())
            / float(total)
            for col in feature_cols
            if col in df.columns
        }
        if "instrument_id" in df.columns:
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
            counts = df.lazy().group_by("instrument_id").agg(pl.len().alias("count")).collect()
            count_map = {str(row[0]): int(row[1]) or 1 for row in counts.iter_rows()}
            for row in grouped.iter_rows(named=True):
                instrument = str(row["instrument_id"])
                denom = count_map.get(instrument, 1)
                by_symbol[instrument] = {
                    key: float(value) / float(denom)
                    for key, value in row.items()
                    if key != "instrument_id"
                }
        return {"overall": overall, "by_symbol": by_symbol}
    if pd is not None:
        total = len(df) or 1
        overall = {
            col: float(df[col].notna().sum()) / float(total)
            for col in feature_cols
            if col in df.columns
        }
        by_symbol = {}
        if "instrument_id" in df.columns:
            for instrument, subset in df.groupby("instrument_id"):
                denom = len(subset) or 1
                by_symbol[str(instrument)] = {
                    col: float(subset[col].notna().sum()) / float(denom)
                    for col in feature_cols
                    if col in subset.columns
                }
        return {"overall": overall, "by_symbol": by_symbol}
    check_ml_dependencies(["polars", "pandas"])
    raise RuntimeError("No dataframe engine available")


def _target_stats(df: Any) -> dict[str, Any]:
    if pl is not None and isinstance(df, pl.DataFrame):
        if "y" not in df.columns:
            return {}
        total = df.height
        positives = int(df.select(pl.col("y").sum().alias("p")).item()) if total > 0 else 0
        overall = {
            "total": int(total),
            "positives": positives,
            "positive_rate": float(positives) / float(total) if total else 0.0,
        }
        by_symbol = {}
        if "instrument_id" in df.columns and total > 0:
            grouped = (
                df.lazy()
                .group_by("instrument_id")
                .agg([pl.col("y").sum().alias("positives"), pl.len().alias("total")])
                .collect()
            )
            for row in grouped.iter_rows(named=True):
                instrument_total = int(row["total"]) or 1
                positives_sym = int(row["positives"])
                instrument = str(row["instrument_id"])
                by_symbol[instrument] = {
                    "total": int(row["total"]),
                    "positives": positives_sym,
                    "positive_rate": float(positives_sym) / float(instrument_total),
                }
        return {"overall": overall, "by_symbol": by_symbol}
    if pd is not None:
        if "y" not in df.columns:
            return {}
        total = len(df)
        positives = int(df["y"].sum()) if total > 0 else 0
        overall = {
            "total": total,
            "positives": positives,
            "positive_rate": float(positives) / float(total) if total else 0.0,
        }
        by_symbol = {}
        if "instrument_id" in df.columns and total > 0:
            grouped = df.groupby("instrument_id")["y"].agg(["sum", "count"]).reset_index()
            for _, row in grouped.iterrows():
                total_sym = int(row["count"]) or 1
                positives_sym = int(row["sum"])
                instrument = str(row["instrument_id"])
                by_symbol[instrument] = {
                    "total": int(row["count"]),
                    "positives": positives_sym,
                    "positive_rate": float(positives_sym) / float(total_sym),
                }
        return {"overall": overall, "by_symbol": by_symbol}
    check_ml_dependencies(["polars", "pandas"])
    raise RuntimeError("No dataframe engine available")


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Dataset Report"]
    macro = report.get("macro_null_rates", {})
    if macro:
        lines.append("\n## Macro Null Rates")
        for key, value in sorted(macro.items()):
            lines.append(f"- {key}: {value:.4f}")
    target = report.get("target", {})
    if target:
        overall = target.get("overall", {})
        lines.append("\n## Target Distribution")
        lines.append(
            "- total: {total}; positives: {positives}; rate: {rate:.4f}".format(
                total=overall.get("total", 0),
                positives=overall.get("positives", 0),
                rate=overall.get("positive_rate", 0.0),
            ),
        )
    return "\n".join(lines) + "\n"


def generate_dataset_report(config: DatasetReportConfig) -> DatasetReport:
    if not config.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")

    df: Any
    suffix = config.dataset_path.suffix.lower()
    if pl is not None and suffix == ".parquet":
        df = pl.read_parquet(str(config.dataset_path))
    elif pl is not None and suffix == ".csv":
        df = pl.read_csv(str(config.dataset_path))
    elif pd is not None:
        if suffix == ".parquet":
            df = pd.read_parquet(str(config.dataset_path))
        else:
            df = pd.read_csv(str(config.dataset_path))
    else:
        check_ml_dependencies(["polars", "pandas"])
        raise RuntimeError("Unable to load dataset: no dataframe engine available")

    if pl is not None and isinstance(df, pl.DataFrame):
        columns = list(df.columns)
    else:
        if pd is None:
            raise RuntimeError("pandas is required to compute dataset columns")
        columns = list(df.columns)

    macro_cols = _infer_macro_columns(columns)
    feature_cols = _infer_feature_columns(columns)

    report_data = {
        "shape": [int(getattr(df, "height", len(df))), len(columns)],
        "macro_null_rates": _compute_macro_null_rates(df, macro_cols),
        "feature_coverage": _feature_coverage(df, feature_cols),
        "target": _target_stats(df),
    }

    markdown = _render_markdown(report_data)
    report = DatasetReport(data=report_data, markdown=markdown)

    if config.output_json is not None:
        config.output_json.write_text(report.to_json() + "\n", encoding="utf-8")
    if config.output_markdown is not None:
        config.output_markdown.write_text(markdown, encoding="utf-8")

    return report
