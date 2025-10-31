"""
Reporting utilities for the playground backtest harness.

The helpers in this module transform aggregated CSV outputs into presentation
artefacts such as PDF summaries so Phase 4 deliverables remain reproducible
from committed datasets.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl


def generate_sensitivity_summary_pdf(
    summary_path: Path,
    output_path: Path,
    *,
    title: str | None = None,
    notes: Sequence[str] | None = None,
) -> None:
    """
    Render the parameter sensitivity summary CSV into a PDF report.

    Parameters
    ----------
    summary_path : Path
        Path to the ``summary.csv`` produced by the Phase 4 sensitivity suite.
    output_path : Path
        Destination PDF path, e.g. ``reports/backtesting/sensitivity/sensitivity_analysis.pdf``.
    title : str | None, optional
        Optional report title. Defaults to
        ``"Phase 4 Parameter Sensitivity Summary"``.
    notes : Sequence[str] | None, optional
        Additional bullet notes rendered beneath the title.

    Raises
    ------
    FileNotFoundError
        If ``summary_path`` does not exist.
    ValueError
        If the loaded dataframe is empty.
    """
    if not summary_path.exists():
        msg = f"Sensitivity summary not found: {summary_path}"
        raise FileNotFoundError(msg)

    frame = pl.read_csv(summary_path)
    if frame.is_empty():
        msg = f"Sensitivity summary is empty: {summary_path}"
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Format data for table rendering.
    formatted_rows = [_format_table_row(row) for row in frame.iter_rows(named=True)]
    column_labels = [column.replace("_", " ").title() for column in frame.columns]

    # Lazy import to avoid heavy dependencies when unused.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # Approximate A4 portrait.
    ax.axis("off")
    report_title = title or "Phase 4 Parameter Sensitivity Summary"
    fig.text(0.05, 0.96, report_title, fontsize=16, fontweight="bold", ha="left")
    timestamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
    fig.text(0.05, 0.93, f"Generated: {timestamp}", fontsize=9, ha="left")

    if notes:
        y_position = 0.90
        for note in notes:
            fig.text(0.05, y_position, f"• {note}", fontsize=9, ha="left")
            y_position -= 0.02

    table = ax.table(
        cellText=formatted_rows,
        colLabels=column_labels,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.9))
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _format_table_row(row: dict[str, object]) -> list[str]:
    """Format a CSV row for presentation."""
    formatted: list[str] = []
    for value in row.values():
        if isinstance(value, float):
            formatted.append(f"{value:.4f}")
        elif isinstance(value, (int, bool)):
            formatted.append(str(value))
        else:
            formatted.append(str(value))
    return formatted
