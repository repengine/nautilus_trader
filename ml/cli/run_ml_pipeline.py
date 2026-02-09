#!/usr/bin/env python
"""Thin CLI wrapper for the ML pipeline runner."""

from __future__ import annotations

import uuid as _uuid

import click

from ml.common.logging_config import bind_log_context
from ml.orchestration.pipeline_runner import PipelineRunConfig
from ml.orchestration.pipeline_runner import run_pipeline
from ml.orchestration.pipeline_runner import setup_logging


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["backfill", "daily", "realtime"]),
    required=True,
    help="Pipeline execution mode",
)
@click.option("--start-date", type=str, help="Start date for backfill mode (YYYY-MM-DD)")
@click.option("--end-date", type=str, help="End date for backfill mode (YYYY-MM-DD)")
@click.option("--config", type=click.Path(exists=True), help="Path to YAML/JSON configuration")
@click.option("--dry-run", is_flag=True, default=False, help="Run without performing actions")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logging")
def main(
    mode: str,
    start_date: str | None,
    end_date: str | None,
    config: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    setup_logging(verbose)
    bind_log_context(
        run_id=f"cli_run_ml_pipeline_{_uuid.uuid4().hex[:8]}",
        component="ml.cli.run_ml_pipeline",
    )

    try:
        run_pipeline(
            PipelineRunConfig(
                mode=mode,
                start_date=start_date,
                end_date=end_date,
                config_path=config,
                dry_run=dry_run,
                verbose=verbose,
            ),
        )
    except Exception as exc:  # pragma: no cover - surfaced to shell
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
