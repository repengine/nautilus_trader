# MLPipelineOrchestrator Refactor Plan

## Goals

1. Offer a single entry point that can run ingestion, dataset build, training, and promotions independently or in sequence without requiring verbose flags.
2. Introduce a structured configuration file (TOML/JSON) so recurring runs do not require CLI flag duplication.
3. Preserve the new binding safeguards (availability + cost filters, informative logging) and surface ingestion failures immediately.
4. Maintain backwards compatibility where practical, but prefer stage-centric invocations for new workflows.

## High-level Strategy

1. **Configuration Layer**
   - Define a typed `OrchestratorRunConfig` dataclass with nested sections for ingestion, dataset, training, and integration.
   - Provide a loader (`ml.orchestration.config_loader`) that can decode TOML/JSON into that dataclass, honoring environment overrides (e.g. cast enums, parse duration fields).
   - Allow CLI to accept `--config path` plus limited overrides (symbols, start/end overrides) to keep one-off runs flexible.

2. **Stage Dispatcher**
   - Add a `--stage` flag (choices: `ingest`, `dataset`, `train`, `full`) or separate subcommands via `argparse`.
   - Map each stage to a clear orchestrator method (`run_ingestion`, `run_dataset`, `run_training`, `run_full`), making it obvious what will run.
   - Ensure each method logs start/finish with binding context and metrics/validator results so operators get immediate feedback.

3. **Incremental Execution**
   - Allow the orchestrator to short-circuit after ingestion (e.g. return early once ingestion completes, no dataset build).
   - For dataset-only runs, resolve bindings, build the dataset, and exit without invoking ingestion (unless `auto_ingest` flag is set).
   - Training-only mode should verify that dataset metadata exists (guarding with a helpful error if not).

4. **Safer Defaults**
   - Provide default symbol universe (e.g., load `ml/config/universe_l0_macro_47.txt` if `symbols` empty).
   - Default `data_dir` to `data/tier1`, `out_dir` to `ml_out/<run_id>` when not supplied.
   - Compute `start/end` from config lookbacks unless explicitly provided.

5. **Telemetry & Validation**
   - Keep structured logging with the new INFO-level binding statements.
   - Ensure ingestion failures raise and halt the pipeline (already in place via error logging & re-raise).
   - Optionally add a summary block at the end of each stage (rows ingested, store rows vs catalogue, macro validation status).

## Deliverables

1. New config dataclasses with loader in `ml/orchestration/config_loader.py` (backward compatible with existing loader but favoring the new structure).
2. Refactored CLI (`ml/cli/pipeline_orchestrator.py`) supporting `--config` plus `--stage`.
3. Reworked `MLPipelineOrchestrator` methods split into stage functions, orchestrated by `_run_stage` helper.
4. Updated docs (`ml/docs/orchestration_refactor_plan.md`, plus README/DEV guides) explaining stage usage.
5. Unit tests covering:
   - Config loading (TOML/JSON)
   - Stage dispatch (ingest-only, dataset-only, full)
   - Binding resolution remains intact when running via config-based orchestrator.

## Status (2025-09-13)

- ✅ Config loader + dataclasses landed (`config_loader.py`, `config_types.py`) with environment overrides and translation back to CLI flags.
- ✅ CLI now honors `--config`/`--stage`; dataset/train stages short-circuit through `_execute_stage` helpers.
- ✅ `MLPipelineOrchestrator.run_training_only` enables training-only flows with dataset validation and promotions.
- ✅ Deterministic tests cover config overrides, stage dispatch, and config argument extraction.
- ✅ Ingestion stage now emits structured metrics, surfaces credential gaps, and advances the progressive fallback chain (bindings → cached coverage → manual lookback → dummy).
- ✅ Ingestion-only runs now consume `IngestionStageConfig` values directly and exit with non-zero status when all fallbacks fail.
- ✅ Dashboard control polls pipeline progress and reflects orchestrator stages (e.g. `running:stage:ingest`).
- ✅ Ingestion-only stage now consumes descriptor-driven binding plans and fails fast when fallbacks exhaust.

## Stage-Oriented CLI Examples

```bash
# Ingestion only (auto-fill + Databento backfill with fallback metrics)
uv run --active --no-sync python -m ml.cli.pipeline_orchestrator \
  --config configs/orchestrator/nightly.toml \
  --stage ingest \
  --ingest

# Dataset refresh (stops after dataset validation/registration)
uv run --active --no-sync python -m ml.cli.pipeline_orchestrator \
  --config configs/orchestrator/nightly.toml \
  --stage dataset

# Training resume (requires existing dataset artifacts)
uv run --active --no-sync python -m ml.cli.pipeline_orchestrator \
  --config configs/orchestrator/nightly.toml \
  --stage train
```

Ingestion runs increment `nautilus_ml_ingestion_stage_runs_total` and tag fallback activations via `ml_fallback_activations_total{component="pipeline_orchestrator_ingestion", level="<fallback>"}` for cached/manually-triggered/dummy degradations.

## Risks / Considerations

- Need to guard against breaking existing automation that relies on the legacy CLI. Consider keeping the old flag behavior as a fallback (`--legacy-flags` or detect when `--config` is absent and operate in compatibility mode).
- Ensure that the new configuration still respects environment-provided credentials (Databento API key, DB connection).
- Test the interaction between `auto_fill` and stage selection to avoid unintended ingestion during dataset-only runs.
- Postgres connectivity is auto-resolved across the Compose port (5433) and the legacy localhost default (5432). All runtime components now share the same resolver, so configuring `NAUTILUS_DB` or `--db` once is sufficient.
- Ingestion auto-registers any dataset manifests required for DataStore writes, removing the manual `EQUS.MINI` bootstrap step for Databento flows.
- Partition management runs during orchestrator start-up via `PartitionManager.run_maintenance`, ensuring the `market_data` partition exists before SQL writers persist new frames.

## Next Steps

1. Prototype the new config dataclasses and loader (commit behind feature flag).
2. Refactor the CLI to accept `--config` and `--stage`, keeping legacy path intact for now.
3. Split orchestrator methods into stage-specific wrappers; ensure each stage reuses the existing building blocks.
4. Update docs/tests once the new flow stabilizes, then remove the deprecated flags if desired.
