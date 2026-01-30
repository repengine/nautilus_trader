# Staged Diff Group Review (2026-01-30)

## Purpose
Capture the full scope of the current staged changes and split them into digestible groups. Successive Codex sessions should tackle one group at a time, update this document with their findings, and append their contribution to the commit-message log below.

## Workflow
1. Choose the lowest-numbered group that still needs review and run the `git diff --staged` command limited to that group’s paths (see each section for the recommended command).
2. Summarize what changed, note any open questions, and add or refine the commit-message entry for that group in the table at the bottom.
3. Mark the group as reviewed (manually edit the section header or add a ✔) before moving on to the next group.
4. Repeat until all groups are marked complete and the staged diff is fully understood.

---

### ✔ Group 1 – Signal surface & actor inference (reviewed)
**Key files to review:** `ml/common/prediction_surface.py`, `ml/actors/base.py`, `ml/actors/common/signal_strategy.py`, `ml/actors/common/signal_metadata.py`, `ml/actors/multi_signal.py`, `ml/actors/signal_facade_impl.py`, `ml/common/timestamps.py`, plus `ml/tests/unit/common/test_prediction_surface.py`, `ml/tests/property/test_prediction_surface_policy.py`, `ml/tests/utils/stubs.py`, and the actor-focused property/unit tests that now depend on probability metadata.

**Summary:**
- Introduced the shared `prediction_surface` helpers that normalize raw outputs/logits into canonical probability-and-confidence pairs, resolve logits semantics, and derive BUY/SELL/HOLD decisions using neutral bands plus tighter timestamp normalization for bar metadata.
- All inference entry points (Facade, Base/Enhanced ONNX, sklearn, multi-signal batching, and signal strategies) now treat predictions as calibrated probabilities, route through `normalize_prediction_output/batch`, and surface the new `signal_metadata` payload instead of ad-hoc clipping.
- Updated strategy signals (threshold, extremes, momentum, ensemble, adaptive) to merge actor metadata, adjust probability arithmetic, and log sanitized metadata for downstream facades/store writes.
- Added stub helpers for bar metadata rendering and property/units tests for the new prediction surface policies and signal metadata expectations.

**Commit-message section:** `Normalize ML signal surface and metadata so every actor emits calibrated probabilities, neutral-band decisions, and shared tests for the prediction surface helpers.` 

**Next session action:** run `git diff --staged -- ml/common/prediction_surface.py ml/actors/base.py ml/actors/common/signal_strategy.py ml/actors/common/signal_metadata.py ml/actors/multi_signal.py ml/actors/signal_facade_impl.py` and update this section once you’ve digested file contents.

---

### ✔ Group 2 – Strategy store/services and returns support (reviewed)
**Key files:** `ml/stores/services/strategy_services.py`, `ml/stores/strategy_store.py`, `ml/strategies/common/returns_updater.py`, `ml/strategies/common/lifecycle.py`, `ml/strategies/common/position_management.py`, `ml/strategies/base_facade.py`, `ml/strategies/ml_strategy.py`, `ml/stores/providers.py`, `ml/stores/adapters.py`, `ml/stores/base.py`, `ml/strategies/common/decision_persistence.py`, plus the companion unit tests under `ml/tests/unit/stores/`, `ml/tests/unit/strategies/`, and property/contract suites touching strategy stores.

**Summary:**
- Strategy persistence now tracks `run_id`/`ingested_at_ns`, surfaces new `StrategyRiskHaltEvent` and `StrategyReplaySummary` dataclasses, and flushes risk-halt batches before returning; each table exposes the helper services and emits registry events with auto-registration fallback.
- The new `ReturnsUpdater` resolves signal/bar prices with cadence controls, feeds position sizers/portfolio managers, and is wired into `BaseMLStrategyFacade` / `MLTradingStrategy` so decision persistence honors neutral-band HOLDs, risk-halt transitions, and sizing rejection hold persistence.
- Position management tracks rejection reasons, values-based sizing flows, and share check-state metrics; decision persistence/strategy store APIs accept optional `run_id`, `persist_hold`, and richer risk metrics so tests can assert end-to-end behavior.
- Added unit/property coverage for the new services, returns updater, and store protocols (e.g., `ml/tests/unit/strategies/common/test_returns_updater.py`, `ml/tests/unit/stores/test_strategy_services_unit.py`, `ml/tests/unit/strategies/test_ml_trading_strategy_sizing_reject.py`), ensuring reason strings, neutral-band decisions, and risk-halt logs behave as expected.

**Next session action:** Move to Group 3 by reviewing `ml/cli/feature_dataset_mirror_refresh.py`, `ml/stores/feature_dataset_mirror.py`, `ml/config/feature_dataset_mirror.py`, `ml/data/**`, and the docs/deployment changes so the dataset mirror tooling and pipeline entrypoint are understood.

---

### ✔ Group 3 – Dataset mirror, coverage config, CLI, docs, and pipeline (reviewed)
**Key files:** `ml/cli/feature_dataset_mirror_refresh.py`, `ml/stores/feature_dataset_mirror.py`, `ml/config/feature_dataset_mirror.py`, `ml/config/base.py`, `ml/config/coverage_datasets_tier1.toml`, the new `ml/config/macro_*.txt` lists, `ml/data/coverage/*.py`, `ml/data/rehydration/catalog_rehydrator.py`, `ml/data/tft_dataset_builder.py`, `ml/data/ingest/*`, `ml/config/market_data.py`, `ml/deployment/entrypoint_pipeline.py`, `ml/deployment/README.md`, and the new docs under `ml/docs/implementation/` plus `.gitignore` entries for the new configs.

**Summary:**
- Added the `feature_dataset_mirror_refresh` CLI plus `ml.stores.feature_dataset_mirror` so SQL-backed macro observations/releases/events can be dumped to Parquet mirrors driven by macro-series lists (`macro_fred_series.txt`, `macro_alfred_fallback_series.txt`), with pandas lazily required and export totals logged.
- Coverage tooling now warns on catalog/SQL parity gaps, stops restoring certain datasets after targets, and rehydrates quotes/trades via streaming reads that reuse `MarketDataTableConfig`; restored buckets emit dataset events and watermarks, and the market-data config gained batch-size/sentinel overrides needed by the reingestion flow.
- The new ML pipeline entrypoint orchestrates coverage bucket classification/restoration/reingestion, exposes health/coverage metrics, loads the feature coverage manifest, triggers manifest-driven dataset updates, and documents the decision/execution plans in `ml/docs/implementation/{decision_target_execution_plan,decision_target_review_plan}.md` along with updated deployment readme notes to explain the new mirror/coverage lifecycle.

**Next session action:** Move on to Group 4 (schema/orchestration/migrations) — begin with `ml/stores/migrations/02*.sql`, `ml/schema.py`, and `ml/registry/common/sql_utils.py` and update that section after absorbing those diffs.

---

### ✔ Group 4 – Registry/orchestration/schema/migrations (reviewed)
**Key files:** `ml/registry/common/sql_utils.py`, `ml/registry/*`, `ml/orchestration/*`, `ml/schema.py`, `ml/config/events.py`, `ml/stores/migrations/*.sql`, `ml/stores/migrations_bootstrap/001_bootstrap.sql`, `ml/stores/migrations_runner*.py` (if touched), and `ml/tests/unit/registry` + schema/manifest/property tests.

**Summary:**
- Added `set_instrumentation_search_path` plus search-path enforcement inside the registry/event helpers so the new instrumentation tables (risk halts/replay summaries) resolve in `public`, and the bootstrap manifests now expect non-negative probabilities, `signal_type` strings, and surfaced prediction-surface metadata from `PREDICTION_SURFACE_V1`.
- `ml/schema` now ships `PredictionSurfaceSpec`, registers prediction/signal schemas, and exposes the constant surface metadata used by manifests; dataset types enumerations include `RISK_HALT_EVENTS` and `REPLAY_SUMMARY` with primary keys defined in the manifest defaults.
- The orchestration flow propagates `provider_schema`, and the replay harness now sanitizes NaNs, records replay summaries, persists counts from the `StrategyStore`, and wires returns/positions configs to drive the new store columns.
- The migrations add `run_id/ingested_at` to signals/order tables, create the new risk-halt/replay tables, and tighten data event/dataset registry constraints to allow the new stages/dataset types, with partition-aware updates in both `public` and `ml_registry`.

**Next session action:** move to Group 5 (Portfolio & Nautilus core) by reviewing the Cython portfolio/cache patches, Nautilus analyzer updates, and the corresponding tests to ensure account/bar return tracking is fully understood.

---

### ✔ Group 5 – Portfolio & Nautilus core adjustments (reviewed)
**Key files:** `nautilus_trader/portfolio/config.py`, `nautilus_trader/portfolio/portfolio.pxd`, `nautilus_trader/portfolio/portfolio.pyx`, `nautilus_trader/analysis/analyzer.py`, `tests/unit_tests/analysis/test_analyzer.py`, `tests/unit_tests/portfolio/test_portfolio.py`, `nautilus_trader/cache/cache.pyx`, `nautilus_trader/execution/engine.pyx`, and the `ml/tests/validation/system_validation_smoke_test.py` guard.

**Summary:**
- Portfolio gained `track_account_returns` and `track_bar_returns` knobs, persists last balances per account, and computes returns for the analyzer when updates arrive so replays/backtests can capture performance without affecting live throughput; analyzer now restores existing returns before re-adding positions.
- Added unit tests ensuring the analyzer preserves prior returns and that portfolio bar updates actually record returns when tracking is enabled, and the smoke test unwraps dummy stores to avoid false positives from proxy attributes.
- Cache snapshotting now catches exceptions/logs them (to keep NETTING OMS from crashing), and the execution engine only snapshots when `snapshot_positions` is configured; this keeps backtests safe while still capturing history when desired.

**Next session action:** None – all groups reviewed, the commit-message log is ready for summarizing the full changes in sequence.

---

## Commit-message log
| Group | Current draft bullet (IPv4-friendly) |
|---|---|
| 1 | Normalize ML signal surface inputs/metadata and treat every actor prediction as a probability with decision-neutral bands. |
| 2 | Refactor StrategyStore persistence/services to track run/ingest metadata, add risk-halt/replay summaries, and wire a returns updater into sizing/portfolio flows. |
| 3 | Export SQL macro datasets/events into Parquet mirrors, harden coverage restoration/reingestion, and document the new pipeline entrypoint/decisions. |
| 4 | Add schema/registry helpers plus migrations for risk-halt/replay datasets, keep manifests aligned with the prediction surface, and wire orchestration to emit the new stages. |
| 5 | Add optional account/bar return tracking to Portfolio/Analyzer, stabilize cache snapshots, and gate engine history snapshots with config. |

*(Feel free to edit any bullet as you learn more; once a group is fully reviewed, prepend the section header with “✔” to indicate completion.)*
