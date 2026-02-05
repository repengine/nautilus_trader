**Plan Overview**
This plan restores the intended facade‑component architecture for the TFT dataset builder, re‑establishes a single canonical builder path, and reduces `ml/data/__init__.py` to a thin export layer. It explicitly reverses the post‑`83c83d9` collapse of the facade into a shim and aligns the codebase with the Phase 3.1 reports. [source: `reports/PHASE_3_1_COMPLETION_CERTIFICATE.md`, `reports/tasks/phase_3_1_tft_dataset_builder_facade_task_report.md`, commit `83c83d9ce42b7be79f25245100dcdf1afd0ee5f0`, commit `7d01a06bc03a4e1cfcb0b786310b4f7e45947c62`, `ml/data/tft_dataset_builder.py`, `ml/data/tft_dataset_builder_facade.py`, `ml/data/__init__.py`]

**Constraints**
- No production code should import from `ml/cli` or `ml/scripts`; only CLI/shims should import domain code. [source: `ml/cli/*`, `ml/scripts/*`, `ml/pipelines/build_runner.py`, `ml/orchestration/pipeline_orchestrator_cli.py`]
- Use `ml/data/common` components where the dataset builder orchestration lives; only a small set of production modules import these components today, so the blast radius is limited. [source: `ml/data/common/*`, `ml/data/tft_dataset_builder_facade.py`, `ml/data/scheduler.py`, `ml/features/common/tft_realtime_feature_calculator.py`, `ml/training/autogluon/soft_label_generator.py`]
- Preserve backward‑compatible import paths via shims where required, but reduce shims to thin delegations. [source: `ml/scripts/build_tft_dataset.py`, `ml/scripts/apply_migrations.py`, `ml/data/tft_dataset_builder_facade.py`]

**Phase 0: Baseline And Safety**
- [x] Capture the current state with `git status`, `git diff`, and a quick note of file sizes for `ml/data/tft_dataset_builder.py`, `ml/data/tft_dataset_builder_facade.py`, and `ml/data/__init__.py` to establish the baseline. [source: `ml/data/tft_dataset_builder.py`, `ml/data/tft_dataset_builder_facade.py`, `ml/data/__init__.py`]
- [x] Snapshot current public entry points that rely on the dataset builder (`ml/data/__init__.py`, `ml/tasks/datasets/tft.py`, `ml/cli/build_tft_dataset.py`, `ml/orchestration/dataset_builder.py`, `ml/pipelines/build_runner.py`). [source: `ml/data/__init__.py`, `ml/tasks/datasets/tft.py`, `ml/cli/build_tft_dataset.py`, `ml/orchestration/dataset_builder.py`, `ml/pipelines/build_runner.py`]
- [x] Record the pre‑collapse facade implementation location (commit `5fa2dd417b` or earlier) to serve as the restoration source of truth. [source: commit `5fa2dd417b9530a11a9b223cb72241054375ef62`, `reports/tasks/phase_3_1_tft_dataset_builder_facade_task_report.md`]

**Phase 0 Findings (2026-02-04)**
- Git state: branch `ml` is dirty with multiple modified, deleted, and untracked files (baseline captured via `git status -sb`). [source: `git status -sb`]
  - Modified: `ml/cli/hpo_tft.py`, `ml/cli/train_chronos.py`, `ml/config/autogluon.py`, `ml/config/lightgbm.py`, `ml/config/shared.py`, `ml/config/xgboost.py`, `ml/core/__init__.py`, `ml/docs/implementation/decision_target_execution_plan.md`, `ml/docs/tools/ORCHESTRATION_RUNBOOK.md`, `ml/orchestration/config_loader.py`, `ml/orchestration/config_types.py`, `ml/orchestration/pipeline_orchestrator_cli.py`, `ml/orchestration/training_coordinator.py`, `ml/tasks/datasets/splits.py`, `ml/tests/integration/actors/test_actor_circuit_breaker_integration.py`, `ml/tests/unit/config/test_config_classes.py`, `ml/tests/unit/config/test_env_builders.py`, `ml/tests/unit/tasks/test_purged_splits.py`, `ml/tests/unit/training/autogluon/test_chronos_evaluation.py`, `ml/tests/unit/training/common/test_cross_validation.py`, `ml/tests/unit/training/common/test_evaluation.py`, `ml/tests/unit/training/common/test_hyperparameter.py`, `ml/training/__init__.py`, `ml/training/autogluon/chronos_evaluation.py`, `ml/training/common/cross_validation.py`, `ml/training/common/evaluation.py`, `ml/training/common/hyperparameter.py`, `ml/training/teacher/tft_cli.py`. [source: `git status -sb`]
  - Deleted: `ml/tests/e2e/test_god_class_analysis_e2e.py`, `ml/tests/integration/analysis/test_analyzer_integration.py`. [source: `git status -sb`]
  - Untracked: `ML_DATA_CLEANUP_PLAN.md`, `ml/common/validation_strategies.py`, `ml/tests/unit/data/test_dataset_build_config_validation.py`, `ml/tests/unit/training/teacher/test_tft_cli_validation_strategy.py`, `ml/training/common/trading_costs.py`. [source: `git status -sb`]
- Diff summary captured with `git diff --stat` (30 files changed, 822 insertions, 701 deletions). [source: `git diff --stat`]
- File size baseline (line counts): `ml/data/tft_dataset_builder.py` 2586 lines, `ml/data/tft_dataset_builder_facade.py` 354 lines, `ml/data/__init__.py` 2181 lines. [source: `wc -l ml/data/tft_dataset_builder.py ml/data/tft_dataset_builder_facade.py ml/data/__init__.py`]
- Dataset builder entry points snapshot (current callers): `ml/data/__init__.py`, `ml/tasks/datasets/tft.py`, `ml/cli/build_tft_dataset.py`, `ml/orchestration/dataset_builder.py`, `ml/pipelines/build_runner.py`. [source: `ml/data/__init__.py`, `ml/tasks/datasets/tft.py`, `ml/cli/build_tft_dataset.py`, `ml/orchestration/dataset_builder.py`, `ml/pipelines/build_runner.py`]
- Pre‑collapse facade source of truth: commit `5fa2dd417b9530a11a9b223cb72241054375ef62` (full facade implementation with component orchestration). [source: commit `5fa2dd417b9530a11a9b223cb72241054375ef62`, `reports/tasks/phase_3_1_tft_dataset_builder_facade_task_report.md`]

**Phase 1: Restore The Facade‑Component Builder**
- [x] Re‑introduce the full facade implementation from pre‑`83c83d9` (e.g., `5fa2dd417b`) and make it the canonical builder path. [source: commit `5fa2dd417b9530a11a9b223cb72241054375ef62`, `reports/tasks/phase_3_1_tft_dataset_builder_facade_task_report.md`]
- [x] Move the current monolithic builder (`ml/data/tft_dataset_builder.py`) to `ml/data/tft_dataset_builder_legacy.py` to preserve behavior for parity testing and rollback. [source: `ml/data/tft_dataset_builder.py`, `ml/data/tft_dataset_builder_legacy.py`, `reports/PHASE_3_1_COMPLETION_CERTIFICATE.md`]
- [x] Ensure `ml/data/tft_dataset_builder.py` becomes the facade (either by moving the restored facade there or re‑exporting it from there) so external imports remain stable. [source: `ml/data/tft_dataset_builder.py`, `ml/data/tft_dataset_builder_facade.py`, `reports/PHASE_3_1_COMPLETION_CERTIFICATE.md`]
- [x] Keep `ml/data/tft_dataset_builder_facade.py` as the canonical facade (needed for existing import paths and docs) and re‑export it from `ml/data/tft_dataset_builder.py`. [source: `ml/data/tft_dataset_builder_facade.py`, `ml/data/tft_dataset_builder.py`, `ml/docs/context/context_tasks_pipelines.md`]

**Phase 1 Findings (2026-02-04)**
- Restored the full facade implementation from commit `5fa2dd417b` into `ml/data/tft_dataset_builder_facade.py` (now 1,546 lines, component‑orchestrated). [source: commit `5fa2dd417b9530a11a9b223cb72241054375ef62`, `ml/data/tft_dataset_builder_facade.py`]
- Preserved the monolithic builder as `ml/data/tft_dataset_builder_legacy.py` (2,586 lines) for parity testing and rollback. [source: `ml/data/tft_dataset_builder_legacy.py`]
- Converted `ml/data/tft_dataset_builder.py` into a thin compatibility wrapper that re‑exports `TFTDatasetBuilder` from the facade. [source: `ml/data/tft_dataset_builder.py`]
- Updated the facade to import the legacy builder from `ml.data.tft_dataset_builder_legacy` for fallback paths. [source: `ml/data/tft_dataset_builder_facade.py`, `ml/data/tft_dataset_builder_legacy.py`]

**Phase 2: Port Newer Behavior Into Components And Facade**
- [x] Diff the legacy builder (current `ml/data/tft_dataset_builder_legacy.py`) against the restored facade to identify all deltas introduced after the pre‑collapse facade. [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/tft_dataset_builder_facade.py`, commit `5fa2dd417b9530a11a9b223cb72241054375ef62`]
- [x] For each delta, decide whether it belongs in:
  - `ml/data/common/*` (shared data‑domain components),
  - a thin helper module under `ml/data/`, or
  - the facade orchestration itself (only orchestration logic). [source: `ml/data/common/*`, `ml/data/tft_dataset_builder_facade.py`]
- [x] Explicitly port newer behaviors commonly referenced in the pipeline surface (examples below) into components or shared helpers, not into a monolith:
  - [x] target semantics enforcement and parsing semantics [source: `ml/data/__init__.py`, `ml/tasks/datasets/tft.py`, `ml/orchestration/dataset_builder.py`]
  - [x] dataset CSV controls + sampling [source: `ml/cli/build_tft_dataset.py`, `ml/data/__init__.py`]
  - [x] macro revisions / vintage settings [source: `ml/data/__init__.py`, `ml/config/feature_flags.py` (historical), `ml/data/vintage.py`]
  - [x] market bindings / dataset bindings [source: `ml/data/ingest/market_bindings.py`, `ml/data/__init__.py`]
  - [x] validation and metadata guards [source: `ml/data/validation.py`, `ml/data/__init__.py`]

- [x] Map and port `_append_macro_delta_features_polars` → `ml/data/common/feature_alignment.py` (or a new macro‑delta helper). [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/common/feature_alignment.py`]
- [x] Map and port `_compute_features_polars/_compute_features_pandas` → `FeatureAlignmentComponent`. [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/common/feature_alignment.py`]
- [x] Map and port `_generate_targets_polars/_generate_targets_pandas` → `TargetGenerationComponent`. [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/common/target_generation.py`]
- [x] Map and port `_add_static_features_polars/_add_static_features_pandas` → `FeatureAlignmentComponent` (static feature integration). [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/common/feature_alignment.py`]
- [x] Map and port `_add_known_future_features_polars/_add_known_future_features_pandas` → `KnownFutureFeatureComponent`. [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/common/known_future_features.py`]
- [x] Map and port `_frame_time_bounds/_restrict_df_to_window/_coerce_to_ns` → `TimeSeriesWindowingComponent`. [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/common/time_series_windowing.py`]
- [x] Map and port `_extract_frame_metadata` → dataset metadata helper (new module under `ml/data/` or `ml/data/common/`). [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/validation.py`]
- [x] Map and port `_resolve_binding/_store_enabled` → market binding resolver/helper (likely under `ml/data/ingest/` or a new `ml/data/common/` helper). [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/ingest/market_bindings.py`]
- [x] Map and port `_get_event_provider` → event provider helper (new module or `ml/data/providers/events.py` integration). [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/providers/events.py`, `ml/data/sources/events.py`]
- [x] Map and port `_fetch_earnings_features` → dedicated earnings join helper (new `ml/data/earnings/*` or `ml/data/common/`) to remove reliance on legacy builder. [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/earnings/*`]

**Phase 2 Findings (Preliminary Delta List)**
- Methods present in legacy builder but not in facade (likely behaviors to port into components or helpers): `_add_known_future_features_pandas`, `_add_known_future_features_polars`, `_add_static_features_pandas`, `_add_static_features_polars`, `_append_macro_delta_features_polars`, `_coerce_to_ns`, `_compute_features_pandas`, `_compute_features_polars`, `_extract_frame_metadata`, `_fetch_earnings_features`, `_frame_time_bounds`, `_generate_targets_pandas`, `_generate_targets_polars`, `_get_event_provider`, `_resolve_binding`, `_resolve_target_semantics`, `_restrict_df_to_window`, `_store_enabled`. [source: `ml/data/tft_dataset_builder_legacy.py`, `ml/data/tft_dataset_builder_facade.py`]
- Methods present in facade but not in legacy (facade‑specific orchestration helpers): `_get_instrument_candidates`, `_join_earnings_features_pandas`, `_join_earnings_features_polars`, `_join_optional_features_pandas`, `_join_optional_features_polars`, `_try_parquet_fallback`, plus component accessors (`feature_alignment_component`, `known_future_component`, `schema_validator_component`, `target_generation_component`, `windowing_component`). [source: `ml/data/tft_dataset_builder_facade.py`]
- Facade public API updated to accept `target_semantics` (matching the legacy builder), and call sites were migrated to the canonical builder path for test patching. [source: `ml/data/tft_dataset_builder_facade.py`, `ml/tests/unit/data/test_tft_dataset_builder_facade.py`, `ml/tests/integration/data/test_tft_builder_integration.py`]

**Phase 2 Progress Update (2026-02-04)**
- Added shared helper modules to remove legacy-only behaviors:
  - `ml/data/common/event_provider.py` for `EventScheduleProvider` initialization.
  - `ml/data/common/earnings_join.py` for `_fetch_earnings_features` logic.
  - `ml/data/common/feature_alignment.py` now exposes `append_macro_delta_features_polars`.
  - `ml/data/common/time_series_windowing.py` now exposes `restrict_df_to_window`. [source: `ml/data/common/event_provider.py`, `ml/data/common/earnings_join.py`, `ml/data/common/feature_alignment.py`, `ml/data/common/time_series_windowing.py`]
- Centralized target semantics resolution and frame metadata extraction:
  - `ml/data/common/target_semantics.py` for target semantics enforcement + binary column resolution.
  - `ml/data/common/frame_metadata.py` for `source_dataset` metadata extraction shared by facade + legacy. [source: `ml/data/common/target_semantics.py`, `ml/data/common/frame_metadata.py`, `ml/data/tft_dataset_builder_facade.py`, `ml/data/tft_dataset_builder_legacy.py`, `ml/data/__init__.py`]
- Centralized dataset CSV controls and sampling:
  - `ml/data/common/dataset_csv.py` for write policy + sample emission shared by dataset builds. [source: `ml/data/common/dataset_csv.py`, `ml/data/__init__.py`]
- Introduced binding helpers in `ml/data/ingest/market_bindings.py`:
  - `build_binding_index`, `resolve_binding`, and `store_enabled` to consolidate binding lookup and store checks. [source: `ml/data/ingest/market_bindings.py`]
- Facade builder now uses binding helpers, parquet fallback guard, and centralized helpers:
  - `_load_bars_dataframe` aligns with legacy binding stats + fallback behavior.
  - `_append_macro_delta_features_polars` wired into macro join (direct + store).
  - Event provider init uses `build_event_provider`.
  - Earnings joins use `fetch_earnings_features` (no legacy dependency).
  - Per-symbol micro/L2/earnings/event joins restored in `_process_symbol_*`.
  - `_restrict_df_to_window` added to facade for parity tests. [source: `ml/data/tft_dataset_builder_facade.py`]
- Legacy builder now delegates to shared helpers:
  - `_store_enabled`, `_resolve_binding`, `_frame_time_bounds`, `_coerce_to_ns`, `_append_macro_delta_features_polars`, `_restrict_df_to_window`, `_get_event_provider`, `_fetch_earnings_features` routed through common helpers.
  - Macro deltas applied to direct (polars) path after macro join. [source: `ml/data/tft_dataset_builder_legacy.py`]
- Legacy builder now delegates feature/target/known-future logic to components:
  - `_compute_features_*` + `_add_static_features_*` via `FeatureAlignmentComponent`.
  - `_generate_targets_*` via `TargetGenerationComponent`.
  - `_add_known_future_features_*` via `KnownFutureFeatureComponent`.
  - Event schedule joins use `_get_event_provider` for shared provider setup. [source: `ml/data/tft_dataset_builder_legacy.py`]
- Facade polish for parity and CLI test shims:
  - Event joins now respect `include_calendar_lags/include_clustering_tags/include_context_features` in the pandas path and share provider initialization with the polars path. 
  - Parquet fallback in the facade now honors `ML_TFT_ALLOW_PARQUET_FALLBACK` like the legacy builder.
  - `ml/data/tft_dataset_builder.py` re-exports `bars_to_dataframe` so fixtures can patch the facade path without touching the catalog module directly. [source: `ml/data/tft_dataset_builder_facade.py`, `ml/data/tft_dataset_builder.py`, `ml/tests/fixtures/datasets.py`]

**Phase 3: Parity Testing Between Facade And Legacy**
- [x] Add or extend parity tests that exercise both the restored facade and the legacy builder with identical configs, comparing outputs and metadata. [source: `ml/tests/unit/data/test_tft_dataset_builder_facade.py`, `ml/tests/e2e/test_tft_dataset_builder_e2e.py`]
- [x] Ensure parity tests cover:
  - row/column counts
  - schema + metadata contracts
  - feature flags and include_* toggles
  - macro/vintage controls
  - dataset CSV sampling behavior [source: `ml/tests/unit/data/test_dataset_build_macro.py`, `ml/tests/unit/data/common/test_tft_schema_validator.py`, `ml/cli/build_tft_dataset.py`]
- [x] Run the existing E2E dataset builder test suite to verify end‑to‑end parity. [source: `ml/tests/e2e/test_tft_dataset_builder_e2e.py`]

**Phase 3 Progress Update (2026-02-05)**
- Added facade vs legacy parity tests (capability flags + macro/vintage + CSV sampling) in `ml/tests/unit/data/test_dataset_build_macro.py`. [source: `ml/tests/unit/data/test_dataset_build_macro.py`]
- Parity checks enforce core schema equality and allow facade to emit optional OHLCV columns (`open`, `high`, `low`, `volume`) since `TFTSchemaValidatorComponent` treats them as optional. [source: `ml/data/common/tft_schema_validator.py`, `ml/tests/unit/data/test_dataset_build_macro.py`]
- Executed the E2E dataset builder suite (`ml/tests/e2e/test_tft_dataset_builder_e2e.py`). [source: `ml/tests/e2e/test_tft_dataset_builder_e2e.py`]
- Decision: keep optional OHLCV columns in the facade (no strict schema parity yet); revisit before Phase 4 if downstream consumers require exact column matching. [source: `ml/data/common/tft_schema_validator.py`, `ml/data/tft_dataset_builder_facade.py`, `ml/tests/unit/data/test_dataset_build_macro.py`]

**Phase 4: Remove The Legacy Builder**
- [x] Remove `ml/data/tft_dataset_builder_legacy.py` after parity tests are green and builders are API‑compatible. [source: `ml/data/tft_dataset_builder_legacy.py`, `reports/PHASE_3_1_COMPLETION_CERTIFICATE.md`]
- [x] Remove legacy fallback logic and the `ML_USE_LEGACY_TFT_BUILDER` selector from the facade; ensure a single canonical path. [source: `ml/data/tft_dataset_builder_facade.py`]
- [x] Update tests to drop legacy/facade toggling and convert parity tests into contract/invariant tests for the canonical builder. [source: `ml/tests/unit/data/test_dataset_build_macro.py`, `ml/tests/unit/data/test_tft_dataset_builder_facade.py`]
- [x] Update docs that reference legacy builder paths or flags (e.g., `ml/docs/context/context_data.md`, `ml/docs/implementation/decision_target_review_plan.md`). [source: `ml/docs/context/context_data.md`, `ml/docs/implementation/decision_target_review_plan.md`]
- [x] Confirm optional OHLCV columns policy (currently allowed in facade) is documented in schema/validation expectations before removal. [source: `ml/data/common/tft_schema_validator.py`, `ml/tests/unit/data/test_dataset_build_macro.py`]

**Phase 4 Progress Update (2026-02-05)**
- Deleted `ml/data/tft_dataset_builder_legacy.py` and removed legacy builder delegation. [source: `ml/data/tft_dataset_builder_facade.py`, `ml/data/tft_dataset_builder.py`]
- Removed `ML_USE_LEGACY_TFT_BUILDER` toggle and legacy selection branches from the facade. [source: `ml/data/tft_dataset_builder_facade.py`]
- Converted facade/legacy parity tests into canonical builder contract tests (macro/vintage + CSV sampling + capability flags). [source: `ml/tests/unit/data/test_dataset_build_macro.py`]
- Updated facade unit/integration tests to patch canonical build paths instead of legacy builder shims. [source: `ml/tests/unit/data/test_tft_dataset_builder_facade.py`, `ml/tests/integration/data/test_tft_builder_integration.py`]
- Updated documentation references from legacy builder paths to the facade implementation. [source: `ml/docs/implementation/decision_target_review_plan.md`, `ml/docs/implementation/decision_target_execution_plan.md`, `FEATURE_PARITY_PLAN.md`, `training_status.md`]

**Phase 5: Thin `ml/data/__init__.py`**
- [ ] Identify implementation blocks in `ml/data/__init__.py` (currently >2k lines) and move them into focused modules (e.g., `ml/data/build.py`, `ml/data/metadata.py`, `ml/data/manifest.py`). [source: `ml/data/__init__.py`]
- [ ] Keep `ml/data/__init__.py` as a thin re‑export surface only, aligning with domain conventions. [source: `ml/data/__init__.py`, `ml/data/common/__init__.py`]
- [ ] Update imports across the codebase to the new module locations while preserving public API exports. [source: `ml/data/__init__.py`, `ml/tasks/datasets/tft.py`]

**Phase 5 Progress Update (2026-02-04)**
- Extracted dataset metadata structures and helpers into `ml/data/metadata.py` and re‑exported from `ml/data/__init__.py`. [source: `ml/data/metadata.py`, `ml/data/__init__.py`]
- Extracted macro/vintage refresh + validation glue into `ml/data/common/macro_vintage.py` and rewired dataset build orchestration to use it. [source: `ml/data/common/macro_vintage.py`, `ml/data/__init__.py`]
- Extracted capability flag derivation into `ml/data/common/capability_flags.py` and consolidated window-based metadata assembly via `build_dataset_metadata_from_windows`. [source: `ml/data/common/capability_flags.py`, `ml/data/metadata.py`, `ml/data/__init__.py`]
- Centralized dataset metadata JSON serialization via `write_dataset_metadata` and moved ALFRED refresh window selection into `ml/data/common/macro_vintage.py`. [source: `ml/data/metadata.py`, `ml/data/common/macro_vintage.py`, `ml/data/__init__.py`]
- Moved dataset build orchestration into `ml/data/build.py`, leaving `ml/data/__init__.py` as a thin re‑export surface. [source: `ml/data/build.py`, `ml/data/__init__.py`]
- Extracted dataset build event emission into `ml/data/common/dataset_events.py` and routed `build_tft_dataset` through the helper. [source: `ml/data/common/dataset_events.py`, `ml/data/build.py`]

**Phase 6: Enforce CLI Import Boundaries**
- [ ] Replace any non‑CLI imports of `ml/cli/*` in domain code with calls to canonical domain APIs. [source: `ml/pipelines/build_runner.py`, `ml/core/common/database_lifecycle.py`, `ml/orchestration/pipeline_orchestrator_cli.py`]
- [ ] Keep `ml/scripts/*` as thin shims that only invoke CLI entry points or domain APIs. [source: `ml/scripts/build_tft_dataset.py`, `ml/scripts/apply_migrations.py`]

**Phase 7: Validation And Coverage**
- [x] Run `poetry run mypy ml --strict` and fix any signature drift introduced by refactors. [source: `AGENTS.md`]
- [x] Run `poetry ruff check ml` and fix lint violations. [source: `AGENTS.md`]
- [x] Run `make validate-fixtures` to ensure fixtures remain compliant. [source: `AGENTS.md`]
- [x] Run focused tests for dataset builder, orchestration, and pipelines impacted by refactors (e.g., `pytest -k tft_dataset_builder`). [source: `ml/tests/e2e/test_tft_dataset_builder_e2e.py`, `ml/tests/unit/data/test_tft_dataset_builder_facade.py`]
- [ ] Verify coverage thresholds: ML modules ≥90%, general Python ≥80%. [source: `AGENTS.md`]

**Phase 7 Validation Update (2026-02-04)**
- Ran `poetry run mypy ml --strict` (success; 0 errors). [source: `poetry run mypy ml --strict`]
- Ran `poetry run ruff check ml` (success; 0 violations). [source: `poetry run ruff check ml`]
- Ran `make validate-fixtures` (success; export guard passed). [source: `make validate-fixtures`]
- Focused tests executed:
  - `poetry run pytest ml/tests/unit/data/common/test_pipeline_batch_executor.py -x` (pass). [source: `ml/tests/unit/data/common/test_pipeline_batch_executor.py`]
  - `poetry run pytest ml/tests/metamorphic/test_feature_transforms.py -x` (pass). [source: `ml/tests/metamorphic/test_feature_transforms.py`]
  - `poetry run pytest ml/tests/unit/features/test_feature_parity_fix.py -x` (pass; 1 xfail expected). [source: `ml/tests/unit/features/test_feature_parity_fix.py`]
  - `poetry run pytest ml/tests/integration/test_registry_store_l2_integration.py -x` (pass). [source: `ml/tests/integration/test_registry_store_l2_integration.py`]
  - `poetry run pytest ml/tests/property/test_config_combinations.py -x` (pass). [source: `ml/tests/property/test_config_combinations.py`]
  - `poetry run pytest ml/tests/combinatorial/test_config_combinations.py -x` (pass). [source: `ml/tests/combinatorial/test_config_combinations.py`]
  - `poetry run pytest ml/tests/property/test_signal_actor_bounds.py -x` (pass). [source: `ml/tests/property/test_signal_actor_bounds.py`]
  - `poetry run pytest -q ml/tests/e2e/test_tft_dataset_builder_e2e.py` (pass; warnings from polars concat deprecation). [source: `ml/tests/e2e/test_tft_dataset_builder_e2e.py`, `ml/data/tft_dataset_builder_facade.py`]
- Coverage: ran `poetry run coverage report`; overall total reported 63.96% (below target; follow-up required). [source: `poetry run coverage report`]

**Phase 8: Documentation And Report Updates**
- [ ] Update or add a follow‑up report describing the restored facade‑component architecture and the removal of legacy code. [source: `reports/PHASE_3_1_COMPLETION_CERTIFICATE.md`, `reports/tasks/phase_3_1_tft_dataset_builder_facade_task_report.md`]
- [ ] Refresh any ops or developer docs that reference dataset builder entry points or file locations. [source: `ml/docs/context/context_tasks_pipelines.md`, `ml/docs/context/context_examples_scripts.md`, `ml/docs/README.md`]

**Exit Criteria**
- [ ] Canonical dataset builder path is the facade implementation, with no legacy fallback in production. [source: `ml/data/tft_dataset_builder.py`, `ml/data/tft_dataset_builder_facade.py`]
- [ ] `ml/data/__init__.py` is a thin export file (no large implementation blocks). [source: `ml/data/__init__.py`]
- [ ] No production modules import from `ml/cli` or `ml/scripts`. [source: `ml/pipelines/build_runner.py`, `ml/core/common/database_lifecycle.py`, `ml/orchestration/pipeline_orchestrator_cli.py`]
- [ ] Parity tests and E2E builder tests pass, with coverage thresholds met. [source: `ml/tests/e2e/test_tft_dataset_builder_e2e.py`, `ml/tests/unit/data/test_tft_dataset_builder_facade.py`, `AGENTS.md`]
