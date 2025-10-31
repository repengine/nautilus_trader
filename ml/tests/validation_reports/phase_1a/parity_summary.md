# Phase 1a – Component vs Legacy Builder Parity

## Overview

This report captures validation runs comparing the legacy `ml.data.tft_dataset_builder_legacy.TFTDatasetBuilder`
against the component-based `ml.data.tft_dataset_builder.TFTDatasetBuilder`. The scenarios mirror the feature
families now surfaced through `TFTStreamingConfig`, `DatasetServiceConfig`, and CLI toggles.

## Scenarios Covered

| Scenario | Flags | Notes |
|----------|-------|-------|
| `macro_only_v1` | `include_macro=True`, `include_macro_revisions=False` | Baseline parity check used by unit tests. |
| `macro_revisions_core` | `include_macro=True`, `include_macro_revisions=True`, `macro_revision_windows=[]` | Confirms revision defaults match (see macro tests). |
| `student_lightweight` | `student_mode=True`, `include_macro=True` (requested) | Verifies student builder disables heavy augmenters. |
| `calendar_events` | `include_macro=True`, `include_calendar=True`, `include_events=True` | Calendar/event joins align; capability flags propagate through manifests. |
| `earnings` | `include_macro=True`, `include_earnings=True`, `earnings_lag_days=1` | Earnings publication lags respected; no schema drift. |
| `micro` | `include_macro=True`, `include_micro=True` | Component vs. legacy matched on 1,113,763 rows; component additionally emits a `close` column. |
| `l2` | `include_macro=True`, `include_micro=True`, `include_l2=True` | Component vs. legacy matched on 1,113,763 rows with the same `close` column difference as above. |
| `phase_one_signals` | `include_macro=True`, `include_macro_deltas=True`, `include_calendar_lags=True`, `include_clustering_tags=True`, `include_context_features=True` | 2024-01-01 → 2024-01-08 SPY cohort compares component vs. legacy builders. Component output adds `PAYEMS_delta_1d`, `UNRATE_delta_1d`, and retains the extra `close` column; all shared numeric fields remain identical and metadata captures the PhaseOne feature families. |

## Artefacts

- `ml/tests/validation_reports/phase_1a/macro_only_v1_summary.md` — component and legacy builders match on 1,114,671 rows with no numeric deltas.
- `ml/tests/validation_reports/phase_1a/macro_revisions_core_summary.md` — revision defaults align; no schema or metric drift detected.
- `ml/tests/validation_reports/phase_1a/student_lightweight_summary.md` — student mode disables heavy augmenters while keeping numerical parity across shared columns.
- `ml/tests/validation_reports/phase_1a/calendar_events_summary.md` — calendar/event toggles maintain parity on 1,114,671 rows.
- `ml/tests/validation_reports/phase_1a/earnings_summary.md` — earnings join matches legacy output with identical metrics.
- `ml/tests/validation_reports/phase_1a/micro_summary.md` — regenerated 2025-10-22T22:51:17Z UTC; all shared columns match exactly with the component builder emitting an extra `close` column across 1,113,763 joined rows.
- `ml/tests/validation_reports/phase_1a/l2_summary.md` — regenerated 2025-10-22T22:51:17Z UTC; identical shared-column parity and the same `close` column difference across 1,113,763 joined rows.
- `ml/tests/validation_reports/phase_1a/phase_one_signals_summary.md` — generated from `ml_out/phase_one_signals_cohort` (SPY, 2024-01-01 → 2024-01-08). Component output adds macro delta columns plus the canonical `close` column; all shared numeric features match exactly and metadata enumerates PhaseOne signal families with capability flags set.

## Evidence

- Unit parity assertions:  
  - `ml/tests/unit/data/test_tft_dataset_builder_store.py::test_component_builder_respects_include_flags`  
  - `ml/tests/unit/data/test_tft_dataset_builder_store.py::test_component_builder_macro_revision_defaults`  
  - `ml/tests/unit/data/test_tft_dataset_builder_store.py::test_component_builder_student_mode_forces_feature_flags`
- PhaseOne signal enrichment unit coverage:  
  - `ml/tests/unit/data/test_tft_dataset_builder_phase_one.py::test_append_macro_delta_features_polars_computes_differences`  
  - `ml/tests/unit/data/test_tft_dataset_builder_phase_one.py::test_append_macro_delta_features_pandas_computes_differences`  
  - `ml/tests/unit/data/test_tft_dataset_builder_phase_one.py::test_event_features_join_when_calendar_lags_enabled`
- Planner coverage enforcing service-level overrides:  
  - `ml/tests/unit/training/event_driven/test_dataset_service.py::test_streaming_dataset_planner_merges_feature_flags`
- CLI/sidecar integration with capability flags:  
  - `ml/tests/integration/pipeline/test_tft_pipeline_sidecar.py::test_pipeline_reads_sidecar_when_args_missing`

## Next Steps

1. Automate dataset diffs (schema + histograms) for the scenarios above and store the artefacts alongside this summary.
2. Backfill the remaining Tier 1 symbol without Databento coverage (`BRK.XNAS`) once an alternate feed is available, then rerun parity to confirm the cohort is complete.
3. Extend coverage to Phase 1b feature families once their pipelines are wired through FeatureStore, and capture a dedicated PhaseOne signal parity report after the enriched dataset is promoted, persisting the artefact in this directory.
