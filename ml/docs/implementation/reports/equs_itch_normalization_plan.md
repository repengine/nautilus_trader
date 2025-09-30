# EQUS vs XNAS Normalization Improvement Plan

## Current Observations

- **Coverage gap** – `EQUS.MINI` begins on 2023‑03‑28 while `XNAS.ITCH` reaches back to 2018‑05‑01. Any pre‑2023 backfill must rely entirely on ITCH-derived data.
- **Parity drift** – The Tier‑1 parity suite (`make parity-report`) still shows mean feature deltas ≈0.05 and max deltas up to ≈19 (INTC, 2023‑09‑18). Volumes remain the dominant source of error.
- **Different products** – EQUS aggregates multiple venues, applies corporate-action adjustments, filters sale conditions, and trims sessions. Raw ITCH bars mirror only NASDAQ executions with no adjustment.
- **Entitlement limit** – Our Databento subscription exposes only the last 30 days of depth (`mbo`, `mbp-10`). We must use that window for calibration, then apply the learned rules to historical trades.

## Objectives

1. Learn EQUS normalization behaviour from recent depth data and encode it as reusable calibration artefacts.
2. Apply those rules to historical ITCH trades so fallback bars converge toward EQUS parity (price, volume, derived features).
3. Instrument parity metrics, calibration freshness, and provenance so regressions halt the pipeline before promotion.

The current refactor continuation focuses on transforming these objectives into three concrete, tightly scoped workstreams: calibration capture automation, ingestion alignment, and validation guardrails. Each workstream must preserve the 4-store architecture, keep hot-path execution allocation-free, and expose only typed public APIs through the appropriate `__init__.py` facades.

## Proposed Actions

### 1. Measurement & Instrumentation
- Expand the parity harness to sample multiple Tier‑1 symbols per quarter, emitting mean/max feature deltas, price/volume correlations, percentile residuals, and the worst offending timestamps.
- Persist suite output in `ml/tests/validation_reports/` and fail `make parity-report` when thresholds are exceeded or calibration artefacts are stale.
- Surface calibration metadata (version, generated_at, symbol coverage) via structured logs and Prometheus (`ml_canonicalization_volume_residual`, calibration freshness gauges).

### 2. Calibration Pipeline (Rolling 30-Day Depth Window)
- Build a CLI (or parity CLI mode) that captures overlapping EQUS.MINI and ITCH slices (`ohlcv`, `trades`, `depth`) for the Tier‑1 universe, applies sale-condition eligibility discovery, and writes artefacts under the configured output directory.
- Emit calibration artefacts as JSON documents with `generated_at`, symbol-indexed payloads, and schema compatibility with `ml/data/ingest/calibration.py:SymbolCalibration` (fields: `sale_condition_allowlist`, `volume_scale_by_minute`, `price_scale_by_minute`, `split_events`, `exclude_auction_minutes`).
- Learn minute-of-day price/volume scalers and auction exclusions by comparing calibrated ITCH reaggregations against EQUS bars inside the 30-day window; persist monotonic split factors derived from ITCH `definition` messages or curated corporate-action feeds.
- Ship CLI output via `ML_EQUS_CALIBRATION_PATH` so ingestion services, parity tooling, and validators consume a consistent bundle; publish provenance (`calibration_version`, `source_window`) alongside the artefact.
- Keep CLI execution on the cold path, wrap message bus publishes with `try/except`, and use `ml.common.metrics_bootstrap` for metrics (no direct Prometheus collectors).

### 3. Enhanced Fallback Reconstruction (2018–Present)
- Extend `DatabentoIngestionService` (via `_attempt_fallback_to_itch`/`_apply_calibration_to_bars`) to filter trades by the allowlist, drop auction minutes, scale price/volume per minute, and apply split adjustments before canonicalization.
- Skip legacy global scaling when calibration artefacts exist while maintaining the progressive fallback chain (PRIMARY → CACHED → FILE → DUMMY) and preserving hot-path zero-allocation behaviour.
- Record calibration application in provenance columns (`source_dataset`, `aggregation_mode`, `calibration_version`) and emit structured telemetry for residuals via `ml_canonicalization_volume_residual`.
- Schedule periodic calibration refreshes (monthly minimum) and enforce expiry guards: stale artefacts should fail ingestion parity checks and prevent fallback promotion until regenerated.

### 4. Metadata & Provenance
- Continue populating `source_dataset`, `aggregation_mode`, `scaling_factor`, and add `calibration_version` so downstream consumers understand how each bar was produced.
- Emit structured logs and events whenever calibration inputs change; include parity suite results in release notes.

### 5. Testing & Validation
- Add CLI contract tests that load generated JSON and validate against `SymbolCalibration` (including split/auction metadata) and parity CLI path coverage.
- Expand ingestion unit tests (`ml/tests/unit/ingest/test_ingestion_service.py`) to verify price/volume scaling, split handling, and sale-condition filtering; ensure calibration absence falls back to legacy behaviour deterministically.
- Update parity orchestration tests (`ml/tests/unit/scripts/test_verify_eq_itch_parity.py`) to fail when calibration artefacts are stale, missing symbols, or produce residuals beyond thresholds; surface diagnostics in `ml/tests/validation_reports/`.
- Document the calibration refresh workflow and guardrails in runbooks, linking to `make parity-report`, `make validate-metrics`, and `make validate-events`.

### 6. Operational Guardrails
- Message bus interactions must use `ml.common.message_topics.build_topic_for_stage` with enums from `ml.config.events.{Stage, Source, EventStatus}`; keep publishes off the hot path.
- Persist observability via DTO builders and `MetricsManager`; avoid direct Prometheus collector manipulation.
- Maintain progressive fallback safety nets, circuit breaker hooks, and structured logging for calibration state transitions.
- Respect the 4-store pattern: actors continue to depend on pre-initialized stores and registries rather than instantiating ad-hoc resources.

### Status (2025-10-05)
- ✅ Trade-level reaggregation (`aggregation_mode=reaggregated_trades`) and volume scaling (`aggregation_mode=scaled_volume`) are available with telemetry.
- ✅ Calibration capture CLI ships SymbolCalibration-compatible JSON and persists provenance metadata (`calibration_version`).
- ✅ Databento ingestion fallback applies calibration bundles (allowlists, scalers, splits) and records calibration version in canonicalized EQUS rows.
- ✅ Tier‑1 parity suite enforces calibration freshness via CLI guardrails and persists reports.

## Open Questions
- Cost/latency impact of monthly depth calibration across the full Tier‑1 universe.
- Precise EQUS sale-condition matrix and venue coverage (confirm with Databento if documentation is incomplete).
- Target thresholds per symbol for acceptable mean/max feature deltas (aligned with TFT feature sensitivity and downstream risk tolerances).

## Next Steps
1. Automate calibration capture scheduling (monthly) and bundle distribution so ingestion and parity jobs stay in sync.
2. Tighten parity thresholds using fresh calibration runs and fail builds when symbol-level metrics exceed agreed deltas.
3. Document operator workflows for calibration refresh + parity verification in the runbooks and add CI hooks that surface calibration version drift.
