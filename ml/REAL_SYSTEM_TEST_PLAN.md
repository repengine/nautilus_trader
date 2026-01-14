# Real-System Testing Plan (Algorithmic Trading)

## Purpose
Build test coverage that matches the real system path: ingest data, generate features,
build datasets, train models, backtest, forward-test (no lookahead), verify/promote models,
serve inference, apply portfolio/strategy logic, and emit broker-ready trade messages.

This plan pushes beyond wiring checks by validating numeric stability, data quality,
and end-to-end behavior using real market data where possible.

## Guiding Principles

- Prefer real data slices over mocks for integration/E2E paths.
- Fail fast on NaN/inf, schema drift, or lookahead leakage.
- Keep tests deterministic, bounded, and resource-conscious.
- Use existing fixtures and shared utilities; avoid one-off test helpers.
- Preserve typed, strict behavior (mypy --strict) and ruff cleanliness.

## Data Sources

- Primary offline source: `data/catalog` (symlink to backup Parquet catalog).
- Use small, deterministic windows and a few instruments (e.g., AAPL, SPY).
- Avoid writing back into the catalog; tests should be read-only.

## Coverage Map (Pipeline Stages)

1. Ingestion (catalog -> store)
2. Feature generation (store/catalog -> feature outputs)
3. Dataset build (features -> training datasets)
4. Training (dataset -> model + metrics)
5. Backtest (no lookahead, deterministic results)
6. Forward-test (walk-forward, time-sliced)
7. Model verification/promotion (quality gates)
8. Inference serving (model -> predictions, finite outputs)
9. Strategy/portfolio (predictions -> positions)
10. Broker message emission (order intents/acks)

## Current Gaps (Observed)

- Real-data numeric stability checks are thin (NaN/inf can slip).
- Training/inference smoke tests are mostly mocked.
- No explicit walk-forward/no-lookahead enforcement on real timestamps.
- Promotion gates don’t hard-fail on NaN metrics in full E2E paths.
- Strategy/broker boundary tests focus on wiring, not order validity.

## Current Tasks (Short-Term)

- [x] Add real-catalog dataset smoke:
  - Build dataset from `data/catalog` with a small time window.
  - Assert: no NaN/inf, monotonic timestamps, required columns present.
- [x] Add train->eval->infer smoke for one backend (start with XGBoost or LightGBM):
  - Train on real-data slice, compute metrics, export artifact, infer on holdout.
  - Assert: metrics finite, predictions finite, confidence bounds respected.
- [x] Add walk-forward backtest/no-lookahead test:
  - Train on window A, predict on window B, enforce time ordering.
  - Assert: no lookahead, watermark monotonicity, no future-feature use.
- [x] Add promotion gate test for NaN metrics:
  - Ensure model verification rejects NaN/inf metrics or missing artifacts.
- [x] Add inference-serving smoke with real inputs:
  - Load exported artifact, run inference, assert finite outputs and latency bounds.
- [x] Add broker message contract test:
  - Validate order intent schema, idempotency, and retry behavior.

## Ongoing Investigation (Expandable)
Use this section to add findings and follow-up tasks as issues surface.

- [ ] Identify model-specific NaN sources (feature scaling, outliers, empty windows).
- [x] Add data-quality property tests for real catalog edge cases (gaps, spikes).
- [x] Expand E2E to multi-instrument portfolios and cross-asset features.
- [ ] Add streaming ingestion E2E (when Databento subscription is active).

### Recent Findings

- Real-catalog smoke + walk-forward integration tests now exercise bounded windows off `data/catalog`.
- Promotion gating now explicitly fails on NaN/inf metrics via stage2 gate validation.
- Inference-serving smoke now exercises ONNX export + runtime against real catalog features.
- Broker intent contracts now cover client-order idempotency and retry plans.
- Real-catalog validation now asserts forward-return alignment, timestamp monotonicity, and gap/spike handling.
- Multi-instrument dataset integration now validates alignment and cross-asset correlation output.
- Scheduler targeted collection now supports injectable DBN loaders for deterministic, tz-aware tests.
- Strategy decision persistence now syncs store/publisher dependencies and breaker failures to emit PARTIAL events.

## Notes

- Tests should reuse `ml/tests/fixtures` where possible.
- Keep heavy tests marked and bounded; use narrow time windows and few symbols.
- When adding new metrics, use `ml.common.metrics_bootstrap`.
