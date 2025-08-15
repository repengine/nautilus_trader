# TFT Teacher: Data Curation & Implementation Plan

Status: draft

This plan specifies the data, features, and implementation work to train a high‑quality TFT “teacher” model and distill “students.” It covers data roles (static, known‑future, unknown), point‑in‑time curation, feature engineering, registry and CLI integration, testing/quality gates, and milestones.

---

## Objectives

- Build a robust TFT teacher trained on enriched inputs beyond bars (L1/L2/L3, calendars, metadata, event schedules).
- Ensure strict point‑in‑time (PIT) correctness with purge/embargo CV and leakage guards.
- Expand feature engineering with explicit “roles”: static covariates, time‑varying known‑future, time‑varying unknown.
- Register feature sets and teacher artifacts; distill to L1‑only students with parity checks.
- Maintain strict typing and ≥90% coverage for `ml/` per repo gates.

---

## Data Model & Roles

Required columns in training frames (batch CSVs or DataFrames):

- `time_index` (int or timestamp): monotonically increasing per `instrument_id`.
- `instrument_id` (str/int): grouping key.
- `y` (float/int): target label (e.g., binary up/down or real return over horizon).
- Static covariates (per instrument): e.g., `tick_size`, `lot_size`, `exchange`, `asset_class`, `contract_size`, `currency`.
- Known‑future reals/categoricals (deterministic for future horizons): calendars (ToD/DoW harmonics, holidays, auctions), contract schedule (days_to_expiry), event calendars (timestamps only), fees/funding/roll flags.
- Unknown reals (observed up to “now” only): bars/transforms, microstructure, order flow, trade prints, cross‑asset context, L2/L3 aggregates.

Principle: Anything not deterministically known at prediction time must be in unknown reals.

---

## Data Sources & PIT Curation

- Market data:
  - L1 bars: O/H/L/C/V (already supported by `ml.features.engineering`).
  - L2/L3: book depth snapshots, top‑of‑book spread, imbalance, queue turnover, cancel/submit rates, auction imbalance.
  - Trade prints: signed volume, aggressor ratio, trade intensity, VWAP deviation.
- Metadata (static): instrument spec (tick size, lot size, currency, contract size, venue, asset class), fee class.
- Calendars and schedules (known‑future): exchange sessions, holidays/early closes, auctions, maintenance, expiries/rolls, earnings/economic event timestamps.
- Context (optional/phase 2): cross‑asset leaders, futures basis, term‑structure slopes, option surface summaries, open interest, funding/borrow series.

Point‑in‑time joins (no leakage):

- As‑of joins keyed by (`instrument_id`, `time_index`).
- For known‑future, include only information public at or before `time_index` for the forecast horizon; never include values revealed after.
- Event schedules: include timestamps/flags only (exclude the content until publish time).
- Use embargo windows around impactful events for evaluation.

---

## Feature Engineering Extensions

Existing blocks (bars): returns, momentum, vol, RSI/BB/ATR/EMA/MACD, volume ratios.

New components to add (with strict typing and role annotations):

1) Known‑Future encodings (`ml/features/known_future.py`)
- Exchange calendar features: `tod_sin`, `tod_cos`, `dow_sin`, `dow_cos`, `is_holiday`, `is_early_close`, `is_open`, `minutes_to_close`.
- Contract schedule: `days_to_expiry`, `is_roll_window`, `roll_d_minus_k` flags.
- Event calendars: `has_event_15m`, `has_event_1h`, `has_earnings`, `is_auction_window`.
- API: pure functions operating on `time_index`, `instrument_id`, and schedule tables;
  deterministic for future timesteps.

2) Static Covariates (`ml/features/static_covariates.py`)
- Load and encode per‑instrument numeric and categorical attributes; optional one‑hot or embedding index mapping.
- Cache by `instrument_id`; expose `build_static_frame(instruments: list[str]) -> DataFrame`.

3) L2/L3 & Microstructure (`ml/features/l2l3.py`)
- Aggregations over sliding windows (encoder history):
  - Top‑of‑book: `spread`, `mid_price`, `depth_imbalance`, `queue_turnover`.
  - Book pressure: bid/ask volume sums for N levels; imbalance ratios; cancel/submit rates.
  - Auction imbalance (if available): normalized and clipped features.
- Trade flow: `signed_volume_k`, `trade_intensity`, `aggressor_ratio`, `vwap_dev`.
- Implementation should be incremental/streaming with batch parity (mirror `IndicatorManager` approach).

4) PIT utilities (`ml/preprocessing/joins.py`, `ml/preprocessing/leakage.py`)
- `asof_join(left, right, on=[time_index, instrument_id], right_ts, tolerance)` for robust joins.
- Embargo helpers integrated with `PurgedCrossValidator`.

5) Feature pipeline integration (`ml/features/pipeline.py`)
- Add `TransformSpec` names: `calendar`, `static_metadata`, `l2l3`, `trade_flow`.
- `PipelineRunner` computes role‑aware output names in stable order; update `FeatureEngineer.build_pipeline_spec_from_config`.

---

## Registry & Manifests

- Feature Registry (`ml/registry/feature_registry.py`):
  - Create two feature sets:
    - Teacher set (role `TEACHER`, `DataRequirements.L1_L2_L3`): enriched features (bars + l2/l3 + calendars + static).
    - Student set (role `STUDENT`, `DataRequirements.L1_ONLY`): L1‑only subset with calendar + static.
  - Record `pipeline_signature`, `schema_hash`, `capability_flags` (e.g., `{"microstructure": true, "calendar": true}`), and constraints (e.g., warmup bars, latency budget).

- Model Registry linkage (`ml/registry/base.py`):
  - Teacher manifest: `role=TEACHER`, `serveable=False`, feature schema = teacher feature set.
  - Student manifest: `role=STUDENT`, `parent_id=<teacher_id>`, feature schema = student feature set.

---

## Training: TFT Teacher

- `ml/training/teacher/tft_cli.py` (already scaffolds):
  - Ensure arguments accept `--static_categoricals`, `--static_reals`, `--known_future_reals` and validate columns exist.
  - Training slice: last 20% as validation; ensure each series has `max_encoder_length` history.
  - Optional: save interpretability artifacts if available (already guarded).

- `ml/training/teacher/tft_teacher.py`:
  - Keep defaults but validate that role‑split columns are correctly assigned; warn if any known‑future/unknown overlap.
  - Add input normalization/robust scaling toggles when helpful (leave off by default).

- Labels & horizons:
  - Define consistent targets in data prep: e.g., `y = sign(return_{horizon})` or regression returns.
  - Support multi‑horizon training through multiple targets in future phase; start with single horizon.

---

## Student Distillation

- Use `ml/training/student/lightgbm.py` with objectives `logit_mse` or `hybrid`.
- Distill L1‑only features + known‑future + static; bake Platt calibration into ONNX.
- Validate parity (ONNXRuntime vs framework) on a validation slice; record `feature_schema_hash` and student metadata.

---

## Quality Gates & Testing

Quality/Leakage
- `PurgedCrossValidator` usage in teacher evaluation and student validation.
- Event embargo windows around important schedules (econ releases, auctions).

Parity & Determinism
- Batch/online parity tests for new feature blocks within 1e‑10 tolerance where possible.
- Deterministic seeds in training CLIs.

Unit/Property Tests (new under `ml/tests/`)
- `tests/features/test_known_future.py`: no‑leakage properties; deterministic encodings; edge cases around market open/close/holidays.
- `tests/features/test_static_covariates.py`: correct typing/encoding, stable ordering, missing metadata handling.
- `tests/features/test_l2l3.py`: windowed aggregates, book imbalance, spread/impact features; PIT joins; NaN handling.
- `tests/preprocessing/test_joins.py`: `asof_join` correctness and tolerance windows.
- `tests/training/test_tft_teacher_integration.py`: tiny synthetic dataset, fit/predict path smoke test (skipping heavy deps if unavailable).
- `tests/training/test_student_distill.py`: fit student on synthetic teacher soft labels; ONNX export; ORT parity.

Coverage & Typing
- `make pytest` reaching ≥90% coverage for modified `ml/`.
- `uv run --active --no-sync mypy ml --strict` clean.

---

## Implementation Tasks (by module)

1) Known‑future features
- Add: `ml/features/known_future.py`
  - Public: `build_calendar_frame(df/index, calendars)`, `encode_cyclic_time(df)`.
  - Public: `build_contract_schedule_frame(instruments, time_index, schedule_table)`.
  - Public: `annotate_event_windows(time_index, event_table, horizons=[15m, 1h])`.

2) Static covariates
- Add: `ml/features/static_covariates.py`
  - Public: `load_static(instruments) -> DataFrame` (typed columns, stable ordering).
  - Encoding helpers for categoricals; imputation policy.

3) L2/L3 & trade flow
- Add: `ml/features/l2l3.py`
  - Streaming + batch parity API, similar to `IndicatorManager`.
  - Windowed aggregates; configurable levels and windows; numeric stability guards.

4) PIT utilities
- Add: `ml/preprocessing/joins.py` and `ml/preprocessing/leakage.py`
  - `asof_join`, embargo helpers, validation utilities.

5) Pipeline integration
- Update: `ml/features/pipeline.py`
  - Register new `TransformSpec` names and routing to new modules.
  - Extend `PipelineRunner.compute_feature_names()` to include new blocks in stable order.

6) Feature engineer
- Update: `ml/features/engineering.py`
  - Add config flags: `include_calendar`, `include_static`, `include_l2l3`, `include_trade_flow`.
  - Wire new transforms to batch and online paths; maintain exact parity.

7) Training/CLI
- Update: `ml/training/teacher/tft_cli.py`
  - Validate presence of declared role columns; produce clear error lists.
  - Optionally write a manifest snapshot next to outputs for reproducibility.

8) Registry
- Use `FeatureManifest` to register teacher and student feature sets, with `DataRequirements`.
- Student registration via existing `ml/training/student/lightgbm_cli.py` with manifest helpers.

9) Documentation & examples
- Add examples under `ml/examples/` for:
  - Building a curated CSV with roles.
  - Running `ml-teacher-tft` training.
  - Distilling a student and verifying ONNX.

---

## Milestones & Deliverables

Phase 0: Foundations (PIT + roles)
- [ ] `joins.py`, `leakage.py` utilities with tests
- [ ] Role validation in CLI (known vs unknown vs static)

Phase 1: Known‑future & static
- [ ] `known_future.py` with calendar + schedule encodings
- [ ] `static_covariates.py` with loaders/encoders
- [ ] Pipeline integration and tests

Phase 2: L2/L3 + trade flow
- [ ] `l2l3.py` windowed aggregates (batch + streaming parity)
- [ ] Microstructure features (spread/imbalance/impact) with tests

Phase 3: Teacher training & registry
- [ ] End‑to‑end TFT training on synthetic sample (or guarded real deps)
- [ ] Feature Registry entries (teacher/student) with capability flags

Phase 4: Student distillation & export
- [ ] Student fit (L1‑only), ONNX export with calibration
- [ ] ORT parity + metadata checks

Phase 5: Quality gates & docs
- [ ] Purged/embargoed CV in evaluation
- [ ] Leakage audits around event windows
- [ ] Examples + README updates

---

## Risks & Mitigations

- Heavy dependencies availability (PyTorch Forecasting/Lightning): guard with `_imports` flags and fallbacks (already in CLI).
- Data sparsity for L2/L3: robust NA handling and windowed defaults; feature toggles per venue.
- Leakage via schedules: strict PIT utilities, event embargo tests.
- Performance/latency: streaming implementations with preallocation; keep student L1‑only.

---

## Acceptance Criteria

- Feature sets registered with clear roles and `DataRequirements`.
- TFT trains and produces calibrated logits on curated validation slice.
- Student distilled and exported to ONNX with schema hash; passes ORT parity test.
- `make ruff` clean; `mypy --strict` clean in `ml/`; tests ≥90% coverage for new/changed code.

---

## Next Actions

1) Implement PIT utilities and tests.
2) Implement known‑future + static modules; integrate pipeline and tests.
3) Implement L2/L3 aggregates; add microstructure/trade flow features and tests.
4) Wire CLI validation for roles; run a synthetic end‑to‑end TFT training.
5) Distill a student on L1‑only features; export and register.

