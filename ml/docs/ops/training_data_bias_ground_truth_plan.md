# Training Data Bias: Ground Truth and Action Plan

Date: 2026-02-08

## Scope
This document records a code-first ground truth for training-data bias risk and a staged remediation plan.

This version intentionally uses current refactored event-driven streaming paths only (no legacy path assumptions).

## Code-Grounded Ground Truth (Current Active Paths)
1. Planner-side limits are applied during planning, then guardrails run.
   - `ml/training/event_driven/dataset_service.py:51`
   - `ml/training/event_driven/dataset_service.py:89`
2. Global-run planning pre-splits metadata into train/validation slices and emits both slices in the plan.
   - `ml/training/event_driven/global_run.py:323`
   - `ml/training/event_driven/global_run.py:464`
   - `ml/training/event_driven/global_run.py:465`
3. Worker context preparation currently re-applies limits and may limit split metadata again.
   - `ml/training/event_driven/worker.py:980`
   - `ml/training/event_driven/worker.py:996`
   - `ml/training/event_driven/worker.py:1000`
   - `ml/training/event_driven/worker.py:1023`
   - `ml/training/event_driven/worker.py:1027`
4. Streaming limit enforcement is shard-level; shards that exceed remaining budget are dropped whole.
   - `ml/training/teacher/streaming_loader.py:885`
   - `ml/training/teacher/streaming_loader.py:887`
5. Dataset guardrails currently validate positive-rate bounds, schema drift, and known-future pairs, but not instrument/time coverage.
   - `ml/training/event_driven/guardrails/dataset.py:53`
   - `ml/training/event_driven/guardrails/dataset.py:59`
   - `ml/training/event_driven/guardrails/dataset.py:66`
6. Positive-rate guardrail uses `metadata.numeric_stats[target_col].mean`; limited metadata retains original numeric stats.
   - `ml/training/event_driven/guardrails/dataset.py:77`
   - `ml/training/teacher/streaming_loader.py:935`
7. Event-driven dataset plan contracts currently carry `target_col` but no explicit target semantics payload.
   - `ml/training/event_driven/services.py:59`
   - `ml/training/event_driven/services.py:101`
8. Strict target-semantics validators already exist and can be reused.
   - `ml/data/metadata.py:501`
   - `ml/data/metadata.py:636`
   - `ml/data/metadata.py:808`
9. Short-entry fallback remains implicit when strategy config omits explicit short policy.
   - `ml/strategies/base_facade.py:260`
   - `ml/strategies/base_facade.py:278`
   - `ml/strategies/ml_strategy.py:417`

## Risk Interpretation
Highest near-term risk is selection integrity, not only directional regime drift.

1. Sampling collapse risk: shard-level hard caps plus repeated limiting can distort instrument/time coverage.
2. Contract drift risk: `target_col` can diverge from declared target semantics because plan contracts do not currently carry semantics.
3. Runtime policy ambiguity: omitted short-entry policy can alter live behavior relative to intended model usage.

## Prioritized Plan
### P0: Integrity and Selection Risk (Concrete)
P0 is implementation-ready and should be executed first. Full implementation detail is in:

- `ml/docs/ops/training_data_bias_p0_implementation_spec.md`

### P0.1 Selection Path Determinism
- Use a single authoritative limiting stage per plan path.
- Preserve planner-provided pre-split train/validation shard assignments by default.
- Eliminate independent train/validation re-limiting that can change coverage composition.

### P0.2 Coverage Guardrails (Hard Fail)
- Add minimum distinct instrument coverage guardrail.
- Add maximum single-instrument share guardrail (sampling-collapse guard).
- Add minimum temporal span guardrail for selected shards.
- Fail planning with `DatasetGuardrailError` when configured thresholds are violated.

### P0.3 Target Contract Consistency
- Extend plan request/event contracts to carry explicit target semantics payload.
- Validate target semantics contract and target-column membership during plan guardrails.
- Fail fast on target mismatch before worker execution.

### P0.4 Runtime Policy Explicitness
- Require explicit short-entry policy in run/deploy configs for strategies that may short.
- Keep fallback behavior for backward compatibility, but make policy intent explicit in rollout docs/config templates.

### P1: Target and Evaluation Upgrades
1. Extend target basis options (market-relative and volatility-normalized) behind config flags.
2. Add regime-conditioned evaluation slices in worker outputs.
3. Keep strict backward compatibility with explicit legacy-compatible target config.

### P2: Robustness Enhancements
1. Add regime-balanced sampling experiments.
2. Add directional symmetry augmentation experiments where feature semantics allow it.
3. Evaluate regime-conditioned heads/routing only after P0 and P1 stabilize.

## Execution Sequence
1. Implement P0 contract plumbing and guardrail metrics behind non-breaking defaults.
2. Implement P0 hard-fail guardrails and single-pass limit semantics.
3. Validate P0 via targeted tests, telemetry checks, and required lint/type gates.
4. Roll out P1 changes behind config flags.
5. Run P2 experiments against stable P1 baseline.

## Acceptance Criteria
1. No plan proceeds when configured instrument/time/target-contract guardrails fail.
2. Worker telemetry exposes selected-vs-total instrument rows/sequences for train and validation.
3. Planner/worker no longer re-shape pre-split shard assignments by default.
4. Event-driven plans include explicit target contract payload when enabled.
5. Short-entry policy is explicit in deployment configs for short-capable strategies.

## Notes
P0 is safety-critical and should complete before objective/evaluation extensions. P1 and P2 remain valuable, but should not start until P0 acceptance criteria are satisfied.
