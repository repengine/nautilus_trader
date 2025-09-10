# ML Strategy Arbiter + Trust Layer Plan

This document records the concrete plan to integrate a trust-aware arbiter layer
and (later) a lightweight execution bandit/RL into the ML strategy stack, aligned
with the current architecture (actors → strategies → stores/registry/monitoring).

## Goals

- Separate prediction from decision: actor produces calibrated predictive targets; strategy decides.
- Add an online, trust-aware gate and smooth sizing in the strategy (arbiter layer).
- Enable A/B evaluation (heuristic vs. bandit/RL) with propensity logging for unbiased analysis.
- Preserve hot-path constraints: actor remains minimal, strategy avoids heavy compute.

## Context Fit (from docs)

- Actors: MLSignalActor (ONNX, hot-path, store triad) stays focused on inference + signal emission.
- Strategies: MLTradingStrategy handles decisions and persistence to StrategyStore; extend here.
- Training: teacher (L2/L3, 30d) → student (L1, ONNX) nightly distillation; student serves hot path.
- Registry: model/feature manifests with schema hashing and lineage; canary/A/B supported.
- Stores: JSONB payloads; StrategyStore suitable for logging arm/propensity and decision context.

## Interfaces and Data Contracts

### MLSignal.metadata (actor → strategy)

Add these keys to `MLSignal.metadata` (all optional; actor remains compatible without them):

- `mu`: float, expected short-horizon mark-out (post-costs if available)
- `sigma`: float, uncertainty of mark-out (ensemble SD or residual proxy)
- `p`: float, probability-of-profit at the specified horizon
- `fill_curve`: dict[str,int|float] or dict[str,float], offset→fill probability (e.g., "0t":0.7,"1t":0.5)
- `horizon_ms`: int, prediction horizon in milliseconds
- `latency_ms`: float, measured end-to-end latency estimate for fills
- `spread`: float, current spread (ticks or price units)
- `volatility`: float, short-horizon volatility proxy
- `teacher_id`/`student_id`: lineage/provenance
- `feature_set_id`/`feature_schema_hash`: for parity/traceability

Note: Actor continues persisting features/predictions in FeatureStore/ModelStore; metadata augments
downstream decisioning and logging without changing model inputs.

### StrategyStore.execution_params (decision persistence)

Extend execution params for decisions with:

- `arm`: "A" (heuristic) or "B" (bandit/RL)
- `propensity`: float in [0,1] logged at decision time
- `action`: e.g., "market", "limit@0t", "limit@1t"
- `offset_ticks`: int | null, limit offset in ticks (if applicable)
- `size`: float, chosen order size
- `LCB`: float, lower confidence bound used for gating
- `m_t`: float, trust margin applied at time t
- `u0`: float, minimal edge baseline for sizing curve

## Arbiter Components (inside strategy)

Implement as lightweight classes, updated online and refreshed daily:

- Calibrator: per instrument/horizon/regime probability/score calibration.
  - Methods: `fit/update/apply`, supports Platt or isotonic, purged/embargoed folds.
  - Tracks ECE/Brier and stores params (registry or StrategyStore).

- TrustScheduler: computes dynamic gate margin `m_t` using evidence size.
  - `m_t = m_max * exp(-kappa * n_eff) + m_min` (start skeptical, relax over time).
  - `n_eff`: EWMA-based effective sample size.

- Sizer: smooth mapping from edge to size (logistic S-curve).
  - `q = cap * sigmoid(k * (LCB - u0))`, inventory/risk-capped.

- ABRouter: Bernoulli(p) routing for A/B with logged propensities.
  - Configurable exploration rate; per opportunity randomization, not per day.

- Policies:
  - HeuristicPolicy (now): chooses order type/offset by maximizing `mu * fill_prob - fees - slippage_guard`.
  - BanditPolicy (stub): placeholder for contextual bandit/off-policy-updated executor.

## Decision Flow (strategy on signal)

1) Extract state from `MLSignal` and `metadata` (mu, sigma, p, fill_curve, horizon, spread, vol, inventory, latency).
2) Apply calibration to `p`/`mu` as needed.
3) Compute `LCB = mu_hat - z_alpha * sigma_hat / sqrt(max(1, n_eff))`.
4) Gate: trade only if `LCB > m_t` (trust margin from TrustScheduler).
5) If gated-in: route via ABRouter (propensity-logged), call selected policy for action + size.
6) Persist decision to StrategyStore with full `execution_params` and `risk_metrics`.
7) If gated-out: persist HOLD with `LCB`, `m_t`, and calibration diagnostics.

## Daily Loop (offline, fast)

- Update calibrators with purged/embargoed folds; log ECE/Brier.
- Recompute TrustScheduler params and update `n_eff` with forgetting factor.
- Off-policy evaluation for Arm B vs Arm A:
  - IPS/self-normalized IPS; doubly robust using the student’s Q^(s,a) (mark-out model).
  - Report realized spread capture and net PnL per decision with clustered/Newey–West SEs.
- Adjust `m_t`/`u0` conservatively based on gates and evaluation metrics.

## Actor Changes (minimal)

- Keep strategies (threshold/adaptive/etc.) only for signal existence gate in actor.
- Populate `MLSignal.metadata` with the predictive targets and context fields above.
- Continue persisting to FeatureStore/ModelStore; no execution policy inside actor.

## Teacher → Student Targets

- Teacher (L2/L3, 30d): predict microprice drift (mu), depletion probabilities, limit fill curves,
  and mark-outs (50–500 ms / 1–5 s) with strong augmentation.
- Student (L1/L0): distill teacher targets to L1-accessible features (+optional representation `z_t`).
- Student outputs in live: `mu`, `sigma`, `p`, `fill_curve`, `markout_k`.

## Guardrails

- Purged + embargoed CV for all supervised fits/calibration.
- Event-time evaluation and wall-clock evaluation.
- Realistic fees, latency, partial fills, queue priority.
- Non-stationarity: EWMA forgetting; CUSUM on residuals to raise `m_t` and shrink sizes on regime breaks.

## Metrics & Persistence

- Strategy metrics: calibration_ECE, markout_RMSE, accept/filtered ratio, realized spread capture,
  net PnL/decision, inventory variance.
- Persist decisions (with arms/propensities), thresholds (`LCB`, `m_t`, `u0`), action, offset, size.

## Config Additions (sketch)

```yaml
arbiter:
  enabled: true
  z_alpha: 1.645              # 95% one-sided
  m_max: 0.10                 # initial trust margin
  m_min: 0.00                 # floor
  kappa: 0.01                 # decay rate for m_t
  u0: 0.00                    # minimal edge for sizing
  cap: 1.0                    # max size scale
  sizer_k: 5.0                # steepness of S-curve
ab_test:
  enabled: true
  arm_b_prob: 0.5             # propensity for Arm B
  log_all_decisions: true
policy:
  heuristic:
    fee_rate: 0.0
    slippage_guard: 0.0
  bandit:
    enabled: false            # stub initially
```

## Near-Term Tasks (while universe loads)

- [ ] Add Arbiter skeleton to strategy: Calibrator, TrustScheduler, Sizer, ABRouter, HeuristicPolicy.
- [ ] Extend `MLTradingStrategy` to read `MLSignal.metadata` and compute `LCB` + gate.
- [ ] Persist `arm`, `propensity`, `LCB`, `m_t`, `u0`, `action`, `offset_ticks`, `size` in StrategyStore.
- [ ] Add nightly evaluation stub to compute IPS/SNIPS/DR and summarize metrics.
- [ ] Add minimal calibrator store/registry serialization (parameters per instrument/horizon/regime).
- [ ] Feature flags to preserve current behavior if `arbiter.enabled=false`.

## Open Questions

- Where to store calibrator params long-term: StrategyStore JSONB vs. Registry manifest section?
- How to parameterize horizon(s) per instrument for heterogeneous universes?
- Which uncertainty proxy to use initially (ensemble SD vs. residual EWMA) for `sigma`?
