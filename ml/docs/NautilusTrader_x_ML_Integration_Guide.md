# NautilusTrader × ML Integration Guide
**Version:** 1.0
**Audience:** Engineering (trading platform + ML)
**Purpose:** Show exactly how `nautilus_trader` runs and how `ml/` should run alongside it so the whole system works together, with clear implementation details and source links.

---

## TL;DR
We run a **Teacher–Student + Context Gating** pipeline on top of NautilusTrader’s **actor + message-bus** runtime:

- **Teachers (nightly, offline):** train on T+1 (incl. L2/L3 when available) to learn microstructure outcomes (next‑tick up, spread widen, fill prob, expected slippage).
- **Students (live, L1-only):** distilled from teachers; score real-time L1 features with minimal latency; export compact scores + confidence.
- **Context gating (low‑frequency):** macro/liquidity regime actor publishes a regime vector; execution scales signals by regime.
- **Integration:** ML actors/strategies publish **signals** (via message bus) or submit orders; execution strategies subscribe and act.

---

## How NautilusTrader runs (what we plug into)
NautilusTrader is an **event-driven engine** with **Actors** communicating over a **MessageBus**:

- **MessageBus:** core pub/sub, point‑to‑point, and request/response messaging; message types: **Data, Events, Commands**. See docs: Message Bus (signals and publish/subscribe) [[docs]](https://nautilustrader.io/docs/nightly/concepts/message_bus).
- **Actors & Strategies:** `Strategy` **inherits from** `Actor`; strategies get all actor capabilities **plus** order management (submit/modify/cancel), and they’re invoked by the engine via lifecycle + data handlers (`on_start`, `on_stop`, `on_*tick`, `on_bar`, etc.). [[Strategies concept]](https://nautilustrader.io/docs/latest/concepts/strategies/) · [[Strategy API]](https://nautilustrader.io/docs/latest/api_reference/trading/)
- **Signals:** lightweight bus notifications for values like `"BullishScore"` or `"RiskMultiplier"` **(I don't love these)** via `publish_signal(...)` / `subscribe_signal(...)` (see “Signals” on the Message Bus page). [[docs]](https://nautilustrader.io/docs/nightly/concepts/message_bus/)
- **Backtest ↔ Live parity:** same strategy code runs in both modes; event ordering is deterministic in backtests. [[Docs home]](https://nautilustrader.io/docs/nightly/) · [[Repo]](https://github.com/nautechsystems/nautilus_trader) · [[Releases (messaging v3)]](https://github.com/nautechsystems/nautilus_trader/releases)

**Implication for ML:** put fast inference **in‑process** (as an Actor/Strategy) for minimal latency; use the **bus** to decouple prediction from execution and to inject macro/regime state.

---

## What exists in `ml/` today (as described)
- **Data loading & prep:** historical readers; alignment of features/labels.
- **Feature generation:** microstructure features (L1 rollups), technical indicators; room to add macro/cross‑asset feeds.
- **Training modules:** LightGBM/XGBoost (plus others), basic trackers.
- **Model tracking:** MLflow/metrics hooks.
- **Base ML strategy class:** scaffolding to run models inside a Strategy.
- **Signal actor:** publishes model outputs on the MessageBus for other actors to consume.

(Teacher–Student separation not yet implemented.)

---

## How `ml/` should run alongside NautilusTrader
Below is the **operational path** that keeps the two systems in lockstep.

### 1) Offline (nightly) training jobs
**Goal:** Train teachers on rich T+1 data, distill to students that only need L1 features.

- **Datasets:** build daily parquet partitions per instrument (`/symbol=XYZ/dt=YYYY‑MM‑DD/…`). Align L2/L3 snapshots with L1 trades/quotes for labels.
- **Features:** reuse the same transformation functions for both training and live (avoid train‑serve skew).
- **Teachers:** fit models for: `p(next_tick_up)`, `p(spread_widen)`, `E(slippage|size)`, `p(fill_within_t)`.
- **Students:** train on **L1‑derivable** features to emulate teacher targets (knowledge distillation).
- **Artifacts:** export students as **ONNX** (or **Treelite** for tree models) + normalization metadata + calibration (isotonic/Platt).
- **Tracking:** log runs to MLflow (params, metrics, artifacts).
- **Regime actor inputs:** train/update a simple macro/liquidity regime classifier (hourly/daily cadence).

> FreqAI shows the “data kitchen → models → artifacts” pattern, sliding‑window retraining, and feature expansion conventions. See: Feature engineering [[FreqAI]](https://www.freqtrade.io/en/stable/freqai-feature-engineering/), Configuration [[FreqAI]](https://www.freqtrade.io/en/stable/freqai-configuration/), Running [[FreqAI]](https://www.freqtrade.io/en/stable/freqai-running/), Developer guide (file structure) [[FreqAI]](https://www.freqtrade.io/en/stable/freqai-developers/).

### 2) Startup (engine boot)
- **MLStrategy actor**: loads latest `student.onnx` (+ scaler/calibration), subscribes to required L1 feeds; initializes rolling feature buffers.
- **Context/Regime actor**: loads the latest regime model/state; schedules minute‑updates; publishes `RiskMultiplier` (0–1).
- **Execution strategy**: subscribes to ML signals and regime signals; enforces risk caps and places orders.

### 3) Live scoring (event loop)
- **Handlers:** `on_trade_tick` / `on_quote_tick` update rolling features (e.g., OFI, imbalance, short‑window vol, microprice vs mid).
- **Score:** call ONNXRuntime (or Treelite) for **Student‑Direction**, **Student‑Impact**, **Student‑Fill**. Keep end‑to‑end per‑tick work under a few milliseconds.
- **Publish:** `publish_signal("BullishScore", p_up)` and friends; or call order APIs directly if using “single‑actor” design.
- **Execute:** execution strategy multiplies by `RiskMultiplier`, checks `do_predict`/confidence, and decides **trade? size? urgency?**
- **Telemetry:** push Prom metrics: inference latency, error rates, Brier/log‑loss, drift statistics; record realized slippage vs predicted.

### 4) Backtesting (same strategies)
- Use NautilusTrader’s backtest engine to **replay exact handlers** and bus signals; verify calibration and decision metrics with **purged/walk‑forward** schedules.

---

## Implementation details (code-level checklist)

### A) Module layout under `nautilus_trader/ml`
```
ml/
  datasets/            # readers/writers; parquet layout helpers
  features/
    l1.py              # fast L1 rollups: OFI, imbalance, microprice-mid, short-vol, spread-state
    macro.py           # yields, credit, FX baskets; minute cache
    nowcasts.py        # teacher-trained proxies: widening prob, slippage E[·]
  labelers/            # next_tick_up, fill prob, widen-in-t, etc.
  students/            # lightgbm.py, xgboost.py, onnx export, calibration
  teachers/            # richer L2/L3 teachers; distillation routines
  gating/              # regime class, minute scheduler, signal publisher
  outliers/            # DI / one-class SVM / DBSCAN gates (optional)
  train/               # nightly jobs (CLI)
  serve/               # optional Ray Serve endpoints (HTTP /score)
  eval/                # walk-forward, calibration, drift reports
  base_strategy.py     # MLStrategy base: load model, subscribe, roll features, predict, publish
  signal_actor.py      # thin actor that only publishes signals (if you decouple)
  registry.py          # model/feature registry; versioning
  config/              # YAML/JSON per instrument
  tests/
```
NOT 100% SOLD ON THIS (differs from current greatly) ^^

### B) Base ML strategy (`base_strategy.py`)
- **Inherits** `Strategy` (so it has all actor + order capabilities).
- `on_start`: load ONNXRuntime session; build feature buffers; **subscribe** to required data; cache pointers.
- `on_trade_tick`/`on_quote_tick`: update features **incrementally** (deque / ring buffers); **avoid Python recompute** per tick.
- `_score()`: assemble current feature vector → ONNXRuntime → post‑process (calibration, clipping) → return dict.
- `_publish_or_execute()`: either `publish_signal` or invoke order APIs if using single‑actor mode.
- **Configurable** thresholds, cool‑downs, and feature windows via `StrategyConfig`.

**Docs:** Strategy concept [[link]](https://nautilustrader.io/docs/latest/concepts/strategies/), Strategy API [[link]](https://nautilustrader.io/docs/latest/api_reference/trading/).

### C) Signals over the bus
- Use `publish_signal("BullishScore", p)` and `subscribe_signal("BullishScore", handler)` to decouple prediction from execution.
- Consider separate topics for **Direction**, **Impact**, **FillProb**, **RiskMultiplier**.
- Signals are **Data** messages intended for lightweight notifications; multiple subscribers receive them. [[Message Bus / Signals]](https://nautilustrader.io/docs/nightly/concepts/message_bus)

### D) Inference speed
- Prefer **ONNXRuntime** (trees and small nets) or **Treelite** (compiled trees). Batch if you run multiple models per tick.
- Keep Python path thin: incremental buffers, NumPy arrays, avoid Pandas in hot loops.
- If a net is heavy, **offload** via thread/process or **Ray Serve** (`/score`), but measure network overhead.

### E) Nightly training jobs
- Sliding windows; embargo + purged CV; CPCV for leakage‑sensitive work.
- Export: `student.onnx`, `scaler.json`, `calibration.pkl`, `meta.json` (feature schema, versions).
- MLflow run with tags: model‑id, instrument, horizon, data‑cut.

### F) Context / regime actor
- Minute scheduler; compute regime from macro/credit/liquidity feeds; publish `RiskMultiplier ∈ [0,1]`.
- Allow strategy override (e.g., lock at 0 during maintenance).

### G) Telemetry & health
- Prom metrics: inference latency, QPS, error rate, Brier/log‑loss, drift (PSI/Wasserstein).
- Data health: L1 staleness, spread explosions; raise a **signal** (e.g., `"DataDegraded"`) to pause execution.

---

## Example (primitive) end-to-end flow (live)
1. **Engine start:** add `MLStrategy`, `ContextActor`, and `ExecutionStrategy` to engine → `engine.run()`.
2. **MLStrategy.on_start:** load `student.onnx`, subscribe to ticks/quotes, init buffers.
3. **Ticks arrive:** `on_trade_tick` updates OFI/imbalance/vol; `_score()` predicts; `publish_signal("BullishScore", 0.82)`.
4. **ContextActor:** publishes `RiskMultiplier=0.6`.
5. **ExecutionStrategy:** receives both signals → 0.82 × 0.6 = 0.492 < 0.5 threshold → **skip** (risk‑off).
6. Later, `RiskMultiplier=1.0` → same 0.82 crosses threshold → **place order** via order APIs.
7. Metrics recorded; fills/slippage fed back into nightly training.

---

## Backtesting
Use the same strategies in a NautilusTrader backtest. Because event ordering and bus delivery are deterministic in backtests, you can **validate calibration, hit‑rate, slippage vs predicted impact**, and policy behavior without changing code. (That parity is a design goal of NautilusTrader.)

---

## References (source links)
- NautilusTrader **Message Bus** (signals, pub/sub, messaging styles): https://nautilustrader.io/docs/nightly/concepts/message_bus
- NautilusTrader **Strategies** (Strategy inherits Actor; lifecycle): https://nautilustrader.io/docs/latest/concepts/strategies/
- NautilusTrader **Strategy API** (order management, config): https://nautilustrader.io/docs/latest/api_reference/trading/
- NautilusTrader **Docs home**: https://nautilustrader.io/docs/nightly/
- NautilusTrader **GitHub (releases, messaging v3)**: https://github.com/nautechsystems/nautilus_trader/releases

- FreqAI **Feature engineering**: https://www.freqtrade.io/en/stable/freqai-feature-engineering/
- FreqAI **Configuration**: https://www.freqtrade.io/en/stable/freqai-configuration/
- FreqAI **Running**: https://www.freqtrade.io/en/stable/freqai-running/
- FreqAI **Developer guide (data handling, file structure)**: https://www.freqtrade.io/en/stable/freqai-developers/

- Internal blueprint: `ML_GENERAL.md` (project root – internal)

---

## Notes
- Keep **train ↔ serve parity** on feature code to avoid skew.
- Favor **decoupled actors** (prediction vs execution vs regime) using the bus.
- Measure latency and correctness first; widen scope (multi‑instrument, more models) only after stable calibration & policy controls.
