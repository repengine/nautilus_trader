# ML Dashboard Services – Planning Layer

## Overview

This directory is currently a planning hub for the future Nautilus ML dashboard
integration layer. The long-term goal is to bridge the dashboard UI with the ML
system, but the concrete Python code in this package is intentionally minimal and
stubbed. The authoritative source of integration details lives in the
`PLAN_*.md` documents that accompany this README.

The intended workflow is:

1. **Dashboard/UI contract** – buttons, forms, and widgets define the desired
   behaviour.
2. **PLAN documents** – capture how those UI promises should map onto ML stores,
   registries, actors, and orchestration.
3. **Implementation phase (future)** – concrete services will be created to honour
   the plans. Today, `integration_layer.py` exists only to sketch the public
   surfaces and support UI prototyping.

## Planned Architecture

```
┌─────────────────────┐
│   Dashboard UI      │  ← UI interactions
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Flask Routes      │  ← HTTP API endpoints
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Integration Layer   │  ← FUTURE services implementation
├─────────────────────┤
│ • StoreIntegration  │
│ • ActorIntegration  │
│ • PipelineIntegr.   │
│ • TradingIntegr.    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ML System         │
├─────────────────────┤
│ • 4 Stores          │
│ • 4 Registries      │
│ • ML Actors         │
│ • Orchestrator      │
│ • TradingNode       │
└─────────────────────┘
```

## Planned Services

The services described below now ship with typed implementations. Behaviour will
continue to evolve toward the full scope captured in the PLAN documents, so keep
the specifications handy when extending the modules.

### StoreIntegrationService
- **Plan:** feed KPI cards, ingestion stats, and portfolio metrics from the
  4-store architecture.
- **Current code:** aggregates portfolio KPIs, ingestion throughput, and store
  health snapshots using the integration manager and database fallbacks.
- **Reference:** `PLAN_metrics_monitoring.md`.

### ActorIntegrationService
- **Plan:** deploy/pause actors, perform hot reloads, and surface health status via
  `BaseMLInferenceActor`.
- **Current code:** provides typed lifecycle DTOs, registry validation, and
  pause/resume/stop orchestration with metrics instrumentation.
- **Reference:** `PLAN_actor_management.md`.

### PipelineIntegrationService
- **Plan:** trigger dataset builds, training jobs, and monitor pipeline progress
  through an orchestrator/queue.
- **Current code:** submits orchestrator runs, tracks job state with persistence
  hooks, and exposes cancellation + progress queries for the dashboard.
- **Reference:** `PLAN_pipeline_orchestration.md`.

### TradingIntegrationService
- **Plan:** bridge live trading toggles, emergency stop, and risk checks to the
  Nautilus `TradingNode`.
- **Current code:** typed service with safety checks, controller integration,
  and metrics. Additional tasks from `PLAN_trading_controls.md` (e.g. trading
  mode analytics) remain to be implemented.
- **Reference:** `PLAN_trading_controls.md`.

### SystemConnectorService
- **Plan:** orchestrate the dashboard "Connect System" flow, instantiate the
  `TradingNode`, and report component status with progressive fallbacks.
- **Current code:** instantiates a trading node from `TradingNodeConfig()`,
  connects data/execution engines with guarded timeouts, emits Prometheus
  metrics, and surfaces health snapshots for the dashboard.
- **Reference:** `PLAN_trading_controls.md` ("Connect System" section).

## Implementation Status Snapshot

| Service  | Planning Source                  | Code Status                            |
|----------|----------------------------------|----------------------------------------|
| Store    | `PLAN_metrics_monitoring.md`     | Aggregation implementation (cold path) |
| Actor    | `PLAN_actor_management.md`       | Lifecycle management implemented       |
| Pipeline | `PLAN_pipeline_orchestration.md` | Job submission + persistence in place  |
| Trading  | `PLAN_trading_controls.md`       | Toggle/emergency stop implemented      |
| System   | `PLAN_trading_controls.md`       | Connect/disconnect service implemented |

`IMPLEMENTATION_PROGRESS.md` tracks the to-do items extracted from the PLAN
files. Values remain advisory until the tracker is regenerated against the
implemented services.

## Current Usage

The exported classes in `integration_layer.py` are safe to import for UI mocks and
tests and now proxy real integration logic with guarded fallbacks. Extend them in
line with the PLAN documents when wiring new dashboard features.

## Design Principles (from plans)

The planning documents outline the principles the eventual implementation must
follow:

- **Progressive enhancement** – ship minimal functionality first, then layer on
  richer control and monitoring.
- **Graceful degradation** – prefer cached or safe fallbacks when ML dependencies
  are unavailable.
- **Contract-first development** – model UI expectations explicitly before wiring
  back-end behaviour.
- **Observability** – use `ml.common.metrics_bootstrap` for Prometheus metrics and
  keep instrumentation off the hot path.

See `GAP_ANALYSIS.md` for additional security, real-time, and performance
considerations that will guide the implementation.

## Next Steps

1. Finalise requirements in the PLAN documents as the UI evolves.
2. Decide sequencing (store metrics, actor control, etc.) and break work into
   scoped stories.
3. Implement real integrations, updating this README alongside the code.
4. Add contract and integration tests that validate behaviour against live
   components instead of stubbed responses.

Until then, treat this directory as documentation and planning scaffolding.
