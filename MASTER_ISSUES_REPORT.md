# Master Issues Report – Updated Assessment (2025-03-31)

**Codebase:** Nautilus Trader ML  
**Context:** Active mid-refactor; multiple legacy facades and opt-in components  
**Scope of Review:** Targeted validation of previously reported blockers, high-risk items, and infrastructure gaps

---

## Executive Summary

The system remains functional but is carrying mid-refactor complexity. Several high-impact defects are confirmed (schema mismatch, configuration hard-coding, missing dependency guards), while some previously reported blockers were not reproducible. Tests and observability continue to rely on optional dependencies and feature flags, which leaves blind spots in CI. The dependency graph also needs clean-up (duplicate Lightning stack, stdlib asyncio pinned, missing Redis declaration).

**Overall health judgment:** Moderately risky until the confirmed issues below are addressed, followed by renewed attention to test fidelity and dependency hygiene.

---

## Confirmed High-Severity Issues

| # | Status | Area | File | Finding | Impact |
|---|--------|------|------|---------|--------|
| 1 | ✅ Fix applied (awaiting compiled-ext test run) | Schema/Migrations | `ml/stores/migrations/001_bootstrap_schema.sql:327-554` | `ml_data_watermarks` helpers were writing `ts_max/count/last_seq`. SQL functions now upsert `last_success_ns/last_attempt_ns/last_count/completeness_pct`; rerun targeted DB tests once Nautilus C extensions are available. | Deployment blocker for new environments (guarded by targeted migrations test). |
| 2 | ✅ Fix merged (`ml/common/event_emitter.py`, unit coverage) | Events/Observability | `ml/common/event_emitter.py:68-188` | Metadata fallback now emits a structured warning (`exc_info=True`) and increments `ml_dataset_event_metadata_fallback_total`; unit tests assert logging and metrics behavior. | High; production telemetry now surfaces metadata drops. |
| 3 | ✅ Fix merged (`ml/config/actors.py`, docs & tests) | Configuration | `ml/config/actors.py:120-170` | `MLSignalActorConfig` now resolves the default connection via `collect_postgres_candidates` (honouring `ML_DB_CONNECTION`, `DATABASE_URL`, etc.), falling back to Compose defaults only when needed. Unit docs and tests updated to cover the env-driven path. | High security and deployment risk mitigated; defaults now respect deployment env. |
| 4 | ⏳ Planned | Dependencies | `ml/common/message_bus.py:56`, `ml/consumers/redis_streams_consumer.py:27`, `ml/dashboard/service.py:967` | Redis is imported directly with no `HAS_REDIS` guard and no `redis` dependency in `pyproject.toml`. Fresh installs crash instead of falling back. | High; optional backend unusable in clean environments. |
| 5 | ⏳ Planned | Actors | `ml/actors/signal.py:1270-1271` & `ml/actors/signal.py:2447-2450` | `_prediction_history` and `_confidence_history` lists grow without bounds on non-optimized paths. | Medium/High; memory leak (~MB/week for active models). |

---

## Confirmed Medium-Severity Issues & Gaps

- **Testing blind spots**
  - 25 test modules skip entirely under default settings, mostly when optional dependencies (`hypothesis`, `pandera`, `Nautilus` core, facade feature flags) are absent. Some suites—for example the orchestrator tests guarded by `ML_ENABLE_COMPONENT_FACADES`—remain permanently disabled in CI.
  - Several smoke/integration tests assert nothing beyond “did not crash” (`ml/tests/test_smoke.py:19`, `ml/tests/integration/test_end_to_end_pipeline.py:751`, `ml/tests/integration/test_postgres_integration.py:94`, `ml/tests/unit/strategies/test_strategy_integration.py:211`), providing little coverage confidence.

- **Dependency hygiene**
  - `pyproject.toml` pins both `lightning==2.5.4` (core dependency) and `pytorch-lightning>=2.1,<3.0` (optional group), inviting conflicts.
  - The project dependency list redundantly includes the stdlib `asyncio`.

- **Dormant code**
  - `ml/config/defaults.py`, `ml/registry/_typing_utils.py`, `ml/core/bus_integration.py`, and `ml/config/version.py` have no runtime importers. Decide whether to wire them back into the refactored entry points or remove them to reduce noise.

---

## Previously Reported Issues Not Reproduced

| Claim | Original Severity | Updated Assessment |
|-------|-------------------|--------------------|
| **P99 alert threshold 100× too high** (`ml/deployment/alerts.yml:5-16`) | P0 | The alert intentionally fires at 200 ms; runbook documentation confirms the same target. No issue. |
| **`GPUMemoryMonitor.stop()` can hang** (`ml/common/gpu_monitor.py:94-103`) | P0 | The sampler is daemonized, and the run loop already times out probe calls; adding a join timeout is unnecessary. |
| **`LiveDataRecorder` race condition** (`ml/stores/writers.py:424-512`) | P0 | Flush defers the async work only after swapping out the buffer, so the unsynchronized mutation risk is low. |
| **Circular import between `promotions` and `stage2_engine`** | P1 | Runtime imports succeed; only `TYPE_CHECKING` references cross modules. |

These items remain on watchlists but are not blocking based on current evidence.

---

## Testing & Coverage Recommendations

1. Replace placeholder `assert True` statements with explicit behavioural assertions or outcome checks.
2. Rework module-level skips so CI executes the component orchestrator suite and property tests under controlled environments (e.g., enabling `ML_ENABLE_COMPONENT_FACADES`, ensuring `hypothesis` is available in the test image).
3. Extend coverage around event metadata handling once logging/metrics are added, to ensure the fallback path is observable.

---

## Dependency & Configuration Actions

1. ✅ Declare `redis>=5,<6` in `pyproject.toml` (core or optional group) and expose `HAS_REDIS` / `check_ml_dependencies("redis")` pattern in `ml/_imports.py`.
2. ✅ Drop the stdlib `asyncio` from runtime dependencies.
3. ✅ Align Lightning dependencies (keep either `lightning` or `pytorch-lightning`, not both).
4. Replace hard-coded DB URLs in actor configs with environment-driven defaults (`MessageBusConfig.from_env`, `ActorDatabaseConfig`, etc.).

---

## Cleanup Opportunities

- Decide on the fate of unused helper modules (`ml/config/defaults.py`, `ml/registry/_typing_utils.py`, `ml/core/bus_integration.py`, `ml/config/version.py`, `ml/stores/_strict_conformance_check.py`) to reduce maintenance surface.
- Document the intentional use of `ml/stores/schedule_partitions.py` (CLI script) and ensure it has appropriate test coverage or monitoring since it is designed for cron usage.

---

## Suggested Next Wave (1–2 sprints)

1. **Migration Health (verification stage)**
   - [ ] Run targeted DB tests once Nautilus C extensions are available (`pytest -k watermark`) to confirm the updated SQL helpers.
   - [ ] Add a SQLite-backed unit test (or fixture) that exercises `emit_data_event_ext`/`update_watermark` against an in-memory database if feasible.

2. **Observability**
   - [x] Add structured warning + counter for metadata fallback in `ml/common/event_emitter.py`.
   - [x] Cover the new path with a unit test asserting log emission and metric increment.

3. **Configuration & Security**
   - [x] Replace hard-coded Postgres URL in `ml/config/actors.py` with env-driven defaults and validation (consider `MessageBusConfig.from_env` patterns).
   - [x] Document migration path in `README` / relevant docs and add a regression test for default config creation.

4. **Dependencies**
   - [x] Introduce `HAS_REDIS` guard in `_imports`, update bus/consumer/dashboard modules to honor it, and add `redis>=5,<6` to `pyproject.toml`.
   - [x] Drop the stdlib `asyncio` pin and align the Lightning stack (choose `lightning` vs `pytorch-lightning` consistently).

5. **Actors / Memory**
   - [ ] Convert `_prediction_history` / `_confidence_history` to bounded deques and extend actor tests to assert history length caps.

6. **Testing Signal-to-Noise**
   - [ ] Upgrade smoke/integration tests (`ml/tests/test_smoke.py`, `ml/tests/integration/test_end_to_end_pipeline.py`, etc.) to assert meaningful outcomes.
   - [ ] Revisit module-level skips to ensure CI runs orchestrator/property suites under a known-good dependency set.

---

## Appendix: Files Referenced

- `ml/stores/migrations/001_bootstrap_schema.sql:327-487`
- `ml/common/event_emitter.py:68-128`
- `ml/config/actors.py:80-118`
- `ml/common/message_bus.py:40-120`
- `ml/consumers/redis_streams_consumer.py:1-78`
- `ml/dashboard/service.py:934-1013`
- `ml/actors/signal.py:1270-1283`, `ml/actors/signal.py:2397-2458`
- `ml/tests/test_smoke.py:1-70`
- `ml/tests/integration/test_end_to_end_pipeline.py:720-760`
- `ml/tests/integration/test_postgres_integration.py:80-120`
- `ml/tests/unit/strategies/test_strategy_integration.py:180-211`
- `ml/config/defaults.py:1-31`
- `ml/registry/_typing_utils.py:1-86`
- `ml/core/bus_integration.py:1-28`
- `ml/config/version.py:1-32`
