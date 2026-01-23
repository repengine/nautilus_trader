Triage - Coverage Flake Follow-up
================================

Context
-------
- Coverage run surfaced parallel-only failures and a teardown warning from `LiveDataRecorder`.
- These are rooted in module reload side effects, patching targets that can be stale, and
  a background task started during deployment integration tests.

Root Causes
-----------
- Patching by string path can resolve stale module objects after `sys.modules` cleanup
  (e.g., `ml.tests.test_no_circular_imports` clears `ml.*` submodules).
- `ml.data.__getattr__` does not expose `fred_join`, so string-path monkeypatching can
  fail when the package is partially loaded.
- Log capture depends on logger levels; the session fixture sets `ml` logger to INFO,
  so debug logs may not be captured if the logger is not targeted.
- `LiveDataRecorder.start()` schedules an async flush task that is not stopped in
  deployment integration tests, leaving a pending task at shutdown.

Implementation Plan
-------------------
- Patch module objects directly via `importlib.import_module(...)` in affected tests to
  avoid stale module attributes.
- Patch `fred_join` by importing the module and patching its symbol (avoid `__getattr__`).
- Scope `caplog` to the specific logger for debug-level assertions.
- Disable live recording in deployment integration fixtures to prevent background tasks.

Status
------
- Implemented: patched tests to use module objects directly, scoped logging capture,
  and disabled live recording in deployment integration fixtures.
- Pending: full suite/coverage rerun once baseline lint/mypy issues are addressed.
