# ML Coding Standards

This document defines the coding standards for the `ml/` package. The goals are:

- Safety and clarity through explicit, precise typing
- Consistency with MyPy strict mode and Ruff linters
- Maintainability under parallel testing and production constraints

## Typing (Mandatory)

- Enable and pass `mypy` with `--strict` on `ml/` at all times.
  - Preferred (uses the project Poetry venv): `poetry run mypy ml --strict`
  - Alternative (uv): `uv run --active --no-sync mypy ml --strict`
  - Why Poetry? It guarantees mypy resolves third‑party imports (e.g., the `google` namespace from protobuf) against the project venv instead of system site‑packages. If you see `mypy: can't read file '/usr/lib/python3/dist-packages//google'`, switch to `poetry run mypy ml --strict` or pass `--python-executable "$(poetry env info -p)/bin/python"`.
- Functions, methods, and variables MUST be fully annotated.
  - No implicit `Any`. Use precise types from `typing`/`collections.abc`.
- Prefer Protocols over `Any` for duck-typed interfaces.
- Use `TypedDict`/`dataclass(slots=True)` for structured records.
- Avoid `type: ignore`. If unavoidable, add a comment with rationale.
- Prefer immutable data (tuples, frozensets) where appropriate.

### Store Protocols (Adoption Plan)

- New public ML components should type against the strict store protocols:
  - `FeatureStoreStrictProtocol`, `ModelStoreStrictProtocol`, `StrategyStoreStrictProtocol`.
- Existing components may continue to use the non‑strict variants for compatibility, but prefer adapters when adding new behavior.
- Avoid widening types to `Any` at these boundaries. Keep enums for events and narrow mappings for features/metrics.

### Service Protocols (Strict)

- Use strict service dependency protocols for services under `ml/stores/services/`.
  - Definitions live in `ml/stores/protocols.py`:
    - `ModelWriteDepsStrict`, `ModelReadDepsStrict`, `ModelEventDepsStrict`, `ModelClearDepsStrict`
    - `StrategyWriteDepsStrict`, `StrategyReadDepsStrict`, `StrategyEventDepsStrict`, `StrategyClearDepsStrict`
  - Also provided: `LoggerLike` and `TableLike` helper protocols to avoid `Any`.
- Migration rules (typing-only):
  - Update service `deps` annotations to the strict protocols; keep method bodies identical.
  - Narrow `values`, `params`, and `columns` to `list[dict[str, object]]`, `Mapping[str, object]`, and `Sequence[str]`.
  - Return types from `_execute_read(...)` should be `object` to avoid forcing heavy imports in cold paths.
- Registry typing:
  - Event services should expose `_get_data_registry() -> RegistryProtocol | None`.
- Performance: No runtime changes. These are type-level contracts only.

## Imports, Layout, and Style

- Use Ruff + Black formatting. Run `make ruff` locally.
- Organize imports (stdlib, third-party, local) and avoid circular imports.
- Keep module-level side effects minimal. Avoid heavy work at import time.
- Use `__all__` where a public API surface is intended. Keep it sorted.

## Logging (Structured)

- Prefer structured logging with stdlib interop using `structlog`.
- Configure logging once at process start (CLIs/entrypoints) via a central helper.
- Hot path:
  - Avoid logging inside tight loops; if unavoidable, keep at DEBUG and avoid allocations.
  - Never allow logging to affect control flow. Use non-blocking best‑effort patterns.
- Cold path:
  - Use key/value fields for context (component, operation, run_id, correlation_id, dataset_id, instrument_id, stage, source).
  - Use placeholders or key/value args; avoid f-string building when log level is disabled.
- Best‑effort pattern examples:

```python
try:
    do_optional_work()
except Exception:
    logger.debug("Optional work failed (ignored)", exc_info=True)

# Service variant (defensive)
except Exception:
    try:
        self.logger.debug("Non-blocking operation failed (ignored)", exc_info=True)
    except Exception:
        ...
```

## Module Structure Standards

All ML modules (especially actors, stores, features, monitoring) MUST follow a consistent structure for clarity, performance, and discoverability.

Required elements

- Module docstring: brief purpose, hot/cold path separation, performance targets (e.g., P99 < 5 ms), and integration points.
- Imports grouped by category: stdlib → third-party → ml → nautilus; use `TYPE_CHECKING` for type-only imports.
- Minimal import-time work: no I/O, no network, no DataFrame construction, no heavy init; only idempotent, light setup allowed.
- Module-level metrics initialization: use `ml.common.metrics_bootstrap` via `MetricsManager` (never import `prometheus_client` directly). Initialize collectors once at import-time or during cold-path initialization; only call `.inc()/.observe()/.set()` in hot loops.
- Explicit public API: define `__all__` and keep it alphabetically sorted; expose only intended symbols.
- Section headers: use clear separators in code to improve scanability (e.g., “Enums”, “Metrics”, “Implementation”).
 - Logging: acquire via `logging.getLogger(__name__)` or `structlog.get_logger(__name__)`; both route through the central config.

Minimal template (example)

```python
"""
<Module Name> — brief description.

Key features: ...
Performance targets: P99 < 5 ms; zero allocations in hot path
Hot/Cold path separation: hot = <x>, cold = <y>
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Final

# ===== Standard library =====
import time

# ===== Third-party =====
import numpy as np

# ===== ML imports =====
from ml.common.metrics_manager import MetricsManager

# ===== Nautilus imports =====
from nautilus_trader.model.data import Bar

if TYPE_CHECKING:
    pass

# ===== Constants =====
MAX_LATENCY_MS: Final = 5.0

# ===== Module metrics (idempotent) =====
_metrics_init = False
_ops_total = None
_latency_seconds = None


def _init_module_metrics() -> None:
    """Initialize module-level metrics once (idempotent)."""
    global _metrics_init, _ops_total, _latency_seconds
    if _metrics_init:
        return
    mm = MetricsManager.default()
    _ops_total = mm.counter(
        "ml_module_operations_total",
        "Total operations performed",
        ["operation"],
    )
    _latency_seconds = mm.histogram(
        "ml_module_latency_seconds",
        "Operation latency (seconds)",
        ["operation"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
    )
    _metrics_init = True


_init_module_metrics()  # Import-time, light and idempotent

# ===== Public API =====
__all__ = [
    "do_work",
]


def do_work(bar: Bar) -> float:
    start = time.perf_counter()
    # Hot path logic (no allocations, no I/O)
    # ...
    _latency_seconds.labels(operation="do_work").observe(time.perf_counter() - start)
    _ops_total.labels(operation="do_work").inc()
    return 0.0
```

Metrics rules

- Never import `prometheus_client` directly; use `MetricsManager` or `ml.common.metrics_bootstrap`.
- Keep creation/registration off the hot path; do light, idempotent initialization at import time or in actor/service initialization.
- Use clear names/units: counters end with `_total`; histograms suffix with `_seconds` for latency/duration; gauges include units (e.g., `_bytes`).

Validation checklist

- [ ] Docstring documents purpose, hot/cold separation, performance targets
- [ ] Imports grouped and `TYPE_CHECKING` used for type-only deps
- [ ] Module-level metrics initialized via `MetricsManager`/bootstrap (no prometheus imports)
- [ ] `__all__` defined and alphabetically sorted
- [ ] No cold-path work at import time (I/O, DataFrames, network, training)
- [ ] Hot-path functions avoid allocations and wrap publish/observability off-path

## Error Handling

- Catch only specific exceptions. Avoid bare `except:`.
- Log warnings with context and structured data where possible.
- Surface actionable error messages that help diagnose issues quickly.

## Database and Migrations

- Use `EngineManager.get_engine(...)` for all SQLAlchemy engines to prevent pool exhaustion.
- Never naïvely split SQL on semicolons; use a splitter that respects dollar-quoted bodies.
  - Tests should import and reuse the splitter used by the migration runner.
- Mark DDL-heavy tests `@pytest.mark.serial` and keep integration tests serial.

## Testing Standards

- Split tests: unit (fast, isolated), property, metamorphic, integration, performance.
- Use markers consistently:
  - `@pytest.mark.integration` for integration tests
  - `@pytest.mark.serial` for tests that must not run under xdist
- Default non-integration tests may run with `-n auto --dist=loadscope`.
- Property tests should use profiles (`ci`, `dev`, `debug`).
  - Default `HYPOTHESIS_PROFILE=ci` to keep runs deterministic and fast.

## Performance Benchmarks

- Benchmarks must be robust to environment variance. Support `ML_BENCH_RELAX`.
- Avoid microbenchmarks that are sensitive to noisy neighbors unless scoped behind relax guards.

## Validation Suite (Static Analysis & Compliance)

- Pre-commit (enforced on changed files):
  - Custom pattern checker: `.pre-commit-hooks/check_nautilus_patterns.py`
    - Hot path: no `open()`/network calls/pandas `DataFrame(...)`/`.fit(...)` in `on_*` handlers and actor paths
    - Security: no `pickle`/`joblib` in actors/strategies/deployment/inference
    - Events/Topics: enforce `EventStatus.<...>.value`; require `build_topic_for_stage(...)` in stores/actors
    - Metrics: no direct `prometheus_client` imports; use `ml.common.metrics_bootstrap`
    - Architecture: forbid direct store instantiation in actors; warn on god-class sizes
  - Semgrep rules: `tools/semgrep/ml-rules.yml` (mirrors the above; fast and CI-friendly)

- Manual/advisory (run locally or in CI):
  - Duplication hotspots: `python tools/duplication/check_duplication.py`
  - Import Linter contracts: `lint-imports` (see `importlinter.ini`)
  - Complexity/maintainability: `xenon --max-absolute B --max-modules B --max-average B ml/`
  - Security: `bandit -q -r ml -x ml/tests`
  - Dead code: `vulture ml --min-confidence 90 --exclude ml/tests/*`
  - SQL lint: `sqlfluff lint schema ml/stores/migrations`

- One-shot suite: `make validate-nautilus-patterns` (advisory)

Policy

- Keep pre-commit hooks clean before committing; address advisory suite warnings before opening a PR.
- These checks align with the Roadmap gates and the Comprehensive Issue Checklist; regressions in hot paths, events/topics, or security will block.

## File Organization

- Ad-hoc scripts belong under `ml/scripts/` (subfolders like `analysis/`).
- Hand-run examples belong under `ml/tests/examples/`.
- Test artifacts (JSON/MD outputs) must write under `ml/tests/validation_reports/`.

## Commit and PR Expectations

- Keep commits focused; include a concise summary of changes and their rationale.
- PRs must be green on CI, including `ruff`, `mypy --strict`, tests, and docs (where applicable).

## Dependency Management

We use Poetry as the authoritative dependency manager for this repository.

- Single source of truth: pyproject.toml under the [tool.poetry.*] sections.
  - Runtime deps: [tool.poetry.dependencies]
  - Dev/tooling: [tool.poetry.group.dev.dependencies]
  - Test-only deps: [tool.poetry.group.test.dependencies]
- Do not use uv "dependency-groups" in this repo. They were removed to avoid drift.
- When adding environment or test utilities, prefer adding them to the test group instead
  of runtime dependencies (e.g., pytest, hypothesis, pytest-timeout).

### Known Package Naming

- Use python-dotenv, not dotenv (the import is `from dotenv import load_dotenv`).
- Avoid duplicating ML stacks: prefer `lightning` OR `pytorch-lightning`, not both. We use
  `lightning` pinned alongside `torchmetrics`.

### Timeouts and Test Stability

- The test group includes pytest-timeout and ML suites assume a default 300s per-test
  timeout (configured in ml/pytest.ini). Ensure your environment installs test deps when
  running the suite.
- For Postgres-backed tests, prefer bounded DB waits. If needed, configure a
  statement_timeout for test engines via EngineManager.

### Installing

Typical flows:

- Runtime only:
  - `poetry install --only main`
- Dev + Test:
  - `poetry install --with dev,test`

If you choose to run with uv for speed, do not add dependency-groups; instead, install
from Poetry’s resolved venv or let uv read the Poetry sections in pyproject.toml.
