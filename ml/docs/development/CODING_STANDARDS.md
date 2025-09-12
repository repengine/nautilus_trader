# ML Coding Standards

This document defines the coding standards for the `ml/` package. The goals are:

- Safety and clarity through explicit, precise typing
- Consistency with MyPy strict mode and Ruff linters
- Maintainability under parallel testing and production constraints

## Typing (Mandatory)

- Enable and pass `mypy` with `--strict` on `ml/` at all times.
  - Command: `uv run --active --no-sync mypy ml --strict`
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

## Imports, Layout, and Style

- Use Ruff + Black formatting. Run `make ruff` locally.
- Organize imports (stdlib, third-party, local) and avoid circular imports.
- Keep module-level side effects minimal. Avoid heavy work at import time.
- Use `__all__` where a public API surface is intended. Keep it sorted.

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
