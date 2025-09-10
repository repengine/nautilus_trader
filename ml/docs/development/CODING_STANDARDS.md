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

## File Organization

- Ad-hoc scripts belong under `ml/scripts/` (subfolders like `analysis/`).
- Hand-run examples belong under `ml/tests/examples/`.
- Test artifacts (JSON/MD outputs) must write under `ml/tests/validation_reports/`.

## Commit and PR Expectations

- Keep commits focused; include a concise summary of changes and their rationale.
- PRs must be green on CI, including `ruff`, `mypy --strict`, tests, and docs (where applicable).
