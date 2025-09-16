# ML Logging Standardization Plan

Checklist plan to adopt structured logging (structlog + stdlib interop), enforce best‑effort patterns, and keep hot paths safe.

## Principles

- Hot path safety: logging must be non‑blocking and typically disabled (DEBUG) in production.
- Structured output: machine‑readable logs with consistent fields and minimal overhead.
- Incremental adoption: preserve existing stdlib logging; migrate where it adds value.

## Phase 0 — Readiness

- [ ] Dependency available: `structlog` added to Poetry and locked
- [ ] Decide default format by environment:
  - [ ] Dev: human/console renderer
  - [ ] CI/Prod: JSON renderer (`ML_LOG_FORMAT=json`)

## Phase 1 — Central Configuration

- [ ] Add `ml/common/logging_config.py` providing:
  - [ ] Stdlib + structlog interop via `structlog.stdlib.ProcessorFormatter`
  - [ ] Minimal processor chain: `filter_by_level`, `add_logger_name`, `add_log_level`, `contextvars.merge_contextvars`, timestamp, `format_exc_info`, renderer
  - [ ] `configure_logging(level: str|None = None, json: bool|None = None)`
- [ ] Call `configure_logging(...)` in entrypoints (cold path only):
  - [ ] `ml/deployment/entrypoint_*.py`
  - [ ] CLI tools under `ml/cli/`

## Phase 2 — Best‑Effort Patterns

- [ ] Replace `except: pass` with non‑blocking debug logging pattern
- [ ] Ensure all logging within hot loops is DEBUG and allocation‑free
- [ ] Add semgrep rule to flag `except: pass` and suggest standard snippet

## Phase 3 — Context Propagation

- [ ] Introduce `bind_log_context(**fields)` helper (contextvars) for run_id, correlation_id, component, dataset_id, instrument_id
- [ ] Bind at orchestration/actor boundaries; unbind on exit
- [ ] Add typed `LogExtra` (`TypedDict`) documenting standard fields

## Phase 4 — Output & Shipping

- [ ] JSON logs enabled via env (`ML_LOG_FORMAT=json`)
- [ ] Document recommended shippers (e.g., Loki/Promtail, Elastic, Cloud logs)
- [ ] Keep logging handlers centralized; no per‑module handlers

## Phase 5 — Linting/Enforcement

- [ ] Semgrep rules:
  - [ ] Require `exc_info=True` when logging exceptions
  - [ ] Forbid f‑string building on error paths; prefer placeholders or key/value fields
  - [ ] Flag per‑module handler creation
- [ ] Pre‑commit hook includes logging rules alongside existing metrics/events checks

## Phase 6 — Hot‑Path Audit

- [ ] Review `on_*` handlers and tight loops for logging
- [ ] Ensure logs are either removed or guarded by disabled levels
- [ ] Verify P99 < 5 ms with micro‑bench tests after changes

## Phase 7 — Targeted Migrations (by area)

- [ ] Stores/Services (cold path):
  - [ ] Convert warning/error logs to structured key/value with standard fields
  - [ ] Use `log_best_effort` helper for optional work
- [ ] Actors:
  - [ ] Keep hot loop logs minimal; bind `run_id`, `correlation_id`
  - [ ] Use non‑blocking patterns for bus publish failures
- [ ] Orchestration/CLI:
  - [ ] Configure logging early; include run metadata
- [ ] Training/Evaluation (cold path):
  - [ ] Structured summaries and error logs with context

## Phase 8 — Documentation

- [ ] Update `ml/docs/development/CODING_STANDARDS.md` (structured logging section)
- [ ] Update `ml/docs/architecture/universal_patterns_guide.md` (observability/logging practices)
- [ ] Add examples for context binding and best‑effort patterns

## Phase 9 — Observability Correlation

- [ ] Include `run_id`/`correlation_id` in both logs and metrics labels
- [ ] (Optional) Add exemplars/tracing correlation via OpenTelemetry if enabled

## Phase 10 — CI/Validation

- [ ] Extend `make validate-nautilus-patterns` to include logging rules
- [ ] Add simple contract tests asserting JSON log structure in entrypoints

---

### Code Patterns (Quick Copy)

Best‑effort (cold path):

```python
try:
    do_optional_work()
except Exception:
    logger.debug("Optional work failed (ignored)", exc_info=True)
```

Context binding:

```python
from structlog.contextvars import bind_contextvars

bind_contextvars(run_id=run_id, component="feature_store", instrument_id=instr)
```

Entry configuration:

```python
from ml.common.logging_config import configure_logging

configure_logging()  # respects ML_LOG_FORMAT/ML_LOG_LEVEL env
```

