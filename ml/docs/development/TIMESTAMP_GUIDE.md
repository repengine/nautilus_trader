# ML Timestamp Usage Guide

This guide defines how timestamps must be handled across the `ml/` package.
It complements the Coding Standards and ensures consistent, safe handling of
time across stores, registries, orchestration, and CLI utilities.

## Principles

- All persisted timestamps and time boundaries are nanoseconds (ns).
- Normalize any external or ambiguous unit via `sanitize_timestamp_ns`.
- Prefer clock-provided ns when available (`self.clock.timestamp_ns()`), falling
  back to `time.time_ns()` for “now” in non‑hot paths.
- Avoid import‑time work; import utilities lazily inside functions.
- Do not block hot paths with logging or I/O. Use best‑effort logging only.

## The Normalizer

```python
from ml.common.timestamps import sanitize_timestamp_ns

ns_value = sanitize_timestamp_ns(
    int(possibly_seconds_or_ms),
    logger=logger,                     # optional
    context="module.function:purpose", # required when passing logger
)
```

- Modes: controlled via `ML_TS_NORMALIZATION_MODE` (`warn`|`normalize`|`reject`).
- Always provide a `context` string when a `logger` is passed.

## Patterns by Area

- Stores (Feature/Model/Strategy/Data)
  - Write paths: sanitize inbound `ts_event` and any computed window
    bounds (`ts_min`/`ts_max`) before persistence, event emission, or bus payloads.
  - Lateness/freshness checks: sanitize both “now” and the latest record ts.
  - Use `self.clock.timestamp_ns()` when a clock is wired; otherwise `time.time_ns()`.

- Registry
  - Manifest timestamps (`created_at`, `last_modified`, `deprecated_at`) should
    be set using `time.time_ns()` for “now” or sanitized values when converting
    external inputs.

- Orchestration/CLI
  - Range bounds derived from `datetime`/`timestamp()` must be sanitized.
  - Single “marker” timestamps (e.g., refresh signals) may use `time.time_ns()`
    sanitized for consistency in logs/validators.

- Observability
  - Retention/cutoff calculations: compute with ns arithmetic and sanitize the
    final values sent to queries.

## Do and Don’t

- Do: `sanitize_timestamp_ns(int(start.timestamp() * 1e9), context="...start")`
- Do: `sanitize_timestamp_ns(time.time_ns(), context="...now")`
- Don’t: `int(datetime.now().timestamp() * 1e9)` without sanitization.
- Don’t: Log or raise from hot paths unless policy requires (`reject`).

## Validation

- Types: `uv run --active --no-sync mypy ml --strict`
- Lint: `make ruff`
- Tests (focused): `pytest -q -k 'stores or registry or orchestrator' ml/tests`
- Validators: `make validate-metrics` and `make validate-events`

