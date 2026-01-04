# ML Fixture Guide

The ML test suite exposes fixtures through dedicated modules under
`ml/tests/fixtures/`. Every test package now registers the shared plug-in
(`pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)`), so fixtures only
need to be declared as parameters or requested via `pytest.mark.usefixtures`.
`ml/tests/fixtures/__init__.py` is the canonical index – it is guarded by
`ml/tests/fixtures/test_exports.py` to ensure alphabetical ordering and that no
exports go missing.

Do not declare `pytest_plugins` inside non-top-level ``conftest.py`` files;
keep registration in package ``__init__.py`` modules or the top-level
`ml/tests/conftest.py` to avoid pytest deprecation warnings.

## Author Workflow (TL;DR)

1. **Discover** – look up the fixture name in `ml/tests/fixtures/__init__.py` (or
   the module listed in the table below). If the fixture does not exist, add it
   under `ml/tests/fixtures/` instead of defining it inside a test file.
2. **Use** – request the fixture by name in your test function/class signature or
   attach it with `pytest.mark.usefixtures`. Do **not** import fixtures directly
   from `ml.tests.fixtures` inside test modules—the plug-in handles registration.
3. **Validate** – run `make validate-fixtures` (which executes
   `tools/validate_fixture_plugins.py`) plus the guard rails from `AGENTS.md`
   (`poetry run mypy ml --strict`, `poetry ruff check ml`, targeted `pytest`
   shards, and coverage requirements).
4. **Additions** – when creating a new fixture, update the relevant module’s
   `__all__`, ensure `ml/tests/fixtures/test_exports.py` passes, and re-run
   `make validate-fixtures` so CI tracks the new export.

## Where Fixtures Live

| Module | Responsibility |
| ------ | --------------- |
| `database_fixtures.py` | PostgreSQL engines, transactional helpers, compatibility shims (`test_database`, `clean_postgres_db`, etc.). |
| `datasets.py` | Deterministic dataset builders for TFT/macros (`sample_bar_series_config_cls`, `sample_bar_series_config_factory`, `sample_bars_dataframe_factory`, `patch_bars_to_dataframe`, `patch_dataset_bars`). |
| `streaming_events.py` | Streaming plan/result/heartbeat helpers (`streaming_test_payloads_factory`, `StreamingTestPayloads`). |
| `universes.py` | Deterministic tier-1 universe stubs (`tier1_symbol_loader_stub`). |
| `monitoring_collectors.py` | Prometheus-safe metrics fixtures (`metric_name_manager`, `patch_prometheus_registry`, `isolated_prometheus_registry`). |
| `stores.py` | Store bundles, DataStore toggles, persistence helpers (`fresh_store_bundle`, `module_store_bundle`, `component_data_store_factory`). |
| `observability.py` | OpenTelemetry/metrics shims (`mock_tracing_backend`, `patch_tracing_backend`) for tracing contracts. |
| `security.py` | ONNX runtime mocks and dependency toggles (`mock_onnx_runtime`, `patch_onnx_runtime`). |
| `runtime.py` | Environment/runtime utilities (`cleanup_after_test`, logging configuration, Hypothesis SQLite session, registry isolation, orchestrator env helpers). |

## Migration Checklist

Follow this sequence whenever migrating an existing suite onto the shared fixtures:

1. **Register the plug-in** – Ensure the package (or the module if it lives outside `ml/tests`) declares `pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)`.
2. **Remove direct fixture imports** – Replace `from ml.tests.fixtures...` imports with fixture parameters or `pytest.mark.usefixtures`. Only `ml/tests/fixtures/test_exports.py` should import fixture modules directly.
3. **Migrate helpers in DB → store → telemetry → dataset order**:
   - **Database**: Request `test_database`, `clean_postgres_db`, and `real_engine_manager` to isolate connection pools before touching store/stateful tests.
   - **Stores**: Wrap EngineManager access via `patch_engine_manager`/`mock_engine_manager`, and request store bundles from `ml.tests.fixtures.stores`.
   - **Telemetry**: Opt into `isolated_prometheus_registry`, `mock_tracing_backend`, and `isolated_orchestrator_env` so metrics/tracing side-effects stay contained.
   - **Datasets & streaming**: Use `patch_dataset_bars`/`patch_bars_to_dataframe` for ingestion builders and `streaming_test_payloads_factory` when constructing streaming plan/result/heartbeat payloads.
4. **Validate** – Run `make validate-fixtures`, `poetry run mypy ml --strict`, `poetry ruff check ml`, and the targeted pytest shards (plus coverage) before merging.

Thin wrappers (mock services, builders, common data) continue to live beside
their original modules but are exported through `ml.tests.fixtures`.

### Observability & Metrics

- Use `isolated_prometheus_registry` when tests assert on Prometheus collectors. The helper snapshots the global registry, resets metric caches before the test, optionally evicts modules for a clean re-import, and unregisters any collectors created during the test.
- For optional Prometheus installs, `mock_prometheus_when_unavailable` remains an opt-in shim to avoid import errors; combine with `metric_name_manager` to generate unique metric prefixes.

### Dataset Builders

- `sample_bar_series_config_factory` produces deterministic configs for every dataset test; pair it with `sample_bars_dataframe_factory` when builders expect raw DataFrames.
- `patch_dataset_bars` patches both `ml.data.catalog_utils` and `ml.data.tft_dataset_builder` (or whichever modules you pass) so suites no longer need bespoke local helpers. The fixture returns the `SampleBarSeriesConfig` being used, which is handy for assertions around timestamps or frequency.
- When you only need to patch a single module, fall back to `patch_bars_to_dataframe` — both fixtures compose so existing tests can migrate incrementally.

### Streaming Events

- Request `streaming_test_payloads_factory` to obtain a callable that builds deterministic streaming plan/result/heartbeat bundles. The helper returns `StreamingTestPayloads`, which exposes `.plan_message()`, `.result_message()`, and `.heartbeat_message()` plus the underlying events for advanced assertions.
- Prefer the factory over direct imports from `ml.tests.fixtures.streaming_events` so the pytest plug-in mediates setup and future dependency changes remain centralized.

### Tier-1 Universes

- `tier1_symbol_loader_stub` patches both L2 (`ml.tasks.ingest.l2`, `ml.data.ingest.l2_efficient`) and alternative loader modules so every call to `get_tier1_symbols`/`load_tier1_symbols` returns a deterministic tuple (`("SPY", "QQQ")`). Use it in ingestion/task suites to guarantee they never reach out to live progress files or external services.

### Security & Model Integrity

- Use `mock_onnx_runtime` to patch ONNX Runtime imports and dependency checks without hitting the real runtime. The harness exposes `ort.InferenceSession` for assertions, `check_dependencies` for requirement verification, and `set_available(False)` when you need to simulate missing optional dependencies.
- `onnx_session_stub_factory` builds deterministic `InferenceSession` stand-ins (with controllable predictions/confidence or forced failures) so actor/registry tests can exercise inference logic without serializing real ONNX models.

## Quick Start

```python
import pytest

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


def test_strategy_reads_isolated(fresh_store_bundle):
    feature_store = fresh_store_bundle.feature_store
    feature_store.write_features(...)
    # Fresh stores every test: no shared state, no flakes.
```

- Prefer `fresh_store_bundle` for new tests. It spins up fresh Feature/Model/Strategy stores
  per test with full cleanup (flush, cancel timers, reset state, dispose engines, truncate tables).
- `store_bundle` still exists for legacy suites that rely on module-level sharing,
  but the pollution detector will nudge you toward `fresh_store_bundle`.
- For compatibility, `test_database`, `module_test_database`, `clean_postgres_db*`,
  and `database_session` remain available exactly as before — simply request them by name
  in your tests (no manual import required).

## Database Patterns

```python
import pytest
from sqlalchemy import text
from ml.tests.fixtures import database_session, clean_postgres_db

@pytest.mark.database
def test_uses_transaction(database_session):
    database_session.execute(text("SELECT 1"))

def test_requires_clean_tables(clean_postgres_db, test_database):
    # Tables truncated before and after the test
    ...
```

`database_engine` (session scoped), `database_session_factory`, and `isolated_engine`
help with advanced scenarios (connection leak tests, SQLite-only paths).

Need to intercept `EngineManager` safely? Request the `patch_engine_manager` fixture (which
returns the context manager) or use `mock_engine_manager`. Both ensure cached engines get disposed
after use. Pass `record_calls=True` when you need to assert on connection strings or kwargs—inspect
`mock_engine._engine_manager_calls` for the captured arguments. Set `side_effect=Exception(...)`
to simulate failures without touching real engines.

```python
def test_engine_manager_records_connection_strings(
    patch_engine_manager,
    component_data_store_factory,
):
    with patch_engine_manager(record_calls=True) as engine_mock:
        with component_data_store_factory(use_component=True):
            # Exercise whatever code path triggers EngineManager lookups
            pass

    call = engine_mock._engine_manager_calls[0]
    assert call[1]["connection_string"].startswith("postgresql://")
```

### Performance without pollution (PostgreSQL on :5434)

- A session-scoped template DB (default name: `nautilus_template`) can be built once on the 5434 test instance; it is **read-only**. Use the `template_database` fixture only as a seed, never mutate it.
- For writable tests, request `cloned_test_database`: it clones the template into a fresh schema/DB per test and drops it on teardown; EngineManager caches are disposed after each test.
- Store bundles (`fresh_store_bundle`, `component_data_store_factory`) should point at the cloned DB for isolation. Do not write directly to the template.
- Avoid manual schema tweaks (e.g., `feature_store.schema`) in tests; use `fresh_store_bundle` and store APIs like `FeatureStore.read_range` for assertions.
- Pollution-detection/pool-stat tests should rely on `cloned_test_database` and fixture auto-skips instead of sentinel files.
- Artifacts follow the same pattern: session-scoped creation is immutable; copy into `tmp_path` before mutation.
- Hypothesis/property tests that hit the DB must keep example counts/deadlines bounded to avoid long-lived connections and contention.

## Orchestrator & Scheduler Examples

- Pair `isolated_orchestrator_env` with `tier1_symbol_loader_stub` so scheduler state
  never leaks across tests and tier-1 universes stay deterministic.
- Patch dataset lookups with `patch_dataset_bars` (optionally supplying a custom
  `sample_bar_series_config_factory`) so ingestion flows never touch live catalog files.
- Use `component_data_store_factory` or `fresh_store_bundle` when schedulers need a store
  handle—they take care of cleanup and EngineManager cache disposal.

```python
from pathlib import Path

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from ml.data.scheduler import DataScheduler


@pytest.mark.usefixtures("isolated_orchestrator_env")
def test_pipeline_scheduler_bootstrap(
    tier1_symbol_loader_stub,
    patch_dataset_bars,
    sample_bar_series_config_factory,
    tmp_path: Path,
):
    config = sample_bar_series_config_factory(rows=16, instrument_id="SPY")
    patch_dataset_bars(config=config)

    catalog = ParquetDataCatalog(str(tmp_path / "catalog"))
    scheduler = DataScheduler(
        catalog=catalog,
        use_orchestrator=True,
    )
    scheduler.run_daily_update()
    # Assert against scheduler side effects (catalog writes, metrics) as needed.
```

### Earnings Dataset Builders

When porting earnings/TFT dataset suites, rely on the dataset fixtures to keep builders
deterministic and fast:

```python
from ml.data import DatasetBuildConfig, build_tft_dataset

def test_earnings_task_dataset(
    patch_dataset_bars,
    sample_bar_series_config_factory,
    isolated_orchestrator_env,
):
    bar_config = sample_bar_series_config_factory(instrument_id="AAPL", rows=5)
    patch_dataset_bars(config=bar_config)

    dataset = build_tft_dataset(
        DatasetBuildConfig(
            instrument_ids=[bar_config.instrument_id],
            lookback_minutes=bar_config.rows,
        ),
    )
    assert dataset.height == bar_config.rows
```

The same approach works for scheduler/CLI suites: isolate env vars with
`isolated_orchestrator_env`, patch catalog lookups via `patch_dataset_bars`, and rely on
fixture-provided stores instead of ad-hoc builders. This keeps ingestion/e2e suites aligned
with the fixture plug-in and avoids real filesystem or network IO.

## Security & ONNX Harness

- Always request `mock_onnx_runtime` in actor, deployment, and registry suites. It patches
  `ml._imports`, `ml.common.security`, and registry layers so tests no longer require the
  optional ONNX dependency.
- Use `onnx_session_stub_factory` to build deterministic `InferenceSession` stand-ins. You
  can control `prediction`, `confidence`, or simulate failures via `raise_on_run=True`.
- When tests need a file path, write simple bytes to `tmp_path` or request the
  `dummy_onnx_model` fixture—both approaches avoid serializing real models.

```python
import numpy as np
from pathlib import Path

from ml.actors.multi_signal import MultiInstrumentSignalActor, MultiInstrumentSignalActorConfig


def test_actor_vectorized_infer(
    mock_onnx_runtime,
    onnx_session_stub_factory,
    tmp_path: Path,
) -> None:
    feature_dim = 4
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test-model")
    mock_onnx_runtime.ort.InferenceSession.side_effect = lambda *_: onnx_session_stub_factory(
        prediction=0.75,
        confidence=0.88,
    )

    config = MultiInstrumentSignalActorConfig(
        actor_id="fixture-demo",
        max_batch_size=2,
        feature_dim=feature_dim,
        use_dummy_stores=True,
        model_path=str(model_path),
        model_id="demo",
        instrument_id=None,
        bar_type=None,
    )
    actor = MultiInstrumentSignalActor(config)  # type: ignore[arg-type]
    batch = np.zeros((2, feature_dim), dtype=np.float32)
    preds, confs = actor._infer_batch(batch)

    np.testing.assert_allclose(preds, 0.75)
    np.testing.assert_allclose(confs, 0.88)
```

Deployment and orchestrator suites follow the same pattern: point `MODEL_PATH` at a temp file,
request `mock_onnx_runtime`, and let the stub factory supply deterministic sessions so the tests
never touch real runtimes or serialized artifacts.

## Runtime Utilities

- `cleanup_after_test` (autouse) clears caches/config and forces a GC sweep.
- `configure_test_logging` (session autouse) applies a per-run log format and restores the original handlers.
- `_set_isolated_ml_registry_path` guarantees registry IO happens in a temporary directory.
- `hypothesis_database_session` provides an in-memory SQLite session for property tests.
- `isolated_orchestrator_env` clears all `ORCH_*` variables before a test runs, ensuring scheduler/ingestion suites never inherit state from another shard; pair it with `monkeypatch.setenv` for explicit overrides.

## Legacy Fixtures

All legacy names remain import-compatible, but new code should not import fixtures directly. If a test
still depends on a shared bundle or legacy helper, migrate it into `ml/tests/fixtures/**` and reference
the fixture through dependency injection. The only module that intentionally imports from
`ml.tests.fixtures` is `ml/tests/fixtures/test_exports.py`, which enforces the canonical export list.
Direct ``ml.tests.conftest`` imports remain unsupported; rely on the shared plug-in instead.
