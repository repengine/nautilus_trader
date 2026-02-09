from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.core.integration import MLIntegrationManager
from ml.observability.scheduler import ObservabilityFlusher
from ml.observability.scheduler import ObservabilityStartConfig
from ml.observability.scheduler import run_observability_start
from ml.observability.service import ObservabilityService
from ml.tests.utils.stubs import build_integration_manager_stub
from nautilus_trader.model.identifiers import InstrumentId


class TestObservabilityFlusher:
    def test_tick_flushes_when_interval_elapsed(
        self,
        tmp_path: Path,
        default_instrument_id: InstrumentId,
    ) -> None:
        svc = ObservabilityService()
        # Add a row so flush writes something
        svc.add_latency_stage(
            correlation_id="c1",
            instrument_id=str(default_instrument_id),
            pipeline_stage="data_ingested",
            ts_stage_start=1,
            ts_stage_end=2,
        )

        # Fake time provider
        times = [0.0, 10.0]

        def now() -> float:
            return times[0]

        flusher = ObservabilityFlusher(
            service=svc,
            base_path=tmp_path,
            file_format="jsonl",
            interval_seconds=5.0,
            now=now,
        )

        # Not yet due
        wrote = flusher.tick()
        assert wrote == {}

        # Advance time
        times[0] = 10.0
        wrote = flusher.tick()
        assert "latency" in wrote
        latency_path = wrote["latency"]
        assert isinstance(latency_path, Path)
        assert latency_path.exists()


class TestIntegrationFlusher:
    def test_integration_manager_single_flush(self, tmp_path: Path) -> None:
        mgr = build_integration_manager_stub()
        MLIntegrationManager.initialize_observability_pipeline(mgr)
        svc = mgr.observability_service
        assert svc is not None
        svc.add_health(
            component_id="strategy_store",
            health_score=0.95,
            subsystem_scores={"db": 0.95},
            timestamp=5,
            measurement_window_ms=100,
        )

        # Start with single flush (no background thread)
        out = MLIntegrationManager.start_observability_flush(
            mgr,
            base_path=tmp_path,
            interval_seconds=0.0,
            file_format="jsonl",
        )
        assert out is not None
        health_entry = out.get("health")
        assert isinstance(health_entry, Path)
        assert health_entry.exists()


class _StubAsyncWorker:
    def __init__(self) -> None:
        self.stop_calls: list[tuple[bool, float]] = []

    async def stop(self, *, drain: bool, timeout: float) -> None:
        self.stop_calls.append((drain, timeout))


class _StubRuntime:
    def __init__(self) -> None:
        self.start_config_calls: list[object] = []
        self.start_flush_calls: list[dict[str, object]] = []
        self.stop_flush_calls = 0
        self._obs_async_worker: _StubAsyncWorker | None = None

    def start_observability_from_config(self, cfg: object) -> None:
        self.start_config_calls.append(cfg)
        self._obs_async_worker = _StubAsyncWorker()

    def start_observability_flush(
        self,
        *,
        base_path: Path,
        interval_seconds: float | None = 60.0,
        file_format: str = "jsonl",
        sink: str = "file",
        db_connection_string: str | None = None,
    ) -> dict[str, Path] | None:
        self.start_flush_calls.append(
            {
                "base_path": base_path,
                "interval_seconds": interval_seconds,
                "file_format": file_format,
                "sink": sink,
                "db_connection_string": db_connection_string,
            },
        )
        return {}

    def stop_observability_flush(self) -> None:
        self.stop_flush_calls += 1


def test_run_observability_start_when_sync_mode_runs_flush(monkeypatch: Any, tmp_path: Path) -> None:
    runtime = _StubRuntime()
    sleep_calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("ml.observability.scheduler.time.sleep", _fake_sleep)

    rc = run_observability_start(
        runtime,
        ObservabilityStartConfig(
            sink="file",
            base_path=tmp_path,
            interval_seconds=1.5,
            duration_seconds=0.25,
        ),
    )

    assert rc == 0
    assert len(runtime.start_flush_calls) == 1
    assert runtime.start_flush_calls[0]["base_path"] == tmp_path
    assert runtime.start_flush_calls[0]["interval_seconds"] == 1.5
    assert runtime.stop_flush_calls == 1
    assert sleep_calls == [0.25]


def test_run_observability_start_when_async_mode_runs_worker_stop(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    runtime = _StubRuntime()
    sleep_calls: list[float] = []

    async def _fake_async_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("ml.observability.scheduler.asyncio.sleep", _fake_async_sleep)

    rc = run_observability_start(
        runtime,
        ObservabilityStartConfig(
            sink="db",
            base_path=tmp_path,
            db_url="sqlite:///obs.db",
            async_enabled=True,
            duration_seconds=0.1,
        ),
    )

    assert rc == 0
    assert len(runtime.start_config_calls) == 1
    cfg = runtime.start_config_calls[0]
    assert getattr(cfg, "sink") == "db"
    assert getattr(cfg, "db_connection_string") == "sqlite:///obs.db"
    worker = runtime._obs_async_worker
    assert worker is not None
    assert worker.stop_calls == [(True, 1.0)]
    assert sleep_calls == [0.1]
