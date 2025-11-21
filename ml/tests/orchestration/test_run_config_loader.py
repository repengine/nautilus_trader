from __future__ import annotations

from pathlib import Path

import pytest

from ml.config import universes as _universes
from ml.orchestration.config_loader import Stage
from ml.orchestration.config_loader import load_orchestrator_run_config
from ml.orchestration.config_loader import to_pipeline_args

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


def _fixture_path(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "orchestrator" / name


def test_load_orchestrator_run_config_parses_sections() -> None:
    config_path = _fixture_path("minimal.json")
    run_cfg = load_orchestrator_run_config(config_path)

    assert run_cfg.stage == Stage.DATASET
    assert run_cfg.dataset is not None
    assert run_cfg.dataset.include_macro is True
    assert run_cfg.dataset.instrument_ids == ("SPY.NYSE", "QQQ.NASDAQ")

    assert run_cfg.ingestion is not None
    assert run_cfg.ingestion.enabled is True
    assert run_cfg.ingestion.write_mode == "sql+parquet"

    orchestrator_cfg = run_cfg.compose_orchestrator_config()
    assert orchestrator_cfg.dataset.symbols == "SPY,QQQ"
    assert orchestrator_cfg.teacher.model_id == "teacher-v1"
    assert orchestrator_cfg.student.enabled is True

    args = to_pipeline_args(orchestrator_cfg, ingestion=run_cfg.ingestion)
    assert "--ingest" in args
    assert ["--write_mode", "sql+parquet"] == args[
        args.index("--write_mode") : args.index("--write_mode") + 2
    ]


def test_environment_override_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _fixture_path("minimal.json")
    monkeypatch.setenv("ML_ORCH__DATASET__SYMBOLS", "MSFT")
    monkeypatch.setenv("ML_ORCH__INGESTION__ENABLED", "false")

    run_cfg = load_orchestrator_run_config(config_path)
    assert run_cfg.dataset is not None
    assert run_cfg.dataset.symbols == "MSFT"
    assert run_cfg.ingestion is not None
    assert run_cfg.ingestion.enabled is False


def test_universe_alias_expansion() -> None:
    config_path = _fixture_path("universe_alias.toml")
    run_cfg = load_orchestrator_run_config(config_path)

    tier1_full = tuple(str(symbol).strip().upper() for symbol in _universes.TIER1_FULL_95)
    expected_symbols = ",".join(_strip_venue(symbol) for symbol in tier1_full)

    assert run_cfg.dataset is not None
    assert run_cfg.dataset.symbols == expected_symbols

    assert run_cfg.ingestion is not None
    assert run_cfg.ingestion.instruments == tier1_full

    assert run_cfg.auto_fill is not None
    assert run_cfg.auto_fill.instrument_ids == tier1_full


def _strip_venue(symbol: str) -> str:
    head, _, _tail = symbol.partition(".")
    return head or symbol
