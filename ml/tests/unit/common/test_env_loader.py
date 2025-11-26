from __future__ import annotations

import os
from pathlib import Path
from importlib import reload

import pytest

import ml.common.env as env_module


def _reload_env_module() -> None:
    reload(env_module)


def test_load_project_dotenv_loads_nearest_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env_file = repo / ".env"
    env_file.write_text("FRED_API_KEY=test-key\nEXTRA_VAR=123\n", encoding="utf-8")

    nested = repo / "scripts" / "inner"
    nested.mkdir(parents=True)

    monkeypatch.chdir(nested)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("EXTRA_VAR", raising=False)

    _reload_env_module()
    loaded = env_module.load_project_dotenv()

    assert loaded == env_file
    assert os.environ["FRED_API_KEY"] == "test-key"
    assert os.environ["EXTRA_VAR"] == "123"


def test_load_project_dotenv_returns_none_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    _reload_env_module()
    loaded = env_module.load_project_dotenv()

    assert loaded is None
    assert "FRED_API_KEY" not in os.environ
