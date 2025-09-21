from __future__ import annotations

from pathlib import Path

import json
import pytest

from ml.config.databento_policy import DatabentoSafetyConfigError
from ml.config.databento_policy import SchemaSafetyConfig
from ml.config.databento_policy import load_databento_safety_config


def test_load_databento_safety_config(tmp_path: Path) -> None:
    cfg = {
        "datasets": ["EQUS.MINI"],
        "schemas": {"trades": {"max_days": 30, "max_cost_usd": 0.0}},
        "global": {"max_cost_usd": 0.0, "max_symbols": 10},
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    loaded = load_databento_safety_config(path)
    assert loaded.datasets == ("EQUS.MINI",)
    assert loaded.max_cost_usd == 0.0
    assert loaded.max_symbols == 10
    assert isinstance(loaded.schemas["trades"], SchemaSafetyConfig)
    assert loaded.schemas["trades"].max_days == 30


def test_load_databento_safety_config_missing(tmp_path: Path) -> None:
    with pytest.raises(DatabentoSafetyConfigError):
        load_databento_safety_config(tmp_path / "missing.json")
