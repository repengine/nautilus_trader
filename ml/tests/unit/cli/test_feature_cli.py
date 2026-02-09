from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ml.cli import feature_cli as cli
from ml.registry import FeaturePromotionGate


def test_cli_register_default_delegates_to_registry_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_register(
        registry_path: Path,
        *,
        name: str,
        version: str | None,
        role: Any,
        data_requirements: Any,
    ) -> str:
        captured["registry_path"] = registry_path
        captured["name"] = name
        captured["version"] = version
        captured["role"] = role
        captured["data_requirements"] = data_requirements
        return "feature_set_123"

    monkeypatch.setattr(cli, "register_default_feature_set", _fake_register)

    result = cli.cli_register_default(
        registry_path="/tmp/registry",
        name="baseline",
        version="v2",
        role="teacher",
        data_requirements="l1_l2",
    )

    assert result == "feature_set_123"
    assert captured["registry_path"] == Path("/tmp/registry")
    assert str(captured["role"].value) == "teacher"
    assert str(captured["data_requirements"].value) == "l1_l2"


def test_cli_promote_with_gates_builds_typed_gate_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_promote(
        registry_path: Path,
        *,
        feature_set_id: str,
        gates: list[FeaturePromotionGate],
    ) -> bool:
        captured["registry_path"] = registry_path
        captured["feature_set_id"] = feature_set_id
        captured["gates"] = gates
        return True

    monkeypatch.setattr(cli, "promote_feature_set", _fake_promote)

    result = cli.cli_promote_with_gates(
        registry_path="/tmp/registry",
        feature_set_id="fs_42",
        gates=[{"metric_name": "auc", "threshold": 0.8}],
    )

    assert result is True
    assert captured["registry_path"] == Path("/tmp/registry")
    assert captured["feature_set_id"] == "fs_42"
    gate = captured["gates"][0]
    assert isinstance(gate, FeaturePromotionGate)
    assert gate.metric_name == "auc"
    assert gate.threshold == 0.8
    assert gate.comparison == "gte"
    assert gate.required is True
