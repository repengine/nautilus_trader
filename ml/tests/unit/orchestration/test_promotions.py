#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ml.registry.base import ModelManifest

from ml.orchestration.promotions import (
    register_and_promote_model,
    register_or_refresh_features,
)
from ml.registry.dataclasses import QualityGate

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

@dataclass
class _ModelRegStub:
    calls: dict[str, Any]

    def register_model(
        self,
        *,
        model_path: Path,
        manifest: ModelManifest,
        auto_deploy: bool,
        quality_gates: list[QualityGate],
        enforce_quality: bool,
    ) -> str:
        self.calls["register_model"] = {
            "model_path": model_path,
            "manifest": manifest,
            "auto_deploy": auto_deploy,
            "quality_gates": quality_gates,
            "enforce_quality": enforce_quality,
        }
        return manifest.model_id

    def track_performance(self, model_id: str, metrics: dict[str, float]) -> None:
        self.calls.setdefault("track_performance", []).append(
            {"model_id": model_id, "metrics": metrics},
        )

    def deploy_model(
        self,
        model_id: str,
        target: str,
        config: dict[str, Any] | None = None,
    ) -> bool:
        self.calls.setdefault("deploy_model", []).append({"model_id": model_id, "target": target})
        return True

@dataclass
class _FeatureRegStub:
    features: dict[str, dict[str, Any]]

    def get_feature_set(self, fid: str) -> Any:
        info = self.features.get(fid)
        if info is None:
            return None
        return type("Info", (), {"manifest": type("M", (), info)})

    def register_feature_set(self, manifest: Any) -> str:
        self.features[manifest.feature_set_id] = {"stage": manifest.stage}
        return manifest.feature_set_id

    def update_manifest(self, feature_set_id: str, **kwargs: Any) -> None:
        self.features.setdefault(feature_set_id, {}).update({"updated": True, **kwargs})

def test_register_and_promote_model(tmp_path: Path) -> None:
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"dummy")
    metrics_json = tmp_path / "model_metrics.json"
    metrics_json.write_text(
        """
        {"model_path": "model.onnx", "model_id": "m1", "architecture": "lgbm", "feature_schema_hash": "abc", "auc": 0.7}
        """,
        encoding="utf-8",
    )
    reg = _ModelRegStub(calls={})
    freg = _FeatureRegStub(features={})
    gates = [QualityGate(metric_name="auc", threshold=0.6, comparison="gte", required=True)]

    mid = register_and_promote_model(
        model_metrics_path=str(metrics_json),
        out_dir=str(tmp_path),
        registry=reg,
        feature_registry=freg,
        gates=gates,
        auto_promote=True,
        deploy_target="actor_1",
    )
    assert mid == "m1"
    assert "register_model" in reg.calls
    assert reg.calls["register_model"]["quality_gates"]
    assert reg.calls.get("deploy_model") is not None

def test_register_or_refresh_features(tmp_path: Path) -> None:
    m = tmp_path / "features.json"
    m.write_text('{"feature_set_id": "fs1", "pr_auc": 0.8}', encoding="utf-8")
    freg = _FeatureRegStub(features={})
    fid = register_or_refresh_features(str(m), freg, auto_register=True)
    assert fid == "fs1"
    assert "fs1" in freg.features
    assert freg.features["fs1"].get("updated")
