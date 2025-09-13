#!/usr/bin/env python3

from __future__ import annotations


# ruff: noqa: E402  # Allow module docstring preceding imports per project style


"""
Promotion helpers for the pipeline orchestrator (cold path).

These helpers compose existing registries and the centralized event emitter.
They avoid expanding orchestrator complexity and keep all work off hot paths.
"""

from pathlib import Path
from typing import Any

from ml.common.event_emitter import emit_dataset_event
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.dataclasses import QualityGate


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return dict(json.loads(path.read_text(encoding="utf-8")))


def _ensure_parent_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def register_and_promote_model(
    model_metrics_path: str,
    out_dir: str,
    registry: Any,
    feature_registry: Any,
    gates: list[QualityGate],
    auto_promote: bool,
    deploy_target: str | None,
) -> str:
    """
    Register a model from metrics/artifacts and optionally promote/deploy.

    Parameters
    ----------
    model_metrics_path : str
        JSON file with minimally: {"model_path", "model_id", "architecture",
        "feature_schema" (mapping) or "feature_schema_hash" (str), and
        optionally "feature_set_id", "version", and numeric metrics to track}.
    out_dir : str
        Output directory containing artifacts (used for attachments if desired).
    registry : ModelRegistry-like
        Model registry instance (use MLIntegrationManager to obtain).
    feature_registry : FeatureRegistry-like
        Feature registry (used to validate/augment feature schema linkage).
    gates : list[QualityGate]
        Quality gates to validate during registration.
    auto_promote : bool
        If True and gates pass, deploy to ``deploy_target`` when provided.
    deploy_target : str | None
        Target identifier for deployment.
    """
    # Load metrics and resolve model/artifacts
    metrics_path = Path(model_metrics_path)
    data = _load_json(metrics_path)

    model_path = Path(str(data.get("model_path", Path(out_dir) / "model.onnx")))
    model_id = str(data.get("model_id", "model_1"))
    architecture = str(data.get("architecture", "unknown"))
    feature_schema: dict[str, str] | None = data.get("feature_schema")
    feature_schema_hash: str | None = data.get("feature_schema_hash")
    feature_set_id = data.get("feature_set_id")
    serveable = bool(data.get("serveable", True))
    version = str(data.get("version", "1.0.0"))

    if feature_schema_hash is None and feature_schema is not None:
        # Stable hash of names/types
        import hashlib

        h = hashlib.sha256()
        for k in sorted(feature_schema):
            h.update(k.encode("utf-8"))
            h.update(b"::")
            h.update(str(feature_schema[k]).encode("utf-8"))
            h.update(b"\n")
        feature_schema_hash = h.hexdigest()

    if feature_schema_hash is None:
        raise ValueError("feature_schema_hash or feature_schema is required")

    # Construct minimal manifest
    manifest = ModelManifest(
        model_id=model_id,
        role=ModelRole.TEACHER if not serveable else ModelRole.INFERENCE,
        data_requirements=DataRequirements.HISTORICAL if not serveable else DataRequirements.L1_ONLY,
        architecture=architecture,
        feature_schema=feature_schema or {},
        feature_schema_hash=feature_schema_hash,
        feature_set_id=feature_set_id,
        version=version,
        serveable=serveable,
        artifact_format="onnx" if serveable else "none",
    )

    # Register with quality gates enforced
    model_id_out = registry.register_model(
        model_path=model_path,
        manifest=manifest,
        auto_deploy=False,
        quality_gates=gates,
        enforce_quality=True,
    )

    # Track numeric metrics in performance history
    perf_metrics = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    if perf_metrics:
        registry.track_performance(model_id_out, perf_metrics)

    # Optionally deploy if serveable and gates passed
    if auto_promote and serveable and deploy_target:
        try:
            registry.deploy_model(model_id_out, deploy_target)
        except Exception:
            # Non-fatal: continue emitting event below
            pass

    # Emit SUCCESS event for registration/promotion (best-effort)
    try:
        from ml.core.integration import MLIntegrationManager

        mgr = MLIntegrationManager(auto_start_postgres=False, auto_migrate=False, ensure_healthy=False)
        from typing import Any, cast
        data_registry = mgr.data_registry
        emit_dataset_event(
            cast(Any, data_registry),
            dataset_id="model",
            instrument_id="GLOBAL",
            stage=Stage.PREDICTION_EMITTED,
            source=Source.HISTORICAL,
            run_id=f"register_{model_id_out}",
            ts_min=0,
            ts_max=0,
            count=1,
            status=EventStatus.SUCCESS,
            metadata={
                "model_id": model_id_out,
                "auto_promote": bool(auto_promote),
                "deploy_target": str(deploy_target or ""),
            },
            dataset_type="model",
            component="promotions",
        )
    except Exception:
        pass

    return str(model_id_out)


def register_or_refresh_features(
    feature_metrics_path: str,
    feature_registry: Any,
    auto_register: bool,
) -> str | None:
    """
    Create/update a FeatureRegistry manifest and attach metrics/artifacts.

    The metrics JSON must contain at least ``feature_set_id`` and can include
    a mapping of numeric metrics which are persisted into ``perf_digest``.
    """
    data = _load_json(Path(feature_metrics_path))
    feature_set_id = str(data.get("feature_set_id", "")).strip()
    if not feature_set_id:
        return None

    info = feature_registry.get_feature_set(feature_set_id)
    if info is None and auto_register:
        # Minimal manifest for registration; callers can enrich later
        from ml.registry.base import DataRequirements
        from ml.registry.feature_registry import FeatureManifest
        from ml.registry.feature_registry import FeatureRole

        manifest = FeatureManifest(
            feature_set_id=feature_set_id,
            name=feature_set_id,
            version="1.0.0",
            role=FeatureRole.TEACHER,
            data_requirements=DataRequirements.HISTORICAL,
            feature_names=[],
            feature_dtypes=[],
            schema_hash="",
            pipeline_signature="",
            pipeline_version="",
        )
        feature_registry.register_feature_set(manifest)

    # Persist metrics into perf_digest
    numeric_metrics = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    if numeric_metrics:
        feature_registry.update_manifest(feature_set_id, perf_digest=numeric_metrics)

    # Emit SUCCESS event
    try:
        from ml.core.integration import MLIntegrationManager

        mgr = MLIntegrationManager(auto_start_postgres=False, auto_migrate=False, ensure_healthy=False)
        from typing import Any, cast
        data_registry = mgr.data_registry
        emit_dataset_event(
            cast(Any, data_registry),
            dataset_id="features",
            instrument_id="GLOBAL",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.HISTORICAL,
            run_id=f"features_{feature_set_id}",
            ts_min=0,
            ts_max=0,
            count=1,
            status=EventStatus.SUCCESS,
            metadata={"feature_set_id": feature_set_id},
            dataset_type="features",
            component="promotions",
        )
    except Exception:
        pass

    return feature_set_id


__all__ = ["register_and_promote_model", "register_or_refresh_features"]
