"""
Feature manifest export utilities for the TFT training pipeline.

Provides helpers to infer feature columns, compute a pipeline signature,
and register a FeatureManifest with the local FeatureRegistry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml._imports import pd
from ml._imports import pl
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole


def infer_feature_columns(df: Any) -> list[str]:
    """
    Infer numeric feature column names from a Polars or pandas DataFrame.

    Excludes control columns (timestamp/time_index/instrument_id/y/ts_event).
    """
    exclude = {"y", "time_index", "timestamp", "instrument_id", "ts_event"}
    if pl is not None and isinstance(df, pl.DataFrame):
        numeric = [c for c in df.columns if df[c].dtype.is_numeric()]
        return [c for c in numeric if c not in exclude]
    if pd is not None and isinstance(df, pd.DataFrame):  # pragma: no cover
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        return [c for c in numeric if c not in exclude]
    return []


def build_pipeline_signature(flags: dict[str, Any]) -> str:
    """
    Build a deterministic pipeline signature from flags (sorted keys + values).
    """
    items = sorted((str(k), str(v)) for k, v in flags.items())
    h = hashlib.sha256()
    for k, v in items:
        h.update(k.encode("utf-8"))
        h.update(b"=")
        h.update(v.encode("utf-8"))
        h.update(b";")
    return h.hexdigest()


@dataclass(slots=True)
class FeatureExportConfig:
    registry_path: Path
    role: FeatureRole = FeatureRole.TEACHER
    data_requirements: DataRequirements = DataRequirements.L1_ONLY
    version: str = "1.0.0"
    pipeline_version: str = "1.0.0"


def export_feature_manifest(
    *,
    feature_names: list[str],
    feature_dtypes: list[str] | None = None,
    flags: dict[str, Any] | None = None,
    cfg: FeatureExportConfig,
) -> str:
    """
    Create and register a feature manifest in the local FeatureRegistry.
    """
    if not feature_names:
        raise ValueError("feature_names cannot be empty")
    dtypes = feature_dtypes or ["float32"] * len(feature_names)
    if len(dtypes) != len(feature_names):
        raise ValueError("feature_dtypes length must match feature_names length")
    flags = flags or {}
    pipeline_sig = build_pipeline_signature(flags)

    # Compute schema hash in the same spirit as the registry util
    from ml.registry.feature_registry import compute_schema_hash

    schema_hash = compute_schema_hash(feature_names, dtypes, pipeline_sig)

    manifest = FeatureManifest(
        feature_set_id="",  # will be filled by registry
        name="tft_features",
        version=cfg.version,
        role=cfg.role,
        data_requirements=cfg.data_requirements,
        feature_names=feature_names,
        feature_dtypes=dtypes,
        schema_hash=schema_hash,
        pipeline_signature=pipeline_sig,
        pipeline_version=cfg.pipeline_version,
        capability_flags={k: bool(v) if isinstance(v, bool) else False for k, v in flags.items()},
    )
    freg = FeatureRegistry(cfg.registry_path)
    fid = freg.register_feature_set(manifest)
    return fid

