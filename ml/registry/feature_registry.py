#!/usr/bin/env python3

"""
Feature registry with self-describing manifests.

Provides a local, file-based registry for feature sets with lifecycle management, schema
hashing, and simple lineage queries.

"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any

from ml.registry.base import DataRequirements
from ml.registry.dataclasses import QualityGate


class FeatureRole(Enum):
    """
    Role of a feature set in the system.
    """

    TEACHER = "teacher"
    STUDENT = "student"
    INFERENCE_SUPPORT = "inference_support"


class FeatureStage(Enum):
    """
    Lifecycle stage for a feature set.
    """

    CANDIDATE = "candidate"
    STAGING = "staging"
    PROD = "prod"
    DEPRECATED = "deprecated"
    SCRAPPED = "scrapped"


@dataclass(slots=True)
class FeatureManifest:
    """
    Self-describing manifest for a feature set.

    Attributes
    ----------
    feature_set_id : str
        Unique identifier for the feature set.
    name : str
        Human-readable name.
    version : str
        Semantic version string.
    role : FeatureRole
        Intended role (teacher/student/etc.).
    data_requirements : DataRequirements
        Data requirements of the pipeline (e.g. L1_ONLY).
    feature_names : list[str]
        Ordered feature names.
    feature_dtypes : list[str]
        Dtypes for each feature (typically "float32").
    schema_hash : str
        Hash of schema used for validation.
    pipeline_signature : str
        Hash/signature of the transform graph and params.
    pipeline_version : str
        Version of the pipeline engine.
    capability_flags : dict[str, bool]
        Optional capability flags (e.g., microstructure, trade_flow).
    constraints : dict[str, Any]
        Runtime constraints (latency, memory, warmup bars).
    parity_tolerance : float
        Parity tolerance used during validation.
    parity_digest : dict[str, Any]
        Summary of parity validation results.
    perf_digest : dict[str, Any]
        Summary of latency/performance measurements.
    parent_feature_set_id : Optional[str]
        Parent feature set (e.g. teacher for a student set).
    metadata : dict[str, Any]
        Extra metadata for auditability.
    created_at : float
        Creation timestamp.
    last_modified : float
        Last modified timestamp.
    stage : FeatureStage
        Lifecycle stage.

    """

    feature_set_id: str
    name: str
    version: str
    role: FeatureRole
    data_requirements: DataRequirements
    feature_names: list[str]
    feature_dtypes: list[str]
    schema_hash: str
    pipeline_signature: str
    pipeline_version: str
    capability_flags: dict[str, bool] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    parity_tolerance: float = 0.0
    parity_digest: dict[str, Any] = field(default_factory=dict)
    perf_digest: dict[str, Any] = field(default_factory=dict)
    parent_feature_set_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_modified: float = 0.0
    stage: FeatureStage = FeatureStage.CANDIDATE


def compute_schema_hash(
    feature_names: list[str],
    feature_dtypes: list[str],
    pipeline_signature: str,
) -> str:
    """
    Compute a stable schema hash from names, dtypes, and pipeline signature.
    """
    h = hashlib.sha256()
    for n, t in zip(feature_names, feature_dtypes):
        h.update(n.encode("utf-8"))
        h.update(b"::")
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    h.update(b"|sig|")
    h.update(pipeline_signature.encode("utf-8"))
    return h.hexdigest()


@dataclass(slots=True)
class FeatureInfo:
    """
    Container for manifest and storage information.
    """

    manifest: FeatureManifest
    artifacts: dict[str, str] = field(default_factory=dict)


class LocalFeatureRegistry:
    """
    Local, file-based feature registry.

    Thread-safe and JSON-backed for persistence.

    """

    def __init__(self, registry_path: Path) -> None:
        self._root = registry_path
        self._root.mkdir(parents=True, exist_ok=True)
        self._file = self._root / "feature_registry.json"
        self._lock = threading.RLock()
        self._features: dict[str, FeatureInfo] = {}
        self._load()

    # Persistence
    def _load(self) -> None:
        if not self._file.exists():
            self._save()
            return
        with self._file.open() as f:
            raw = json.load(f)
        self._features = {}
        for fid, data in raw.get("features", {}).items():
            m = data["manifest"]
            manifest = FeatureManifest(
                feature_set_id=fid,
                name=m["name"],
                version=m["version"],
                role=FeatureRole(m["role"]),
                data_requirements=DataRequirements(m["data_requirements"]),
                feature_names=m["feature_names"],
                feature_dtypes=m["feature_dtypes"],
                schema_hash=m["schema_hash"],
                pipeline_signature=m["pipeline_signature"],
                pipeline_version=m["pipeline_version"],
                capability_flags=m.get("capability_flags", {}),
                constraints=m.get("constraints", {}),
                parity_tolerance=m.get("parity_tolerance", 0.0),
                parity_digest=m.get("parity_digest", {}),
                perf_digest=m.get("perf_digest", {}),
                parent_feature_set_id=m.get("parent_feature_set_id"),
                metadata=m.get("metadata", {}),
                created_at=m.get("created_at", 0.0),
                last_modified=m.get("last_modified", 0.0),
                stage=FeatureStage(m.get("stage", FeatureStage.CANDIDATE.value)),
            )
            self._features[fid] = FeatureInfo(
                manifest=manifest,
                artifacts=data.get("artifacts", {}),
            )

    def _save(self) -> None:
        with self._lock:
            serial: dict[str, Any] = {
                "features": {
                    fid: {
                        "manifest": {
                            "name": info.manifest.name,
                            "version": info.manifest.version,
                            "role": info.manifest.role.value,
                            "data_requirements": info.manifest.data_requirements.value,
                            "feature_names": info.manifest.feature_names,
                            "feature_dtypes": info.manifest.feature_dtypes,
                            "schema_hash": info.manifest.schema_hash,
                            "pipeline_signature": info.manifest.pipeline_signature,
                            "pipeline_version": info.manifest.pipeline_version,
                            "capability_flags": info.manifest.capability_flags,
                            "constraints": info.manifest.constraints,
                            "parity_tolerance": info.manifest.parity_tolerance,
                            "parity_digest": info.manifest.parity_digest,
                            "perf_digest": info.manifest.perf_digest,
                            "parent_feature_set_id": info.manifest.parent_feature_set_id,
                            "metadata": info.manifest.metadata,
                            "created_at": info.manifest.created_at,
                            "last_modified": info.manifest.last_modified,
                            "stage": info.manifest.stage.value,
                        },
                        "artifacts": info.artifacts,
                    }
                    for fid, info in self._features.items()
                },
                "last_updated": time.time(),
            }
            self._file.parent.mkdir(parents=True, exist_ok=True)
            with self._file.open("w") as f:
                json.dump(serial, f, indent=2)

    # ID generation
    def _gen_id(self) -> str:
        return f"feature_set_{int(time.time() * 1_000_000)}"

    # Public API
    def register_feature_set(
        self,
        manifest: FeatureManifest,
        artifacts: dict[str, str] | None = None,
    ) -> str:
        with self._lock:
            fid = manifest.feature_set_id or self._gen_id()
            now = time.time()
            manifest.feature_set_id = fid
            if manifest.created_at == 0.0:
                manifest.created_at = now
            manifest.last_modified = now
            self._features[fid] = FeatureInfo(manifest=manifest, artifacts=artifacts or {})
            self._save()
            return fid

    def promote(self, feature_set_id: str, stage: FeatureStage) -> None:
        with self._lock:
            info = self._features[feature_set_id]
            info.manifest.stage = stage
            info.manifest.last_modified = time.time()
            self._save()

    def deprecate(self, feature_set_id: str, reason: str | None = None) -> None:
        with self._lock:
            info = self._features[feature_set_id]
            info.manifest.stage = FeatureStage.DEPRECATED
            if reason:
                info.manifest.metadata["deprecation_reason"] = reason
            info.manifest.last_modified = time.time()
            self._save()

    def scrap(self, feature_set_id: str) -> None:
        with self._lock:
            if feature_set_id in self._features:
                self._features[feature_set_id].manifest.stage = FeatureStage.SCRAPPED
                self._save()

    def get_feature_set(self, feature_set_id: str) -> FeatureManifest | None:
        info = self._features.get(feature_set_id)
        return None if info is None else info.manifest

    def resolve_by_schema_hash(self, schema_hash: str) -> list[FeatureManifest]:
        return [
            fi.manifest for fi in self._features.values() if fi.manifest.schema_hash == schema_hash
        ]

    def list_by_role(self, role: FeatureRole) -> list[FeatureManifest]:
        return [fi.manifest for fi in self._features.values() if fi.manifest.role == role]

    def list_all(self) -> list[FeatureManifest]:
        return [fi.manifest for fi in self._features.values()]

    def get_lineage(self, feature_set_id: str) -> list[FeatureManifest]:
        """
        Return the lineage chain from parent to child including the requested ID if
        linked.
        """
        result: list[FeatureManifest] = []
        # Simple one-hop: include parent then the requested manifest and any children pointing to it
        target = self.get_feature_set(feature_set_id)
        if target is None:
            return result
        if target.parent_feature_set_id:
            parent = self.get_feature_set(target.parent_feature_set_id)
            if parent is not None:
                result.append(parent)
        result.append(target)
        result.extend(
            [
                fi.manifest
                for fi in self._features.values()
                if fi.manifest.parent_feature_set_id == feature_set_id
            ],
        )
        return result

    # Quality gating
    def validate_and_promote(self, feature_set_id: str, gates: list[QualityGate]) -> bool:
        """
        Validate a feature set against quality gates and promote to PROD if passed.

        Gates can reference keys present in manifest.perf_digest,
        manifest.parity_digest, or manifest.constraints. The first matching key is used.

        """
        info = self._features.get(feature_set_id)
        if info is None:
            raise KeyError(f"Unknown feature_set_id: {feature_set_id}")

        def _get_metric(name: str) -> float | None:
            # Look into perf, then parity, then constraints
            for src in (
                info.manifest.perf_digest,
                info.manifest.parity_digest,
                info.manifest.constraints,
            ):
                if name in src:
                    val = src[name]
                    if isinstance(val, (int, float)):
                        return float(val)
            return None

        passed_all = True
        for gate in gates:
            val = _get_metric(gate.metric_name)
            ok = False
            if val is None:
                ok = False
            elif gate.comparison == "gte":
                ok = val >= gate.threshold
            elif gate.comparison == "lte":
                ok = val <= gate.threshold
            elif gate.comparison == "gt":
                ok = val > gate.threshold
            elif gate.comparison == "lt":
                ok = val < gate.threshold
            elif gate.comparison == "eq":
                ok = abs(val - gate.threshold) < 1e-12
            else:
                ok = False

            if gate.required and not ok:
                passed_all = False

        if passed_all:
            self.promote(feature_set_id, FeatureStage.PROD)
        return passed_all
