
"""
Model artifact lifecycle helpers.

Provides typed utilities to compute SHA-256 digests for model artifacts, perform
lightweight validation (e.g., ONNX load), and update the model registry with
artifact metadata. Off hot-path; designed for training/export and deployment
staging.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Any as _Any

import structlog

from ml.common.metrics_manager import MetricsManager
from ml.registry.persistence import PersistenceConfig
from ml.registry.persistence import PersistenceManager


_SASession = _Any

logger = structlog.get_logger(__name__)
mm = MetricsManager.default()
_artifact_updates_total = mm.counter(
    "ml_artifact_updates_total",
    "Total artifact registry updates",
    ["status"],
)

def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_onnx(path: Path) -> bool:
    """
    Return True if the ONNX file loads successfully via onnxruntime; False otherwise.
    """
    try:
        import onnxruntime as ort
        _ = ort.InferenceSession(path.as_posix())
        return True
    except Exception:  # pragma: no cover - optional dependency
        return False

@dataclass(slots=True)
class ArtifactUpdateRequest:
    model_id: str
    artifact_path: Path
    artifact_format: str = "onnx"
    validate: bool = True

def update_model_artifact(
    *,
    request: ArtifactUpdateRequest,
    registry_config: PersistenceConfig,
) -> dict[str, Any]:
    """
    Compute digest/validate an artifact and update registry model row.
    """
    pm = PersistenceManager(registry_config)
    session: _SASession | None = pm.get_session()
    try:
        digest = _sha256(request.artifact_path)
        ok = True
        if request.validate and request.artifact_format.lower() == "onnx":
            ok = validate_onnx(request.artifact_path)
        from sqlalchemy import text as _text
        if session is None:
            raise RuntimeError("PersistenceManager returned no active session")
        session.execute(
            _text(
                """
                UPDATE ml_registry.models
                SET artifact_sha256_digest = :digest,
                    artifact_format = :fmt,
                    model_path = :path,
                    last_modified = NOW()
                WHERE model_id = :model_id
                """,
            ),
            {
                "digest": digest,
                "fmt": request.artifact_format,
                "path": request.artifact_path.as_posix(),
                "model_id": request.model_id,
            },
        )
        session.commit()
        _artifact_updates_total.labels(status="ok").inc()
        logger.info(
            "Updated model artifact",
            model_id=request.model_id,
            path=request.artifact_path.as_posix(),
            digest=digest,
            valid=ok,
        )
        return {"model_id": request.model_id, "digest": digest, "valid": ok}
    except Exception as exc:  # pragma: no cover - defensive
        try:
            session.rollback() if session is not None else None
        except Exception:
            ...
        _artifact_updates_total.labels(status="error").inc()
        logger.warning(
            "Artifact update failed",
            model_id=request.model_id,
            path=request.artifact_path.as_posix(),
            error=str(exc),
        )
        return {"model_id": request.model_id, "error": str(exc)}
    finally:
        try:
            session.close() if session is not None else None
        except Exception:
            ...
