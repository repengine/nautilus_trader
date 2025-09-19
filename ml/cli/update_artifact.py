#!/usr/bin/env python3

"""
Update/validate model artifact metadata in the local ModelRegistry.

Operations
----------
- Compute and persist SHA-256 digests for ONNX artifacts
- Set a specific digest value
- Validate ONNX artifacts against stored/provided digest

Notes
-----
- This CLI operates on the cold path only.
- All operations are best-effort and avoid heavy imports at module import time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _compute_digest(path: Path) -> str:
    from ml.common.security import calculate_file_sha256

    return calculate_file_sha256(path)


def _load_registry(registry_dir: Path) -> Any:
    from ml.registry.model_registry import ModelRegistry

    return ModelRegistry(registry_dir)


def _update_manifest_digest(registry: Any, model_id: str, digest: str) -> None:
    info = registry.get_model(model_id)
    if info is None:
        raise ValueError(f"Model not found: {model_id}")
    info.manifest.artifact_sha256_digest = digest
    # Persist by saving the registry immediately
    # JSON backend flushes on change via batch save; ensure immediate save when possible
    try:
        registry._save_registry(immediate=True)
    except Exception:
        # Fallback: re-register same model to force save
        registry.log_audit(
            entity_type="model",
            entity_id=model_id,
            action="updated_digest",
            changes={"artifact_sha256_digest": digest},
        )


def _validate_onnx(path: Path, digest: str | None) -> bool:
    from ml.common.security import verify_artifact_integrity

    return bool(verify_artifact_integrity(path, digest, strict=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Update/validate model artifact metadata")
    ap.add_argument("--registry-dir", required=True, type=Path, help="Registry root directory")
    ap.add_argument("--model-id", required=True, help="Model identifier")
    ap.add_argument(
        "--artifact-path",
        type=Path,
        default=None,
        help="Optional artifact path override (defaults to registry model path)",
    )
    ap.add_argument(
        "--compute-digest",
        action="store_true",
        help="Compute SHA-256 digest for the artifact and store in manifest",
    )
    ap.add_argument(
        "--set-digest",
        type=str,
        default=None,
        help="Explicitly set the SHA-256 digest on manifest",
    )
    ap.add_argument(
        "--validate-onnx",
        action="store_true",
        help="Validate ONNX artifact integrity using stored/provided digest",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    registry_dir: Path = args.registry_dir
    model_id: str = args.model_id
    registry = _load_registry(registry_dir)

    info = registry.get_model(model_id)
    if info is None:
        raise ValueError(f"Model not found: {model_id}")

    artifact_path: Path = args.artifact_path or info.model_path
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact path not found: {artifact_path}")

    updated = False
    if args.compute_digest:
        digest = _compute_digest(artifact_path)
        _update_manifest_digest(registry, model_id, digest)
        print(f"Computed and updated digest: {digest}")
        updated = True

    if args.set_digest is not None:
        _update_manifest_digest(registry, model_id, args.set_digest)
        print(f"Set digest to: {args.set_digest}")
        updated = True

    if args.validate_onnx:
        # Prefer stored digest if available unless explicitly set in the same run
        digest_val: str | None = args.set_digest
        if digest_val is None:
            digest_val = getattr(info.manifest, "artifact_sha256_digest", None)
        ok = _validate_onnx(artifact_path, digest_val)
        print(
            f"Integrity check: {'PASS' if ok else 'FAIL'} (digest={'present' if digest_val else 'missing'})",
        )

    if not (args.compute_digest or args.set_digest or args.validate_onnx):
        print("No operation selected; use --compute-digest/--set-digest/--validate-onnx")
        return 2

    # Force save if we updated and backend supports it
    if updated:
        try:
            registry._save_registry(immediate=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI entry
    raise SystemExit(main())
