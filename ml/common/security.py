#!/usr/bin/env python3

"""
Security utilities for ML model integrity verification.

This module provides functions for verifying the integrity of ML model artifacts to
prevent tampering and ensure secure model deployment.

"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable as _Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Re-export optional ML dependency flags for backward-compatible test patching
# Use consistent signatures and avoid conditional redefinitions that confuse mypy.


HAS_ONNX: bool
ort: Any | None
check_ml_dependencies: _Callable[[list[str]], None]

try:  # pragma: no cover - import depends on environment
    from ml._imports import HAS_ONNX as _HAS_ONNX
    from ml._imports import check_ml_dependencies as _check_ml_dependencies
    from ml._imports import ort as _ORT

    HAS_ONNX = bool(_HAS_ONNX)
    check_ml_dependencies = _check_ml_dependencies
    ort = _ORT
except Exception:  # pragma: no cover - fallback
    HAS_ONNX = False
    ort = None

    def check_ml_dependencies(packages: Sequence[str]) -> None:
        missing = ", ".join(packages)
        raise ImportError(f"Missing ML dependencies: {missing}")


class ArtifactIntegrityError(Exception):
    """
    Exception raised when artifact integrity verification fails.

    This indicates that a model artifact may have been tampered with and should be
    rejected for security reasons.

    """

    def __init__(
        self,
        message: str,
        expected_digest: str | None = None,
        actual_digest: str | None = None,
    ):
        """
        Initialize artifact integrity error.

        Parameters
        ----------
        message : str
            Error message
        expected_digest : str | None
            Expected SHA-256 digest
        actual_digest : str | None
            Actual SHA-256 digest

        """
        super().__init__(message)
        self.expected_digest = expected_digest
        self.actual_digest = actual_digest


def calculate_file_sha256(file_path: Path) -> str:
    """
    Calculate SHA-256 digest of a file for integrity verification.

    Parameters
    ----------
    file_path : Path
        Path to the file

    Returns
    -------
    str
        Hexadecimal SHA-256 digest of the file

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    IOError
        If the file cannot be read

    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large models efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
    except OSError as e:
        raise OSError(f"Failed to read file {file_path}: {e}") from e

    return sha256_hash.hexdigest()


def verify_artifact_integrity(
    file_path: Path,
    expected_digest: str | None,
    strict: bool = True,
) -> bool:
    """
    Verify artifact integrity using SHA-256 digest.

    Parameters
    ----------
    file_path : Path
        Path to the artifact file
    expected_digest : str | None
        Expected SHA-256 digest. If None and strict=False, skip verification.
    strict : bool
        If True, raise exception on verification failure.
        If False, return False on verification failure.

    Returns
    -------
    bool
        True if verification passes, False if verification fails and strict=False

    Raises
    ------
    ArtifactIntegrityError
        If digest verification fails and strict=True
    ValueError
        If file cannot be read or other validation errors occur
    FileNotFoundError
        If the file doesn't exist

    """
    if expected_digest is None:
        message = f"No SHA-256 digest available for {file_path.name}"
        if strict:
            logger.warning(f"{message}, proceeding without verification in strict mode")
        else:
            logger.info(f"{message}, skipping integrity verification")
        return True

    if not expected_digest:
        message = f"Empty SHA-256 digest for {file_path.name}"
        if strict:
            logger.warning(f"{message}, proceeding without verification in strict mode")
        else:
            logger.info(f"{message}, skipping integrity verification")
        return True

    try:
        actual_digest = calculate_file_sha256(file_path)
    except (OSError, FileNotFoundError) as e:
        raise ValueError(f"Cannot verify artifact integrity: {e}") from e

    if actual_digest != expected_digest:
        # Security: Log the verification failure for auditing
        error_msg = (
            f"SECURITY ALERT: Artifact integrity verification failed for {file_path}\n"
            f"Expected SHA-256: {expected_digest}\n"
            f"Actual SHA-256:   {actual_digest}\n"
            f"This indicates the model artifact may have been tampered with!"
        )
        logger.error(error_msg)

        user_msg = (
            f"Artifact integrity verification failed for {file_path.name}. "
            f"Expected digest: {expected_digest[:16]}..., "
            f"but got: {actual_digest[:16]}... "
            f"The model artifact may have been tampered with and is rejected for security."
        )

        if strict:
            raise ArtifactIntegrityError(
                user_msg,
                expected_digest=expected_digest,
                actual_digest=actual_digest,
            )
        else:
            return False

    logger.debug(f"Artifact integrity verified for {file_path.name}: {actual_digest[:16]}...")
    return True


def secure_onnx_load(
    file_path: Path,
    expected_digest: str | None = None,
    session_options: Any = None,
    providers: list[str] | None = None,
    strict_integrity: bool = True,
) -> Any:
    """
    Securely load an ONNX model with integrity verification.

    Parameters
    ----------
    file_path : Path
        Path to the ONNX model file
    expected_digest : str | None
        Expected SHA-256 digest for integrity verification
    session_options : Any
        ONNX Runtime session options
    providers : list[str] | None
        ONNX Runtime providers
    strict_integrity : bool
        If True, raise exception on integrity verification failure

    Returns
    -------
    Any
        ONNX InferenceSession

    Raises
    ------
    ArtifactIntegrityError
        If integrity verification fails and strict_integrity=True
    ImportError
        If ONNX Runtime is not available
    ValueError
        If file cannot be read or model loading fails

    """
    # Verify integrity before loading
    verify_artifact_integrity(file_path, expected_digest, strict=strict_integrity)

    # Import ONNX Runtime (use module-level re-exports for patchability in tests)
    try:
        if not HAS_ONNX:
            check_ml_dependencies(["onnxruntime"])
        # Help static analysis: ensure ort is non-None here
        assert ort is not None
    except ImportError as e:
        raise ImportError(f"ONNX Runtime not available: {e}") from e

    # Create ONNX Runtime session
    try:
        if session_options is not None and providers is not None:
            session = ort.InferenceSession(
                str(file_path),
                sess_options=session_options,
                providers=providers,
            )
        else:
            session = ort.InferenceSession(str(file_path))

        logger.info(f"Securely loaded ONNX model: {file_path.name}")
        return session

    except Exception as e:
        raise ValueError(f"Failed to load ONNX model from {file_path}: {e}") from e
