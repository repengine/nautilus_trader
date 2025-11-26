#!/usr/bin/env python3

"""
Pipeline signature computation and validation for data consistency.

This module provides deterministic signature computation for ML pipelines to ensure
data consistency across runs. Signatures are based on feature lists, pipeline versions,
and optional configuration hashes.

Universal ML Architecture Patterns Compliance:
- Pattern 2: Protocol-first design for extensibility
- Pattern 3: Cold-path only - no hot-path operations
- Pattern 4: Progressive fallback chains for registry access
- Pattern 5: N/A (no metrics needed for signature computation)

Examples
--------
>>> # Compute pipeline signature
>>> signature = compute_pipeline_signature(
...     features=["returns_1", "rsi_14"],
...     version="1.0",
... )
>>> assert len(signature) == 64  # SHA256 hex digest

>>> # Validate signature compatibility
>>> validator = PipelineSignatureValidator()
>>> validator.validate_signature(
...     dataset_id="my_dataset",
...     new_signature=signature,
...     registry=data_registry,
... )  # Raises ValueError if mismatch

Notes
-----
- Signatures are order-invariant (features sorted before hashing)
- Uses SHA256 for deterministic, collision-resistant hashing
- Validation prevents training on datasets with incompatible feature sets

"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    class PipelineSignatureRegistry(Protocol):
        def get_pipeline_signature(self, dataset_id: str) -> str | None: ...


logger = logging.getLogger(__name__)


def compute_pipeline_signature(
    features: list[str],
    version: str,
    *,
    config_hash: str | None = None,
) -> str:
    """
    Compute deterministic SHA256 signature for pipeline specification.

    The signature is computed from:
    - Sorted feature names (order-invariant)
    - Pipeline version string
    - Optional configuration hash

    This ensures that pipelines with identical feature sets produce identical
    signatures, regardless of the order in which features are specified.

    Parameters
    ----------
    features : list[str]
        List of feature names (will be sorted for determinism)
    version : str
        Pipeline version string (e.g., "1.0", "2.1.3")
    config_hash : str | None, optional
        Optional configuration hash for additional specificity

    Returns
    -------
    str
        SHA256 hex digest (64 characters)

    Raises
    ------
    ValueError
        If features list is empty or version is empty

    Examples
    --------
    >>> # Basic signature computation
    >>> sig1 = compute_pipeline_signature(["returns_1", "rsi_14"], "1.0")
    >>> sig2 = compute_pipeline_signature(["rsi_14", "returns_1"], "1.0")
    >>> assert sig1 == sig2  # Order-invariant

    >>> # With config hash
    >>> sig_with_config = compute_pipeline_signature(
    ...     features=["returns_1", "rsi_14"],
    ...     version="1.0",
    ...     config_hash="abc123",
    ... )
    >>> assert len(sig_with_config) == 64

    """
    if not features:
        msg = "Features list cannot be empty"
        raise ValueError(msg)
    if not version:
        msg = "Version string cannot be empty"
        raise ValueError(msg)

    # Sort features for determinism (order-invariant)
    sorted_features = sorted(features)

    # Build signature content
    content: dict[str, str | list[str]] = {
        "features": sorted_features,
        "version": version,
    }
    if config_hash:
        content["config_hash"] = config_hash

    # Compute SHA256
    content_str = json.dumps(content, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()


@dataclass
class PipelineSignatureValidator:
    """
    Validates pipeline signature compatibility against stored signatures.

    This validator ensures that dataset builds use consistent pipeline configurations
    by comparing new signatures against previously stored signatures. This prevents
    training on datasets with incompatible feature sets or pipeline versions.

    Examples
    --------
    >>> validator = PipelineSignatureValidator()
    >>> validator.validate_signature(
    ...     dataset_id="my_dataset",
    ...     new_signature="abc123...",
    ...     registry=data_registry,
    ... )  # Raises ValueError if mismatch

    """

    def validate_signature(
        self,
        dataset_id: str,
        new_signature: str,
        registry: "PipelineSignatureRegistry",
    ) -> None:
        """
        Validate new signature against stored signature for dataset.

        This method checks if a new pipeline signature is compatible with the
        signature stored in the registry for the given dataset. On first run
        (no stored signature), validation passes. On subsequent runs, signatures
        must match exactly.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier to validate against
        new_signature : str
            New pipeline signature to validate (SHA256 hex digest)
        registry : PipelineSignatureRegistry
            Data registry to retrieve stored signature

        Raises
        ------
        ValueError
            If signatures don't match (incompatible pipelines)

        Examples
        --------
        >>> # First run - no stored signature
        >>> validator.validate_signature(
        ...     dataset_id="new_dataset",
        ...     new_signature="abc123...",
        ...     registry=data_registry,
        ... )  # Passes (no signature stored yet)

        >>> # Second run - signatures match
        >>> validator.validate_signature(
        ...     dataset_id="new_dataset",
        ...     new_signature="abc123...",
        ...     registry=data_registry,
        ... )  # Passes (signatures match)

        >>> # Incompatible pipeline
        >>> validator.validate_signature(
        ...     dataset_id="new_dataset",
        ...     new_signature="def456...",
        ...     registry=data_registry,
        ... )  # Raises ValueError

        """
        stored_signature = registry.get_pipeline_signature(dataset_id)

        if stored_signature is None:
            # First run - no signature stored yet
            logger.debug(
                "No stored signature for dataset_id=%s, validation passes",
                dataset_id,
            )
            return

        if stored_signature != new_signature:
            msg = (
                f"Pipeline signature mismatch for dataset {dataset_id}: "
                f"stored={stored_signature[:16]}..., new={new_signature[:16]}..."
            )
            logger.error(
                "Pipeline signature validation failed: dataset_id=%s, stored=%s, new=%s",
                dataset_id,
                stored_signature,
                new_signature,
            )
            raise ValueError(msg)

        logger.debug(
            "Pipeline signature validation passed: dataset_id=%s, signature=%s...",
            dataset_id,
            new_signature[:16],
        )


__all__ = [
    "PipelineSignatureValidator",
    "compute_pipeline_signature",
]
