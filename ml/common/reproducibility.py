"""
Shared reproducibility helpers.

This module centralizes deterministic seeding and provenance metadata capture
for training and export workflows.
"""

from __future__ import annotations

import logging
import platform
import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import Any, cast

import numpy as np

from ml._imports import HAS_TORCH
from ml._imports import torch as _torch


logger = logging.getLogger(__name__)

ReproducibilityValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class DeterministicSeedResult:
    """
    Result payload returned from deterministic seed application.

    Parameters
    ----------
    seed : int
        Seed value applied to deterministic RNG paths.
    python_random_seeded : bool
        Whether Python's ``random`` module was seeded.
    numpy_seeded : bool
        Whether NumPy RNG was seeded.
    torch_seeded : bool
        Whether torch CPU RNG was seeded.
    torch_cuda_seeded : bool
        Whether torch CUDA RNGs were seeded.
    """

    seed: int
    python_random_seeded: bool
    numpy_seeded: bool
    torch_seeded: bool
    torch_cuda_seeded: bool


class ReproducibilityHelper:
    """
    Utility entry points for deterministic seed and provenance behaviors.
    """

    @staticmethod
    def apply_seed(
        seed: int,
        *,
        deterministic_mode: bool = False,
        include_torch: bool = True,
    ) -> DeterministicSeedResult:
        """
        Apply deterministic seeds for available random number generators.

        Parameters
        ----------
        seed : int
            Non-negative seed value.
        deterministic_mode : bool, default False
            When ``True``, enables torch deterministic algorithm flags when
            torch is available.
        include_torch : bool, default True
            Whether to attempt torch seeding.

        Returns
        -------
        DeterministicSeedResult
            Seed-application summary for telemetry/testing.

        Examples
        --------
        >>> result = ReproducibilityHelper.apply_seed(42, include_torch=False)
        >>> result.numpy_seeded
        True
        """
        seed_value = int(seed)
        if seed_value < 0:
            raise ValueError("seed must be >= 0")

        random.seed(seed_value)
        np.random.seed(seed_value)

        torch_seeded = False
        torch_cuda_seeded = False
        if include_torch and HAS_TORCH and _torch is not None:
            torch_seeded, torch_cuda_seeded = _seed_torch(
                seed=seed_value,
                deterministic_mode=deterministic_mode,
            )

        return DeterministicSeedResult(
            seed=seed_value,
            python_random_seeded=True,
            numpy_seeded=True,
            torch_seeded=torch_seeded,
            torch_cuda_seeded=torch_cuda_seeded,
        )

    @staticmethod
    def build_provenance(
        *,
        seed: int | None,
        deterministic_mode: bool,
        extra: Mapping[str, ReproducibilityValue] | None = None,
    ) -> dict[str, ReproducibilityValue]:
        """
        Build deterministic/provenance metadata for manifests or run records.

        Parameters
        ----------
        seed : int | None
            Seed value used for the run.
        deterministic_mode : bool
            Whether deterministic mode was enabled.
        extra : Mapping[str, ReproducibilityValue] | None, optional
            Additional scalar metadata to merge into the payload.

        Returns
        -------
        dict[str, ReproducibilityValue]
            Provenance dictionary with environment and dependency metadata.
        """
        payload: dict[str, ReproducibilityValue] = {
            "seed": int(seed) if seed is not None else None,
            "deterministic_mode": deterministic_mode,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "has_torch": HAS_TORCH,
            "torch_version": _resolve_torch_version(),
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        if extra is not None:
            for key, value in extra.items():
                payload[str(key)] = value
        return payload


def apply_reproducibility_seed(
    seed: int,
    *,
    deterministic_mode: bool = False,
    include_torch: bool = True,
) -> DeterministicSeedResult:
    """
    Convenience wrapper around :meth:`ReproducibilityHelper.apply_seed`.

    Parameters
    ----------
    seed : int
        Non-negative seed value.
    deterministic_mode : bool, default False
        Whether deterministic mode should be enabled where supported.
    include_torch : bool, default True
        Whether to attempt torch seeding.

    Returns
    -------
    DeterministicSeedResult
        Seed application result payload.
    """
    return ReproducibilityHelper.apply_seed(
        seed,
        deterministic_mode=deterministic_mode,
        include_torch=include_torch,
    )


def resolve_configured_seed(
    *,
    primary_seed: int | None,
    fallback_seed: int | None = None,
    required: bool = False,
    context: str = "seed",
) -> int | None:
    """
    Resolve and validate a deterministic seed value from config inputs.

    Parameters
    ----------
    primary_seed : int | None
        Preferred configured seed value.
    fallback_seed : int | None, default None
        Secondary seed used when ``primary_seed`` is unset.
    required : bool, default False
        Whether a seed must be present.
    context : str, default "seed"
        Human-readable context included in validation errors.

    Returns
    -------
    int | None
        The resolved non-negative seed, or ``None`` when optional and unset.

    Raises
    ------
    ValueError
        If no seed is available while ``required=True``.
    ValueError
        If the provided seed is non-integer or negative.
    """
    raw_seed = primary_seed if primary_seed is not None else fallback_seed
    if raw_seed is None:
        if required:
            raise ValueError(f"{context} must be configured")
        return None

    try:
        seed_value = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer seed value") from exc

    if seed_value < 0:
        raise ValueError(f"{context} must be >= 0")

    return seed_value


def build_reproducibility_provenance(
    *,
    seed: int | None,
    deterministic_mode: bool,
    extra: Mapping[str, ReproducibilityValue] | None = None,
) -> dict[str, ReproducibilityValue]:
    """
    Convenience wrapper around :meth:`ReproducibilityHelper.build_provenance`.

    Parameters
    ----------
    seed : int | None
        Seed value used for the run.
    deterministic_mode : bool
        Whether deterministic mode was enabled.
    extra : Mapping[str, ReproducibilityValue] | None, optional
        Additional scalar metadata to merge into the payload.

    Returns
    -------
    dict[str, ReproducibilityValue]
        Provenance payload.
    """
    return ReproducibilityHelper.build_provenance(
        seed=seed,
        deterministic_mode=deterministic_mode,
        extra=extra,
    )


def build_configured_reproducibility_provenance(
    *,
    primary_seed: int | None,
    fallback_seed: int | None = None,
    deterministic_mode: bool,
    context: str = "reproducibility seed",
    require_seed_when_deterministic: bool = True,
    extra: Mapping[str, ReproducibilityValue] | None = None,
) -> dict[str, ReproducibilityValue]:
    """
    Build canonical reproducibility provenance from config-driven seed inputs.

    Parameters
    ----------
    primary_seed : int | None
        Preferred configured seed.
    fallback_seed : int | None, default None
        Optional fallback seed when primary seed is unset.
    deterministic_mode : bool
        Whether deterministic mode is enabled.
    context : str, default "reproducibility seed"
        Error context used during seed validation.
    require_seed_when_deterministic : bool, default True
        Require a resolved seed whenever deterministic mode is enabled.
    extra : Mapping[str, ReproducibilityValue] | None, optional
        Additional scalar payload fields.

    Returns
    -------
    dict[str, ReproducibilityValue]
        Normalized reproducibility provenance payload.
    """
    resolved_seed = resolve_configured_seed(
        primary_seed=primary_seed,
        fallback_seed=fallback_seed,
        required=bool(deterministic_mode and require_seed_when_deterministic),
        context=context,
    )
    payload = build_reproducibility_provenance(
        seed=resolved_seed,
        deterministic_mode=deterministic_mode,
        extra=extra,
    )
    return validate_reproducibility_provenance(
        payload=payload,
        context=context,
        require_seed_when_deterministic=require_seed_when_deterministic,
    )


def validate_reproducibility_provenance(
    *,
    payload: Mapping[str, object],
    context: str = "reproducibility",
    require_seed_when_deterministic: bool = True,
    require_runtime_fields: bool = True,
) -> dict[str, ReproducibilityValue]:
    """
    Validate and normalize a reproducibility provenance payload.

    Parameters
    ----------
    payload : Mapping[str, object]
        Raw provenance payload to validate.
    context : str, default "reproducibility"
        Context label for validation errors.
    require_seed_when_deterministic : bool, default True
        Require a seed when deterministic mode is enabled.
    require_runtime_fields : bool, default True
        Require canonical runtime/version fields.

    Returns
    -------
    dict[str, ReproducibilityValue]
        Normalized provenance payload.

    Raises
    ------
    ValueError
        If required fields are missing or invalid.
    """
    deterministic_mode_raw = payload.get("deterministic_mode")
    if not isinstance(deterministic_mode_raw, bool):
        raise ValueError(f"{context}.deterministic_mode must be bool")

    seed_value = resolve_configured_seed(
        primary_seed=cast(int | None, payload.get("seed")),
        required=False,
        context=f"{context}.seed",
    )
    if bool(deterministic_mode_raw and require_seed_when_deterministic and seed_value is None):
        raise ValueError(f"{context}.seed must be configured when deterministic_mode is true")

    if require_runtime_fields:
        for key in ("python_version", "platform", "numpy_version", "generated_at_utc"):
            raw_value = payload.get(key)
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(f"{context}.{key} must be non-empty string")

        has_torch_raw = payload.get("has_torch")
        if not isinstance(has_torch_raw, bool):
            raise ValueError(f"{context}.has_torch must be bool")

        torch_version_raw = payload.get("torch_version")
        if torch_version_raw is not None and (
            not isinstance(torch_version_raw, str) or not torch_version_raw.strip()
        ):
            raise ValueError(f"{context}.torch_version must be null or non-empty string")

    normalized: dict[str, ReproducibilityValue] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            normalized[str(key)] = value
            continue
        raise ValueError(f"{context}.{key} must be scalar JSON value")

    normalized["seed"] = seed_value
    normalized["deterministic_mode"] = deterministic_mode_raw
    return normalized


def _seed_torch(*, seed: int, deterministic_mode: bool) -> tuple[bool, bool]:
    torch_seeded = False
    torch_cuda_seeded = False
    torch_mod = cast(Any, _torch)
    try:
        torch_mod.manual_seed(seed)
        torch_seeded = True

        cuda = getattr(torch_mod, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        manual_seed_all = getattr(cuda, "manual_seed_all", None)
        if callable(is_available) and bool(is_available()) and callable(manual_seed_all):
            manual_seed_all(seed)
            torch_cuda_seeded = True

        if deterministic_mode:
            use_deterministic_algorithms = getattr(
                torch_mod,
                "use_deterministic_algorithms",
                None,
            )
            if callable(use_deterministic_algorithms):
                use_deterministic_algorithms(True)
            backends = getattr(torch_mod, "backends", None)
            cudnn = getattr(backends, "cudnn", None) if backends is not None else None
            if cudnn is not None and hasattr(cudnn, "deterministic"):
                setattr(cudnn, "deterministic", True)
            if cudnn is not None and hasattr(cudnn, "benchmark"):
                setattr(cudnn, "benchmark", False)
    except Exception as exc:  # pragma: no cover - optional dependency behavior
        logger.debug(
            "torch seed application failed: %s",
            exc,
            extra={"seed": seed, "deterministic_mode": deterministic_mode},
            exc_info=True,
        )
    return torch_seeded, torch_cuda_seeded


def _resolve_torch_version() -> str | None:
    if not HAS_TORCH or _torch is None:
        return None
    version = getattr(_torch, "__version__", None)
    return str(version) if version is not None else None


__all__ = [
    "DeterministicSeedResult",
    "ReproducibilityHelper",
    "ReproducibilityValue",
    "apply_reproducibility_seed",
    "build_configured_reproducibility_provenance",
    "build_reproducibility_provenance",
    "resolve_configured_seed",
    "validate_reproducibility_provenance",
]
