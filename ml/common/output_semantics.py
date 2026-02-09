"""
Shared output semantics validation helpers.

This module centralizes validation for model output schema and calibration
metadata so registry and direct-load paths can share one implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard


@dataclass(frozen=True, slots=True)
class OutputSemanticsValidationResult:
    """
    Validation result for output schema + calibration metadata.

    Parameters
    ----------
    is_valid : bool
        ``True`` when no validation errors were found.
    errors : tuple[str, ...]
        Ordered list of validation error messages.
    normalized_output_schema : dict[str, Any] | None
        Normalized output schema payload.
    normalized_calibration : dict[str, Any] | None
        Normalized calibration payload.
    """

    is_valid: bool
    errors: tuple[str, ...]
    normalized_output_schema: dict[str, Any] | None
    normalized_calibration: dict[str, Any] | None


class OutputSemanticsValidator:
    """
    Validate model output semantics in a backward-compatible way.

    Examples
    --------
    >>> validator = OutputSemanticsValidator()
    >>> result = validator.validate(
    ...     output_schema={"kind": "binary_proba", "shape": [None, 1]},
    ...     calibration={"kind": "platt", "params": {"coef": 1.2}},
    ...     require_output_semantics=True,
    ... )
    >>> result.is_valid
    True
    """

    def validate(
        self,
        *,
        output_schema: object,
        calibration: object = None,
        require_output_semantics: bool = False,
    ) -> OutputSemanticsValidationResult:
        """
        Validate output schema and calibration payloads.

        Parameters
        ----------
        output_schema : object
            Output schema payload (expected mapping).
        calibration : object, optional
            Calibration metadata payload (expected mapping).
        require_output_semantics : bool, default False
            Whether missing output schema should be treated as an error.

        Returns
        -------
        OutputSemanticsValidationResult
            Typed validation result with normalized payloads.
        """
        errors: list[str] = []

        normalized_output_schema = _as_mapping(output_schema)
        if output_schema is None:
            if require_output_semantics:
                errors.append("output_schema is required")
        elif normalized_output_schema is None:
            errors.append("output_schema must be a mapping")
        else:
            _validate_output_schema(normalized_output_schema, errors)

        normalized_calibration = _as_mapping(calibration)
        if calibration is not None and normalized_calibration is None:
            errors.append("calibration must be a mapping when provided")
        elif normalized_calibration is not None:
            _validate_calibration(normalized_calibration, errors)

        return OutputSemanticsValidationResult(
            is_valid=not errors,
            errors=tuple(errors),
            normalized_output_schema=normalized_output_schema,
            normalized_calibration=normalized_calibration,
        )


def validate_output_semantics(
    *,
    output_schema: object,
    calibration: object = None,
    require_output_semantics: bool = False,
) -> OutputSemanticsValidationResult:
    """
    Convenience function wrapper for output semantics validation.

    Parameters
    ----------
    output_schema : object
        Output schema payload (expected mapping).
    calibration : object, optional
        Calibration metadata payload (expected mapping).
    require_output_semantics : bool, default False
        Whether missing output schema should be treated as an error.

    Returns
    -------
    OutputSemanticsValidationResult
        Validation result from :class:`OutputSemanticsValidator`.
    """
    validator = OutputSemanticsValidator()
    return validator.validate(
        output_schema=output_schema,
        calibration=calibration,
        require_output_semantics=require_output_semantics,
    )


def _as_mapping(value: object) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _validate_output_schema(schema: Mapping[str, Any], errors: list[str]) -> None:
    if "kind" in schema:
        kind = schema["kind"]
        if not isinstance(kind, str) or not kind.strip():
            errors.append("output_schema.kind must be a non-empty string")

    if "shape" in schema and not _is_valid_shape(schema["shape"]):
        errors.append("output_schema.shape must be a non-empty sequence of int or None")

    classes: Sequence[Any] | None = None
    if "classes" in schema:
        raw_classes = schema["classes"]
        if _is_sequence_payload(raw_classes):
            classes = raw_classes
        else:
            errors.append("output_schema.classes must be a sequence when provided")

    if "positive_class_index" in schema:
        index = schema["positive_class_index"]
        if not isinstance(index, int) or index < 0:
            errors.append("output_schema.positive_class_index must be an int >= 0")
        elif classes is not None and index >= len(classes):
            errors.append(
                "output_schema.positive_class_index must be within output_schema.classes",
            )


def _validate_calibration(calibration: Mapping[str, Any], errors: list[str]) -> None:
    if "kind" in calibration:
        kind = calibration["kind"]
        if not isinstance(kind, str) or not kind.strip():
            errors.append("calibration.kind must be a non-empty string")
    if "params" in calibration and not isinstance(calibration["params"], Mapping):
        errors.append("calibration.params must be a mapping when provided")


def _is_valid_shape(shape: object) -> bool:
    if not _is_sequence_payload(shape):
        return False
    if len(shape) == 0:
        return False
    for dim in shape:
        if dim is None:
            continue
        if not isinstance(dim, int) or dim < 0:
            return False
    return True


def _is_sequence_payload(value: object) -> TypeGuard[Sequence[Any]]:
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, Sequence)


__all__ = [
    "OutputSemanticsValidationResult",
    "OutputSemanticsValidator",
    "validate_output_semantics",
]
