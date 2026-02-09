from __future__ import annotations

import pytest

from ml.common.output_semantics import OutputSemanticsValidator
from ml.common.output_semantics import validate_output_semantics


pytestmark = pytest.mark.unit


def test_output_semantics_validator_when_optional_and_missing_is_valid() -> None:
    validator = OutputSemanticsValidator()

    result = validator.validate(output_schema=None)

    assert result.is_valid is True
    assert result.errors == ()
    assert result.normalized_output_schema is None


def test_output_semantics_validator_when_required_and_missing_returns_error() -> None:
    validator = OutputSemanticsValidator()

    result = validator.validate(output_schema=None, require_output_semantics=True)

    assert result.is_valid is False
    assert result.errors == ("output_schema is required",)


def test_output_semantics_validator_when_schema_not_mapping_returns_error() -> None:
    validator = OutputSemanticsValidator()

    result = validator.validate(output_schema=123)

    assert result.is_valid is False
    assert result.errors == ("output_schema must be a mapping",)


def test_output_semantics_validator_when_payload_valid_normalizes_values() -> None:
    result = validate_output_semantics(
        output_schema={
            "kind": "binary_proba",
            "shape": [None, 2],
            "classes": ["NEG", "POS"],
            "positive_class_index": 1,
        },
        calibration={"kind": "platt", "params": {"coef": 1.2}},
        require_output_semantics=True,
    )

    assert result.is_valid is True
    assert result.errors == ()
    assert result.normalized_output_schema == {
        "kind": "binary_proba",
        "shape": [None, 2],
        "classes": ["NEG", "POS"],
        "positive_class_index": 1,
    }
    assert result.normalized_calibration == {"kind": "platt", "params": {"coef": 1.2}}


def test_output_semantics_validator_when_calibration_not_mapping_returns_error() -> None:
    validator = OutputSemanticsValidator()

    result = validator.validate(
        output_schema={"kind": "binary_proba"},
        calibration=123,
    )

    assert result.is_valid is False
    assert result.errors == ("calibration must be a mapping when provided",)


def test_output_semantics_validator_when_schema_invalid_returns_errors() -> None:
    validator = OutputSemanticsValidator()

    result = validator.validate(
        output_schema={
            "kind": "",
            "shape": "not-a-shape",
            "positive_class_index": -1,
        },
        calibration={"kind": "", "params": 123},
    )

    assert result.is_valid is False
    assert "output_schema.kind must be a non-empty string" in result.errors
    assert "output_schema.shape must be a non-empty sequence of int or None" in result.errors
    assert "output_schema.positive_class_index must be an int >= 0" in result.errors
    assert "calibration.kind must be a non-empty string" in result.errors
    assert "calibration.params must be a mapping when provided" in result.errors


def test_output_semantics_validator_when_classes_not_sequence_returns_error() -> None:
    validator = OutputSemanticsValidator()

    result = validator.validate(
        output_schema={"classes": b"NEG,POS", "positive_class_index": 0},
        require_output_semantics=True,
    )

    assert result.is_valid is False
    assert "output_schema.classes must be a sequence when provided" in result.errors


def test_output_semantics_validator_when_shape_empty_or_negative_returns_error() -> None:
    validator = OutputSemanticsValidator()

    empty_shape = validator.validate(
        output_schema={"shape": []},
        require_output_semantics=True,
    )
    negative_shape = validator.validate(
        output_schema={"shape": [None, -1]},
        require_output_semantics=True,
    )

    assert empty_shape.is_valid is False
    assert "output_schema.shape must be a non-empty sequence of int or None" in empty_shape.errors
    assert negative_shape.is_valid is False
    assert (
        "output_schema.shape must be a non-empty sequence of int or None"
        in negative_shape.errors
    )


def test_output_semantics_validator_when_index_exceeds_classes_returns_error() -> None:
    validator = OutputSemanticsValidator()

    result = validator.validate(
        output_schema={"classes": ["NEG", "POS"], "positive_class_index": 2},
        require_output_semantics=True,
    )

    assert result.is_valid is False
    assert (
        "output_schema.positive_class_index must be within output_schema.classes"
        in result.errors
    )
