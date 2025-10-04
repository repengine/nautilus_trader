"""Tests for macro coverage validator."""

from __future__ import annotations

import polars as pl
import pytest

from ml.data.validation import MacroCoverageError
from ml.data.validation import MacroCoverageValidator


def test_macro_coverage_validator_success() -> None:
    """Validator should accept fully populated macro columns."""

    df = pl.DataFrame({
        "DGS10": [3.0, 3.1, 3.2],
        "DGS2": [2.0, 2.1, 2.2],
    })

    validator = MacroCoverageValidator(min_coverage=0.9)
    coverage = validator.validate_macro_coverage(df, ["DGS10", "DGS2"])

    assert pytest.approx(coverage["DGS10"], abs=1e-9) == 1.0
    assert pytest.approx(coverage["DGS2"], abs=1e-9) == 1.0


def test_macro_coverage_validator_missing_series_raises() -> None:
    """Missing macro columns should raise a coverage error."""

    df = pl.DataFrame({
        "DGS10": [3.0, 3.1, 3.2],
    })

    validator = MacroCoverageValidator()
    with pytest.raises(MacroCoverageError):
        validator.validate_macro_coverage(df, ["DGS10", "DGS2"])


def test_macro_coverage_validator_sparse_values_raise() -> None:
    """Sparse macro coverage should trigger an error when below threshold."""

    df = pl.DataFrame({
        "DGS10": [3.0, None, None, 3.1],
        "DGS2": [2.0, 2.1, 2.2, 2.3],
    })

    validator = MacroCoverageValidator(min_coverage=0.75)
    with pytest.raises(MacroCoverageError) as exc:
        validator.validate_macro_coverage(df, ["DGS10", "DGS2"])

    assert "sparse" in str(exc.value)
