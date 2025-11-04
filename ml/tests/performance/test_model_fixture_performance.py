"""Performance tests for model fixture scope promotion.

Verify that promoting model fixtures to session/module scope achieves
target ≥2s speedup through benchmark comparisons.
"""

import pytest


@pytest.mark.skip("Implementation pending")
def test_dummy_onnx_model_creation_benchmark() -> None:
    """Benchmark ONNX model creation time.

    Measure time to create ONNX model once, establishing baseline for
    speedup calculation (~75ms expected).
    """


@pytest.mark.skip("Implementation pending")
def test_dummy_onnx_model_speedup_vs_function_scope() -> None:
    """Verify session-scoped ONNX model achieves ≥10x speedup.

    Compare 10 test invocations with function-scoped (10 × 75ms = 750ms)
    vs session-scoped (75ms + 10 × <1ms ≈ 85ms) fixture creation.

    Target: ≥10x speedup (session-scoped ≤75ms total for 10 tests).
    """


@pytest.mark.skip("Implementation pending")
def test_xgboost_model_creation_benchmark() -> None:
    """Benchmark XGBoost model creation time.

    Measure time to create and train XGBoost model once, establishing
    baseline for speedup calculation (~75ms expected).
    """


@pytest.mark.skip("Implementation pending")
def test_total_suite_speedup() -> None:
    """Verify total test suite achieves ≥2s speedup.

    Compare full test suite runtime before/after fixture scope promotion:
    - dummy_onnx_model: 28 tests × 75ms = 2.1s → 75ms = ~2s saved
    - xgboost_test_model: 2 tests × 75ms = 150ms → 75ms = ~75ms saved

    Total target: ≥2s faster test suite.
    """
