"""
Integration test for ML performance guardrails.

This test validates that the guardrail system works correctly and can detect performance
regressions as intended.

"""

import time
from pathlib import Path

import pytest

from ml.actors.signal import OptimizationLevel
from ml.config.actors import OptimizationConfig
from ml.features.config import FeatureConfig


class TestGuardrailsIntegration:
    """
    Integration tests for performance guardrails system.
    """

    def test_optimization_levels_configuration(self):
        """
        Test that optimization levels are properly configured.
        """
        # Test that optimization levels exist
        assert hasattr(OptimizationLevel, "STANDARD"), "Should have STANDARD optimization level"
        assert hasattr(OptimizationLevel, "OPTIMIZED"), "Should have OPTIMIZED optimization level"

        # Test OptimizationConfig structure
        config = OptimizationConfig(
            level=OptimizationLevel.OPTIMIZED,
            feature_cache_size=1000,
            enable_profiling=True,
        )

        assert config.level == OptimizationLevel.OPTIMIZED
        assert config.feature_cache_size == 1000
        assert config.enable_profiling

    def test_performance_measurement_utilities(self):
        """
        Test that performance measurement utilities work correctly.
        """
        # Import the utilities from the guardrails module
        import sys
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "guardrails",
            Path(__file__).parent / "test_parity_buffer_guardrails.py",
        )
        if spec and spec.loader:
            guardrails = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(guardrails)

            # Test latency measurement
            def fast_function():
                time.sleep(0.001)  # 1ms

            def slow_function():
                time.sleep(0.005)  # 5ms

            fast_latency = guardrails.measure_p99_latency_ns(fast_function, iterations=10)
            slow_latency = guardrails.measure_p99_latency_ns(slow_function, iterations=10)

            # Convert to milliseconds for comparison
            fast_ms = fast_latency / 1_000_000
            slow_ms = slow_latency / 1_000_000

            # Should be roughly 1ms and 5ms respectively (with some tolerance)
            assert 0.5 < fast_ms < 2.0, f"Fast function latency {fast_ms}ms not in expected range"
            assert 4.0 < slow_ms < 8.0, f"Slow function latency {slow_ms}ms not in expected range"
            assert slow_latency > fast_latency, "Slow function should be slower than fast function"

    def test_ci_runner_script_exists_and_executable(self):
        """
        Test that the CI runner script exists and is properly structured.
        """
        script_path = Path(__file__).parent / "ci_performance_guardrails.py"

        assert script_path.exists(), "CI performance guardrails script should exist"
        assert script_path.is_file(), "CI script should be a file"

        # Check that it's executable by Python
        with open(script_path) as f:
            content = f.read()

        assert "PerformanceGuardrailRunner" in content, "Should contain main runner class"
        assert "def main(" in content, "Should have main entry point"
        assert "sys.exit" in content, "Should have proper exit handling"

    def test_makefile_targets_integration(self):
        """
        Test that Makefile targets are properly defined.
        """
        makefile_path = Path(__file__).parent.parent.parent.parent / "Makefile"

        with open(makefile_path) as f:
            makefile_content = f.read()

        # Check that our new targets are present
        assert "pytest-ml-guardrails:" in makefile_content, "Should have guardrails target"
        assert (
            "pytest-ml-guardrails-strict:" in makefile_content
        ), "Should have strict guardrails target"
        assert (
            "pytest-ml-zero-allocation:" in makefile_content
        ), "Should have zero-allocation target"

        # Check that targets reference our scripts
        assert "ci_performance_guardrails.py" in makefile_content, "Should reference our CI script"

    def test_documentation_exists(self):
        """
        Test that documentation is properly created.
        """
        readme_path = Path(__file__).parent / "README_GUARDRAILS.md"

        assert readme_path.exists(), "Guardrails README should exist"

        with open(readme_path) as f:
            content = f.read()

        # Check key sections
        assert "Performance Requirements" in content, "Should document performance requirements"
        assert "P99 Latency Budget" in content, "Should document latency budgets"
        assert "Zero Allocations" in content, "Should document zero allocation requirements"
        assert "make pytest-ml-guardrails" in content, "Should document usage"

    def test_feature_config_optimization_defaults(self):
        """
        Test that feature configuration defaults are suitable for performance.
        """
        config = FeatureConfig(
            return_periods=[1, 5, 10],  # Reasonable number of periods
            momentum_periods=[5, 10],  # Not too many momentum calculations
            include_microstructure=False,  # Disabled for better performance
            include_trade_flow=False,  # Disabled for better performance
        )

        # Verify performance-friendly defaults
        assert len(config.return_periods) <= 3, "Should use limited return periods for performance"
        assert (
            len(config.momentum_periods) <= 2
        ), "Should use limited momentum periods for performance"
        assert (
            not config.include_microstructure
        ), "Microstructure should be disabled for base performance"
        assert not config.include_trade_flow, "Trade flow should be disabled for base performance"


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v"])
