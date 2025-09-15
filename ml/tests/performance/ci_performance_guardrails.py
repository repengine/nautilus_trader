#!/usr/bin/env python3
"""
CI Performance Guardrails Runner.

This script runs performance guardrail tests and ensures CI fails if any
performance regressions are detected. It integrates with the existing
CI pipeline to enforce production reliability requirements.

Usage:
    python ci_performance_guardrails.py [--strict] [--report-file <path>]

Exit codes:
    0: All guardrails passed
    1: Performance regressions detected (fails CI)
    2: Test execution failed
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class PerformanceGuardrailRunner:
    """
    Runner for ML performance guardrail tests in CI environments.
    """

    def __init__(self, strict: bool = False, report_file: Path | None = None):
        """
        Initialize the guardrail runner.

        Parameters
        ----------
        strict : bool
            If True, use stricter performance requirements
        report_file : Path, optional
            Path to write performance report JSON
        """
        self.strict = strict
        self.report_file = report_file
        self.results: dict[str, Any] = {
            "timestamp": time.time(),
            "environment": self._get_environment_info(),
            "tests": [],
            "summary": {},
        }

    def _get_environment_info(self) -> dict[str, Any]:
        """Get CI environment information."""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "ci": bool(os.getenv("CI")),
            "github_actions": bool(os.getenv("GITHUB_ACTIONS")),
            "pytest_xdist": bool(os.getenv("PYTEST_XDIST_WORKER")),
            "ml_bench_relax": float(os.getenv("ML_BENCH_RELAX", "1.0")),
            "strict_mode": self.strict,
        }

    def run_guardrail_tests(self) -> bool:
        """
        Run all performance guardrail tests.

        Returns
        -------
        bool
            True if all tests passed, False if any failed
        """
        print("🚀 Running ML Performance Guardrails...")
        print(f"Environment: {self.results['environment']}")
        print("=" * 80)

        # Configure environment for strict mode
        env = os.environ.copy()
        if self.strict:
            env["ML_BENCH_RELAX"] = "0.8"  # Stricter requirements
            print("⚡ STRICT MODE: Using tighter performance requirements")

        # Performance test categories
        class _Category(TypedDict):
            name: str
            pattern: str
            critical: bool

        test_categories: list[_Category] = [
            {
                "name": "Feature Computation Guardrails",
                "pattern": "test_parity_buffer_guardrails.py::TestFeatureComputationGuardrails",
                "critical": True,
            },
            {
                "name": "Feature Parity Guardrails",
                "pattern": "test_parity_buffer_guardrails.py::TestFeatureParityGuardrails",
                "critical": True,
            },
            {
                "name": "Model Inference Guardrails",
                "pattern": "test_parity_buffer_guardrails.py::TestModelInferenceGuardrails",
                "critical": True,
            },
            {
                "name": "End-to-End Guardrails",
                "pattern": "test_parity_buffer_guardrails.py::TestEndToEndGuardrails",
                "critical": True,
            },
            {
                "name": "Buffer Reuse Guardrails",
                "pattern": "test_parity_buffer_guardrails.py::TestBufferReuseGuardrails",
                "critical": False,
            },
            {
                "name": "Performance Regression Detection",
                "pattern": "test_parity_buffer_guardrails.py::TestPerformanceRegressionGuardrails",
                "critical": True,
            },
        ]

        all_passed = True
        test_dir = Path(__file__).parent

        for category in test_categories:
            print(f"\n📊 {category['name']}")
            print("-" * 60)

            # Run pytest for this category
            cmd: list[str] = [
                sys.executable, "-m", "pytest",
                str(test_dir / category["pattern"]),
                "-v", "--tb=short",
                "-m", "performance",
                "--maxfail=1",  # Stop on first failure for critical tests
                "--disable-warnings",
            ]

            # Add additional flags for CI stability
            if os.getenv("CI"):
                cmd.extend([
                    "--timeout=300",  # 5 minute timeout per test
                    "--timeout-method=thread",
                ])

            start_time = time.time()
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            duration = time.time() - start_time

            # Parse results
            test_result = {
                "category": category["name"],
                "pattern": category["pattern"],
                "critical": category["critical"],
                "passed": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            self.results["tests"].append(test_result)

            if test_result["passed"]:
                print(f"✅ {category['name']} PASSED ({duration:.1f}s)")
            else:
                print(f"❌ {category['name']} FAILED ({duration:.1f}s)")
                if category["critical"]:
                    print("🚨 CRITICAL TEST FAILURE - Will fail CI")
                    all_passed = False
                    # Print failure details
                    print("\nFailure Details:")
                    print(result.stdout[-1000:])  # Last 1000 chars
                    if result.stderr:
                        print("\nErrors:")
                        print(result.stderr[-1000:])

        # Generate summary
        self._generate_summary()

        # Write report if requested
        if self.report_file:
            self._write_report()

        return all_passed

    def _generate_summary(self) -> None:
        """Generate test execution summary."""
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for t in self.results["tests"] if t["passed"])
        failed_tests = total_tests - passed_tests
        critical_failures = sum(
            1 for t in self.results["tests"]
            if not t["passed"] and t["critical"]
        )

        total_duration = sum(t["duration"] for t in self.results["tests"])

        self.results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "critical_failures": critical_failures,
            "total_duration": total_duration,
            "overall_passed": failed_tests == 0,
        }

        # Print summary
        print("\n" + "=" * 80)
        print("📈 PERFORMANCE GUARDRAILS SUMMARY")
        print("=" * 80)
        print(f"Total test categories: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Critical failures: {critical_failures}")
        print(f"Total duration: {total_duration:.1f}s")

        if self.results["summary"]["overall_passed"]:
            print("\n🎉 ALL PERFORMANCE GUARDRAILS PASSED")
            print("✅ Production reliability requirements satisfied")
        else:
            print(f"\n💥 {failed_tests} PERFORMANCE GUARDRAIL(S) FAILED")
            if critical_failures > 0:
                print(f"🚨 {critical_failures} CRITICAL FAILURE(S) - CI MUST FAIL")
            print("❌ Production reliability requirements NOT met")

    def _write_report(self) -> None:
        """Write performance report to file."""
        if not self.report_file:
            return

        try:
            self.report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.report_file, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"\n📄 Performance report written to: {self.report_file}")
        except Exception as e:
            print(f"\n⚠️  Failed to write report: {e}")

    def run_zero_allocation_validation(self) -> bool:
        """
        Run specialized zero-allocation validation tests.

        Returns
        -------
        bool
            True if validation passed
        """
        print("\n🔍 Running Zero-Allocation Validation...")
        print("-" * 60)

        test_dir = Path(__file__).parent
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_dir / "test_zero_allocation.py"),
            "-v", "--tb=short",
            "-k", "allocation",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Zero-allocation validation PASSED")
            return True
        else:
            print("❌ Zero-allocation validation FAILED")
            print(result.stdout[-500:])
            return False


def main() -> int:
    """
    Main entry point for CI performance guardrails.

    Returns
    -------
    int
        Exit code (0=success, 1=performance failure, 2=execution error)
    """
    parser = argparse.ArgumentParser(
        description="Run ML performance guardrail tests for CI"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use stricter performance requirements"
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        help="Path to write performance report JSON"
    )
    parser.add_argument(
        "--zero-allocation-only",
        action="store_true",
        help="Run only zero-allocation validation tests"
    )

    args = parser.parse_args()

    try:
        runner = PerformanceGuardrailRunner(
            strict=args.strict,
            report_file=args.report_file
        )

        if args.zero_allocation_only:
            success = runner.run_zero_allocation_validation()
        else:
            success = runner.run_guardrail_tests()

            # Also run zero-allocation validation
            if success:
                success = runner.run_zero_allocation_validation()

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        return 2
    except Exception as e:
        print(f"\n💥 Execution error: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
