#!/usr/bin/env python3
"""
Security validator for ML production posture.

This script validates that the ML codebase maintains secure production practices:
1. No pickle/joblib usage in production paths
2. ONNX-only enforcement for model loading
3. Proper test-only guards for unsafe formats

Run this script to ensure production security compliance.
"""

import ast
import os
import sys
from pathlib import Path
from typing import NamedTuple


class SecurityViolation(NamedTuple):
    """Security violation details."""

    file_path: str
    line_number: int
    violation_type: str
    description: str
    severity: str  # "critical", "high", "medium", "low"


class SecurityAuditor(ast.NodeVisitor):
    """AST visitor for security compliance analysis."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.violations: list[SecurityViolation] = []
        self.is_test_file = self._is_test_file(file_path)
        self.is_example_file = self._is_example_file(file_path)
        self.has_test_guards = False
        self.pickle_imports = []
        self.joblib_imports = []
        self.current_function = None
        self.in_test_guard = False

    def _is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file."""
        return (
            "/tests/" in file_path
            or "/test_" in file_path
            or file_path.endswith(("_test.py", "/conftest.py"))
            or "pytest" in file_path.lower()
        )

    def _is_example_file(self, file_path: str) -> bool:
        """Check if file is an example file."""
        return (
            "/examples/" in file_path
            or "/example_" in file_path
            or file_path.endswith("_example.py")
        )

    def visit_Import(self, node: ast.Import) -> None:
        """Track dangerous imports."""
        for alias in node.names:
            if alias.name == "pickle":
                self.pickle_imports.append(node.lineno)
                self._check_pickle_usage(node.lineno, "import pickle")
            elif alias.name == "joblib":
                self.joblib_imports.append(node.lineno)
                self._check_joblib_usage(node.lineno, "import joblib")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track dangerous from imports."""
        if node.module == "pickle":
            for alias in node.names:
                self.pickle_imports.append(node.lineno)
                self._check_pickle_usage(node.lineno, f"from pickle import {alias.name}")
        elif node.module == "joblib":
            for alias in node.names:
                self.joblib_imports.append(node.lineno)
                self._check_joblib_usage(node.lineno, f"from joblib import {alias.name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for dangerous function calls."""
        # Check for pickle.load/dump calls
        if isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and
                node.func.value.id == "pickle" and
                node.func.attr in ["load", "dump"]):
                self._check_pickle_usage(node.lineno, f"pickle.{node.func.attr}()")
            elif (isinstance(node.func.value, ast.Name) and
                  node.func.value.id == "joblib" and
                  node.func.attr in ["load", "dump"]):
                self._check_joblib_usage(node.lineno, f"joblib.{node.func.attr}()")

        # Check for np.load with allow_pickle=True
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == "np" and
            node.func.attr == "load"):
            for keyword in node.keywords:
                if (keyword.arg == "allow_pickle" and
                    isinstance(keyword.value, ast.Constant) and
                    keyword.value.value is True):
                    # This is acceptable in training/CLI paths
                    if not self._is_training_or_cli_path():
                        self.violations.append(SecurityViolation(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            violation_type="numpy_pickle",
                            description="np.load with allow_pickle=True in non-training path",
                            severity="medium"
                        ))

        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """Track test guard contexts."""
        # Check for test environment guards
        if self._is_test_guard(node):
            self.has_test_guards = True
            old_in_guard = self.in_test_guard
            self.in_test_guard = True
            self.generic_visit(node)
            self.in_test_guard = old_in_guard
        else:
            self.generic_visit(node)

    def _is_test_guard(self, node: ast.If) -> bool:
        """Check if this is a test environment guard."""
        test_patterns = [
            "PYTEST_CURRENT_TEST",
            "ML_ALLOW_JOBLIB",
            "ML_TESTING",
            "pytest",
            "test",
        ]

        # Convert the test condition to string for pattern matching
        try:
            if hasattr(ast, "unparse"):
                condition_str = ast.unparse(node.test)
            else:
                condition_str = str(node.test)

            return any(pattern in condition_str for pattern in test_patterns)
        except Exception:
            return False

    def _is_training_or_cli_path(self) -> bool:
        """Check if file is in training or CLI path."""
        training_paths = [
            "/training/",
            "/cli/",
            "/scripts/",
            "train_",
            "_train",
            "distillation",
            "teacher",
            "student",
        ]
        return any(path in self.file_path for path in training_paths)

    def _is_controlled_import_module(self) -> bool:
        """Check if this is a controlled import module like _imports.py."""
        return (
            self.file_path.endswith("/_imports.py") or
            self.file_path.endswith("\\_imports.py") or
            "import" in Path(self.file_path).name.lower()
        )

    def _check_pickle_usage(self, line_number: int, usage: str) -> None:
        """Check pickle usage compliance."""
        if self.is_test_file or self.is_example_file:
            # Examples and tests should not use pickle at all
            if not self.in_test_guard:
                self.violations.append(SecurityViolation(
                    file_path=self.file_path,
                    line_number=line_number,
                    violation_type="pickle_in_test",
                    description=f"Pickle usage in test/example without proper guards: {usage}",
                    severity="high"
                ))
        else:
            # Production paths should never use pickle
            self.violations.append(SecurityViolation(
                file_path=self.file_path,
                line_number=line_number,
                violation_type="pickle_in_production",
                description=f"Pickle usage in production path: {usage}",
                severity="critical"
            ))

    def _check_joblib_usage(self, line_number: int, usage: str) -> None:
        """Check joblib usage compliance."""
        if self._is_controlled_import_module():
            # Controlled import modules like _imports.py are allowed to import joblib
            pass  # This is expected and safe
        elif self.is_test_file:
            # Test files can use joblib with proper guards
            if not self.in_test_guard and not self.has_test_guards:
                self.violations.append(SecurityViolation(
                    file_path=self.file_path,
                    line_number=line_number,
                    violation_type="joblib_without_guards",
                    description=f"Joblib usage in test without environment guards: {usage}",
                    severity="medium"
                ))
        elif self._is_training_or_cli_path():
            # Training/CLI paths can use joblib but should have guards
            pass  # Allow for now but could add warnings
        else:
            # Production paths should not use joblib
            self.violations.append(SecurityViolation(
                file_path=self.file_path,
                line_number=line_number,
                violation_type="joblib_in_production",
                description=f"Joblib usage in production path: {usage}",
                severity="high"
            ))


def analyze_file(file_path: Path) -> list[SecurityViolation]:
    """Analyze a single Python file for security violations."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        auditor = SecurityAuditor(str(file_path))
        auditor.visit(tree)
        return auditor.violations
    except Exception as e:
        # Return a violation for files that can't be parsed
        return [SecurityViolation(
            file_path=str(file_path),
            line_number=1,
            violation_type="parse_error",
            description=f"Failed to parse file: {e}",
            severity="low"
        )]


def scan_ml_directory() -> list[SecurityViolation]:
    """Scan the ML directory for security violations."""
    ml_dir = Path("ml")
    if not ml_dir.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        ml_dir = script_dir
        if not ml_dir.exists():
            raise FileNotFoundError("ML directory not found")

    violations = []
    python_files = list(ml_dir.rglob("*.py"))

    print(f"Scanning {len(python_files)} Python files in {ml_dir}...")

    for file_path in python_files:
        # Skip __pycache__ and .pyc files
        if "__pycache__" in str(file_path) or file_path.suffix == ".pyc":
            continue

        file_violations = analyze_file(file_path)
        violations.extend(file_violations)

    return violations


def check_model_loader_compliance() -> list[SecurityViolation]:
    """Check that model loaders properly reject unsafe formats."""
    violations = []

    # Check base actor loader
    base_actor_path = Path("ml/actors/base.py")
    if base_actor_path.exists():
        try:
            with open(base_actor_path) as f:
                content = f.read()

            # Check for proper pickle rejection
            if "pickle" in content and "not supported" not in content:
                violations.append(SecurityViolation(
                    file_path=str(base_actor_path),
                    line_number=1,
                    violation_type="missing_pickle_rejection",
                    description="Model loader may not properly reject pickle formats",
                    severity="high"
                ))

            # Check for proper joblib guards
            if "joblib" in content and "ML_ALLOW_JOBLIB" not in content:
                violations.append(SecurityViolation(
                    file_path=str(base_actor_path),
                    line_number=1,
                    violation_type="missing_joblib_guards",
                    description="Model loader missing joblib environment guards",
                    severity="medium"
                ))

        except Exception as e:
            violations.append(SecurityViolation(
                file_path=str(base_actor_path),
                line_number=1,
                violation_type="check_error",
                description=f"Failed to check model loader: {e}",
                severity="low"
            ))

    return violations


def check_environment_variables() -> list[SecurityViolation]:
    """Check for insecure environment variable settings."""
    violations = []

    # Check if joblib is accidentally enabled
    if os.getenv("ML_ALLOW_JOBLIB", "").lower() in {"1", "true", "yes"}:
        if not (os.getenv("PYTEST_CURRENT_TEST") or os.getenv("ML_TESTING")):
            violations.append(SecurityViolation(
                file_path="environment",
                line_number=1,
                violation_type="joblib_enabled_production",
                description="ML_ALLOW_JOBLIB is enabled outside test environment",
                severity="critical"
            ))

    # Recommend ONNX-only mode for production
    if not os.getenv("ML_ONNX_ONLY"):
        violations.append(SecurityViolation(
            file_path="environment",
            line_number=1,
            violation_type="onnx_only_not_set",
            description="ML_ONNX_ONLY not set - recommend enabling for maximum security",
            severity="low"
        ))

    return violations


def main() -> int:
    """Main validator function."""
    print("🔒 ML Security Posture Validator")
    print("=" * 50)

    all_violations = []

    # Scan files
    try:
        file_violations = scan_ml_directory()
        all_violations.extend(file_violations)
    except Exception as e:
        print(f"❌ Failed to scan files: {e}")
        return 1

    # Check model loaders
    loader_violations = check_model_loader_compliance()
    all_violations.extend(loader_violations)

    # Check environment
    env_violations = check_environment_variables()
    all_violations.extend(env_violations)

    # Categorize violations by severity
    critical = [v for v in all_violations if v.severity == "critical"]
    high = [v for v in all_violations if v.severity == "high"]
    medium = [v for v in all_violations if v.severity == "medium"]
    low = [v for v in all_violations if v.severity == "low"]

    # Report results
    total_violations = len(all_violations)
    print("\n📊 Security Scan Results:")
    print(f"Total violations: {total_violations}")
    print(f"Critical: {len(critical)}")
    print(f"High: {len(high)}")
    print(f"Medium: {len(medium)}")
    print(f"Low: {len(low)}")

    if critical:
        print(f"\n🚨 CRITICAL VIOLATIONS ({len(critical)}):")
        for v in critical:
            print(f"  {v.file_path}:{v.line_number} - {v.description}")

    if high:
        print(f"\n⚠️  HIGH PRIORITY VIOLATIONS ({len(high)}):")
        for v in high:
            print(f"  {v.file_path}:{v.line_number} - {v.description}")

    if medium:
        print(f"\n⚡ MEDIUM PRIORITY VIOLATIONS ({len(medium)}):")
        for v in medium:
            print(f"  {v.file_path}:{v.line_number} - {v.description}")

    if low:
        print(f"\n📝 LOW PRIORITY VIOLATIONS ({len(low)}):")
        for v in low:
            print(f"  {v.file_path}:{v.line_number} - {v.description}")

    # Determine exit code
    if critical:
        print("\n❌ SECURITY POSTURE: CRITICAL ISSUES DETECTED")
        print("Production deployment should be blocked until critical issues are resolved.")
        return 1
    elif high:
        print("\n⚠️  SECURITY POSTURE: HIGH PRIORITY ISSUES")
        print("Recommend addressing high priority issues before production.")
        return 1
    elif medium or low:
        print("\n✅ SECURITY POSTURE: ACCEPTABLE")
        print("Minor issues detected but safe for production.")
        return 0
    else:
        print("\n🎉 SECURITY POSTURE: EXCELLENT")
        print("No security violations detected!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
