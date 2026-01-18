#!/usr/bin/env python3
"""
Validate pytest plug-in adoption plus fixture export guards.

The ML test suites rely on ``ml.tests.fixtures.pytest_plugins`` for canonical
fixtures.  This script keeps two invariants in place:

1. Every ``ml/tests/**/test_*.py`` belongs to a package (or module) that
   registers the shared plug-in.
2. The fixture export guard (`ml/tests/fixtures/test_exports.py`) stays in sync,
   so adding new fixture modules forces the canonical export list to update.

The script exits non-zero when either invariant is violated.
It also enforces fixture-guide alignment by prohibiting direct fixture
imports in tests (except helper shims) and disallowing pytest plug-in
declarations in non-top-level conftest files.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import ast
import importlib
import inspect
import re
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "ml" / "tests"
PLUGIN_PATTERN = re.compile(
    r'pytest_plugins\s*=\s*\(\s*["\']ml\.tests\.fixtures\.pytest_plugins["\']\s*,\s*\)',
)
PLUGIN_EXCLUDES = frozenset({"fixtures"})
ALLOWED_RUNTIME_FIXTURE_IMPORTS = frozenset({"ml.tests.fixtures.pandera"})


def _iter_test_packages(root: Path) -> Iterable[Path]:
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue
        if directory.name == "__pycache__":
            continue
        if not any(child.suffix == ".py" for child in directory.iterdir() if child.is_file()):
            continue
        rel = directory.relative_to(root)
        if rel.parts and rel.parts[0] in PLUGIN_EXCLUDES:
            continue
        yield directory


def _iter_test_modules(root: Path) -> Iterable[Path]:
    for file_path in root.rglob("test_*.py"):
        rel = file_path.relative_to(root)
        if rel.parts and rel.parts[0] in PLUGIN_EXCLUDES:
            continue
        yield file_path


def _iter_conftest_files(root: Path) -> Iterable[Path]:
    for file_path in root.rglob("conftest.py"):
        rel = file_path.relative_to(root)
        if rel.parts and rel.parts[0] in PLUGIN_EXCLUDES:
            continue
        yield file_path


def _top_level_package(file_path: Path, root: Path) -> str:
    rel = file_path.relative_to(root)
    if len(rel.parts) <= 1:
        return "."
    return rel.parts[0]


def _format_package_name(name: str) -> str:
    return "(root)" if name == "." else name


def _load_plugin_directories(root: Path) -> set[Path]:
    plugin_dirs: set[Path] = set()
    for init_file in root.rglob("__init__.py"):
        try:
            text = init_file.read_text()
        except UnicodeDecodeError:
            continue
        if PLUGIN_PATTERN.search(text):
            plugin_dirs.add(init_file.parent.resolve())
    return plugin_dirs


def _file_has_plugin(file_path: Path) -> bool:
    try:
        text = file_path.read_text()
    except UnicodeDecodeError:
        return False
    return PLUGIN_PATTERN.search(text) is not None


def _contains_type_checking(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == "TYPE_CHECKING":
            return True
        if isinstance(child, ast.Attribute) and child.attr == "TYPE_CHECKING":
            return True
    return False


def _collect_direct_fixture_imports(root: Path) -> list[str]:
    violations: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            super().__init__()
            self._type_checking_stack: list[bool] = [False]

        @property
        def _in_type_checking(self) -> bool:
            return any(self._type_checking_stack)

        def visit_If(self, node: ast.If) -> None:
            is_type_checking = _contains_type_checking(node.test)
            self._type_checking_stack.append(self._in_type_checking or is_type_checking)
            for child in node.body:
                self.visit(child)
            self._type_checking_stack.pop()

            self._type_checking_stack.append(self._in_type_checking)
            for child in node.orelse:
                self.visit(child)
            self._type_checking_stack.pop()

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = node.module or ""
            if not module.startswith("ml.tests.fixtures"):
                return
            if self._in_type_checking:
                return
            if module in ALLOWED_RUNTIME_FIXTURE_IMPORTS:
                return
            violations.append(
                f"{current_file}:{node.lineno}: direct fixture import ({module})",
            )

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if not alias.name.startswith("ml.tests.fixtures"):
                    continue
                if self._in_type_checking:
                    continue
                if alias.name in ALLOWED_RUNTIME_FIXTURE_IMPORTS:
                    continue
                violations.append(
                    f"{current_file}:{node.lineno}: direct fixture import ({alias.name})",
                )

    for file_path in _iter_test_modules(root):
        current_file = file_path.relative_to(REPO_ROOT)
        try:
            tree = ast.parse(file_path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        visitor = _Visitor()
        visitor.visit(tree)

    return violations


def _collect_nested_conftest_plugins(root: Path) -> list[str]:
    violations: list[str] = []
    root_conftest = root / "conftest.py"
    for file_path in _iter_conftest_files(root):
        if file_path == root_conftest:
            continue
        if _file_has_plugin(file_path):
            violations.append(f"{file_path.relative_to(REPO_ROOT)} declares pytest_plugins")
    return violations


def _is_package_covered(file_path: Path, plugin_dirs: set[Path], root: Path) -> bool:
    current = file_path.parent.resolve()
    while True:
        if current in plugin_dirs:
            return True
        if current == root:
            break
        current = current.parent
    return False


def _collect_missing_plugin_files(plugin_dirs: set[Path]) -> list[Path]:
    missing: list[Path] = []
    for file_path in _iter_test_modules(TESTS_ROOT):
        if _file_has_plugin(file_path):
            continue
        if _is_package_covered(file_path, plugin_dirs, TESTS_ROOT.resolve()):
            continue
        missing.append(file_path)
    return missing


def _collect_package_bootstrap_violations(root: Path) -> list[str]:
    violations: list[str] = []
    for package_dir in _iter_test_packages(root):
        init_file = package_dir / "__init__.py"
        rel = init_file.relative_to(REPO_ROOT)
        if not init_file.exists():
            violations.append(f"{rel} missing canonical pytest plug-in declaration")
            continue
        try:
            text = init_file.read_text()
        except UnicodeDecodeError:
            violations.append(f"{rel} is not readable as text")
            continue
        if not PLUGIN_PATTERN.search(text):
            violations.append(f"{rel} missing `pytest_plugins = (\"ml.tests.fixtures.pytest_plugins\",)`")
    return violations


def _count_plugin_declarations(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for file_path in root.rglob("*.py"):
        try:
            text = file_path.read_text()
        except UnicodeDecodeError:
            continue
        if not PLUGIN_PATTERN.search(text):
            continue
        top_level = _top_level_package(file_path, root)
        counts[top_level] += 1
    return counts


def _count_test_modules_by_package(root: Path) -> Counter[str]:
    totals: Counter[str] = Counter()
    for file_path in _iter_test_modules(root):
        top_level = _top_level_package(file_path, root)
        totals[top_level] += 1
    return totals


def _count_missing_by_package(missing_files: Iterable[Path], root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for file_path in missing_files:
        top_level = _top_level_package(file_path, root)
        counts[top_level] += 1
    return counts


def _run_fixture_export_guards() -> list[str]:
    """Execute the guard tests from ml.tests.fixtures.test_exports."""

    sys.path.insert(0, str(REPO_ROOT))
    guard_module = importlib.import_module("ml.tests.fixtures.test_exports")
    failures: list[str] = []
    for name in sorted(dir(guard_module)):
        if not name.startswith("test_"):
            continue
        func = getattr(guard_module, name)
        if not inspect.isfunction(func):
            continue
        try:
            func()
        except AssertionError as exc:
            failures.append(f"{name} failed: {exc}")
    return failures


def main() -> int:
    plugin_dirs = _load_plugin_directories(TESTS_ROOT)
    missing_files = _collect_missing_plugin_files(plugin_dirs)
    bootstrap_violations = _collect_package_bootstrap_violations(TESTS_ROOT)
    fixture_import_violations = _collect_direct_fixture_imports(TESTS_ROOT)
    conftest_violations = _collect_nested_conftest_plugins(TESTS_ROOT)
    test_totals = _count_test_modules_by_package(TESTS_ROOT)
    missing_counts = _count_missing_by_package(missing_files, TESTS_ROOT)
    failures: list[str] = []
    if missing_files:
        rel_paths = "\n".join(
            f"  - {path.relative_to(REPO_ROOT)}" for path in sorted(missing_files)
        )
        failures.append(
            "The following test modules do not register "
            "`ml.tests.fixtures.pytest_plugins` via the module or package:\n"
            f"{rel_paths}",
        )
    if bootstrap_violations:
        failures.append(
            "Test packages missing canonical pytest plug-in bootstraps:\n"
            + "\n".join(f"  - {item}" for item in sorted(bootstrap_violations)),
        )
    if fixture_import_violations:
        failures.append(
            "Direct fixture imports detected (see FIXTURE_GUIDE.md):\n"
            + "\n".join(f"  - {item}" for item in sorted(fixture_import_violations)),
        )
    if conftest_violations:
        failures.append(
            "pytest_plugins declared in non-top-level conftest.py (see FIXTURE_GUIDE.md):\n"
            + "\n".join(f"  - {item}" for item in sorted(conftest_violations)),
        )

    failures.extend(_run_fixture_export_guards())

    plugin_counts = _count_plugin_declarations(TESTS_ROOT)
    total = sum(plugin_counts.values())
    print(f"[fixtures] pytest plug-in declarations: {total}")
    for key in sorted(plugin_counts):
        print(f"  - {_format_package_name(key)}: {plugin_counts[key]}")

    all_packages = sorted(set(test_totals) | set(plugin_counts))
    print("[fixtures] per-package plug-in adoption:")
    for package in all_packages:
        total_tests = test_totals.get(package, 0)
        missing = missing_counts.get(package, 0)
        covered = total_tests - missing
        coverage = 100.0 if total_tests == 0 else (covered / total_tests) * 100.0
        plugin_decl = plugin_counts.get(package, 0)
        label = _format_package_name(package)
        print(
            f"  - {label}: {covered}/{total_tests} files ({coverage:.1f}%) | plug-in declarations={plugin_decl}"
        )

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    print("[fixtures] export guard passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
