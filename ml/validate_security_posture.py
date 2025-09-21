"""
Static analysis helpers to enforce ML security posture.

These utilities provide lightweight AST-based scanning to detect disallowed
imports such as ``pickle``/``joblib`` that could compromise model artifact
integrity. They are intentionally simple so they can run during unit tests
without adding heavy dependencies.

"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(slots=True)
class SecurityViolation:
    """
    Represents a violation detected while analyzing a Python source file.
    """

    violation_type: str
    line_number: int
    message: str


_BANNED_IMPORT_PREFIXES: Final[dict[str, str]] = {
    "pickle": "pickle_import",
    "joblib": "joblib_import",
    "dill": "dill_import",
    "cloudpickle": "cloudpickle_import",
}


def _scan_imports(node: ast.AST, path: Path) -> list[SecurityViolation]:
    violations: list[SecurityViolation] = []

    if isinstance(node, ast.Import):
        for alias in node.names:
            root_name = alias.name.split(".", 1)[0]
            violation_key = _BANNED_IMPORT_PREFIXES.get(root_name)
            if violation_key:
                violations.append(
                    SecurityViolation(
                        violation_type=violation_key,
                        line_number=node.lineno,
                        message=(f"Disallowed import '{alias.name}' detected in {path.name}"),
                    ),
                )
    elif isinstance(node, ast.ImportFrom) and node.module:
        root_name = node.module.split(".", 1)[0]
        violation_key = _BANNED_IMPORT_PREFIXES.get(root_name)
        if violation_key:
            violations.append(
                SecurityViolation(
                    violation_type=violation_key,
                    line_number=node.lineno,
                    message=(f"Disallowed import from '{node.module}' detected in {path.name}"),
                ),
            )

    return violations


def analyze_file(path: Path) -> list[SecurityViolation]:
    """
    Analyze ``path`` for banned imports and return any violations discovered.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [
            SecurityViolation(
                violation_type="syntax_error",
                line_number=exc.lineno or 0,
                message=f"Unable to parse {path.name}: {exc.msg}",
            ),
        ]

    violations: list[SecurityViolation] = []
    for node in ast.walk(tree):
        violations.extend(_scan_imports(node, path))

    return violations


__all__ = ["SecurityViolation", "analyze_file"]
