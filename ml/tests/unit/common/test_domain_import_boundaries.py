from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_PREFIXES = ("scripts", "ml.cli")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ml"


def _module_parts_for(path: Path) -> list[str]:
    relative = path.relative_to(ML_ROOT)
    parts = list(relative.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ["ml", *parts]


def _is_forbidden(module_name: str) -> bool:
    for forbidden in FORBIDDEN_PREFIXES:
        if module_name == forbidden or module_name.startswith(f"{forbidden}."):
            return True
    return False


def _resolve_import_from(module_parts: list[str], node: ast.ImportFrom) -> str | None:
    target = (node.module or "").lstrip(".")
    if node.level == 0:
        return target
    base = module_parts[:-node.level] if node.level <= len(module_parts) else []
    target_parts = target.split(".") if target else []
    resolved_parts = base + target_parts
    return ".".join(part for part in resolved_parts if part)


def test_domain_modules_do_not_import_scripts_or_cli() -> None:
    violations: list[str] = []

    for path in ML_ROOT.rglob("*.py"):
        rel_parts = path.relative_to(ML_ROOT).parts
        if rel_parts and rel_parts[0] == "cli":
            continue  # CLI modules are allowed to import scripts/CLI helpers.

        module_parts = _module_parts_for(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - fail fast on invalid files
            violations.append(f"{path}: syntax error while parsing ({exc})")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden(alias.name):
                        violations.append(f"{path}: import {alias.name!r} is forbidden")
            elif isinstance(node, ast.ImportFrom):
                resolved = _resolve_import_from(module_parts, node)
                if resolved and _is_forbidden(resolved):
                    violations.append(
                        f"{path}: from-import of {resolved!r} is forbidden",
                    )

    assert not violations, (
        "Domain modules must not import CLI or scripts packages. "
        "Violations:\n- " + "\n- ".join(violations)
    )
