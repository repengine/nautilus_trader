from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
ML_ROOT = REPO_ROOT / "ml"
TASKS_ROOT = ML_ROOT / "tasks"
LEDGER_PATH = ML_ROOT / "docs" / "development" / "DRY_DEDUP_TASK_MIGRATION_LEDGER.md"

_LEDGER_ROW = re.compile(
    r"^\|\s*`(ml/tasks/[^`]+\.py)`\s*\|.*\|\s*(planned|in_progress|done)\s*\|$",
)

_FROZEN_LEGACY_TASK_SHA256: dict[str, str] = {}

_EXPECTED_RUNTIME_TASK_IMPORTERS: frozenset[str] = frozenset()
_EXPECTED_TASK_MODULE_INVENTORY: frozenset[str] = frozenset()
_ALLOWED_PYTHON_TASK_REFERENCE_FILES: frozenset[str] = frozenset(
    {
        "ml/tests/unit/common/test_task_namespace_package_shims.py",
        "ml/tests/unit/common/test_tasks_migration_guardrails.py",
        "ml/tests/unit/data/test_task_cache_shims.py",
        "ml/tests/unit/registry/test_task_registry_shim.py",
        "ml/tests/unit/stores/test_migrations_runner.py",
        "ml/tests/unit/training/teacher/test_task_quick_shim.py",
    },
)
_ALLOWED_PYTHON_TASK_REFERENCE_PREFIXES: tuple[str, ...] = (
    "ml/tests/unit/tasks/",
)


def _all_task_modules() -> set[str]:
    return {
        path.relative_to(REPO_ROOT).as_posix()
        for path in TASKS_ROOT.rglob("*.py")
    }


def _parse_ledger_statuses() -> dict[str, str]:
    text = LEDGER_PATH.read_text(encoding="utf-8")
    statuses: dict[str, str] = {}
    for line in text.splitlines():
        match = _LEDGER_ROW.match(line.strip())
        if not match:
            continue
        module_path = match.group(1)
        status = match.group(2)
        statuses[module_path] = status
    return statuses


def _is_docstring_expr(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_allowed_shim_assignment(node: ast.stmt) -> bool:
    if isinstance(node, ast.Assign):
        return all(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == "__all__"
    return False


def _module_is_shim_only(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if _is_docstring_expr(node):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if _is_allowed_shim_assignment(node):
            continue
        return False
    return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_parts_for(path: Path) -> list[str]:
    relative = path.relative_to(ML_ROOT)
    parts = list(relative.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ["ml", *parts]


def _resolve_import_from(module_parts: list[str], node: ast.ImportFrom) -> str | None:
    target = (node.module or "").lstrip(".")
    if node.level == 0:
        return target
    base = module_parts[:-node.level] if node.level <= len(module_parts) else []
    target_parts = target.split(".") if target else []
    return ".".join([*base, *target_parts])


def _imports_ml_tasks(path: Path) -> bool:
    module_parts = _module_parts_for(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ml.tasks" or alias.name.startswith("ml.tasks."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_from(module_parts, node)
            if resolved and (resolved == "ml.tasks" or resolved.startswith("ml.tasks.")):
                return True
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                target = node.args[0].value
                if target == "ml.tasks" or target.startswith("ml.tasks."):
                    return True
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                target = node.args[0].value
                if target == "ml.tasks" or target.startswith("ml.tasks."):
                    return True

    return False


def _contains_task_reference_tokens(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "ml.tasks" in text or "ml/tasks" in text


def _is_allowed_task_reference_file(path: str) -> bool:
    if path in _ALLOWED_PYTHON_TASK_REFERENCE_FILES:
        return True
    return any(path.startswith(prefix) for prefix in _ALLOWED_PYTHON_TASK_REFERENCE_PREFIXES)


def test_task_ledger_tracks_every_task_module() -> None:
    ledger_statuses = _parse_ledger_statuses()
    task_modules = _all_task_modules()
    assert set(ledger_statuses) == task_modules, (
        "Task migration ledger must track every task module.\n"
        f"Missing in ledger: {sorted(task_modules - set(ledger_statuses))}\n"
        f"Stale in ledger: {sorted(set(ledger_statuses) - task_modules)}"
    )

    assert task_modules == set(_EXPECTED_TASK_MODULE_INVENTORY), (
        "Task module inventory changed.\n"
        f"Expected: {sorted(_EXPECTED_TASK_MODULE_INVENTORY)}\n"
        f"Actual: {sorted(task_modules)}"
    )


def test_done_task_modules_are_shim_only() -> None:
    ledger_statuses = _parse_ledger_statuses()
    done_modules = sorted(path for path, status in ledger_statuses.items() if status == "done")

    violations: list[str] = []
    for module_path in done_modules:
        path = REPO_ROOT / module_path
        if not _module_is_shim_only(path):
            violations.append(module_path)

    assert not violations, (
        "Task modules marked done must remain shim-only.\n"
        f"Violations: {violations}"
    )


def test_non_done_task_modules_match_frozen_baseline() -> None:
    ledger_statuses = _parse_ledger_statuses()
    non_done_modules = {
        path
        for path, status in ledger_statuses.items()
        if status in {"planned", "in_progress"}
    }
    assert set(_FROZEN_LEGACY_TASK_SHA256) == non_done_modules, (
        "Frozen baseline must track all non-done task modules.\n"
        f"Missing hashes: {sorted(non_done_modules - set(_FROZEN_LEGACY_TASK_SHA256))}\n"
        f"Unexpected hashes: {sorted(set(_FROZEN_LEGACY_TASK_SHA256) - non_done_modules)}"
    )

    drift: list[str] = []
    for module_path, expected_hash in sorted(_FROZEN_LEGACY_TASK_SHA256.items()):
        actual_hash = _sha256(REPO_ROOT / module_path)
        if actual_hash != expected_hash:
            drift.append(f"{module_path}: expected={expected_hash} actual={actual_hash}")

    assert not drift, (
        "Legacy task modules are frozen. Move logic to canonical owners or mark module done.\n"
        + "\n".join(drift)
    )


def test_runtime_import_boundary_for_ml_tasks() -> None:
    runtime_importers: set[str] = set()
    for path in ML_ROOT.rglob("*.py"):
        rel = path.relative_to(ML_ROOT).parts
        if not rel:
            continue
        if rel[0] in {"cli", "tasks", "tests"}:
            continue
        if _imports_ml_tasks(path):
            runtime_importers.add(path.relative_to(REPO_ROOT).as_posix())

    assert runtime_importers == set(_EXPECTED_RUNTIME_TASK_IMPORTERS), (
        "Non-CLI runtime imports of ml.tasks changed.\n"
        f"Expected: {sorted(_EXPECTED_RUNTIME_TASK_IMPORTERS)}\n"
        f"Actual: {sorted(runtime_importers)}"
    )


def test_python_ml_tasks_references_are_quarantined() -> None:
    references: set[str] = set()
    for path in ML_ROOT.rglob("*.py"):
        if _contains_task_reference_tokens(path):
            references.add(path.relative_to(REPO_ROOT).as_posix())

    unexpected = sorted(path for path in references if not _is_allowed_task_reference_file(path))
    missing_explicit = sorted(
        path
        for path in _ALLOWED_PYTHON_TASK_REFERENCE_FILES
        if path not in references
    )

    assert not unexpected and not missing_explicit, (
        "Python references to ml.tasks must stay quarantined to migration-history tests.\n"
        f"Allowed explicit files: {sorted(_ALLOWED_PYTHON_TASK_REFERENCE_FILES)}\n"
        f"Allowed prefixes: {list(_ALLOWED_PYTHON_TASK_REFERENCE_PREFIXES)}\n"
        f"Unexpected: {unexpected}\n"
        f"Missing explicit: {missing_explicit}"
    )
