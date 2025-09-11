#!/usr/bin/env python3
"""
Generate a JSON index of public API symbols (functions, classes, methods).

By default, scans the `ml/` directory tree, enumerates public symbols
(names not starting with `_`), and emits a JSON array with:

- module: dotted module path
- path: repo‑relative file path
- kind: "function" | "class" | "method"
- name: short name
- qualname: fully qualified name (e.g., ml.module.Class.method)
- line_start, line_end: 1-based line range in source
- summary: first line of docstring (if any)
- docstring: full docstring (if any)
- link: path with line anchors (e.g., path#L10-L42); use --base-url for web links

Usage:
  python tools/api_index.py --root ml --output ml/tests/validation_reports/public_api_index.json

Optional:
  --include-private    Include private symbols (names starting with '_')
  --base-url URL       Prefix for file links (e.g., https://github.com/org/repo/blob/HEAD/)
  --git-root PATH      Repo root for relative paths (auto-detected via `git rev-parse --show-toplevel`)

Notes:
- This script uses AST (no imports) to avoid side effects.
- Methods listed only for classes with public names (unless --include-private).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal


Kind = Literal["function", "class", "method"]


@dataclass(slots=True)
class Symbol:
    module: str
    path: str
    kind: Kind
    name: str
    qualname: str
    line_start: int
    line_end: int
    summary: str | None
    docstring: str | None
    link: str


def _detect_git_root() -> Path | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        return Path(out)
    except Exception:
        return None


def _rel_path(p: Path, repo_root: Path | None) -> str:
    try:
        if repo_root is not None:
            return str(p.resolve().relative_to(repo_root.resolve()))
        return str(p)
    except Exception:
        return str(p)


def _file_link(path: str, start: int, end: int, base_url: str | None) -> str:
    anchor = f"#L{start}-L{end}" if start and end else ""
    if base_url:
        sep = "" if base_url.endswith("/") else "/"
        return f"{base_url}{sep}{path}{anchor}"
    return f"{path}{anchor}"


def _is_public(name: str, include_private: bool) -> bool:
    if include_private:
        return True
    # Exclude dunder and private by default
    return not (name.startswith("_") or name.startswith("__"))


def _iter_py_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*.py"):
        # Skip cache and tests by default
        if any(part in {"__pycache__", ".venv", ".git"} for part in p.parts):
            continue
        yield p


def _module_name(py_file: Path, repo_root: Path | None) -> str:
    rel = _rel_path(py_file, repo_root)
    mod = rel[:-3].replace(os.sep, ".")  # strip .py
    # Drop leading path elements up to first package root if needed
    return mod


def _doc_summary(doc: str | None) -> str | None:
    if not doc:
        return None
    first = doc.strip().splitlines()[0] if doc.strip() else ""
    return first or None


def _collect_symbols_from_file(
    py_file: Path,
    *,
    repo_root: Path | None,
    base_url: str | None,
    include_private: bool,
) -> list[Symbol]:
    try:
        source = py_file.read_text(encoding="utf-8")
    except Exception:
        return []

    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError:
        return []

    module = _module_name(py_file, repo_root)
    rel_path = _rel_path(py_file, repo_root)
    symbols: list[Symbol] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # module-level function
            if isinstance(getattr(node, "parent", None), ast.Module) and _is_public(node.name, include_private):
                doc = ast.get_docstring(node)
                start = getattr(node, "lineno", 0)
                end = getattr(node, "end_lineno", start)
                symbols.append(
                    Symbol(
                        module=module,
                        path=rel_path,
                        kind="function",
                        name=node.name,
                        qualname=f"{module}.{node.name}",
                        line_start=int(start),
                        line_end=int(end),
                        summary=_doc_summary(doc),
                        docstring=doc,
                        link=_file_link(rel_path, int(start), int(end), base_url),
                    ),
                )
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
            # Treat async def as functions
            self.visit_FunctionDef(node)  # type: ignore[arg-type]

        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            if not _is_public(node.name, include_private):
                # Still visit children to catch public nested definitions if private inclusion is on
                for child in node.body:
                    setattr(child, "parent", node)
                for child in node.body:
                    self.visit(child)
                return

            # Class symbol
            doc = ast.get_docstring(node)
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start)
            symbols.append(
                Symbol(
                    module=module,
                    path=rel_path,
                    kind="class",
                    name=node.name,
                    qualname=f"{module}.{node.name}",
                    line_start=int(start),
                    line_end=int(end),
                    summary=_doc_summary(doc),
                    docstring=doc,
                    link=_file_link(rel_path, int(start), int(end), base_url),
                ),
            )

            # Methods
            for child in node.body:
                setattr(child, "parent", node)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(
                    child.name,
                    include_private,
                ):
                    mdoc = ast.get_docstring(child)
                    mstart = getattr(child, "lineno", 0)
                    mend = getattr(child, "end_lineno", mstart)
                    symbols.append(
                        Symbol(
                            module=module,
                            path=rel_path,
                            kind="method",
                            name=child.name,
                            qualname=f"{module}.{node.name}.{child.name}",
                            line_start=int(mstart),
                            line_end=int(mend),
                            summary=_doc_summary(mdoc),
                            docstring=mdoc,
                            link=_file_link(rel_path, int(mstart), int(mend), base_url),
                        ),
                    )
            # Visit children for nested classes etc.
            for child in node.body:
                self.visit(child)

    # Wire parent pointers for module-level traversal
    for n in ast.walk(tree):
        for child in ast.iter_child_nodes(n):
            setattr(child, "parent", n)

    Visitor().visit(tree)
    return symbols


def build_index(
    *,
    root: Path,
    repo_root: Path | None,
    base_url: str | None,
    include_private: bool,
) -> list[dict[str, Any]]:
    items: list[Symbol] = []
    for py in _iter_py_files(root):
        items.extend(
            _collect_symbols_from_file(
                py, repo_root=repo_root, base_url=base_url, include_private=include_private
            ),
        )
    # Sort alphabetically by qualname
    items.sort(key=lambda s: s.qualname)
    return [asdict(symbol) for symbol in items]


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate public API JSON index")
    ap.add_argument("--root", default="ml", help="Directory tree to scan (default: ml)")
    ap.add_argument("--output", default="-", help="Output JSON file path or '-' for stdout")
    ap.add_argument("--include-private", action="store_true", help="Include private symbols")
    ap.add_argument(
        "--base-url",
        default=None,
        help="Optional base URL for file links (e.g., https://github.com/org/repo/blob/HEAD)",
    )
    ap.add_argument("--git-root", default=None, help="Override git root path for relative links")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        ap.error(f"root not found: {root}")

    repo_root: Path | None
    if args.git_root:
        repo_root = Path(args.git_root)
    else:
        repo_root = _detect_git_root() or Path.cwd()

    index = build_index(
        root=root,
        repo_root=repo_root,
        base_url=args.base_url,
        include_private=bool(args.include_private),
    )

    data = json.dumps(index, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(data)
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(data + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
