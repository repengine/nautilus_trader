#!/usr/bin/env python3
"""
Emit a JSONL symbol index for Python files under `ml/`.

Each line: {kind: class|function, name, file, lineno, col_offset, ...}

"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def index_file(path: Path) -> list[dict[str, Any]]:
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    items: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            items.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "file": str(path),
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                    "args": [a.arg for a in node.args.args],
                    "returns": (
                        ast.unparse(node.returns) if getattr(node, "returns", None) else None
                    ),
                },
            )
        elif isinstance(node, ast.ClassDef):
            items.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "file": str(path),
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                    "bases": [ast.unparse(b) for b in node.bases],
                },
            )
    return items


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "ml"
    for p in root.rglob("*.py"):
        for entry in index_file(p):
            print(json.dumps(entry))


if __name__ == "__main__":
    main()
