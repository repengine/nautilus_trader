#!/usr/bin/env python3
"""
Emit a JSONL usage map: call sites found in `ml/`.

Each line: {callee: str, file: str, lineno: int, arg_count: int, keywords: [str]}

Note: This is a static heuristic; it doesn't resolve imports or attributes.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path


def map_file(path: Path) -> list[dict[str, object]]:
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    calls: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Attempt to unparse the callee as a string (may be attr or name)
            try:
                callee = ast.unparse(node.func)
            except Exception:
                callee = "<unknown>"
            calls.append(
                {
                    "callee": callee,
                    "file": str(path),
                    "lineno": getattr(node, "lineno", 0),
                    "arg_count": len(getattr(node, "args", [])),
                    "keywords": [kw.arg or "" for kw in getattr(node, "keywords", [])],
                }
            )
    return calls


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "ml"
    for p in root.rglob("*.py"):
        for entry in map_file(p):
            print(json.dumps(entry))


if __name__ == "__main__":
    main()

