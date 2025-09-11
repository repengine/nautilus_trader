#!/usr/bin/env python3
"""
Lightweight duplication detector for ML core modules.

Flags repeated code blocks across ml/{actors,stores,registry} to surface copy/paste hotspots
related to C001–C007/H021–H023 in the checklist.

Heuristic: normalized line shingles of length N across files; report blocks that occur >=2 times
in different files with at least M non-empty lines in each block.

This tool is advisory; wire as pre-commit manual stage or CI gate.

"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOTS = [Path("ml/actors"), Path("ml/stores"), Path("ml/registry")]
MIN_LINES = 8  # min non-empty lines per block
WINDOW = 12  # shingle window size (lines)
MAX_REPORTS = 50
EXCLUDES = {"__init__.py"}


def normalize_line(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    # Basic normalization: collapse spaces, drop inline comments (# ...)
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    return " ".join(s.split())


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if p.name in EXCLUDES:
                continue
            files.append(p)
    return files


def shingle_file(path: Path) -> list[tuple[str, int, int]]:
    """
    Return list of (hash, start_line, end_line) shingles for a file.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [normalize_line(l) for l in text.splitlines()]
    # Filter leading/trailing blanks for windowing
    n = len(lines)
    results: list[tuple[str, int, int]] = []
    for i in range(0, max(0, n - WINDOW + 1)):
        block = lines[i : i + WINDOW]
        non_empty = [l for l in block if l]
        if len(non_empty) < MIN_LINES:
            continue
        h = hashlib.sha1("\n".join(non_empty).encode("utf-8")).hexdigest()
        results.append((h, i + 1, i + WINDOW))
    return results


def main() -> int:
    files = iter_python_files()
    index: dict[str, list[tuple[Path, int, int]]] = {}

    for f in files:
        for h, start, end in shingle_file(f):
            index.setdefault(h, []).append((f, start, end))

    # Report shingles seen in 2+ distinct files
    reports: list[str] = []
    for h, occs in index.items():
        paths = {p for p, _, _ in occs}
        if len(paths) < 2:
            continue
        # Summarize first few occurrences
        parts = ["Duplicate block across files ({} occurrences):".format(len(occs))]
        for p, s, e in occs[:5]:
            parts.append(f"  - {p}:{s}-{e}")
        reports.append("\n".join(parts))
        if len(reports) >= MAX_REPORTS:
            break

    if reports:
        print("Found potential duplication hotspots:\n")
        print("\n\n".join(reports))
        print("\nHint: Extract common helpers/mixins or base classes to remove duplication.")
        return 1

    print("No significant duplication hotspots detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
