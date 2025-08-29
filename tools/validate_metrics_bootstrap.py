#!/usr/bin/env python3
"""
Validate that metrics are acquired via ml.common.metrics_bootstrap or ml.common.metrics.

Flags any direct usages of prometheus_client or direct instantiation of Counter/Gauge/Histogram
within the ml/ package, excluding allowed modules.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "ml"

ALLOWLIST = {
    str(ML_DIR / "common" / "metrics.py"),
    str(ML_DIR / "common" / "metrics_bootstrap.py"),
    str(ML_DIR / "_imports.py"),
}

PROM_IMPORT_RE = re.compile(r"^\s*from\s+prometheus_client\s+import\s+", re.M)
DIRECT_INSTANTIATE_RE = re.compile(r"(?<!get_)(Counter|Histogram|Gauge)\s*\(")


def scan_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    violations: list[str] = []

    if PROM_IMPORT_RE.search(text):
        if "TYPE_CHECKING" not in text:
            violations.append("direct prometheus_client import")

    # Only flag direct instantiation; allow via bootstrap get_* helpers
    for m in DIRECT_INSTANTIATE_RE.finditer(text):
        violations.append(f"direct instantiation: {m.group(1)}(")

    return violations


def main() -> int:
    violations: list[str] = []
    for py in ML_DIR.rglob("*.py"):
        # Skip allowlist
        if str(py) in ALLOWLIST:
            continue
        # Skip tests and docs
        if "/tests/" in str(py) or "/docs/" in str(py):
            continue
        v = scan_file(py)
        if v:
            for msg in v:
                violations.append(f"{py}: {msg}")

    if violations:
        print("Metrics bootstrap validation FAILED:\n" + "\n".join(violations))
        return 1
    print("Metrics bootstrap validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
