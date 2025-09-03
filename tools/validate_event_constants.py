#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "ml"

ALLOWLIST = {
    str(ML_DIR / "config" / "events.py"),
}

STAGE_LITERALS = (
    "CATALOG_WRITTEN",
    "FEATURE_COMPUTED",
    "PREDICTION_EMITTED",
    "SIGNAL_EMITTED",
    "INGESTED",
)

STAGE_ASSIGN_RE = re.compile(
    r"stage\s*=\s*['\"](" + "|".join(map(re.escape, STAGE_LITERALS)) + r")['\"]"
)


def main() -> int:
    violations: list[str] = []
    stores_dir = ML_DIR / "stores"
    data_dir = ML_DIR / "data"
    for py in ML_DIR.rglob("*.py"):
        p = str(py)
        if p in ALLOWLIST:
            continue
        # Scope: only validate core emitters in stores/ and data/
        if not (p.startswith(str(stores_dir)) or p.startswith(str(data_dir))):
            continue
        if "/tests/" in p or "/docs/" in p or "/migrations/" in p:
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for m in STAGE_ASSIGN_RE.finditer(text):
            violations.append(f"{py}: raw stage literal '{m.group(1)}'")

    if violations:
        print("Event constants validation FAILED:\n" + "\n".join(violations))
        return 1
    print("Event constants validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
