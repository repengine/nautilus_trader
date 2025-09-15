#!/usr/bin/env python3
"""
Lightweight codebase sanity sweep (advisory, fast, DRY/SOLID-leaning).

Runs quick checks and prints a concise report:
- Gate checks: ruff (S608/C901), mypy (ml/ strict)
- Duplicate/drift: legacy schema references
- Safety: SQL f-strings, broad excepts
- Architecture: stores importing actors
- Optional tools: pip-audit/bandit/vulture/deptry if installed

Exit code 0 always (advisory). Use Make target: `make sanity`.

"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output


def header(title: str) -> None:
    print("\n== " + title)


def check_ruff() -> None:
    header("Ruff checks (S608/C901 on ml/)")
    if shutil.which("ruff"):
        _code, out = run(["ruff", "check", "ml", "--select", "S608,C901", "--quiet"])
        print(out.strip() or "No S608/C901 issues detected in ml/")
    else:
        print("ruff not installed; skipping")


def check_mypy() -> None:
    header("Mypy strict (ml/)")
    if shutil.which("uv"):
        cmd = [
            "uv",
            "run",
            "--active",
            "--no-sync",
            "mypy",
            "ml",
            "--strict",
        ]
        _code, out = run(cmd)
        print(out.strip())
    elif shutil.which("mypy"):
        _code, out = run(["mypy", "ml", "--strict"])
        print(out.strip())
    else:
        print("mypy not installed; skipping")


def rg(pattern: str) -> list[str]:
    """
    Return matching lines (path:line:content).
    """
    if shutil.which("rg"):
        _code, out = run(["rg", "-n", "-S", pattern])
        return [l for l in out.splitlines() if l.strip()]
    # Fallback: walk and search text files
    matches: list[str] = []
    rx = re.compile(pattern)
    for p in ROOT.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            if rx.search(line):
                matches.append(f"{p}:{i}:{line}")
    return matches


def check_legacy_schema_refs() -> None:
    header("Legacy schema/ references (should use ml/stores/migrations)")
    hits = rg(
        r"\bml/schema/|/schema/|docker-entrypoint-initdb\.d/00_init\.sql|features\.sql|models\.sql|strategies\.sql",
    )
    filtered = [h for h in hits if "/schema/sql" not in h and "crates/" not in h]
    if filtered:
        print("Found legacy references:\n" + "\n".join(filtered))
    else:
        print("No legacy schema references detected")


def check_sql_fstrings_and_broad_excepts() -> None:
    header("Potential SQL f-strings and broad excepts in ml/")
    fstrings = rg(r"ml/.*(text\(f\"|f\"\"\"[\s\S]*SELECT|pl\.read_database\(f\")")
    f_ml = [l for l in fstrings if "/ml/" in l]
    if f_ml:
        print("Possible SQL f-strings:\n" + "\n".join(f_ml))
    else:
        print("No obvious SQL f-strings in ml/")

    broad = rg(r"ml/.*except Exception:|ml/.*except:\s*$")
    if broad:
        print("Broad exceptions to review:\n" + "\n".join(broad))
    else:
        print("No broad exceptions detected in ml/")


def check_architecture_imports() -> None:
    header("Architecture: stores importing actors (should be avoided)")
    hits = rg(r"ml/stores/.*(from ml\.actors|import ml\.actors)")
    if hits:
        print("Found potential layering violations:\n" + "\n".join(hits))
    else:
        print("No store→actor imports detected")


def optional_tools() -> None:
    header("Optional tools (advisory)")
    if shutil.which("pip-audit"):
        _code, out = run(["pip-audit", "-r", "pyproject.toml"])
        print("pip-audit:\n" + out.strip())
    else:
        print("pip-audit not installed; skipping")

    if shutil.which("bandit"):
        _code, out = run(["bandit", "-q", "-r", "ml"])  # quiet recursive
        print("bandit:\n" + out.strip())
    else:
        print("bandit not installed; skipping")

    if shutil.which("vulture"):
        _code, out = run(["vulture", "ml", "--min-confidence", "80"])
        print("vulture:\n" + out.strip())
    else:
        print("vulture not installed; skipping")

    if shutil.which("deptry"):
        _code, out = run(["deptry", "."])  # repo root
        print("deptry:\n" + out.strip())
    else:
        print("deptry not installed; skipping")


def main() -> None:
    check_ruff()
    check_mypy()
    check_legacy_schema_refs()
    check_sql_fstrings_and_broad_excepts()
    check_architecture_imports()
    optional_tools()
    print("\nSanity sweep complete (advisory).")


if __name__ == "__main__":
    main()
