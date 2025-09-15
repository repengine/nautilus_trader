#!/usr/bin/env python3
"""
Ratcheting test coverage gate for pre-commit.

Behavior
- Measures total coverage for one or more packages (default: ml).
- Reads baseline from tools/coverage_target.txt (integer percent).
- Fails if current coverage < baseline.
- If current coverage >= baseline + 1, writes (baseline + 1) to the file and
  fails to prompt the developer to include the baseline bump in the commit.
- If the baseline file does not exist, creates it at the current coverage and
  fails to prompt inclusion in the commit.

This pattern ensures:
- The threshold starts at the team’s current reality (not aspirational 90%).
- Each commit has an opportunity to raise the bar by 1 percentage point.

Notes
- This hook intentionally exits non-zero when it writes/updates the baseline
  file, so that pre-commit stops and the developer can `git add` the updated
  file and re-run the commit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tools" / "coverage_target.txt"


def run_pytest_with_coverage(packages: list[str]) -> Path:
    """Run pytest with coverage for the given packages and produce coverage.xml.

    Returns the path to the generated coverage.xml.
    """
    # Ensure we produce an XML report we can parse deterministically.
    cov_args = [f"--cov={pkg}" for pkg in packages]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *cov_args,
        "--cov-report=term-missing:skip-covered",
        "--cov-report=xml",
        "--no-header",
    ]
    # Avoid overly long runs by not forcing full failure traces; rely on other
    # hooks/tests for strict pass/fail. We only need the coverage numbers here.
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        # We still may have produced coverage.xml; but indicate the test failures.
        print("Tests did not fully pass during coverage run (coverage will still be parsed if available).", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    xml_path = ROOT / "coverage.xml"
    if not xml_path.exists():
        print("coverage.xml not found. Ensure pytest-cov is installed and enabled.", file=sys.stderr)
        sys.exit(2)
    return xml_path


def parse_total_coverage(xml_path: Path) -> float:
    """Parse total line-rate percent from coverage.xml (0-100)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Coverage.py writes attributes on the root <coverage> element
    # including 'line-rate' (0.0-1.0) and totals. Prefer computing from 
    # lines-valid and lines-covered when available for accuracy.
    lines_valid = root.attrib.get("lines-valid") or root.attrib.get("lines_valid")
    lines_covered = root.attrib.get("lines-covered") or root.attrib.get("lines_covered")
    if lines_valid and lines_covered:
        try:
            valid = float(lines_valid)
            covered = float(lines_covered)
            if valid > 0:
                return round(covered / valid * 100.0, 2)
        except ValueError:
            pass
    line_rate = root.attrib.get("line-rate") or root.attrib.get("line_rate")
    if line_rate:
        try:
            return round(float(line_rate) * 100.0, 2)
        except ValueError:
            pass
    print("Unable to parse total coverage from coverage.xml", file=sys.stderr)
    sys.exit(2)


def read_baseline(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
        return int(text)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def write_baseline(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n")


def main() -> int:
    # Allow packages to be overridden via env var; default to ml only.
    packages_env = os.environ.get("COVERAGE_PACKAGES", "ml").strip()
    packages = [p.strip() for p in packages_env.split(",") if p.strip()]

    xml_path = run_pytest_with_coverage(packages)
    total = parse_total_coverage(xml_path)
    total_floor = int(total)  # integer percentage floor

    baseline = read_baseline(BASELINE_PATH)

    if baseline is None:
        # Initialize baseline to current coverage (floored)
        write_baseline(BASELINE_PATH, total_floor)
        print(f"Initialized coverage baseline at {total_floor}% (current total: {total}%).")
        print("Baseline file created: tools/coverage_target.txt. Add it to your commit.")
        return 1  # fail to prompt inclusion

    print(f"Current coverage: {total}% | Baseline: {baseline}%")

    if total_floor < baseline:
        print(f"❌ Coverage {total}% is below baseline {baseline}%. Add tests or reduce scope.")
        return 1

    # If we can ratchet up by 1, do so and require the bump be committed.
    if total_floor >= baseline + 1:
        new_baseline = baseline + 1
        write_baseline(BASELINE_PATH, new_baseline)
        print(f"✅ Coverage improved. Raising baseline to {new_baseline}%.")
        print("Re-add tools/coverage_target.txt and re-run commit.")
        return 1

    print("✅ Coverage meets baseline (no ratchet this commit).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

