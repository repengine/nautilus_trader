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
-----
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
    """
    Run fast test subsets with coverage and emit coverage.xml.

    Uses coverage.py directly to avoid pytest-cov plugin interactions that can fail to
    generate reports in pre-commit environments.

    """
    tests_env = os.environ.get(
        "COVERAGE_TEST_PATHS",
        "ml/tests/unit ml/tests/property ml/tests/contracts",
    )
    test_paths = [p for p in tests_env.split() if p]

    env = os.environ.copy()

    # Create a minimal temporary coverage config to avoid repository plugins.
    import tempfile

    cfg = """
    [run]
    branch = True
    parallel = False
    source = {sources}

    [report]
    fail_under = 0
    show_missing = true
    precision = 2
    skip_covered = false
    skip_empty = true
    """.strip().format(
        sources=",".join(packages),
    )

    with tempfile.NamedTemporaryFile("w", delete=False, prefix="covratchet_", suffix=".ini") as tf:
        tf.write(cfg)
        tf.flush()
        rcfile = tf.name
    # Prefer passing configuration via environment to maximize CLI compatibility
    env["COVERAGE_RCFILE"] = rcfile

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, text=True, capture_output=True, env=env)

    # Erase any prior data (best effort)
    run(["uv", "run", "--active", "--no-sync", "coverage", "erase"])

    # Run tests under coverage
    cmd_run = [
        "uv",
        "run",
        "--active",
        "--no-sync",
        "coverage",
        "run",
        "-m",
        "pytest",
        "-q",
        "-m",
        "not prototype",
        *test_paths,
    ]
    result_run = run(cmd_run)
    if result_run.returncode != 0:
        print(
            "Tests did not fully pass during coverage run (continuing to report).",
            file=sys.stderr,
        )
        if result_run.stderr:
            print(result_run.stderr, file=sys.stderr)

    # Always attempt to produce XML report
    result_xml = run(
        [
            "uv",
            "run",
            "--active",
            "--no-sync",
            "coverage",
            "xml",
            "-i",
        ],
    )
    if result_xml.returncode != 0:
        print("Failed to generate coverage.xml", file=sys.stderr)
        if result_xml.stderr:
            print(result_xml.stderr, file=sys.stderr)
        sys.exit(2)

    xml_path = ROOT / "coverage.xml"
    if not xml_path.exists():
        print("coverage.xml not found after coverage xml step.", file=sys.stderr)
        sys.exit(2)
    return xml_path


def parse_total_coverage(xml_path: Path) -> float:
    """
    Parse total line-rate percent from coverage.xml (0-100).
    """
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
