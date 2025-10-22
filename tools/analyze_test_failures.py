#!/usr/bin/env python3
"""Analyze test failures and categorize them for systematic restoration.

Usage:
    poetry run pytest ml/tests --tb=line -q > test_output.txt 2>&1
    python tools/analyze_test_failures.py test_output.txt
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class FailureCategory:
    """Categorized test failure."""
    name: str
    pattern: str
    confidence: str  # HIGH, MEDIUM, LOW
    fix_strategy: str
    agent_type: str


CATEGORIES = [
    FailureCategory(
        name="DB_CONNECTION",
        pattern=r"OperationalError.*connection.*refused|connection to server.*failed",
        confidence="HIGH",
        fix_strategy="Environmental - Start PostgreSQL",
        agent_type="Manual (1-time fix)"
    ),
    FailureCategory(
        name="MISSING_ATTRIBUTES",
        pattern=r"AttributeError.*'(\w+)' object has no attribute '([_\w]+)'",
        confidence="HIGH",
        fix_strategy="Diff-based restoration of internal state",
        agent_type="Diff-Based Restoration Agent"
    ),
    FailureCategory(
        name="IMPORT_ERRORS",
        pattern=r"NameError: name '(\w+)' is not defined|ImportError|ModuleNotFoundError",
        confidence="HIGH",
        fix_strategy="Add missing imports from phase0",
        agent_type="Diff-Based Restoration Agent"
    ),
    FailureCategory(
        name="METRICS_TELEMETRY",
        pattern=r"assert.*call_count.*[><=]|assert.*== 0.*Expected.*> 0|'labels'.*have been called",
        confidence="MEDIUM",
        fix_strategy="Restore event emission wiring",
        agent_type="Event Wiring Analysis Agent"
    ),
    FailureCategory(
        name="ASSERTION_FAILURES",
        pattern=r"AssertionError: assert|assert .* == .*",
        confidence="LOW",
        fix_strategy="Behavioral analysis - may require test update",
        agent_type="Human Review Required"
    ),
    FailureCategory(
        name="FILE_NOT_FOUND",
        pattern=r"FileNotFoundError|Configuration file not found|Config file not found",
        confidence="HIGH",
        fix_strategy="Fix file paths or create missing configs",
        agent_type="Diff-Based Restoration Agent"
    ),
    FailureCategory(
        name="TEST_INFRASTRUCTURE",
        pattern=r"FlakyFailure|OSError.*stdin.*captured|RuntimeError: boom",
        confidence="MEDIUM",
        fix_strategy="Test-specific fixes (hypothesis settings, mocking)",
        agent_type="Test Infrastructure Agent"
    ),
]


def extract_failures(output_file: Path) -> List[str]:
    """Extract FAILED lines from pytest output."""
    failures = []
    with open(output_file) as f:
        for line in f:
            if line.startswith("FAILED "):
                failures.append(line.strip())
    return failures


def categorize_failure(failure_line: str, full_output: str) -> tuple[str, str, str]:
    """Categorize a failure and extract key info."""
    # Extract test name
    test_match = re.search(r"FAILED ([\w\./]+)::([\w\[\]_-]+)", failure_line)
    if not test_match:
        return "UNKNOWN", "Unknown test", ""

    test_path = test_match.group(1)
    test_name = test_match.group(2)

    # Find error message in full output
    error_pattern = f"{test_path}::{test_name}.*?- (.*?)(?=\nFAILED|$)"
    error_match = re.search(error_pattern, full_output, re.DOTALL)
    error_msg = error_match.group(1).strip() if error_match else ""

    # Categorize based on error message
    for category in CATEGORIES:
        if re.search(category.pattern, error_msg, re.IGNORECASE):
            return category.name, f"{test_path}::{test_name}", error_msg

    # Default to assertion failure category
    return "ASSERTION_FAILURES", f"{test_path}::{test_name}", error_msg


def analyze_failures(output_file: Path) -> Dict[str, List[tuple[str, str]]]:
    """Analyze all failures and group by category."""
    with open(output_file) as f:
        full_output = f.read()

    failures = extract_failures(output_file)
    categorized = defaultdict(list)

    for failure_line in failures:
        category, test_name, error_msg = categorize_failure(failure_line, full_output)
        categorized[category].append((test_name, error_msg))

    return categorized


def generate_report(categorized: Dict[str, List[tuple[str, str]]]) -> str:
    """Generate restoration taxonomy report."""
    category_map = {c.name: c for c in CATEGORIES}

    report = ["# Test Failure Restoration Taxonomy\n"]
    report.append(f"**Generated:** {Path.cwd()}\n")
    report.append(f"**Total Failures:** {sum(len(tests) for tests in categorized.values())}\n\n")

    # Summary table
    report.append("## Summary by Category\n\n")
    report.append("| Category | Count | Confidence | Fix Strategy | Agent Type |\n")
    report.append("|----------|-------|------------|--------------|------------|\n")

    total = 0
    for cat_name in sorted(categorized.keys(), key=lambda k: len(categorized[k]), reverse=True):
        tests = categorized[cat_name]
        total += len(tests)
        cat = category_map.get(cat_name, FailureCategory(
            cat_name, "", "LOW", "Unknown", "Human Review"
        ))
        report.append(
            f"| {cat.name} | {len(tests)} | {cat.confidence} | {cat.fix_strategy} | {cat.agent_type} |\n"
        )

    report.append(f"\n**Total:** {total} failures\n\n")

    # Detailed breakdown
    report.append("## Detailed Breakdown\n\n")

    for cat_name in sorted(categorized.keys(), key=lambda k: len(categorized[k]), reverse=True):
        tests = categorized[cat_name]
        cat = category_map.get(cat_name)

        report.append(f"### {cat_name} ({len(tests)} failures)\n\n")
        if cat:
            report.append(f"**Confidence:** {cat.confidence}\n")
            report.append(f"**Fix Strategy:** {cat.fix_strategy}\n")
            report.append(f"**Agent Type:** {cat.agent_type}\n\n")

        report.append("**Tests:**\n")
        for test_name, error_msg in tests[:10]:  # Show first 10
            report.append(f"- `{test_name}`\n")
            if error_msg:
                # Show first line of error
                first_line = error_msg.split('\n')[0][:100]
                report.append(f"  - Error: `{first_line}...`\n")

        if len(tests) > 10:
            report.append(f"\n... and {len(tests) - 10} more\n")
        report.append("\n")

    # Prioritization
    report.append("## Recommended Execution Order\n\n")
    report.append("1. **DB_CONNECTION** (Environmental - fix once, recover many)\n")
    report.append("2. **MISSING_ATTRIBUTES, IMPORT_ERRORS, FILE_NOT_FOUND** (Parallel - high confidence, mechanical)\n")
    report.append("3. **METRICS_TELEMETRY** (Sequential - after mechanical fixes)\n")
    report.append("4. **ASSERTION_FAILURES** (Human review - requires domain knowledge)\n")
    report.append("5. **TEST_INFRASTRUCTURE** (Case-by-case)\n\n")

    # Quick wins calculation
    high_conf = sum(
        len(categorized[cat.name])
        for cat in CATEGORIES
        if cat.confidence == "HIGH" and cat.name in categorized
    )

    report.append(f"## Quick Win Potential\n\n")
    report.append(f"**High confidence fixes:** {high_conf} tests (~{high_conf/total*100:.1f}%)\n")
    report.append(f"**Expected 1-day recovery:** {high_conf} tests\n")
    report.append(f"**Expected 3-day recovery:** ~{high_conf + len(categorized.get('METRICS_TELEMETRY', []))} tests\n\n")

    return "".join(report)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_test_failures.py <pytest_output.txt>")
        print("\nFirst run: poetry run pytest ml/tests --tb=line -q > test_output.txt 2>&1")
        sys.exit(1)

    output_file = Path(sys.argv[1])
    if not output_file.exists():
        print(f"Error: {output_file} not found")
        sys.exit(1)

    print(f"Analyzing {output_file}...")
    categorized = analyze_failures(output_file)

    report = generate_report(categorized)

    # Save report
    report_file = Path("restoration_taxonomy.md")
    report_file.write_text(report)
    print(f"\n✓ Report saved to {report_file}")

    # Also print to stdout
    print("\n" + report)

    # Generate category-specific test lists
    for cat_name, tests in categorized.items():
        category_file = Path(f"restoration_tasks/{cat_name.lower()}_tests.txt")
        category_file.parent.mkdir(exist_ok=True)
        category_file.write_text("\n".join(test for test, _ in tests))
        print(f"✓ {cat_name} tests saved to {category_file}")


if __name__ == "__main__":
    main()
