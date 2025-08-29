#!/usr/bin/env python3
"""
Script to clean up redundant tests identified by the redundancy analysis.

This script helps identify and optionally remove:
1. Exact duplicate test files
2. Tests that are subsumed by property tests
3. Tests that could be parameterized
"""

import ast
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import click


def get_file_hash(filepath):
    """Get hash of file content to identify exact duplicates."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def find_duplicate_files(test_dir):
    """Find test files with identical content."""
    file_hashes = defaultdict(list)

    for path in Path(test_dir).rglob("test_*.py"):
        file_hash = get_file_hash(path)
        file_hashes[file_hash].append(str(path))

    duplicates = {h: files for h, files in file_hashes.items() if len(files) > 1}
    return duplicates


def find_redundant_example_tests(test_dir):
    """Find example tests that are covered by property tests."""
    redundant = []

    # Known patterns where property tests subsume examples
    property_test_patterns = [
        ("test_.*_property", "test_.*_example"),
        ("test_.*_hypothesis", "test_.*_simple"),
        ("test_.*_invariant", "test_.*_specific"),
    ]

    for path in Path(test_dir).rglob("test_*.py"):
        with open(path) as f:
            content = f.read()

        # Check if file has both property and example tests
        has_hypothesis = "@given" in content or "from hypothesis" in content
        has_simple_tests = "def test_" in content and not has_hypothesis

        if has_hypothesis and has_simple_tests:
            # Parse to find specific redundant tests
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    # Check if this is a simple example that's covered by property test
                    if "_example" in node.name or "_simple" in node.name or "_specific" in node.name:
                        redundant.append((str(path), node.name, node.lineno))

    return redundant


def find_parameterizable_tests(test_dir):
    """Find tests that could be converted to parametrized tests."""
    parameterizable = defaultdict(list)

    for path in Path(test_dir).rglob("test_*.py"):
        with open(path) as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue

        # Look for test functions with similar names (numbered or varied)
        test_functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                # Strip common suffixes to find base name
                import re
                base_name = re.sub(r"_(valid|invalid|empty|none|small|large|1|2|3)$", "", node.name)

                if base_name not in test_functions:
                    test_functions[base_name] = []
                test_functions[base_name].append({
                    "name": node.name,
                    "line": node.lineno,
                    "file": str(path)
                })

        # Find groups that could be parameterized
        for base, tests in test_functions.items():
            if len(tests) > 2:  # 3 or more similar tests
                parameterizable[base].extend(tests)

    return parameterizable


def generate_cleanup_report(test_dir):
    """Generate a report of redundant tests to clean up."""
    report = []

    # Find exact duplicate files
    duplicates = find_duplicate_files(test_dir)
    if duplicates:
        report.append("## Exact Duplicate Files\n")
        for hash_val, files in duplicates.items():
            report.append("\nDuplicate group (keep first, remove others):\n")
            for f in files:
                report.append(f"  - {f}\n")

    # Find redundant example tests
    redundant_examples = find_redundant_example_tests(test_dir)
    if redundant_examples:
        report.append("\n## Redundant Example Tests (covered by property tests)\n")
        for filepath, test_name, line_no in redundant_examples:
            report.append(f"  - {filepath}:{line_no} - {test_name}\n")

    # Find parameterizable tests
    parameterizable = find_parameterizable_tests(test_dir)
    if parameterizable:
        report.append("\n## Tests That Should Be Parameterized\n")
        for base_name, tests in sorted(parameterizable.items(),
                                      key=lambda x: len(x[1]), reverse=True)[:10]:
            report.append(f"\n### {base_name} ({len(tests)} tests)\n")
            for test in tests[:5]:
                report.append(f"  - {test['file']}:{test['line']} - {test['name']}\n")
            if len(tests) > 5:
                report.append(f"  ... and {len(tests) - 5} more\n")

    return "".join(report)


def estimate_reduction(test_dir):
    """Estimate potential test reduction."""
    total_files = len(list(Path(test_dir).rglob("test_*.py")))

    # Count potential reductions
    duplicates = find_duplicate_files(test_dir)
    duplicate_files_to_remove = sum(len(files) - 1 for files in duplicates.values())

    redundant_examples = find_redundant_example_tests(test_dir)
    redundant_tests_to_remove = len(redundant_examples)

    parameterizable = find_parameterizable_tests(test_dir)
    # Estimate reduction from parameterization (keep 1/3 of tests)
    parameterizable_reduction = sum(len(tests) * 2 // 3 for tests in parameterizable.values())

    total_reduction = duplicate_files_to_remove + redundant_tests_to_remove + parameterizable_reduction

    return {
        "total_files": total_files,
        "duplicate_files": duplicate_files_to_remove,
        "redundant_tests": redundant_tests_to_remove,
        "parameterizable": parameterizable_reduction,
        "total_reduction": total_reduction,
    }


@click.command()
@click.option("--test-dir", default="ml/tests", help="Directory containing tests")
@click.option("--output", default=None, help="Output file for cleanup report")
@click.option("--dry-run", is_flag=True, help="Show what would be removed without removing")
def main(test_dir, output, dry_run):
    """Analyze and optionally clean up redundant tests."""
    print(f"Analyzing {test_dir} for redundant tests...")

    # Generate cleanup report
    report = generate_cleanup_report(test_dir)

    if output:
        with open(output, "w") as f:
            f.write("# Test Cleanup Report\n\n")
            f.write(report)
        print(f"Report written to {output}")
    else:
        print(report)

    # Estimate reduction
    stats = estimate_reduction(test_dir)

    print("\n=== Potential Reduction ===")
    print(f"Total test files: {stats['total_files']}")
    print(f"Duplicate files to remove: {stats['duplicate_files']}")
    print(f"Redundant example tests: {stats['redundant_tests']}")
    print(f"Tests that could be parameterized: {stats['parameterizable']}")
    print(f"Total potential reduction: {stats['total_reduction']} tests")

    if not dry_run:
        response = input("\nWould you like to see specific files to remove? (y/n): ")
        if response.lower() == "y":
            duplicates = find_duplicate_files(test_dir)
            print("\n=== Files to Remove (duplicates) ===")
            for files in duplicates.values():
                # Keep first, remove rest
                for f in files[1:]:
                    print(f"  rm {f}")


if __name__ == "__main__":
    main()
