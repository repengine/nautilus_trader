#!/usr/bin/env python3
"""
Analyze test suite for redundancy and overlapping coverage.

This script identifies:
1. Tests with similar names (potential duplicates)
2. Tests that test the same functions/methods
3. Tests with overlapping assertions
4. Property tests that subsume example tests
5. Tests that could be consolidated

"""

import ast
import functools
import operator
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import click


class TestAnalyzer(ast.NodeVisitor):
    """
    Analyze test files for redundancy patterns.
    """

    def __init__(self):
        self.test_functions = []
        self.test_targets = defaultdict(list)  # What each test is testing
        self.assertions = defaultdict(list)  # Assertions per test
        self.fixtures_used = defaultdict(set)  # Fixtures per test
        self.decorators = defaultdict(list)  # Decorators per test
        self.current_class = None
        self.current_function = None

    def visit_ClassDef(self, node):
        """
        Track test classes.
        """
        if node.name.startswith("Test"):
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = None

    def visit_FunctionDef(self, node):
        """
        Track test functions.
        """
        if node.name.startswith("test_"):
            test_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
            self.test_functions.append(
                {
                    "name": test_name,
                    "file": self.current_file,
                    "line": node.lineno,
                    "docstring": ast.get_docstring(node),
                },
            )

            # Track decorators (hypothesis, pytest.mark, etc.)
            for decorator in node.decorator_list:
                decorator_name = self._get_decorator_name(decorator)
                if decorator_name:
                    self.decorators[test_name].append(decorator_name)

            # Track fixtures
            for arg in node.args.args:
                if arg.arg not in ["self", "cls"]:
                    self.fixtures_used[test_name].add(arg.arg)

            self.current_function = test_name
            self.generic_visit(node)
            self.current_function = None

    def visit_Call(self, node):
        """
        Track what functions/methods are being called in tests.
        """
        if self.current_function:
            # Track assertions
            if isinstance(node.func, ast.Name) and node.func.id == "assert":
                self.assertions[self.current_function].append("assert")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in [
                    "assertEqual",
                    "assertTrue",
                    "assertFalse",
                    "assert_allclose",
                ]:
                    self.assertions[self.current_function].append(node.func.attr)
                # Track what's being tested
                if hasattr(node.func.value, "id"):
                    self.test_targets[self.current_function].append(node.func.value.id)
        self.generic_visit(node)

    def _get_decorator_name(self, decorator):
        """
        Extract decorator name.
        """
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return None


def find_similar_tests(tests, threshold=0.8):
    """
    Find tests with similar names that might be duplicates.
    """
    from difflib import SequenceMatcher

    similar_groups = []
    seen = set()

    for i, test1 in enumerate(tests):
        if test1["name"] in seen:
            continue

        similar = [test1]
        for test2 in tests[i + 1 :]:
            if test2["name"] in seen:
                continue

            # Compare test names
            similarity = SequenceMatcher(
                None,
                test1["name"].lower(),
                test2["name"].lower(),
            ).ratio()

            if similarity > threshold:
                similar.append(test2)
                seen.add(test2["name"])

        if len(similar) > 1:
            similar_groups.append(similar)

    return similar_groups


def find_overlapping_coverage(analyzer):
    """
    Find tests that test the same targets.
    """
    target_to_tests = defaultdict(list)

    for test_name, targets in analyzer.test_targets.items():
        for target in targets:
            target_to_tests[target].append(test_name)

    # Find targets tested by multiple tests
    overlapping = {}
    for target, tests in target_to_tests.items():
        if len(tests) > 1:
            overlapping[target] = tests

    return overlapping


def identify_redundant_patterns(analyzer):
    """
    Identify common redundancy patterns.
    """
    patterns = {
        "example_with_property": [],
        "duplicate_assertions": [],
        "similar_fixtures": [],
        "could_be_parameterized": [],
    }

    # Find example tests that could be replaced by property tests
    for test_name, decorators in analyzer.decorators.items():
        if "given" in decorators or "hypothesis" in str(decorators):
            # This is a property test - look for related example tests
            base_name = test_name.replace("_property", "").replace("_hypothesis", "")
            for other_test in analyzer.test_functions:
                if base_name in other_test["name"] and other_test["name"] != test_name:
                    patterns["example_with_property"].append(
                        {
                            "property_test": test_name,
                            "example_test": other_test["name"],
                        },
                    )

    # Find tests with identical assertion patterns
    assertion_patterns = defaultdict(list)
    for test_name, assertions in analyzer.assertions.items():
        pattern = tuple(assertions)
        if pattern:
            assertion_patterns[pattern].append(test_name)

    for pattern, tests in assertion_patterns.items():
        if len(tests) > 2:  # More than 2 tests with same assertion pattern
            patterns["duplicate_assertions"].append(
                {
                    "pattern": pattern,
                    "tests": tests,
                },
            )

    # Find tests that could be parameterized
    test_families = defaultdict(list)
    for test in analyzer.test_functions:
        # Extract base name (remove numbers, variations)
        import re

        base = re.sub(r"_\d+$", "", test["name"])
        base = re.sub(r"_(small|medium|large|valid|invalid)$", "", base)
        test_families[base].append(test["name"])

    for base, tests in test_families.items():
        if len(tests) > 2:
            patterns["could_be_parameterized"].append(
                {
                    "base": base,
                    "tests": tests,
                },
            )

    return patterns


def analyze_test_directory(test_dir):
    """
    Analyze all test files in a directory.
    """
    analyzer = TestAnalyzer()
    test_files = []

    for path in Path(test_dir).rglob("test_*.py"):
        test_files.append(path)
        with open(path) as f:
            try:
                tree = ast.parse(f.read())
                analyzer.current_file = str(path)
                analyzer.visit(tree)
            except SyntaxError as e:
                print(f"Syntax error in {path}: {e}")

    return analyzer, test_files


def generate_report(analyzer, test_files):
    """
    Generate redundancy analysis report.
    """
    report = []
    report.append("# Test Redundancy Analysis Report\n")
    report.append(f"Analyzed {len(test_files)} test files\n")
    report.append(f"Found {len(analyzer.test_functions)} test functions\n\n")

    # Find similar tests
    similar = find_similar_tests(analyzer.test_functions)
    if similar:
        report.append("## Similar Test Names (Potential Duplicates)\n")
        for group in similar:
            report.append("\n### Similar group:\n")
            for test in group:
                report.append(f"  - {test['name']} ({test['file']}:{test['line']})\n")

    # Find overlapping coverage
    overlapping = find_overlapping_coverage(analyzer)
    if overlapping:
        report.append("\n## Tests with Overlapping Coverage\n")
        for target, tests in sorted(overlapping.items(), key=lambda x: len(x[1]), reverse=True):
            if len(tests) > 2:  # Only show if 3+ tests cover same target
                report.append(f"\n### Target: {target}\n")
                report.append(f"  Tested by {len(tests)} tests:\n")
                for test in tests[:5]:  # Show first 5
                    report.append(f"  - {test}\n")
                if len(tests) > 5:
                    report.append(f"  ... and {len(tests) - 5} more\n")

    # Identify redundancy patterns
    patterns = identify_redundant_patterns(analyzer)

    if patterns["example_with_property"]:
        report.append("\n## Example Tests with Property Test Coverage\n")
        report.append("These example tests might be redundant with property tests:\n")
        for item in patterns["example_with_property"]:
            report.append(f"  - {item['example_test']} (covered by {item['property_test']})\n")

    if patterns["duplicate_assertions"]:
        report.append("\n## Tests with Duplicate Assertion Patterns\n")
        for item in patterns["duplicate_assertions"]:
            report.append(f"\n### Pattern: {item['pattern']}\n")
            for test in item["tests"]:
                report.append(f"  - {test}\n")

    if patterns["could_be_parameterized"]:
        report.append("\n## Tests That Could Be Parameterized\n")
        report.append("These test families could potentially use pytest.mark.parametrize:\n")
        for item in patterns["could_be_parameterized"]:
            report.append(f"\n### Base: {item['base']}\n")
            for test in item["tests"]:
                report.append(f"  - {test}\n")

    # Statistics
    report.append("\n## Statistics\n")
    report.append(f"- Total tests: {len(analyzer.test_functions)}\n")
    report.append(f"- Tests using fixtures: {len(analyzer.fixtures_used)}\n")
    report.append(f"- Tests with decorators: {len(analyzer.decorators)}\n")
    report.append(
        f"- Unique test targets: {len(set(functools.reduce(operator.iadd, analyzer.test_targets.values(), [])))}\n",
    )

    # Recommendations
    report.append("\n## Recommendations\n")

    if similar:
        report.append("1. **Review similar test names** - These might be duplicates\n")

    if patterns["example_with_property"]:
        report.append(
            "2. **Remove redundant example tests** - Property tests already cover these\n",
        )

    if patterns["could_be_parameterized"]:
        report.append(
            "3. **Use parametrized tests** - Reduce code duplication with @pytest.mark.parametrize\n",
        )

    if overlapping:
        high_overlap = [t for t, tests in overlapping.items() if len(tests) > 3]
        if high_overlap:
            report.append(
                f"4. **Consolidate overlapping tests** - {len(high_overlap)} targets are tested by 4+ tests\n",
            )

    return "".join(report)


@click.command()
@click.option("--test-dir", default="ml/tests", help="Directory containing tests")
@click.option("--output", default=None, help="Output file for report")
@click.option("--verbose", is_flag=True, help="Verbose output")
def main(test_dir, output, verbose):
    """
    Analyze test suite for redundancy.
    """
    print(f"Analyzing tests in {test_dir}...")

    analyzer, test_files = analyze_test_directory(test_dir)
    report = generate_report(analyzer, test_files)

    if output:
        with open(output, "w") as f:
            f.write(report)
        print(f"Report written to {output}")
    else:
        print(report)

    # Summary statistics
    print("\n=== Quick Summary ===")
    print(f"Total test files: {len(test_files)}")
    print(f"Total test functions: {len(analyzer.test_functions)}")

    similar = find_similar_tests(analyzer.test_functions)
    if similar:
        total_similar = sum(len(group) for group in similar)
        print(f"Similar tests that might be duplicates: {total_similar}")

    patterns = identify_redundant_patterns(analyzer)
    if patterns["example_with_property"]:
        print(
            f"Example tests redundant with property tests: {len(patterns['example_with_property'])}",
        )

    if patterns["could_be_parameterized"]:
        total_param = sum(len(item["tests"]) for item in patterns["could_be_parameterized"])
        print(f"Tests that could be parameterized: {total_param}")


if __name__ == "__main__":
    main()
