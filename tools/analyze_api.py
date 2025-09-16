#!/usr/bin/env python3
"""
Analyze API index for actionable insights.
"""

import json
from collections import defaultdict, Counter
from pathlib import Path
import argparse


def analyze_api_index(index_path: Path):
    """
    Generate actionable reports from API index.
    """

    with open(index_path) as f:
        data = json.load(f)

    reports = {}

    # 1. Find methods that should be private (get/set patterns without property decorators)
    getter_setters = []
    for item in data:
        if item["kind"] == "method" and item["name"].startswith(("get_", "set_")):
            # Could check if there's a corresponding @property
            getter_setters.append(item["qualname"])
    reports["potential_properties"] = getter_setters[:20]

    # 2. Find test files that could be consolidated
    test_modules = defaultdict(list)
    for item in data:
        if "test" in item["module"]:
            base = (
                item["module"].split(".test_")[0] if ".test_" in item["module"] else item["module"]
            )
            test_modules[base].append(item["module"])

    fragmented_tests = {k: v for k, v in test_modules.items() if len(set(v)) > 3}
    reports["fragmented_tests"] = fragmented_tests

    # 3. Find undocumented public APIs (high priority)
    undocumented_public = []
    for item in data:
        if (
            item["kind"] in ["class", "function"]
            and not item.get("docstring")
            and "test" not in item["module"]
            and not item["name"].startswith("_")
        ):
            undocumented_public.append(item["qualname"])
    reports["undocumented_public_apis"] = undocumented_public[:30]

    # 4. Find circular dependency risks (classes with many cross-references)
    class_refs = defaultdict(set)
    for item in data:
        if item.get("docstring"):
            # Simple heuristic: look for other class names in docstrings
            for other in data:
                if other["kind"] == "class" and other["name"] in item.get("docstring", ""):
                    class_refs[item["module"]].add(other["module"])

    circular_risks = {k: list(v) for k, v in class_refs.items() if len(v) > 5}
    reports["circular_dependency_risks"] = circular_risks

    # 5. Generate refactoring candidates
    refactor_candidates = []

    # Large classes
    class_methods = defaultdict(int)
    for item in data:
        if item["kind"] == "method":
            class_name = ".".join(item["qualname"].split(".")[:-1])
            class_methods[class_name] += 1

    for cls, count in class_methods.items():
        if count > 20:
            refactor_candidates.append(
                {
                    "class": cls,
                    "methods": count,
                    "suggestion": "Consider splitting into smaller classes or using composition",
                },
            )

    reports["refactoring_candidates"] = refactor_candidates

    # 6. API stability assessment
    version_patterns = []
    for item in data:
        name_lower = item["name"].lower()
        if any(p in name_lower for p in ["v1", "v2", "_new", "_old", "legacy"]):
            version_patterns.append(item["qualname"])
    reports["unstable_apis"] = version_patterns[:20]

    return reports


def generate_markdown_report(reports: dict) -> str:
    """
    Generate markdown report from analysis.
    """

    md = ["# ML API Analysis Report\n"]

    if reports.get("undocumented_public_apis"):
        md.append("## 🔴 Undocumented Public APIs (Priority 1)")
        md.append("These should have docstrings:")
        for api in reports["undocumented_public_apis"][:10]:
            md.append(f"- `{api}`")
        md.append("")

    if reports.get("refactoring_candidates"):
        md.append("## 🟡 Refactoring Candidates (Priority 2)")
        for candidate in reports["refactoring_candidates"]:
            md.append(f"- **{candidate['class']}**: {candidate['methods']} methods")
            md.append(f"  - {candidate['suggestion']}")
        md.append("")

    if reports.get("unstable_apis"):
        md.append("## 🟠 API Stability Issues")
        md.append("APIs with version indicators:")
        for api in reports["unstable_apis"][:10]:
            md.append(f"- `{api}`")
        md.append("")

    if reports.get("potential_properties"):
        md.append("## 💡 Potential Property Conversions")
        md.append("Consider using @property for these getters/setters:")
        for api in reports["potential_properties"][:10]:
            md.append(f"- `{api}`")
        md.append("")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Analyze API index")
    parser.add_argument("--index", default="ml/public_api_index.json")
    parser.add_argument("--output", default="ml/api_analysis.md")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of markdown")
    args = parser.parse_args()

    reports = analyze_api_index(Path(args.index))

    if args.json:
        # Output detailed JSON
        output = json.dumps(reports, indent=2)
        print(output)
    else:
        # Output markdown summary
        md = generate_markdown_report(reports)
        Path(args.output).write_text(md)
        print(f"Report written to {args.output}")

        # Print summary stats
        print("\nSummary:")
        print(f"- Undocumented APIs: {len(reports.get('undocumented_public_apis', []))}")
        print(f"- Refactoring candidates: {len(reports.get('refactoring_candidates', []))}")
        print(f"- Unstable APIs: {len(reports.get('unstable_apis', []))}")


if __name__ == "__main__":
    main()
