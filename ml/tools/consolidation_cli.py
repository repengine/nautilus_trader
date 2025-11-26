"""Orchestrate concern mapping and architecture analysis tools.

Runs all analysis tools and generates consolidated reports per domain.

Usage:
    python -m ml.tools.consolidation_cli --domain features --output reports/architecture/
    python -m ml.tools.consolidation_cli --all-domains --output reports/architecture/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DOMAINS = ["features", "registry", "stores", "actors", "orchestration"]

DOMAIN_PATHS = {
    "features": "ml/features",
    "registry": "ml/registry",
    "stores": "ml/stores",
    "actors": "ml/actors",
    "orchestration": "ml/orchestration",
}


def run_tool(
    module: str,
    args: list[str],
    *,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a Python module with arguments."""
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        check=False,
    )


def ensure_api_index(output_dir: Path) -> Path:
    """Ensure api_index.json exists, generate if needed."""
    api_index_path = output_dir / "api_index.json"
    if not api_index_path.exists():
        print("Generating API index...")
        result = run_tool(
            "tools.api_index",
            ["--root", "ml", "--output", str(api_index_path)],
            capture_output=False,
        )
        if result.returncode != 0:
            # Try alternative invocation
            subprocess.run(
                [sys.executable, "tools/api_index.py", "--root", "ml", "--output", str(api_index_path)],
                check=True,
            )
    return api_index_path


def run_concern_mapper(
    api_index: Path,
    output_dir: Path,
    domain: str | None = None,
) -> Path:
    """Run concern mapper tool."""
    output_file = output_dir / f"concern_map{'_' + domain if domain else ''}.json"
    args = ["--input", str(api_index), "--output", str(output_file)]
    if domain:
        args.extend(["--domain", DOMAIN_PATHS.get(domain, domain)])

    result = run_tool("ml.tools.concern_mapper", args, capture_output=False)
    if result.returncode != 0:
        print(f"Warning: concern_mapper returned {result.returncode}")
    return output_file


def run_cross_reference(
    api_index: Path,
    output_dir: Path,
    domain: str | None = None,
) -> Path:
    """Run cross-reference finder."""
    output_file = output_dir / f"cross_refs{'_' + domain if domain else ''}.json"
    args = [
        "--input", str(api_index),
        "--output", str(output_file),
        "--include-overlap-pairs",
    ]
    if domain:
        args.extend(["--domain", DOMAIN_PATHS.get(domain, domain)])

    result = run_tool("ml.tools.cross_reference", args, capture_output=False)
    if result.returncode != 0:
        print(f"Warning: cross_reference returned {result.returncode}")
    return output_file


def run_dependency_graph(
    output_dir: Path,
    domain: str | None = None,
) -> Path:
    """Run dependency graph builder."""
    output_file = output_dir / f"dependency_graph{'_' + domain if domain else ''}.json"
    args = [
        "--root", "ml",
        "--output", str(output_file),
        "--find-cycles",
    ]
    if domain:
        args.extend(["--domain", DOMAIN_PATHS.get(domain, domain)])

    result = run_tool("ml.tools.dependency_graph", args, capture_output=False)
    if result.returncode != 0:
        print(f"Warning: dependency_graph returned {result.returncode}")
    return output_file


def generate_domain_report(
    output_dir: Path,
    domain: str,
) -> dict[str, Any]:
    """Generate a consolidated report for a domain."""
    domain_dir = output_dir / domain

    # Load generated files
    concern_map_file = output_dir / f"concern_map_{domain}.json"
    cross_refs_file = output_dir / f"cross_refs_{domain}.json"
    dep_graph_file = output_dir / f"dependency_graph_{domain}.json"

    report: dict[str, Any] = {
        "domain": domain,
        "domain_path": DOMAIN_PATHS.get(domain, f"ml/{domain}"),
    }

    if concern_map_file.exists():
        with concern_map_file.open() as f:
            concern_data = json.load(f)
            report["concerns"] = concern_data.get("summary", {})

    if cross_refs_file.exists():
        with cross_refs_file.open() as f:
            cross_ref_data = json.load(f)
            report["cross_references"] = cross_ref_data.get("summary", {})
            report["overlap_pairs"] = cross_ref_data.get("overlap_pairs", [])[:10]

    if dep_graph_file.exists():
        with dep_graph_file.open() as f:
            dep_data = json.load(f)
            report["dependencies"] = dep_data.get("summary", {})
            report["coupling"] = dep_data.get("coupling", {})
            report["circular_dependencies"] = dep_data.get("circular_dependencies", [])

    return report


def main(argv: list[str] | None = None) -> int:
    """Entry point for consolidation CLI."""
    parser = argparse.ArgumentParser(
        description="Run all architecture analysis tools",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=DOMAINS,
        help="Specific domain to analyze",
    )
    parser.add_argument(
        "--all-domains",
        action="store_true",
        help="Analyze all domains",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--skip-api-index",
        action="store_true",
        help="Skip API index generation (use existing)",
    )

    args = parser.parse_args(argv)

    if not args.domain and not args.all_domains:
        parser.error("Must specify --domain or --all-domains")

    args.output.mkdir(parents=True, exist_ok=True)

    # Ensure API index exists
    if not args.skip_api_index:
        api_index = ensure_api_index(args.output)
    else:
        api_index = args.output / "api_index.json"
        if not api_index.exists():
            parser.error(f"API index not found: {api_index}")

    # Determine domains to analyze
    domains = DOMAINS if args.all_domains else [args.domain]

    # Run analysis for each domain
    reports: dict[str, Any] = {}
    for domain in domains:
        print(f"\n{'=' * 60}")
        print(f"Analyzing domain: {domain}")
        print("=" * 60)

        # Run tools
        run_concern_mapper(api_index, args.output, domain)
        run_cross_reference(api_index, args.output, domain)
        run_dependency_graph(args.output, domain)

        # Generate report
        reports[domain] = generate_domain_report(args.output, domain)

    # Write consolidated report
    consolidated_file = args.output / "consolidated_analysis.json"
    with consolidated_file.open("w") as f:
        json.dump(reports, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Analysis Complete")
    print("=" * 60)
    print(f"Consolidated report: {consolidated_file}")

    # Print summary
    for domain, report in reports.items():
        print(f"\n{domain.upper()}:")
        if "concerns" in report:
            print(f"  Concerns mapped: {report['concerns'].get('total_concerns', 'N/A')}")
            print(f"  Methods mapped: {report['concerns'].get('total_methods', 'N/A')}")
        if "cross_references" in report:
            print(f"  Cross-referenced names: {report['cross_references'].get('total_cross_referenced_names', 'N/A')}")
        if "circular_dependencies" in report:
            cycles = report["circular_dependencies"]
            print(f"  Circular dependencies: {len(cycles)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
