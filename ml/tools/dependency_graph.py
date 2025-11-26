"""AST-based import analysis to build module dependency graph.

Identifies circular dependencies, coupling hotspots, and import patterns.

Usage:
    python -m ml.tools.dependency_graph --root ml \
        --output reports/architecture/dependency_graph.json
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_imports(file_path: Path) -> tuple[list[str], list[str]]:
    """Extract import statements from a Python file.

    Args:
        file_path: Path to Python file

    Returns:
        Tuple of (absolute imports, relative imports)
    """
    try:
        with file_path.open(encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return [], []

    absolute_imports: list[str] = []
    relative_imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                absolute_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level

            if level > 0:
                # Relative import
                relative_imports.append(f"{'.' * level}{module}")
            else:
                absolute_imports.append(module)

    return absolute_imports, relative_imports


def resolve_relative_import(
    source_file: Path,
    relative_import: str,
    root: Path,
) -> str | None:
    """Resolve a relative import to an absolute module path.

    Args:
        source_file: The file containing the import
        relative_import: The relative import string (e.g., '..foo')
        root: Project root directory

    Returns:
        Absolute module path or None if cannot resolve
    """
    dots = 0
    for char in relative_import:
        if char == ".":
            dots += 1
        else:
            break

    remainder = relative_import[dots:]

    # Navigate up directories
    current = source_file.parent
    for _ in range(dots):
        current = current.parent

    # Build module path
    try:
        rel_to_root = current.relative_to(root)
        base = str(rel_to_root).replace("/", ".")
        if remainder:
            return f"{base}.{remainder}"
        return base
    except ValueError:
        return None


def build_dependency_graph(
    root: Path,
    *,
    include_external: bool = False,
) -> dict[str, Any]:
    """Build a dependency graph for all Python files under root.

    Args:
        root: Root directory to scan
        include_external: Include imports outside root package

    Returns:
        Dependency graph data structure
    """
    root = root.resolve()
    root_package = root.name

    # Maps module → list of modules it imports
    dependencies: dict[str, list[str]] = defaultdict(list)
    # Maps module → list of modules that import it
    dependents: dict[str, list[str]] = defaultdict(list)
    # All modules found
    modules: set[str] = set()

    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        # Convert file path to module path
        rel_path = py_file.relative_to(root.parent)
        if py_file.name == "__init__.py":
            module_path = str(rel_path.parent).replace("/", ".")
        else:
            module_path = str(rel_path.with_suffix("")).replace("/", ".")

        modules.add(module_path)

        abs_imports, rel_imports = extract_imports(py_file)

        # Process absolute imports
        for imp in abs_imports:
            if imp.startswith(root_package):
                dependencies[module_path].append(imp)
                dependents[imp].append(module_path)
            elif include_external:
                dependencies[module_path].append(imp)

        # Process relative imports
        for rel_imp in rel_imports:
            resolved = resolve_relative_import(py_file, rel_imp, root.parent)
            if resolved and resolved.startswith(root_package):
                dependencies[module_path].append(resolved)
                dependents[resolved].append(module_path)

    return {
        "modules": sorted(modules),
        "dependencies": {k: sorted(set(v)) for k, v in dependencies.items()},
        "dependents": {k: sorted(set(v)) for k, v in dependents.items()},
    }


def find_circular_dependencies(
    dependencies: dict[str, list[str]],
) -> list[list[str]]:
    """Find circular dependency chains.

    Args:
        dependencies: Module dependency map

    Returns:
        List of circular dependency chains
    """
    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()

    def dfs(module: str, path: list[str]) -> None:
        if module in rec_stack:
            # Found cycle - extract it
            cycle_start = path.index(module)
            cycle = path[cycle_start:] + [module]
            # Normalize cycle (start from lexicographically smallest)
            min_idx = cycle.index(min(cycle[:-1]))
            normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
            if normalized not in cycles:
                cycles.append(normalized)
            return

        if module in visited:
            return

        visited.add(module)
        rec_stack.add(module)

        for dep in dependencies.get(module, []):
            dfs(dep, path + [module])

        rec_stack.discard(module)

    for module in dependencies:
        dfs(module, [])

    return cycles


def analyze_coupling(
    dependencies: dict[str, list[str]],
    dependents: dict[str, list[str]],
) -> dict[str, Any]:
    """Analyze coupling metrics.

    Args:
        dependencies: What each module imports
        dependents: What imports each module

    Returns:
        Coupling analysis
    """
    # Afferent coupling (incoming) - who depends on me
    afferent = {mod: len(deps) for mod, deps in dependents.items()}

    # Efferent coupling (outgoing) - who do I depend on
    efferent = {mod: len(deps) for mod, deps in dependencies.items()}

    # High coupling modules
    high_afferent = sorted(afferent.items(), key=lambda x: -x[1])[:20]
    high_efferent = sorted(efferent.items(), key=lambda x: -x[1])[:20]

    return {
        "high_afferent_coupling": dict(high_afferent),
        "high_efferent_coupling": dict(high_efferent),
        "total_modules": len(set(dependencies.keys()) | set(dependents.keys())),
    }


def filter_by_domain(
    graph: dict[str, Any],
    domain: str,
) -> dict[str, Any]:
    """Filter dependency graph to a specific domain."""
    domain_prefix = domain.replace("/", ".")

    filtered_modules = [m for m in graph["modules"] if m.startswith(domain_prefix)]
    filtered_deps = {
        k: [v for v in vals if v.startswith(domain_prefix)]
        for k, vals in graph["dependencies"].items()
        if k.startswith(domain_prefix)
    }
    filtered_dependents = {
        k: [v for v in vals if v.startswith(domain_prefix)]
        for k, vals in graph["dependents"].items()
        if k.startswith(domain_prefix)
    }

    return {
        "modules": filtered_modules,
        "dependencies": {k: v for k, v in filtered_deps.items() if v},
        "dependents": {k: v for k, v in filtered_dependents.items() if v},
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for dependency graph builder."""
    parser = argparse.ArgumentParser(
        description="Build module dependency graph via AST analysis",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root package directory to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for dependency_graph.json",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter to specific domain (e.g., ml/features)",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include external package imports",
    )
    parser.add_argument(
        "--find-cycles",
        action="store_true",
        help="Find circular dependencies",
    )

    args = parser.parse_args(argv)

    # Build graph
    print(f"Analyzing {args.root}...")
    graph = build_dependency_graph(args.root, include_external=args.include_external)

    # Filter by domain if specified
    if args.domain:
        graph = filter_by_domain(graph, args.domain)

    # Analyze coupling
    coupling = analyze_coupling(graph["dependencies"], graph["dependents"])

    # Find cycles if requested
    cycles: list[list[str]] = []
    if args.find_cycles:
        print("Finding circular dependencies...")
        cycles = find_circular_dependencies(graph["dependencies"])

    # Build output
    output = {
        "summary": {
            "total_modules": len(graph["modules"]),
            "total_dependencies": sum(len(v) for v in graph["dependencies"].values()),
            "circular_dependencies": len(cycles),
        },
        "coupling": coupling,
        "dependencies": graph["dependencies"],
        "dependents": graph["dependents"],
    }

    if cycles:
        output["circular_dependencies"] = cycles

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"Modules analyzed: {len(graph['modules'])}")
    print(f"Total dependencies: {output['summary']['total_dependencies']}")
    if cycles:
        print(f"Circular dependencies found: {len(cycles)}")
        for cycle in cycles[:5]:
            print(f"  {' -> '.join(cycle)}")
    print(f"Output written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
