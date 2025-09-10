#!/usr/bin/env python3
"""
Apply pytest markers to tests based on their characteristics.

This tool analyzes test files and automatically suggests or applies appropriate markers
based on imports, fixtures, and content.

"""

import ast
import re
from pathlib import Path

import click


def detect_test_characteristics(filepath: Path) -> dict[str, bool]:
    """
    Detect characteristics of a test file to determine appropriate markers.
    """
    with open(filepath) as f:
        content = f.read()

    characteristics = {
        "database": False,
        "redis": False,
        "docker": False,
        "mock_heavy": False,
        "property": False,
        "slow": False,
        "integration": False,
        "unit": False,
        "flaky": False,
        "serial": False,
        "parallel_safe": True,  # Default to parallel safe
    }

    # Check imports
    if "from ml.stores" in content or "postgres" in content.lower():
        characteristics["database"] = True
        characteristics["serial"] = True
        characteristics["parallel_safe"] = False

    if "redis" in content.lower():
        characteristics["redis"] = True

    if "docker" in content.lower() or "container" in content.lower():
        characteristics["docker"] = True

    if "from hypothesis import" in content or "@given" in content:
        characteristics["property"] = True

    # Count mocks
    mock_count = content.count("mock") + content.count("Mock") + content.count("patch")
    if mock_count > 10:
        characteristics["mock_heavy"] = True

    # Check for time.sleep or asyncio.sleep (indicates potential flakiness)
    if "time.sleep" in content or "asyncio.sleep" in content:
        characteristics["flaky"] = True
        characteristics["slow"] = True

    # Check fixtures used
    if "database_engine" in content or "postgres_connection" in content:
        characteristics["database"] = True
        characteristics["integration"] = True
        characteristics["serial"] = True
        characteristics["parallel_safe"] = False

    # Check file location
    if "/integration/" in str(filepath):
        characteristics["integration"] = True
    elif "/unit/" in str(filepath):
        characteristics["unit"] = True
    elif "/e2e/" in str(filepath):
        characteristics["integration"] = True
        characteristics["slow"] = True

    # Check for performance tests
    if "benchmark" in content or "@pytest.mark.benchmark" in content:
        characteristics["slow"] = True

    # Parse AST to check test function characteristics
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                # Check for large tests (>100 lines)
                if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
                    if node.end_lineno - node.lineno > 100:
                        characteristics["slow"] = True

                # Check for concurrent/async tests
                if "concurrent" in node.name or "async" in node.name or "thread" in node.name:
                    characteristics["flaky"] = True
    except Exception:
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "AST parse failed while detecting test characteristics",
            exc_info=True,
        )

    return characteristics


def suggest_markers(characteristics: dict[str, bool]) -> list[str]:
    """
    Suggest pytest markers based on test characteristics.
    """
    markers = []

    # Priority order matters - most specific first
    if characteristics.get("property"):
        markers.append("property")

    if characteristics.get("database"):
        markers.append("database")
        markers.append("serial")

    if characteristics.get("redis"):
        markers.append("redis")

    if characteristics.get("docker"):
        markers.append("docker")
        markers.append("slow")

    if characteristics.get("flaky"):
        markers.append("flaky")

    if characteristics.get("slow"):
        markers.append("slow")
    elif characteristics.get("parallel_safe") and not characteristics.get("database"):
        markers.append("parallel_safe")

    if characteristics.get("integration"):
        markers.append("integration")
    elif characteristics.get("unit"):
        markers.append("unit")

    return markers


def apply_markers_to_file(filepath: Path, markers: list[str], dry_run: bool = True) -> str:
    """
    Apply markers to a test file.
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Check if markers already exist
    has_markers = any("@pytest.mark." in line for line in lines[:50])

    if has_markers:
        return f"Skipping {filepath.name} - already has markers"

    # Find the first test function or class
    insert_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def test_") or line.strip().startswith("class Test"):
            # Back up to find decorators if any
            j = i - 1
            while j >= 0 and lines[j].strip().startswith("@"):
                j -= 1
            insert_line = j + 1
            break

    if insert_line is None:
        return f"No tests found in {filepath.name}"

    # Create marker decorators
    marker_lines = [f"@pytest.mark.{marker}\n" for marker in markers]

    if not dry_run:
        # Insert markers
        for marker_line in reversed(marker_lines):
            lines.insert(insert_line, marker_line)

        # Add pytest import if needed
        if "import pytest" not in "".join(lines[:20]):
            # Find appropriate place for import
            for i, line in enumerate(lines):
                if line.startswith(("import ", "from ")):
                    lines.insert(i, "import pytest\n")
                    break

        # Write back
        with open(filepath, "w") as f:
            f.writelines(lines)

    return f"Applied markers to {filepath.name}: {', '.join(markers)}"


def analyze_test_directory(test_dir: Path) -> dict[str, list[str]]:
    """
    Analyze all test files and suggest markers.
    """
    suggestions = {}

    for filepath in test_dir.rglob("test_*.py"):
        characteristics = detect_test_characteristics(filepath)
        markers = suggest_markers(characteristics)
        if markers:
            suggestions[str(filepath.relative_to(test_dir))] = markers

    return suggestions


@click.command()
@click.option("--test-dir", default="ml/tests", help="Test directory to analyze")
@click.option("--apply", is_flag=True, help="Actually apply markers (default is dry run)")
@click.option("--output", default=None, help="Output file for suggestions")
def main(test_dir, apply, output):
    """
    Analyze tests and suggest or apply appropriate markers.
    """
    test_path = Path(test_dir)
    if not test_path.exists():
        click.echo(f"Error: {test_path} does not exist")
        return

    click.echo(f"Analyzing {test_path}...")

    suggestions = analyze_test_directory(test_path)

    # Group by marker for summary
    marker_stats = {}
    for markers in suggestions.values():
        for marker in markers:
            marker_stats[marker] = marker_stats.get(marker, 0) + 1

    click.echo("\n=== Marker Statistics ===")
    for marker, count in sorted(marker_stats.items(), key=lambda x: x[1], reverse=True):
        click.echo(f"  {marker}: {count} files")

    click.echo(f"\n=== File Analysis ({len(suggestions)} files) ===")

    # Apply or display suggestions
    for filepath, markers in sorted(suggestions.items()):
        if apply:
            result = apply_markers_to_file(test_path / filepath, markers, dry_run=False)
            click.echo(result)
        else:
            click.echo(f"{filepath}: {', '.join(markers)}")

    if output:
        with open(output, "w") as f:
            f.write("# Test Marker Suggestions\n\n")
            f.write("## Statistics\n")
            for marker, count in sorted(marker_stats.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {marker}: {count} files\n")
            f.write("\n## File Suggestions\n")
            for filepath, markers in sorted(suggestions.items()):
                f.write(f"- {filepath}: {', '.join(markers)}\n")
        click.echo(f"\nSuggestions written to {output}")

    if not apply:
        click.echo("\n💡 To apply markers, run with --apply flag")
        click.echo("   This will modify test files to add @pytest.mark decorators")


if __name__ == "__main__":
    main()
