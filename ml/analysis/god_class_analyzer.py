"""
God class analysis tools for systematic refactoring.

This module provides comprehensive analysis of large Python classes to identify:
- Method groupings by naming patterns
- Call dependencies between methods
- Variable usage and cohesion metrics
- Shared patterns across multiple classes
- Extraction candidates for refactoring

All analysis is read-only and deterministic.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass
class MethodInfo:
    """Information about a single method."""

    name: str
    line_number: int
    calls: list[str] = field(default_factory=list)
    variables_used: set[str] = field(default_factory=set)
    variables_assigned: set[str] = field(default_factory=set)
    is_private: bool = False
    decorators: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    complexity: int = 0  # Number of branches/loops


@dataclass
class ResponsibilityGroup:
    """A group of related methods representing a responsibility."""

    name: str
    methods: list[str]
    description: str
    cohesion_score: float = 0.0
    extraction_recommendation: str = ""


@dataclass
class SharedPattern:
    """A pattern shared across multiple classes."""

    pattern_name: str
    pattern_type: str  # validation, lifecycle, database, error_handling, metrics
    affected_classes: list[str]
    description: str
    extraction_recommendation: str
    priority: int = 0  # Higher = more important to extract


@dataclass
class ClassAnalysis:
    """Complete analysis results for a single class."""

    class_name: str
    file_path: Path
    line_count: int
    method_count: int
    methods: dict[str, MethodInfo]
    method_groups: dict[str, list[str]]
    call_graph: dict[str, list[str]]
    cohesion_metrics: dict[str, float]
    responsibilities: list[ResponsibilityGroup]
    patterns: list[str]


class MethodGrouper:
    """Groups methods by naming patterns and semantic relationships."""

    def __init__(self) -> None:
        """Initialize method grouper."""
        self.common_prefixes = [
            "read", "write", "get", "set", "fetch", "save", "load",
            "validate", "check", "verify", "ensure",
            "create", "build", "make", "generate",
            "process", "handle", "execute", "run",
            "init", "start", "stop", "close", "open",
            "update", "delete", "remove", "add",
            "parse", "format", "convert", "transform",
            "emit", "publish", "send", "receive"
        ]

        self.common_suffixes = [
            "data", "features", "model", "schema", "config",
            "store", "db", "file", "cache", "registry",
            "event", "message", "metadata", "manifest",
            "error", "exception", "warning", "metrics"
        ]

    def group_by_prefix(self, method_names: list[str]) -> dict[str, list[str]]:
        """
        Group methods by common prefixes.

        Args:
            method_names: List of method names to group

        Returns:
            Dictionary mapping prefix to list of method names

        Example:
            >>> grouper = MethodGrouper()
            >>> names = ["read_data", "read_features", "write_data", "parse_config"]
            >>> groups = grouper.group_by_prefix(names)
            >>> assert "read" in groups
            >>> assert len(groups["read"]) == 2
        """
        groups: dict[str, list[str]] = defaultdict(list)

        for method_name in method_names:
            # Check each common prefix
            for prefix in self.common_prefixes:
                if method_name.startswith(f"{prefix}_"):
                    groups[prefix].append(method_name)
                    break

        # Convert defaultdict to regular dict
        return dict(groups)

    def group_by_suffix(self, method_names: list[str]) -> dict[str, list[str]]:
        """
        Group methods by common suffixes.

        Args:
            method_names: List of method names to group

        Returns:
            Dictionary mapping suffix to list of method names

        Example:
            >>> grouper = MethodGrouper()
            >>> names = ["read_data", "write_data", "parse_data", "read_features"]
            >>> groups = grouper.group_by_suffix(names)
            >>> assert "data" in groups
            >>> assert len(groups["data"]) == 3
        """
        groups: dict[str, list[str]] = defaultdict(list)

        for method_name in method_names:
            # Check each common suffix
            for suffix in self.common_suffixes:
                if method_name.endswith(f"_{suffix}"):
                    groups[suffix].append(method_name)
                    break

        # Convert defaultdict to regular dict
        return dict(groups)

    def group_by_pattern(self, method_names: list[str]) -> dict[str, list[str]]:
        """
        Group methods by combined prefix and suffix patterns.

        Args:
            method_names: List of method names to group

        Returns:
            Dictionary mapping pattern name to list of method names
        """
        prefix_groups = self.group_by_prefix(method_names)
        suffix_groups = self.group_by_suffix(method_names)

        # Combine groups, prefixed with type
        combined: dict[str, list[str]] = {}

        for prefix, methods in prefix_groups.items():
            combined[f"prefix:{prefix}"] = methods

        for suffix, methods in suffix_groups.items():
            combined[f"suffix:{suffix}"] = methods

        return combined


class CallGraphBuilder:
    """Builds call graphs showing method dependencies."""

    def build_call_graph(self, methods: dict[str, MethodInfo]) -> dict[str, list[str]]:
        """
        Build a call graph from method information.

        Args:
            methods: Dictionary mapping method names to MethodInfo objects

        Returns:
            Dictionary mapping caller method to list of called methods

        Example:
            >>> builder = CallGraphBuilder()
            >>> methods = {
            ...     "method_a": MethodInfo(name="method_a", line_number=10, calls=["method_b", "method_c"]),
            ...     "method_b": MethodInfo(name="method_b", line_number=20, calls=[]),
            ...     "method_c": MethodInfo(name="method_c", line_number=30, calls=["method_b"])
            ... }
            >>> graph = builder.build_call_graph(methods)
            >>> assert graph["method_a"] == ["method_b", "method_c"]
        """
        graph: dict[str, list[str]] = {}

        for method_name, method_info in methods.items():
            # Filter calls to only include methods within this class
            calls_in_class = [
                call for call in method_info.calls
                if call in methods
            ]
            graph[method_name] = calls_in_class

        return graph

    def find_transitive_calls(
        self,
        graph: dict[str, list[str]],
        method_name: str,
        visited: set[str] | None = None,
    ) -> set[str]:
        """
        Find all methods transitively called by a method.

        Args:
            graph: Call graph dictionary
            method_name: Starting method
            visited: Set of already visited methods (for cycle detection)

        Returns:
            Set of all method names transitively called

        Example:
            >>> builder = CallGraphBuilder()
            >>> graph = {"A": ["B"], "B": ["C"], "C": []}
            >>> transitive = builder.find_transitive_calls(graph, "A")
            >>> assert transitive == {"B", "C"}
        """
        if visited is None:
            visited = set()

        if method_name in visited:
            return set()  # Cycle detected, stop

        visited.add(method_name)
        transitive = set()

        for called in graph.get(method_name, []):
            # Don't include self-calls in transitive set (they're cycles)
            if called != method_name:
                transitive.add(called)
                # Recursively find what the called method calls
                transitive.update(self.find_transitive_calls(graph, called, visited.copy()))

        return transitive


class CohesionCalculator:
    """Calculates cohesion metrics for methods in a class."""

    def calculate_cohesion(self, methods: dict[str, MethodInfo]) -> float:
        """
        Calculate overall cohesion score for a class.

        Cohesion is measured as the ratio of method pairs that share variables
        to the total number of method pairs.

        Args:
            methods: Dictionary of method information

        Returns:
            Cohesion score between 0.0 (no cohesion) and 1.0 (perfect cohesion)

        Example:
            >>> calc = CohesionCalculator()
            >>> methods = {
            ...     "m1": MethodInfo(name="m1", line_number=1, variables_used={"var_a", "var_b"}),
            ...     "m2": MethodInfo(name="m2", line_number=2, variables_used={"var_a", "var_c"}),
            ...     "m3": MethodInfo(name="m3", line_number=3, variables_used={"var_d"})
            ... }
            >>> score = calc.calculate_cohesion(methods)
            >>> assert 0.0 <= score <= 1.0
        """
        if len(methods) <= 1:
            return 1.0  # Trivially cohesive

        method_list = list(methods.values())
        pairs_sharing = 0
        total_pairs = 0

        # Check all pairs of methods
        for i in range(len(method_list)):
            for j in range(i + 1, len(method_list)):
                total_pairs += 1

                method_i = method_list[i]
                method_j = method_list[j]

                # Count all variables (used + assigned)
                vars_i = method_i.variables_used | method_i.variables_assigned
                vars_j = method_j.variables_used | method_j.variables_assigned

                # Check if they share any variables
                if vars_i & vars_j:
                    pairs_sharing += 1

        if total_pairs == 0:
            return 1.0

        return pairs_sharing / total_pairs

    def calculate_group_cohesion(
        self,
        method_group: list[str],
        all_methods: dict[str, MethodInfo],
    ) -> float:
        """
        Calculate cohesion score for a specific group of methods.

        Args:
            method_group: List of method names in the group
            all_methods: All methods in the class

        Returns:
            Cohesion score for this specific group
        """
        group_methods = {
            name: info for name, info in all_methods.items()
            if name in method_group
        }
        return self.calculate_cohesion(group_methods)


class PatternDetector:
    """Detects common patterns across multiple classes."""

    def __init__(self) -> None:
        """Initialize pattern detector with known pattern signatures."""
        self.pattern_signatures = {
            "validation": [
                "validate", "check", "verify", "ensure", "schema",
                "pandera", "isinstance", "assert", "raise"
            ],
            "lifecycle": [
                "init", "start", "stop", "close", "open", "setup",
                "teardown", "on_start", "on_stop", "__init__", "__del__"
            ],
            "database": [
                "execute", "fetch", "query", "insert", "update",
                "delete", "select", "commit", "rollback", "connection"
            ],
            "error_handling": [
                "try", "except", "finally", "raise", "log_error",
                "handle_error", "error", "exception", "exc_info"
            ],
            "metrics": [
                "counter", "gauge", "histogram", "prometheus",
                "metric", "increment", "observe", "set_gauge"
            ]
        }

    def detect_patterns_in_class(self, analysis: ClassAnalysis) -> list[str]:
        """
        Detect patterns present in a single class.

        Args:
            analysis: ClassAnalysis object for the class

        Returns:
            List of pattern names detected
        """
        detected = []

        # Check method names and docstrings for pattern signatures
        for pattern_name, keywords in self.pattern_signatures.items():
            pattern_found = False

            for method_info in analysis.methods.values():
                method_text = method_info.name.lower()
                if method_info.docstring:
                    method_text += " " + method_info.docstring.lower()

                # Check if any keywords are present
                if any(keyword in method_text for keyword in keywords):
                    pattern_found = True
                    break

            if pattern_found:
                detected.append(pattern_name)

        return detected

    def find_shared_patterns(
        self,
        analyses: list[ClassAnalysis],
        min_classes: int = 2,
    ) -> list[SharedPattern]:
        """
        Find patterns shared across multiple classes.

        Args:
            analyses: List of class analyses
            min_classes: Minimum number of classes that must have a pattern

        Returns:
            List of SharedPattern objects

        Example:
            >>> detector = PatternDetector()
            >>> # Assume we have analyses for 3 classes
            >>> patterns = detector.find_shared_patterns(analyses, min_classes=2)
            >>> # Should find patterns present in at least 2 classes
        """
        # Track which classes have which patterns
        pattern_to_classes: dict[str, list[str]] = defaultdict(list)

        for analysis in analyses:
            detected = self.detect_patterns_in_class(analysis)
            for pattern in detected:
                pattern_to_classes[pattern].append(analysis.class_name)

        # Filter to patterns present in at least min_classes
        shared = []
        for pattern_name, classes in pattern_to_classes.items():
            if len(classes) >= min_classes:
                shared_pattern = SharedPattern(
                    pattern_name=pattern_name,
                    pattern_type=pattern_name,
                    affected_classes=classes,
                    description=self._generate_pattern_description(pattern_name, classes),
                    extraction_recommendation=self._generate_extraction_recommendation(pattern_name),
                    priority=len(classes),  # More classes = higher priority
                )
                shared.append(shared_pattern)

        # Sort by priority (descending)
        shared.sort(key=lambda p: p.priority, reverse=True)

        return shared

    def _generate_pattern_description(self, pattern_name: str, classes: list[str]) -> str:
        """Generate a description for a shared pattern."""
        return (
            f"The '{pattern_name}' pattern appears in {len(classes)} classes: "
            f"{', '.join(classes)}. This pattern handles {pattern_name}-related "
            f"responsibilities that could be centralized."
        )

    def _generate_extraction_recommendation(self, pattern_name: str) -> str:
        """Generate extraction recommendation for a pattern."""
        recommendations = {
            "validation": "Extract to ml/common/validation.py - centralized schema validation utilities",
            "lifecycle": "Extract to ml/common/lifecycle.py - shared lifecycle management patterns",
            "database": "Extract to ml/common/database.py - database operation helpers",
            "error_handling": "Extract to ml/common/error_handlers.py - standardized error handling",
            "metrics": "Already centralized in ml/common/metrics_bootstrap.py - ensure usage"
        }
        return recommendations.get(
            pattern_name,
            f"Extract to ml/common/{pattern_name}.py - shared {pattern_name} utilities"
        )


class ReportGenerator:
    """Generates markdown reports from analysis results."""

    def generate_responsibility_catalog(
        self,
        analyses: list[ClassAnalysis],
        output_path: Path,
    ) -> None:
        """
        Generate comprehensive responsibility catalog report.

        Args:
            analyses: List of class analyses
            output_path: Path to write the report

        Example:
            >>> generator = ReportGenerator()
            >>> generator.generate_responsibility_catalog(analyses, Path("report.md"))
        """
        lines = [
            "# God Class Responsibility Catalog",
            "",
            "Generated analysis of responsibilities across all god classes.",
            "",
            f"**Total Classes Analyzed:** {len(analyses)}",
            "",
            "---",
            ""
        ]

        for analysis in sorted(analyses, key=lambda a: a.line_count, reverse=True):
            lines.extend([
                f"## {analysis.class_name}",
                "",
                f"**File:** `{analysis.file_path}`",
                f"**Lines:** {analysis.line_count}",
                f"**Methods:** {analysis.method_count}",
                f"**Overall Cohesion:** {analysis.cohesion_metrics.get('overall', 0.0):.2f}",
                "",
                "### Responsibilities",
                ""
            ])

            if analysis.responsibilities:
                for resp in analysis.responsibilities:
                    lines.extend([
                        f"#### {resp.name}",
                        "",
                        resp.description,
                        "",
                        f"**Methods ({len(resp.methods)}):** {', '.join(resp.methods)}",
                        f"**Cohesion:** {resp.cohesion_score:.2f}",
                        f"**Extraction:** {resp.extraction_recommendation}",
                        ""
                    ])
            else:
                lines.append("*No distinct responsibilities identified*")
                lines.append("")

            lines.extend([
                "### Method Groups",
                ""
            ])

            if analysis.method_groups:
                for group_name, methods in analysis.method_groups.items():
                    lines.append(f"- **{group_name}:** {', '.join(methods)}")
                lines.append("")
            else:
                lines.append("*No method groups identified*")
                lines.append("")

            lines.append("---")
            lines.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def generate_cross_class_pattern_report(
        self,
        patterns: list[SharedPattern],
        output_path: Path,
    ) -> None:
        """
        Generate cross-class pattern analysis report.

        Args:
            patterns: List of shared patterns
            output_path: Path to write the report
        """
        lines = [
            "# Cross-Class Pattern Analysis",
            "",
            "Patterns shared across multiple god classes.",
            "",
            f"**Total Patterns Found:** {len(patterns)}",
            "",
            "---",
            ""
        ]

        if patterns:
            for pattern in patterns:
                lines.extend([
                    f"## Pattern: {pattern.pattern_name}",
                    "",
                    f"**Type:** {pattern.pattern_type}",
                    f"**Classes Affected:** {len(pattern.affected_classes)}",
                    f"**Priority:** {pattern.priority}",
                    "",
                    "### Description",
                    "",
                    pattern.description,
                    "",
                    "### Affected Classes",
                    ""
                ])

                for class_name in pattern.affected_classes:
                    lines.append(f"- {class_name}")

                lines.extend([
                    "",
                    "### Extraction Recommendation",
                    "",
                    pattern.extraction_recommendation,
                    "",
                    "---",
                    ""
                ])
        else:
            lines.extend([
                "*No shared patterns identified across classes.*",
                ""
            ])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def generate_shared_utilities_report(
        self,
        patterns: list[SharedPattern],
        output_path: Path,
    ) -> None:
        """
        Generate shared utility extraction candidates report.

        Args:
            patterns: List of shared patterns (sorted by priority)
            output_path: Path to write the report
        """
        lines = [
            "# Shared Utility Extraction Candidates",
            "",
            "Prioritized list of utilities to extract from god classes.",
            "",
            f"**Total Candidates:** {len(patterns)}",
            "",
            "Candidates are sorted by priority (number of affected classes).",
            "",
            "---",
            ""
        ]

        if patterns:
            for i, pattern in enumerate(patterns, 1):
                lines.extend([
                    f"## {i}. {pattern.pattern_name} (Priority: {pattern.priority})",
                    "",
                    f"**Affects:** {len(pattern.affected_classes)} classes",
                    "",
                    "### Description",
                    "",
                    pattern.description,
                    "",
                    "### Extraction Target",
                    "",
                    pattern.extraction_recommendation,
                    "",
                    "### Impact",
                    "",
                    f"Extracting this utility would simplify {len(pattern.affected_classes)} classes "
                    f"and eliminate duplication.",
                    "",
                    "---",
                    ""
                ])
        else:
            lines.extend([
                "*No shared utility candidates identified.*",
                ""
            ])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def generate_extraction_strategy(
        self,
        analyses: list[ClassAnalysis],
        output_path: Path,
        title: str,
        description: str,
    ) -> None:
        """
        Generate extraction strategy report.

        Args:
            analyses: List of class analyses to include
            output_path: Path to write the report
            title: Report title
            description: Report description
        """
        lines = [
            f"# {title}",
            "",
            description,
            "",
            f"**Classes Covered:** {len(analyses)}",
            "",
            "---",
            ""
        ]

        for analysis in sorted(analyses, key=lambda a: a.line_count, reverse=True):
            lines.extend([
                f"## {analysis.class_name}",
                "",
                f"**File:** `{analysis.file_path}`",
                f"**Lines:** {analysis.line_count}",
                f"**Methods:** {analysis.method_count}",
                "",
                "### Extraction Approach",
                ""
            ])

            if analysis.responsibilities:
                lines.append("**Recommended Extractions:**")
                lines.append("")
                for resp in analysis.responsibilities:
                    lines.extend([
                        f"- **{resp.name}:** {resp.extraction_recommendation}",
                    ])
                lines.append("")
            else:
                lines.append("*Further analysis needed to identify extraction opportunities.*")
                lines.append("")

            lines.extend([
                "### Dependencies",
                "",
                f"Methods call {len(analysis.call_graph)} other methods internally.",
                "",
                "---",
                ""
            ])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")


class GodClassAnalyzer:
    """Main analyzer for god class decomposition analysis."""

    def __init__(self) -> None:
        """Initialize analyzer with component instances."""
        self.grouper = MethodGrouper()
        self.call_graph_builder = CallGraphBuilder()
        self.cohesion_calculator = CohesionCalculator()
        self.pattern_detector = PatternDetector()
        self.report_generator = ReportGenerator()

    def analyze_file(self, file_path: Path) -> ClassAnalysis:
        """
        Analyze a single Python file containing a god class.

        Args:
            file_path: Path to Python file

        Returns:
            ClassAnalysis object with complete analysis

        Raises:
            FileNotFoundError: If file does not exist
            SyntaxError: If file contains invalid Python syntax

        Example:
            >>> analyzer = GodClassAnalyzer()
            >>> analysis = analyzer.analyze_file(Path("my_class.py"))
            >>> print(f"Found {analysis.method_count} methods")
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read and parse file
        source_code = file_path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source_code, filename=str(file_path))
        except SyntaxError as e:
            raise SyntaxError(
                f"Invalid Python syntax in {file_path} at line {e.lineno}: {e.msg}"
            ) from e

        # Extract class information
        class_node = self._find_main_class(tree)

        if class_node is None:
            # No class found - return empty analysis
            return ClassAnalysis(
                class_name="<empty>",
                file_path=file_path,
                line_count=len(source_code.splitlines()),
                method_count=0,
                methods={},
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            )

        # Analyze methods
        methods = self._extract_methods(class_node)

        # Build method groups
        method_names = list(methods.keys())
        method_groups = self.grouper.group_by_pattern(method_names)

        # Build call graph
        call_graph = self.call_graph_builder.build_call_graph(methods)

        # Calculate cohesion
        overall_cohesion = self.cohesion_calculator.calculate_cohesion(methods)

        # Identify responsibilities from method groups
        responsibilities = self._identify_responsibilities(methods, method_groups)

        # Create analysis object
        analysis = ClassAnalysis(
            class_name=class_node.name,
            file_path=file_path,
            line_count=len(source_code.splitlines()),
            method_count=len(methods),
            methods=methods,
            method_groups=method_groups,
            call_graph=call_graph,
            cohesion_metrics={"overall": overall_cohesion},
            responsibilities=responsibilities,
            patterns=[]  # Will be filled by pattern detector
        )

        # Detect patterns
        analysis.patterns = self.pattern_detector.detect_patterns_in_class(analysis)

        return analysis

    def analyze_all(
        self,
        file_paths: list[Path],
        min_shared_pattern_classes: int = 2,
    ) -> tuple[list[ClassAnalysis], list[SharedPattern]]:
        """
        Analyze multiple files and find cross-class patterns.

        Args:
            file_paths: List of Python file paths to analyze
            min_shared_pattern_classes: Minimum classes for a pattern to be "shared"

        Returns:
            Tuple of (analyses, shared_patterns)

        Example:
            >>> analyzer = GodClassAnalyzer()
            >>> paths = [Path("class1.py"), Path("class2.py")]
            >>> analyses, patterns = analyzer.analyze_all(paths)
            >>> print(f"Found {len(patterns)} shared patterns")
        """
        analyses = []

        for file_path in file_paths:
            try:
                analysis = self.analyze_file(file_path)
                analyses.append(analysis)
            except Exception as e:
                # Log error but continue with other files
                print(f"Error analyzing {file_path}: {e}")

        # Find shared patterns
        shared_patterns = self.pattern_detector.find_shared_patterns(
            analyses,
            min_classes=min_shared_pattern_classes
        )

        return analyses, shared_patterns

    def generate_all_reports(
        self,
        analyses: list[ClassAnalysis],
        shared_patterns: list[SharedPattern],
        output_dir: Path,
    ) -> None:
        """
        Generate all analysis reports.

        Args:
            analyses: List of class analyses
            shared_patterns: List of shared patterns
            output_dir: Directory to write reports
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Responsibility catalog
        self.report_generator.generate_responsibility_catalog(
            analyses,
            output_dir / "god_class_responsibility_catalog.md"
        )

        # 2. Cross-class patterns
        self.report_generator.generate_cross_class_pattern_report(
            shared_patterns,
            output_dir / "cross_class_pattern_analysis.md"
        )

        # 3. Shared utilities
        self.report_generator.generate_shared_utilities_report(
            shared_patterns,
            output_dir / "shared_utility_candidates.md"
        )

        # 4. Phase 2 strategy (Tier 0-1: largest 8 classes)
        tier_0_1 = sorted(analyses, key=lambda a: a.line_count, reverse=True)[:8]
        self.report_generator.generate_extraction_strategy(
            tier_0_1,
            output_dir / "phase2_extraction_strategy.md",
            title="Phase 2 Extraction Strategy (Tier 0-1)",
            description="Extraction plan for the 8 largest god classes (Phase 2.1-2.6)"
        )

        # 5. Phase 3 strategy (Tier 2-3: remaining classes)
        tier_2_3 = sorted(analyses, key=lambda a: a.line_count, reverse=True)[8:]
        self.report_generator.generate_extraction_strategy(
            tier_2_3,
            output_dir / "phase3_extraction_strategy.md",
            title="Phase 3 Extraction Strategy (Tier 2-3)",
            description="Extraction plan for remaining god classes (Phase 3)"
        )

    def _find_main_class(self, tree: ast.Module) -> ast.ClassDef | None:
        """
        Find the main (largest) class definition in an AST.

        Returns the class with the most methods, assuming that's the god class.
        """
        classes = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(node)

        if not classes:
            return None

        # Return the class with the most methods (the "god class")
        return max(classes, key=lambda c: sum(1 for n in c.body if isinstance(n, ast.FunctionDef)))

    def _extract_methods(self, class_node: ast.ClassDef) -> dict[str, MethodInfo]:
        """Extract all methods from a class node."""
        methods = {}

        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_info = self._analyze_method(node)
                methods[node.name] = method_info

        return methods

    def _analyze_method(self, func_node: ast.FunctionDef) -> MethodInfo:
        """Analyze a single method node."""
        # Extract method calls
        calls = self._extract_method_calls(func_node)

        # Extract variable usage
        variables_used, variables_assigned = self._extract_variable_usage(func_node)

        # Check if private
        is_private = func_node.name.startswith("_") and not func_node.name.startswith("__")

        # Extract decorators
        decorators = [
            self._get_decorator_name(dec)
            for dec in func_node.decorator_list
        ]

        # Extract parameters
        parameters = [arg.arg for arg in func_node.args.args if arg.arg != "self"]

        # Extract return type
        return_type = None
        if func_node.returns:
            return_type = ast.unparse(func_node.returns)

        # Extract docstring
        docstring = ast.get_docstring(func_node)

        # Calculate complexity (rough estimate: count if/for/while/try)
        complexity = sum(
            1 for node in ast.walk(func_node)
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try))
        )

        return MethodInfo(
            name=func_node.name,
            line_number=func_node.lineno,
            calls=calls,
            variables_used=variables_used,
            variables_assigned=variables_assigned,
            is_private=is_private,
            decorators=decorators,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            complexity=complexity
        )

    def _extract_method_calls(self, func_node: ast.FunctionDef) -> list[str]:
        """Extract method calls from a function."""
        calls = []

        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                # Check for self.method_name() calls
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                        calls.append(node.func.attr)

        return calls

    def _extract_variable_usage(self, func_node: ast.FunctionDef) -> tuple[set[str], set[str]]:
        """Extract variables used and assigned in a function."""
        used = set()
        assigned = set()

        for node in ast.walk(func_node):
            # Variables used (self.variable)
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    used.add(node.attr)

            # Variables assigned (self.variable = ...)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name) and target.value.id == "self":
                            assigned.add(target.attr)

        return used, assigned

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name as string."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        else:
            return "<unknown>"

    def _identify_responsibilities(
        self,
        methods: dict[str, MethodInfo],
        method_groups: dict[str, list[str]],
    ) -> list[ResponsibilityGroup]:
        """Identify distinct responsibilities from method groups."""
        responsibilities = []

        # Map semantic prefixes to responsibility names
        responsibility_mapping = {
            "prefix:read": "Data Reading",
            "prefix:write": "Data Writing",
            "prefix:validate": "Validation",
            "prefix:process": "Data Processing",
            "prefix:create": "Object Creation",
            "prefix:init": "Initialization",
            "prefix:fetch": "Data Fetching",
            "prefix:load": "Data Loading",
            "prefix:save": "Data Persistence",
            "prefix:emit": "Event Emission",
            "prefix:publish": "Message Publishing",
            "suffix:store": "Store Operations",
            "suffix:db": "Database Operations",
            "suffix:event": "Event Handling",
            "suffix:metrics": "Metrics Collection",
            "suffix:schema": "Schema Management",
        }

        for group_key, method_list in method_groups.items():
            if len(method_list) < 2:
                continue  # Skip groups with only 1 method

            resp_name = responsibility_mapping.get(group_key, group_key.replace("prefix:", "").replace("suffix:", "").title())

            # Calculate cohesion for this group
            cohesion = self.cohesion_calculator.calculate_group_cohesion(method_list, methods)

            # Generate description
            description = f"Handles {resp_name.lower()} responsibilities with {len(method_list)} methods."

            # Generate extraction recommendation
            extraction = self._generate_extraction_recommendation(resp_name, method_list)

            responsibilities.append(ResponsibilityGroup(
                name=resp_name,
                methods=method_list,
                description=description,
                cohesion_score=cohesion,
                extraction_recommendation=extraction
            ))

        return responsibilities

    def _generate_extraction_recommendation(self, responsibility_name: str, methods: list[str]) -> str:
        """Generate extraction recommendation for a responsibility."""
        return (
            f"Extract {len(methods)} methods to a dedicated component "
            f"for {responsibility_name.lower()}."
        )
