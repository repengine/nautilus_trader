"""Unit tests for god class analyzer components."""

from pathlib import Path
from textwrap import dedent

import pytest

from ml.analysis.god_class_analyzer import (
    CallGraphBuilder,
    CohesionCalculator,
    GodClassAnalyzer,
    MethodGrouper,
    MethodInfo,
    PatternDetector,
)


class TestMethodGrouper:
    """Tests for MethodGrouper component."""

    def test_method_grouping_by_prefix(self) -> None:
        """Verify that methods with common prefixes are grouped together."""
        grouper = MethodGrouper()
        method_names = ["read_data", "read_features", "write_data", "write_features"]

        groups = grouper.group_by_prefix(method_names)

        assert "read" in groups
        assert "write" in groups
        assert len(groups["read"]) == 2
        assert len(groups["write"]) == 2
        assert "read_data" in groups["read"]
        assert "read_features" in groups["read"]
        assert "write_data" in groups["write"]
        assert "write_features" in groups["write"]

    def test_method_grouping_by_suffix(self) -> None:
        """Verify that methods with common suffixes are grouped together."""
        grouper = MethodGrouper()
        method_names = ["read_data", "write_data", "parse_data", "read_features"]

        groups = grouper.group_by_suffix(method_names)

        assert "data" in groups
        assert "features" in groups
        assert len(groups["data"]) == 3
        assert "read_data" in groups["data"]
        assert "write_data" in groups["data"]
        assert "parse_data" in groups["data"]

    def test_method_grouping_no_common_prefixes(self) -> None:
        """Verify grouping when no methods share prefixes."""
        grouper = MethodGrouper()
        method_names = ["foo", "bar", "baz"]

        groups = grouper.group_by_prefix(method_names)

        # No recognized prefixes, so groups should be empty
        assert len(groups) == 0

    def test_method_grouping_by_pattern_combines_prefix_and_suffix(self) -> None:
        """Verify pattern grouping combines prefix and suffix groups."""
        grouper = MethodGrouper()
        method_names = ["read_data", "write_data", "read_features"]

        groups = grouper.group_by_pattern(method_names)

        # Should have both prefix and suffix groups
        assert "prefix:read" in groups
        assert "prefix:write" in groups
        assert "suffix:data" in groups
        assert "suffix:features" in groups


class TestCallGraphBuilder:
    """Tests for CallGraphBuilder component."""

    def test_call_graph_analysis_basic(self) -> None:
        """Verify call graph correctly identifies direct method calls."""
        builder = CallGraphBuilder()
        methods = {
            "method_a": MethodInfo(name="method_a", line_number=10, calls=["method_b", "method_c"]),
            "method_b": MethodInfo(name="method_b", line_number=20, calls=[]),
            "method_c": MethodInfo(name="method_c", line_number=30, calls=[])
        }

        graph = builder.build_call_graph(methods)

        assert "method_a" in graph
        assert "method_b" in graph
        assert "method_c" in graph
        assert graph["method_a"] == ["method_b", "method_c"]
        assert graph["method_b"] == []
        assert graph["method_c"] == []

    def test_call_graph_analysis_transitive(self) -> None:
        """Verify transitive call analysis works correctly."""
        builder = CallGraphBuilder()
        graph = {
            "method_a": ["method_b"],
            "method_b": ["method_c"],
            "method_c": []
        }

        transitive = builder.find_transitive_calls(graph, "method_a")

        assert "method_b" in transitive
        assert "method_c" in transitive
        assert len(transitive) == 2

    def test_call_graph_analysis_recursive_calls(self) -> None:
        """Verify analyzer handles recursive method calls."""
        builder = CallGraphBuilder()
        methods = {
            "method_a": MethodInfo(name="method_a", line_number=10, calls=["method_a"])
        }

        graph = builder.build_call_graph(methods)

        assert graph["method_a"] == ["method_a"]
        # Ensure no infinite loop when finding transitive calls
        transitive = builder.find_transitive_calls(graph, "method_a")
        assert "method_a" not in transitive  # Should detect cycle and stop

    def test_call_graph_analysis_circular_calls(self) -> None:
        """Verify analyzer handles circular call chains."""
        builder = CallGraphBuilder()
        graph = {
            "method_a": ["method_b"],
            "method_b": ["method_a"]
        }

        # Should not infinite loop
        transitive_a = builder.find_transitive_calls(graph, "method_a")
        transitive_b = builder.find_transitive_calls(graph, "method_b")

        assert "method_b" in transitive_a
        assert "method_a" in transitive_b


class TestCohesionCalculator:
    """Tests for CohesionCalculator component."""

    def test_cohesion_metrics_calculation(self) -> None:
        """Verify cohesion metrics calculation is correct."""
        calc = CohesionCalculator()

        # High cohesion class (methods share variables)
        high_cohesion_methods = {
            "method_a": MethodInfo(
                name="method_a",
                line_number=1,
                variables_used={"var_a", "var_b"}
            ),
            "method_b": MethodInfo(
                name="method_b",
                line_number=2,
                variables_used={"var_a", "var_c"}
            ),
            "method_c": MethodInfo(
                name="method_c",
                line_number=3,
                variables_used={"var_b", "var_c"}
            )
        }

        high_score = calc.calculate_cohesion(high_cohesion_methods)
        assert 0.0 <= high_score <= 1.0
        assert high_score >= 0.7  # All methods share at least one variable

        # Low cohesion class (methods are independent)
        low_cohesion_methods = {
            "method_a": MethodInfo(
                name="method_a",
                line_number=1,
                variables_used={"var_a"}
            ),
            "method_b": MethodInfo(
                name="method_b",
                line_number=2,
                variables_used={"var_b"}
            ),
            "method_c": MethodInfo(
                name="method_c",
                line_number=3,
                variables_used={"var_c"}
            )
        }

        low_score = calc.calculate_cohesion(low_cohesion_methods)
        assert 0.0 <= low_score <= 1.0
        assert low_score < 0.3  # No methods share variables

    def test_cohesion_metrics_single_method_class(self) -> None:
        """Verify cohesion calculation for trivial class."""
        calc = CohesionCalculator()
        methods = {
            "method_a": MethodInfo(name="method_a", line_number=1)
        }

        score = calc.calculate_cohesion(methods)
        assert score == 1.0  # Trivially cohesive

    def test_variable_usage_clustering(self) -> None:
        """Verify methods are clustered by variable usage."""
        calc = CohesionCalculator()

        # Methods that use same variables should have high cohesion
        methods = {
            "method_a": MethodInfo(
                name="method_a",
                line_number=1,
                variables_used={"shared_var_1", "shared_var_2"}
            ),
            "method_b": MethodInfo(
                name="method_b",
                line_number=2,
                variables_used={"shared_var_1", "shared_var_2"}
            )
        }

        score = calc.calculate_cohesion(methods)
        assert score == 1.0  # Perfect cohesion

    def test_calculate_group_cohesion(self) -> None:
        """Verify group cohesion calculation."""
        calc = CohesionCalculator()

        all_methods = {
            "method_a": MethodInfo(
                name="method_a",
                line_number=1,
                variables_used={"var_a"}
            ),
            "method_b": MethodInfo(
                name="method_b",
                line_number=2,
                variables_used={"var_a"}
            ),
            "method_c": MethodInfo(
                name="method_c",
                line_number=3,
                variables_used={"var_b"}
            )
        }

        # Group with methods a and b should have high cohesion
        group_score = calc.calculate_group_cohesion(["method_a", "method_b"], all_methods)
        assert group_score == 1.0


class TestPatternDetector:
    """Tests for PatternDetector component."""

    def test_pattern_detection_validation(self) -> None:
        """Verify detection of shared validation logic patterns."""
        from ml.analysis.god_class_analyzer import ClassAnalysis

        detector = PatternDetector()

        # Create mock analyses with validation patterns
        analyses = [
            ClassAnalysis(
                class_name="ClassA",
                file_path=Path("class_a.py"),
                line_count=100,
                method_count=5,
                methods={
                    "validate_schema": MethodInfo(
                        name="validate_schema",
                        line_number=10,
                        docstring="Validates schema using Pandera"
                    )
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            ),
            ClassAnalysis(
                class_name="ClassB",
                file_path=Path("class_b.py"),
                line_count=200,
                method_count=8,
                methods={
                    "validate_model": MethodInfo(
                        name="validate_model",
                        line_number=15,
                        docstring="Validates model schema"
                    )
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            ),
            ClassAnalysis(
                class_name="ClassC",
                file_path=Path("class_c.py"),
                line_count=150,
                method_count=6,
                methods={
                    "check_features": MethodInfo(
                        name="check_features",
                        line_number=20,
                        docstring="Validates feature schema"
                    )
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            )
        ]

        patterns = detector.find_shared_patterns(analyses, min_classes=2)

        # Should find validation pattern in all 3 classes
        validation_patterns = [p for p in patterns if p.pattern_type == "validation"]
        assert len(validation_patterns) > 0
        assert len(validation_patterns[0].affected_classes) == 3

    def test_pattern_detection_lifecycle(self) -> None:
        """Verify detection of lifecycle management patterns."""
        from ml.analysis.god_class_analyzer import ClassAnalysis

        detector = PatternDetector()

        analyses = [
            ClassAnalysis(
                class_name="ClassA",
                file_path=Path("class_a.py"),
                line_count=100,
                method_count=3,
                methods={
                    "on_start": MethodInfo(name="on_start", line_number=10)
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            ),
            ClassAnalysis(
                class_name="ClassB",
                file_path=Path("class_b.py"),
                line_count=200,
                method_count=4,
                methods={
                    "on_stop": MethodInfo(name="on_stop", line_number=15)
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            )
        ]

        patterns = detector.find_shared_patterns(analyses, min_classes=2)
        lifecycle_patterns = [p for p in patterns if p.pattern_type == "lifecycle"]
        assert len(lifecycle_patterns) > 0

    def test_pattern_detection_no_patterns_found(self) -> None:
        """Verify analyzer handles case where no shared patterns exist."""
        from ml.analysis.god_class_analyzer import ClassAnalysis

        detector = PatternDetector()

        # Classes with completely different implementations
        analyses = [
            ClassAnalysis(
                class_name="ClassA",
                file_path=Path("class_a.py"),
                line_count=100,
                method_count=2,
                methods={
                    "foo": MethodInfo(name="foo", line_number=10)
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            ),
            ClassAnalysis(
                class_name="ClassB",
                file_path=Path("class_b.py"),
                line_count=200,
                method_count=2,
                methods={
                    "bar": MethodInfo(name="bar", line_number=15)
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            )
        ]

        patterns = detector.find_shared_patterns(analyses, min_classes=2)
        assert len(patterns) == 0

    def test_pattern_detection_single_occurrence(self) -> None:
        """Verify patterns are not flagged if only in 1 class."""
        from ml.analysis.god_class_analyzer import ClassAnalysis

        detector = PatternDetector()

        analyses = [
            ClassAnalysis(
                class_name="ClassA",
                file_path=Path("class_a.py"),
                line_count=100,
                method_count=2,
                methods={
                    "validate_schema": MethodInfo(
                        name="validate_schema",
                        line_number=10,
                        docstring="Validates using Pandera"
                    )
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            ),
            ClassAnalysis(
                class_name="ClassB",
                file_path=Path("class_b.py"),
                line_count=200,
                method_count=2,
                methods={
                    "foo": MethodInfo(name="foo", line_number=15)
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            ),
            ClassAnalysis(
                class_name="ClassC",
                file_path=Path("class_c.py"),
                line_count=150,
                method_count=2,
                methods={
                    "bar": MethodInfo(name="bar", line_number=20)
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            )
        ]

        # Validation pattern only in ClassA, min_classes=2
        patterns = detector.find_shared_patterns(analyses, min_classes=2)
        validation_patterns = [p for p in patterns if p.pattern_type == "validation"]
        assert len(validation_patterns) == 0  # Should not be flagged


class TestGodClassAnalyzer:
    """Tests for main GodClassAnalyzer component."""

    def test_responsibility_detection_empty_class(self, tmp_path: Path) -> None:
        """Verify analyzer handles edge case of empty class gracefully."""
        analyzer = GodClassAnalyzer()

        # Create empty class file
        test_file = tmp_path / "empty_class.py"
        test_file.write_text(dedent("""
            class EmptyClass:
                pass
        """))

        analysis = analyzer.analyze_file(test_file)

        assert analysis.class_name == "EmptyClass"
        assert analysis.method_count == 0
        assert len(analysis.responsibilities) == 0

    def test_responsibility_detection_single_responsibility(self, tmp_path: Path) -> None:
        """Verify analyzer correctly identifies single responsibility."""
        analyzer = GodClassAnalyzer()

        test_file = tmp_path / "single_resp.py"
        test_file.write_text(dedent("""
            class DataReader:
                def read_data(self, path):
                    pass

                def read_features(self, path):
                    pass

                def read_metadata(self, path):
                    pass
        """))

        analysis = analyzer.analyze_file(test_file)

        assert analysis.class_name == "DataReader"
        assert analysis.method_count == 3

        # Should identify "Data Reading" responsibility
        read_responsibilities = [r for r in analysis.responsibilities if "Read" in r.name or "read" in r.name.lower()]
        assert len(read_responsibilities) > 0

    def test_responsibility_detection_multiple_responsibilities(self, tmp_path: Path) -> None:
        """Verify analyzer correctly identifies distinct responsibilities."""
        analyzer = GodClassAnalyzer()

        test_file = tmp_path / "multi_resp.py"
        test_file.write_text(dedent("""
            class MultiResponsibility:
                def read_data(self, path):
                    return self.validate_data(path)

                def read_features(self, path):
                    return self.validate_features(path)

                def write_data(self, data):
                    self.validate_data(data)
                    self.emit_event("data_written")

                def write_features(self, features):
                    self.validate_features(features)
                    self.emit_event("features_written")

                def validate_data(self, data):
                    pass

                def validate_features(self, features):
                    pass

                def emit_event(self, event_type):
                    pass
        """))

        analysis = analyzer.analyze_file(test_file)

        assert analysis.class_name == "MultiResponsibility"
        assert analysis.method_count == 7

        # Should identify multiple responsibilities: Reading, Writing, Validation, Events
        assert len(analysis.responsibilities) >= 3

    def test_analyze_nonexistent_file(self) -> None:
        """Verify analyzer handles missing file gracefully."""
        analyzer = GodClassAnalyzer()

        with pytest.raises(FileNotFoundError, match="File not found"):
            analyzer.analyze_file(Path("/nonexistent/file.py"))

    def test_analyze_class_with_decorators(self, tmp_path: Path) -> None:
        """Verify analyzer handles methods with decorators."""
        analyzer = GodClassAnalyzer()

        test_file = tmp_path / "decorators.py"
        test_file.write_text(dedent("""
            class DecoratedClass:
                @property
                def name(self):
                    return self._name

                @staticmethod
                def helper():
                    pass

                @classmethod
                def factory(cls):
                    return cls()
        """))

        analysis = analyzer.analyze_file(test_file)

        assert analysis.class_name == "DecoratedClass"
        assert analysis.method_count == 3
        # Verify decorators are captured
        assert any("property" in method.decorators for method in analysis.methods.values())

    def test_analyze_class_with_nested_classes(self, tmp_path: Path) -> None:
        """Verify analyzer handles nested class definitions."""
        analyzer = GodClassAnalyzer()

        test_file = tmp_path / "nested.py"
        test_file.write_text(dedent("""
            class OuterClass:
                def outer_method(self):
                    pass

                class InnerClass:
                    def inner_method(self):
                        pass
        """))

        analysis = analyzer.analyze_file(test_file)

        # Should analyze the outer class (first class found)
        assert analysis.class_name == "OuterClass"
        assert "outer_method" in analysis.methods

    def test_analyze_file_invalid_syntax(self, tmp_path: Path) -> None:
        """Verify analyzer handles malformed Python files."""
        analyzer = GodClassAnalyzer()

        test_file = tmp_path / "invalid.py"
        test_file.write_text("def invalid syntax here @#$%")

        with pytest.raises(SyntaxError, match="Invalid Python syntax"):
            analyzer.analyze_file(test_file)

    def test_shared_utility_candidate_identification(self) -> None:
        """Verify shared utility candidates are identified correctly."""
        from ml.analysis.god_class_analyzer import ClassAnalysis

        analyzer = GodClassAnalyzer()

        # Create analyses with shared validation pattern
        analyses = [
            ClassAnalysis(
                class_name="ClassA",
                file_path=Path("class_a.py"),
                line_count=100,
                method_count=3,
                methods={
                    "validate_schema": MethodInfo(
                        name="validate_schema",
                        line_number=10,
                        docstring="Schema validation with Pandera"
                    )
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            ),
            ClassAnalysis(
                class_name="ClassB",
                file_path=Path("class_b.py"),
                line_count=200,
                method_count=4,
                methods={
                    "check_schema": MethodInfo(
                        name="check_schema",
                        line_number=15,
                        docstring="Validates schema"
                    )
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            ),
            ClassAnalysis(
                class_name="ClassC",
                file_path=Path("class_c.py"),
                line_count=150,
                method_count=5,
                methods={
                    "verify_data": MethodInfo(
                        name="verify_data",
                        line_number=20,
                        docstring="Validates data"
                    )
                },
                method_groups={},
                call_graph={},
                cohesion_metrics={},
                responsibilities=[],
                patterns=[]
            )
        ]

        shared_patterns = analyzer.pattern_detector.find_shared_patterns(analyses, min_classes=2)

        # Should identify validation as shared utility candidate
        assert len(shared_patterns) > 0
        validation_patterns = [p for p in shared_patterns if p.pattern_type == "validation"]
        assert len(validation_patterns) > 0
        assert len(validation_patterns[0].affected_classes) == 3
